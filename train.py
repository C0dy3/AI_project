import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.src.ops import dtype

from ReplayBuffer import ReplayBuffer


def check_out_of_track(observation):
    grass_color = np.array([110, 110, 100])
    threshold = 10
    grass_pixels = np.all(np.abs(observation - grass_color) < threshold, axis=-1)
    grass_ratio = np.mean(grass_pixels)
    return grass_ratio > 0.1


def train_model(env, model, episodes=50, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05,
                batch_size=96):
    replay_buffer = ReplayBuffer(max_size=1000)
    total_reward_list = []

    for episode in range(episodes):
        observation, info = env.reset(options={"randomize": False})
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = [0.0, 0.0, 0.0]
            mapped_action = [0.0, 0.0, 0.0]
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Náhodný výběr
                print("Random action:", action)
            else:
                if np.random.random() < epsilon:
                    action_idx = np.random.randint(0, 10)  # Vyber náhodnou akci
                else:
                    action_values = model.predict(tf.convert_to_tensor(np.array([observation]), dtype=tf.float32))
                    action_idx = np.argmax(action_values)

                # Mapujeme index na příslušné řízení
                action_map = {
                    0: [-1.0, 0.0, 0.0],  # Otáčení vlevo
                    1: [1.0, 0.0, 0.0],  # Otáčení doprava
                    2: [0.0, 1.0, 0.0],  # Plynový pedál (maximální plyn)
                    3: [0.0, 0.0, 1.0],  # Maximální brždění
                    4: [-0.5, 0.5, 0.0],  # Mírné otáčení doleva s plynem
                    5: [0.5, 0.5, 0.0],  # Mírné otáčení doprava s plynem
                    6: [-1.0, 0.5, 0.0],  # Plné otáčení doleva s plynem
                    7: [1.0, 0.5, 0.0],  # Plné otáčení doprava s plynem
                    8: [0.0, 0.0, 0.0],  # Žádný pohyb
                    9: [0.0, 0.0, 0.5],  # Lehké brždění
                }

                mapped_action = action_map.get(action_idx, [0.0, 0.0, 0.0])
                print("Mapped action:", mapped_action)


            new_obs, reward, terminated, truncated, info = env.step(mapped_action)
            track_visited_tile_count = info.get('track_tile_visited_count', 1)
            print("info", info)
            print("Tiles visited:", track_visited_tile_count)
            new_obs = preprocess_observation(new_obs)
            print("Current reward:", reward)

            replay_buffer.add((observation, action, reward, new_obs, terminated or truncated))
            observation = new_obs
            total_reward += reward
            print("Total reward: ", total_reward)

            if replay_buffer.size() > batch_size:
                batch = replay_buffer.sample(batch_size)
                update_model(model, batch, gamma)

            epsilon = max(epsilon * epsilon_decay, min_epsilon)

            done = done or terminated or truncated

        total_reward_list.append(total_reward)
        print("Total reward: ",total_reward)
        print(f'Episode {episode + 1} completed. Total reward: {total_reward}')

    # Ukončit simulaci
    env.close()
    return total_reward_list


def update_model(model, batch, gamma):
    states, actions, rewards, new_states, dones = zip(*batch)

    # Convert to NumPy array pro TensorFlow
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions).astype(int)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    new_states = np.array(new_states, dtype=np.float32)

    with tf.GradientTape() as tape:
        q_values = model(states)
        q_values_next = model(new_states)
        target_q = rewards + gamma * np.max(q_values_next, axis=1) * (1 - dones)
        actions = np.array(actions).reshape(-1)
        action_mask = tf.one_hot(actions, depth=5)
        q_values = tf.reduce_sum(q_values, axis=1)  # Izolujeme Q pro vybrané akce
        loss = keras.losses.mean_squared_error(target_q, q_values)

    # Gradiente a optimalizace
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def preprocess_observation(observation, target_size=(96, 96)):
    observation = cv2.resize(observation, target_size)
    normalized_observation = observation / 255.0
    return tf.convert_to_tensor(normalized_observation, dtype=tf.float32)
