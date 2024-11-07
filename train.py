import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Model, load_model

from ReplayBuffer import ReplayBuffer

def save_model(model, epoch, project_path="./"):
    model_save_path = os.path.join(project_path, 'models', f'model_checkpoint_epoch_{epoch}.h5')
    if not os.path.exists(os.path.join(project_path, 'models')):
        os.makedirs(os.path.join(project_path, 'models'))
    model.save(model_save_path)
    print(f'Model saved at {model_save_path}')

def load_trained_model(project_path="./", epoch=10):
    model_path = os.path.join(project_path, 'models', f'model_checkpoint_epoch_{epoch}.h5')
    model = load_model(model_path)
    print(f'Model loaded from {model_path}')
    return model


def train_model(env, model, episodes=500, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, batch_size=32, project_path="./"):
    replay_buffer = ReplayBuffer(max_size=1000)
    total_reward_list = []

    for episode in range(episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()

            if np.random.random() < epsilon:
                action = env.action_space.sample()
                action = np.argmax(action)
            else:
                action_values = model.predict(np.array([observation]))[0]
                action = np.argmax(action_values)

            # Discretize action (map continuous action space to discrete)
            discrete_to_continuous_map = {
                0: [-1.0, 0.0, 0.0],
                1: [0.0, 1.0, 0.0],
                2: [1.0, 0.0, 0.0],
            }

            action = discrete_to_continuous_map.get(action, [0.0, 0.0, 0.0])

            new_obs, reward, terminated, truncated, info = env.step(action)
            new_obs = preprocess_observation(new_obs)

            replay_buffer.add((observation, action, reward, new_obs, terminated or truncated))
            observation = new_obs
            total_reward += reward

            if replay_buffer.size() > batch_size:
                batch = replay_buffer.sample(batch_size)
                update_model(model, batch, gamma)

            epsilon = max(epsilon * epsilon_decay, min_epsilon)

            done = terminated or truncated

        total_reward_list.append(total_reward)
        print(f'Episode {episode + 1} Reward: {total_reward}')
        save_model(model, episode + 1, project_path)
        print("Episode finished. Resetting environment...")
        observation, info = env.reset()

    env.close()
    return total_reward_list

def update_model(model, batch, gamma):
    states, actions, rewards, new_states, dones = zip(*batch)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    new_states = np.array(new_states, dtype=np.float32)

    q_values_next = model.predict(new_states)
    q_values_target = rewards + gamma * np.max(q_values_next, axis=1) * (1 - dones)

    states = tf.convert_to_tensor(states, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(states)

        q_values = model(states)
        q_values = tf.reduce_sum(q_values, axis=-1)
        q_values_target = tf.tile(tf.reshape(q_values_target, [-1, 1]), [1, q_values.shape[0]])
        q_values_selected = tf.reduce_sum(tf.one_hot(actions, depth=q_values.shape[0]) * q_values, axis=1)
        loss = tf.reduce_mean(tf.square(q_values_target - q_values_selected))

    grads = tape.gradient(loss, model.trainable_variables)

    if grads is None or all(grad is None for grad in grads):
        print("Warning: No gradients provided!")
        return loss

    optimizer = Adam(learning_rate=0.001)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def plot_rewards(total_rewards):
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards vs Episodes')
    plt.show()

def preprocess_observation(observation, target_size=(96, 96)):
    observation = np.array(observation)
    resized_observation = cv2.resize(observation, target_size)
    normalized_observation = resized_observation / 255.0
    return normalized_observation
