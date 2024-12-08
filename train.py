import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Model

from ReplayBuffer import ReplayBuffer

def train_model(env, model, episodes=500, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05, batch_size=32):
    replay_buffer = ReplayBuffer(max_size=1000)
    total_reward_list = []

    for episode in range(episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0
        previous_position = observation[:2]  # Uchováme počáteční pozici (nebo jiný relevantní parametr)

        while not done:
            env.render()

            # Epsilon-greedy strategie
            if np.random.random() < epsilon:
                action = env.action_space.sample()
                action = np.argmax(action)  # Zaměříme se na diskretizovanou akci
            else:
                action_values = model.predict(np.array([observation]))[0]
                action = np.argmax(action_values)

            # Diskretizace akce (mapování z 0, 1, 2 na konkrétní akce)
            discrete_to_continuous_map = {
                0: [-1.0, 0.5, 0.0],  # Ostřejší zatáčka doleva
                1: [0.0, 1.0, 0.0],   # Plný plyn vpřed
                2: [1.0, 0.5, 0.0],   # Ostřejší zatáčka doprava
            }

            action = discrete_to_continuous_map.get(action, [0.0, 0.0, 0.0])

            # Nový stav a odměna
            new_obs, reward, terminated, truncated, info = env.step(action)
            new_obs = preprocess_observation(new_obs)

            # Předpokládáme, že observation a new_obs obsahují 2D nebo 3D souřadnice pozice (např. [x, y])
            previous_position = observation[:2]  # Uložíme počáteční pozici

            # Kód v smyčce, kde získáme novou pozici
            current_position = new_obs[:2]  # Nová pozice auta (např. x, y)

            # Výpočet vzdálenosti mezi předchozí a novou pozicí
            distance = np.linalg.norm(current_position - previous_position)

            # Pokud se pozice příliš nezměnila (auto se nepohlo)
            if distance < 0.5:
                reward -= 0.5  # Penalizace za stagnaci, auto se nepohlo
                print("Auto se nepohlo. Penalizace.")
            else:
                print("Auto se pohlo.")

            # Dlouhodobější odměny (můžete přidat i pokrok na trati, pokud je to možné)
            reward += 0.05  # Malá pozitivní odměna za pokrok

            # Přidání do Replay Bufferu
            replay_buffer.add((observation, action, reward, new_obs, terminated or truncated))
            observation = new_obs
            total_reward += reward

            # Trénování modelu, pokud máme dostatek dat v Replay Bufferu
            if replay_buffer.size() > batch_size:
                batch = replay_buffer.sample(batch_size)
                update_model(model, batch, gamma)

            # Snižování epsilon pro postupné přecházení od explorace k exploataci
            epsilon = max(epsilon * epsilon_decay, min_epsilon)

            done = terminated or truncated

        total_reward_list.append(total_reward)
        print(f'Episode {episode + 1} Reward: {total_reward}')

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
