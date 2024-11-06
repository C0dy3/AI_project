import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.lite.python.schema_py_generated import Tensor
from tensorflow.python.keras.backend import epsilon, update, gradients

from ReplayBuffer import ReplayBuffer
import tensorflow as tf
from tensorflow.python.keras import losses

import matplotlib.pyplot as plt
import gym
import cv2


def train_model(env, model, episodes=500, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, batch_size=32):
    replay_buffer = ReplayBuffer(max_size=1000)
    total_reward_list = []

    for episode in range(episodes):
        # Reset environment and get initial observation
        observation, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()

            # Select action (exploration vs exploitation)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action_values = model.predict(np.array([observation]))[0]
                action = np.argmax(action_values)

            # Discretize action (map continuous action space to discrete)
            discrete_to_continuous_map = {
                0: [-1.0, 0.0, 0.0],
                1: [0.0, 1.0, 0.0],
                2: [1.0, 0.0, 0.0],
            }

            if isinstance(action, (int, np.integer)) and action in discrete_to_continuous_map:
                action = discrete_to_continuous_map[action]
            elif isinstance(action, (int, float, np.integer, np.float32, np.float64)):
                action = [action, 0.0, 0.0]
            elif isinstance(action, (list, np.ndarray)) and len(action) != 3:
                action = [0.0, 0.0, 0.0]
            elif not isinstance(action, (list, np.ndarray)):
                action = [0.0, 0.0, 0.0]

            # Perform action
            new_obs, reward, terminated, truncated, info = env.step(action)  # Gym >=0.26
            new_obs = preprocess_observation(new_obs)

            # Update replay buffer
            replay_buffer.add((observation, action, reward, new_obs, terminated or truncated))
            observation = new_obs
            total_reward += reward

            # Train model if replay buffer has enough samples
            if replay_buffer.size() > batch_size:
                batch = replay_buffer.sample(batch_size)
                update_model(model, batch, gamma)

            # Decay epsilon
            epsilon = max(epsilon * epsilon_decay, min_epsilon)

            # Check if the episode is done (terminated or truncated)
            done = terminated or truncated

        total_reward_list.append(total_reward)
        print(f'Episode {episode + 1} Reward: {total_reward}')

        # Reset environment for the next episode
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
    q_values_target = tf.convert_to_tensor(q_values_target, dtype=tf.float32)


    with tf.GradientTape() as tape:
        q_values = model(states)

        actions = tf.squeeze(actions)
        if len(actions.shape) > 1:
            actions = actions[:, 0]

        q_values_selected = tf.reduce_sum(tf.one_hot(actions, depth=q_values.shape[1]), axis=1)


        loss = losses.mean_squared_error(q_values_target, q_values_selected)



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