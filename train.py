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
    replay_buffer = ReplayBuffer(max_size=10000)
    total_reward_list = []


    for episodes in range(episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0


        while not done:
            env.render()


            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model.predict(np.array([observation]))[0]
                steering = np.clip(action[0], -1, 1)
                throttle = np.clip(action[1], 0.1, 1)
                brake = np.clip(action[2], 0.1, 1)

                action = [steering, throttle, brake]
                action = np.array(action, dtype=np.float32)


            new_obs, reward, done, info, truncated = env.step(action)
            new_obs = preprocess_observation(new_obs)
            replay_buffer.add((observation, action, reward, new_obs, done))
            observation = new_obs
            total_reward += reward

            if replay_buffer.size() > batch_size:
                batch = replay_buffer.sample(batch_size)
                update_model(model, batch, gamma)

            epsilon = max(epsilon * epsilon_decay, min_epsilon)


        total_reward_list.append(total_reward)
        print(f'Episode {episodes+1} Reward: {total_reward}')
        env.close()

    return total_reward_list

def update_model(model, batch, gamma):
    states, actions, rewards, new_states, dones = zip(*batch)


    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards)
    dones = np.array(dones)
    new_states = np.array(new_states, dtype=np.float32)


    q_values_next = model.predict(new_states)

    q_values_target = rewards + gamma * np.max(q_values_next, axis=1) * (1 - dones)


    states = tf.convert_to_tensor(states, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(states)

        q_values = model(states)


        print(f"q_values shape: {q_values.shape}")
        print(f"actions shape: {actions.shape}")
        print(f"q_values_target shape: {q_values_target.shape}")


        q_values_selected = tf.reduce_sum(q_values * tf.one_hot(actions, 3), axis=1)
        loss = losses.mean_squared_error(q_values_target, q_values_selected)

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


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