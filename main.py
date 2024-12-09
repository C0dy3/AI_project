import os
import tensorflow as tf
import gym

from model import create_model
from train import train_model


def main():
    tf.config.list_physical_devices('GPU')
    env = gym.make('CarRacing-v2', render_mode="human", continuous=True)

    model = create_model()
    train_model(env, model)

if __name__ == '__main__':

    main()