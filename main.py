import os

import gym

from model import create_model
from train import train_model
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    env = gym.make('CarRacing-v2', render_mode="human")
    model = create_model()
    train_model(env, model)

if __name__ == '__main__':

    main()