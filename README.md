# Car Racing AI Project

This project implements a reinforcement learning agent using Q-learning to play the Car Racing environment from OpenAI's Gym. The agent learns to control a car in a simulated environment, aiming to stay on the track and achieve the highest reward.

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Commands](#commands)

## Installation

<li>To run this project, you'll need to have Python 3.12 installed along with the following dependencies:</li>
<li>You need to have swing installed on you system path to compile some C++ code</li>
<li>Alongsie with C++/C compiler to compile pygame assets</li>

## Requirements
<li>Tensorflow 2.18.0</li>
<li>Keras 3.7.0 <b>!! Need to by installed with tensorflow !!</b></li>
<li>Gym 0.26.1</li>
<li>Cuda 12.6</li>
<li>Swig</li>
<li><b>All can be found in requirements.txt file</b></li>

## Commands

1. Clone this repository experimental branch:
 ```bash
git clone --branch experimental https://github.com/C0dy3/AI_project.git
cd AI_project
```
2.  Set up a virtual environment and activate it:
```bash
python -m venv .venv
.\venv\Scirpts\activate
```
3. Install the required dependencies:
```bash
python pip install -r requirements.txt
```
4. Run main file
```bash
python main.py
```

