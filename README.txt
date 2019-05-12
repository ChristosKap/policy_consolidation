Code for Policy Consolidation for Continual Reinforcement Learning
Accepted at ICML 2019
arXiv link: https://arxiv.org/abs/1902.00255

Python version used: anaconda3-5.1.0 
Requirements: numpy, tensorflow 1.7.0, gym, baselines, gym_extensions, robosumo, gym-compete

- tc.py contains the main classes and functions for the PC model
- run_single.py and run_selfplay.py are the run files used for training agents in single agent and selfplay environments respectively
- play_history.py is the run file for playing a self-play agent against its history or against other agents
