# Script for testing a trained agent on the Robosumo task
# against its historical self or against another trained agent
import tc
import gym, roboschool
import pickle
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from multiagent_monitor import MultiAgentMonitor
from baselines import logger
from gym_extensions.continuous import mujoco
import robosumo
import tensorflow as tf
import os

# Directories containing saved agents
model_dir = './model_dir'
test_dir = './test_dir'

max_eps = 30 # Number of episodes per agent match up

params = pickle.load(open(model_dir+"/params.pkl","rb"))
seed = params['seed']
ncpu = params['ncpu']

long_test = True # 5000 steps per episode  
checkpoint_match = True # To test model and test at equal times during training
checkpoint_time = False # Optionally set to a specific checkpoint number, otherwise use latest

if long_test:
    env_id = "RoboSumo-Ant-vs-Ant-long-v0"
else:
    env_id = "RoboSumo-Ant-vs-Ant-v0"

logger.configure()

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
config.gpu_options.allow_growth = True
tf.Session(config=config).__enter__()

def make_env_from_id(env_id, seed):
    env = gym.make(env_id)
    env = MultiAgentMonitor(env, logger.get_dir(),allow_early_resets=True, file_prefix="")
    env.seed(seed)
    return env

env = SubprocVecEnv([lambda: make_env_from_id(
            env_id, seed + i if seed is not None else None) for i in range(ncpu)])

times, scores, wins = tc.historic_test_play(
    env=env,model_dir=model_dir, test_dir=test_dir,
    max_eps=max_eps,checkpoint_match=checkpoint_match,
    checkpoint_time=checkpoint_time)

test_dict = {}
test_dict['times'] = times
test_dict['scores'] = scores
test_dict['max_eps'] = max_eps

# Saving results
if test_dir==model_dir:
    if long_test:
        pickle.dump((times,scores,wins),open(model_dir+'/long_test_scores.pkl',"wb"))        
    else:
        pickle.dump((times,scores,wins),open(model_dir+'/test_scores.pkl',"wb"))
else:
    model_test_dir = model_dir+test_dir[1:]
    if not os.path.exists(model_test_dir):
        os.mkdir(model_test_dir)
    if checkpoint_match:
        pickle.dump((times,scores,wins),open(model_test_dir+'/matched_test_scores.pkl',"wb"))
    else:
        pickle.dump((times,scores,wins),open(model_test_dir+'/test_scores.pkl',"wb"))
