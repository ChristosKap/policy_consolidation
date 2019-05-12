#!/usr/bin/env python3
# File for running self-play experiments for Policy Consolidation for Continual Reinforcement Learning
import argparse
from baselines.common.cmd_util import mujoco_arg_parser, make_mujoco_env
from baselines import bench, logger
import tensorflow as tf
import pickle
import multiprocessing
import robosumo
from multiagent_monitor import MultiAgentMonitor


def train(env_ids, num_timesteps, seed, load_path=None, cascade_depth=1,
          flow_factor=1, mesh_factor=2.0, num_epochs=1, lr=3e-4, 
          lr_decay=True,var_init='random',ent_coef=0.0,
          imp_sampling='normal',imp_clips=[-5,5],
          dynamic_neglogpacs=False, full_kl=False, kl_beta=1.0,
          separate_value=False, prox_value_fac=False,
          value_cascade=False,kl_type='fixed', adaptive_targ=None,
          targ_leeway=1.5, beta_mult_fac=2.0, traj_length=2048,
          cliprange=0.2, nminibatches=32, lam=0.95, gamma=0.99,
          noptepochs=10, ncpu=1, vf_coef=1.0, reverse_kl=False,
          cross_kls=['new','new'],test_history=False, num_test_eps=30,
          save_interval=25,dense_decay=False):

    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from policies import MlpPolicy
    import gym, roboschool
    from gym_extensions.continuous import mujoco
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    import tc

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
    print('ncpu: '+str(ncpu))
    
    def make_env_from_id(env_id, seed, prefix):
        env = gym.make(env_id)
        env = MultiAgentMonitor(env, logger.get_dir() ,allow_early_resets=True, file_prefix=prefix)
        env.seed(seed)
        return env

    policy = MlpPolicy

    model=None
    
    for epoch in range(num_epochs):
        env_id = env_ids[epoch%len(env_ids)]
        env = SubprocVecEnv([lambda: make_env_from_id(
                    env_id, seed + i if seed is not None else None,"")
                             for i in range(ncpu)])
        if test_history:
            test_env = SubprocVecEnv([lambda: make_env_from_id(
                        env_id, seed + i if seed is not None else None, "test_")
                                      for i in range(ncpu)])
        else:
            test_env=None
        
        load_path, model=tc.learn_multi(policy=policy, env=env, nsteps=traj_length, nminibatches=nminibatches,
                                        lam=lam, gamma=gamma, noptepochs=noptepochs, log_interval=1,
                                        ent_coef=ent_coef, vf_coef=vf_coef,lr=lr, lr_decay = lr_decay,
                                        var_init=var_init, cliprange=cliprange, imp_sampling = imp_sampling,
                                        imp_clips = imp_clips, dynamic_neglogpacs=dynamic_neglogpacs,
                                        full_kl=full_kl, kl_beta=kl_beta, separate_value=separate_value,
                                        prox_value_fac=prox_value_fac, value_cascade=value_cascade,
                                        kl_type=kl_type, reverse_kl=reverse_kl, cross_kls=cross_kls,
                                        adaptive_targ=adaptive_targ, targ_leeway=targ_leeway,
                                        beta_mult_fac=beta_mult_fac, total_timesteps=num_timesteps,
                                        save_interval=save_interval, cascade_depth = cascade_depth,
                                        flow_factor = flow_factor, mesh_factor = mesh_factor,
                                        load_path=load_path,prev_model=model, test_history=test_history,
                                        num_test_eps=num_test_eps,test_env=test_env, dense_decay=dense_decay)
        

def main():

    params={}
    args = mujoco_arg_parser().parse_args()
    params['cascade_depth'] = 1 # Number of policy networks (including visible)
    params['flow_factor'] = 0.25 #  \omega_{1,2}
    params['mesh_factor'] = 4.0 # \omega
    params['env_names'] = ["RoboSumo-Ant-vs-Ant-v0"]
    
    env_string = ''
    for name in params['env_names']:
        env_string += name
    params['load_path'] = False # for loading previous model as start point
    params['num_epochs'] = 1 # num task epochs
    params['num_timesteps'] = 600000000 # num timesteps per task epoch
    params['traj_length'] = 8192 # trajectory length
    params['noptepochs'] = 6 # num training epochs per policy update
    params['nminibatches'] = 32 # num minibatches per policy update
    params['lam'] = 0.95 # lambda for GAE
    params['gamma'] = 0.995 # discount factor
    params['lr'] = 0.0001 # Adam learning rate
    params['lr_decay'] = True # Adam learning rate scaled with cascade depth (deeper policy = lower lr)
    params['var_init'] = 'equal' # how to initialise hidden policy networks 
    params['ent_coef'] = 0.0 # entropy coefficient for ppo, always set to 0
    params['vf_coef'] = 0.5 # value function coefficient in ppo loss
    params['imp_sampling'] = 'none' # Type of importance sampling for optimising cross KL terms in cascade, for minimising KL between trajectories
    params['imp_clips'] = [-5,5] # for clipped importance sampling
    params['dynamic_neglogpacs']=False # Set to True to allow policy targets to move dynamically during policy update, typically leads to instability
    params['full_kl']=True # Set to false for clipped PPO, true for full and adaptive KL
    params['seed'] = 1 # random seed - used seed=1 for single runs and {0,1,2} for multiple PC runs 
    params['separate_value'] = False # Set to true to have a separate value function network for each policy
    params['cliprange'] = 0.2 # Clip coefficient for clipped PPO
    params['kl_beta'] = 0.5 # beta for KL version of PPO and PC model
    params['kl_type'] = 'fixed' # fixed or adaptive KL
    params['adaptive_targ'] = 0.01 # target for adaptive KL
    params['targ_leeway'] = 1.5 # threshold for adjusting KL coeff
    params['beta_mult_fac'] = 2.0 # factor to adjust KL coeff by
    params['ncpu']=1 # how many parallel environments to use
    params['prox_value_fac'] = False # Set value if want to constrain how much value function moves with each policy step
    params['value_cascade'] = False # Set to true to use a cascade of value networks as well as policy networks
    params['reverse_kl'] = False # True for original PPO KL direction
    params['cross_kls'] = ['new','new'] # KL directions for cross-KL terms in PC model: new is KL(pi_k || pi_k-1_old), old is vice versa
    params['test_history'] = False # True to test agent vs history during training
    params['num_test_eps'] = 30 # Number of test episodes vs history
    params['save_interval'] = 10 # How often to save model
    params['dense_decay_time_frac'] = 0.15 # over what percentage of training time dense reward is decayed to zero
    params['dense_decay'] = params['traj_length']*params['ncpu'] / (params['dense_decay_time_frac']*params['num_timesteps'])
    params['drawisloss'] = True # Set true to set reward for drawing equal to reward for losing
    
    log_dir = './test_dir'
    #logger.configure()
    logger.configure(dir=log_dir,format_strs=['stdout','csv'])

    pickle.dump(params,open(log_dir+'/params.pkl',"wb"))
    
    train(params['env_names'], num_timesteps=params['num_timesteps'], seed=params['seed'],
          cascade_depth=params['cascade_depth'], flow_factor=params['flow_factor'],
          mesh_factor=params['mesh_factor'],load_path=params['load_path'],
          num_epochs=params['num_epochs'],lr=params['lr'],lr_decay=params['lr_decay'],
          var_init=params['var_init'],ent_coef=params['ent_coef'], imp_sampling=params['imp_sampling'],
          imp_clips = params['imp_clips'], dynamic_neglogpacs=params['dynamic_neglogpacs'],
          full_kl=params['full_kl'], separate_value=params['separate_value'], kl_beta=params['kl_beta'],
          kl_type=params['kl_type'], adaptive_targ=params['adaptive_targ'], traj_length=params['traj_length'],
          cliprange=params['cliprange'],nminibatches=params['nminibatches'],lam=params['lam'],
          gamma=params['gamma'],noptepochs=params['noptepochs'],targ_leeway=params['targ_leeway'],
          beta_mult_fac=params['beta_mult_fac'],ncpu=params['ncpu'],prox_value_fac=params['prox_value_fac'],
          value_cascade=params['value_cascade'],vf_coef=params['vf_coef'],reverse_kl=params['reverse_kl'],
          cross_kls=params['cross_kls'],test_history=params['test_history'],num_test_eps=params['num_test_eps'],
          save_interval=params['save_interval'],dense_decay=params['dense_decay'])

if __name__ == '__main__':
    main()

