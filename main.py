import time
import argparse
import numpy as np
import torch
import os
import gym
from gym.spaces import Discrete, Box

from genetic import GeneticAlgorithm
import policies

torch.set_num_threads(1)
gym.logger.set_level(40)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():

    root_path = './results/' + args.env_name + '_' + args.policy_type + '_generations_' + \
                     str(args.generations) + '_popsize_' + str(args.popsize)+'_seed_'+str(args.seed)+'/'


    if not os.path.exists(root_path):
        os.makedirs(root_path)

    # Look up observation and action space dimension
    env = gym.make(args.env_name)
    env.seed(args.seed)

    input_dim = env.observation_space.shape[0] 

    if isinstance(env.action_space, Box): 
        action_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete):  
        action_dim = env.action_space.n
    else:
        raise ValueError('Action space not supported')

    policy = getattr(policies, args.policy_type)(input_dim, action_dim)

    # Initialise the EvolutionStrategy class
    print('\nInitilisating GA for ' + str(args.env_name))

    ga = GeneticAlgorithm(policy, policy.get_weights(), env, population_size=args.popsize,sigma=args.sigma)

    # Start the evolution
    tic = time.time()
    print('\nStarting Evolution\n')
    ga.run(args.generations, print_step=args.print_every, path=root_path)
    toc = time.time()
    print('\nEvolution took: ', int(toc - tic), ' seconds\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2', metavar='', choices=['CartPole-v1', 'HalfCheetah-v2', 'Swimmer-v2', 'HumanoidStandup-v2'], 
                        help='Environment: any OpenAI Gym or pyBullet environment may be used')
    parser.add_argument('--policy_type', default="SPN_Connections", choices=["SPN_Connections", "SPN_Weights"])    
    parser.add_argument('--popsize', type=int, default=200, metavar='',
                        help='Population size.')
    parser.add_argument('--print_every', type=int, default=1, metavar='', help='Print and save every N generations.')
    parser.add_argument('--sigma', type=float, default=0.01, metavar='',
                        help='GA sigma: modulates the amount of noise used to populate each new generation')  
    parser.add_argument('--generations', type=int, default=100, metavar='',
                        help='Number of generations that the GA will run.')
    parser.add_argument('--seed', default=10, type=int) 


    args = parser.parse_args()

    seed_everything(args.seed)

    main()



