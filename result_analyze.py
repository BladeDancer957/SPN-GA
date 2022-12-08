import matplotlib.pyplot as plt

import numpy as np



def read_multiple_ppo_mujoco_models(filepath, model_num, steps):
    test_reward_list = np.zeros((steps, len(model_num)))


    for i,num in enumerate(model_num):
        test_reward = np.load(filepath.format(seed=num) + '/elite_returns.npy')

        test_reward_list[:, i] = test_reward

    return test_reward_list.mean(axis=1), test_reward_list.std(axis=1)


def plot_multiple_mean_rewards(label_list, color_list, model_num, steps, env_name,filepaths,scale=2.0):
    plt.rcParams['font.size'] = 14
    plt.figure()
    plt.axhline(y=115686.76,label='DPN-Weights-PPO',color='red') # halfcheetah: 2534.96   swimmer: 107.19   humanoidstandup: 115686.76
    for i, filepath in enumerate(filepaths, 0):

        label = label_list[i]
        color = color_list[i]
        reward_mean, reward_std = read_multiple_ppo_mujoco_models(filepath,model_num, steps)


        plt.plot([num for num in range(1,steps+1)], reward_mean, color, label=label)
        plt.fill_between([num for num in range(1,steps+1)], reward_mean - reward_std/scale, reward_mean + reward_std/scale,
                         alpha=0.2, color=color)


        print("max avg return: ",np.max(reward_mean), reward_std[np.argmax(reward_mean)])
        print("final avg return: ",reward_mean[-1], reward_std[-1])


    plt.xlim([1, steps])

    plt.xlabel("Generations.")
    plt.ylabel("Average return.")
    plt.title(env_name)
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig(fname="./imgs/human-v2_2.svg", format="svg", bbox_inches='tight')

    plt.show()

if __name__ == "__main__":

    env_name = 'HumanoidStandup-v2'
    generations = 100
    popsize = 200


    filepaths = [
                './results/' + env_name + '_SPN_Weights' + '_generations_' + \
                                 str(generations) + '_popsize_' + str(popsize) +'_seed_{seed}',

                './results/' + env_name + '_SPN_Connections' + '_generations_' + \
                                 str(generations) + '_popsize_' + str(popsize) +'_seed_{seed}',

                 ]


    model_num = list(range(10,101,10))

    steps = generations

    labels_list = ['SPN-Weights-GA', 'SPN-Connections-GA']
    color_list = ['blue','purple']

    plot_multiple_mean_rewards(labels_list, color_list, model_num, steps, env_name, filepaths)





