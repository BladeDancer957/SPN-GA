import numpy as np

from fitness_functions import fitness_static

class GeneticAlgorithm(object):
    def __init__(self, policy, weights, env, population_size, sigma):

        self.policy = policy
        self.weights = weights  
        self.env = env
        self.POPULATION_SIZE = population_size
        self.SIGMA = np.float32(sigma)

        self.get_reward = fitness_static

        self.current_population = None

        self.elites_list = [] # 存储历代精英 (第一代+ 每隔print_every代)
        self.elite = None # 追踪至今最好的精英
        self.elite_return = -10000 # 追踪至今最好的精英对应的return
        self.elites_return_list = [] # 存储历代精英对应的avg return (第一代+ 每隔print_every代)
        self.index = 0 # 指示位置

    def _get_population(self):

        if self.index == 0:  # 第一代种群 初始化
            population = []
            for i in range(int(self.POPULATION_SIZE / 2)):  
                weights_try = np.random.normal(scale=self.SIGMA,size=self.weights.shape)
                population.append(weights_try)
                population.append(-weights_try)

            population = np.array(population).astype(np.float32)

            return population
        else: 
            population = self.current_population[:int(self.POPULATION_SIZE*0.25)] 

            population1 = []
            #mutate_num = self.POPULATION_SIZE * 1.0
            sample_index = np.random.choice(np.arange(len(population)),int(self.POPULATION_SIZE /2))
            for index in sample_index:
                weights_try = np.random.normal(scale=self.SIGMA,size=self.weights.shape)
                population1.append(population[index]+weights_try)
                population1.append(population[index]-weights_try)
      
            population1 = np.array(population1).astype(np.float32)

            return population1


    def _get_rewards(self,population):

        rewards = []
        for p in population: 
            weights_try = np.array(p).astype(np.float32)
            rewards.append(self.get_reward(weights_try, self.env, self.policy))

        rewards = np.array(rewards).astype(np.float32)

        return rewards 

    def _elite_candidate(self, rewards, population):

        reverse_index = np.argsort(rewards)[::-1] 
        population_sort = population[reverse_index] 

        candidate_elite = population_sort[:10] 

        avg_return = []
        for elite1 in candidate_elite:  
            return1 =0
            for _ in range(10): 
                return1 += self.get_reward(elite1,self.env,self.policy)
            avg_return.append(return1/10.0)

        avg_return = np.array(avg_return).astype(np.float32)
        avg_return_index = np.argsort(avg_return)[::-1]
        elite_index = avg_return_index[0]  
        max_avg_return = avg_return[elite_index] 
        elite = population_sort[elite_index] 

        return max_avg_return, elite, population_sort



    def run(self, iterations, print_step, path):

        for iteration in range(iterations):
            self.index = iteration

            population = self._get_population() 
            rewards = self._get_rewards(population) 

            current_elite_return, current_elite, self.current_population= self._elite_candidate(rewards, population)

            if iteration == 0:
                self.elite_return = current_elite_return
                self.elites_return_list.append(current_elite_return)
                self.elites_list.append(current_elite)
                print("当前第%d代精英return:%.6lf"%(iteration+1,current_elite_return))

            else:
                if current_elite_return >= self.elite_return:
                    self.elite_return = current_elite_return
                    self.elite = current_elite

                if (iteration+1) % print_step ==0:
                    self.elites_return_list.append(current_elite_return)
                    self.elites_list.append(current_elite)

                    print("当前第%d代精英return:%.6lf"%(iteration + 1, current_elite_return))



        np.save(path + 'elite_returns.npy', np.array(self.elites_return_list).astype(np.float32))
        np.save(path+'elites.npy', np.array(self.elites_list).astype(np.float32))



