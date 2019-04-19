import numpy as np

class GA():
    """
    crossover_method --> Single-point crossover(1), Two-point(2), Uniform crossover(3)
    mutation_method  --> Bit Flip(1), Random Resetting(2), Swap Mutation(3), shuffle(4)
    selection_method --> Roulette Wheel Selection(1), Random Selection(2)
    """
    def __init__(self,population_shape,method,crossover_prob,mutation_prob,crossover_method,mutation_method,selection_method,is_binary,lower_bound, upper_bound, maximum):
        self.population_shape = population_shape
        self.method = method
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method
        self.selection_method =selection_method
        self.is_binary =is_binary
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.maximum = maximum
    def evaluation(self, population):
         return np.array([self.method(i) for i in population])
    def initialization(self):
        if self.is_binary:
            self.population = np.random.randint(low=0, high=2, size=self.population_shape)
        else:
            self.population = np.random.randint(low=0, high=self.population_shape[1], size=self.population_shape)
        self.fitness = self.evaluation(self.population)
        print(self.fitness)
    def crossover(self, individual_0, individual_1):
        if self.crossover_method == 1:
            point = np.random.randint(len(individual_0))
            new1 = np.hstack((individual_0[:point], individual_1[point:]))
            new2 = np.hstack((individual_1[:point], individual_0[point:]))
        if self.crossover_method == 2:
            point1 = np.random.randint(len(individual_0))
            point2 = np.random.randint(low=point1,high=len(individual_0))
            tmp1 = np.hstack((individual_0[:point1], individual_1[point1:point2]))
            tmp2 = np.hstack((individual_1[:point1], individual_0[point1:point2]))
            new1 = np.hstack((tmp1, individual_0[point2:]))
            new2 = np.hstack((tmp1, individual_1[point2:]))
        if self.crossover_method == 3:
            uniform = np.random.randint(low=0, high=2, size=len(individual_0))
            new1 = individual_0.copy()
            new2 = individual_1.copy()
            for it in range(uniform.shape[0]):
                if(uniform[it]):
                    tmp = new1[it]
                    new1[it] = new2[it]
                    new2[it] = tmp
        return new1,new2
    def mutation(self, individual):
        if self.mutation_method == 1:
            point = np.random.randint(len(individual))
            individual[point] = 1 - individual[point]
        if self.mutation_method == 2:
            point = np.random.randint(len(individual))
            individual[point] = 0
        if self.mutation_method == 3:
            point1 = np.random.randint(len(individual))
            point2 = np.random.randint(len(individual))
            temp = individual[point1]
            individual[point1] = individual[point2]
            individual[point2] = temp
        if self.mutation_method == 4:
            np.random.shuffle(individual)
        return individual
    def selection(self, size, fitness):

        if self.selection_method == 1:
            fitness_ = fitness
            for i in range(fitness_.shape[0]):
                if fitness_[i] < 0:
                    fitness_[i] = 1
            idx = np.random.choice(np.arange(len(fitness_)), size=size, replace=True, p=fitness_/fitness_.sum())
        if self.selection_method == 2:
            idx = np.random.choice(np.arange(len(fitness)), size=size, replace=True)
        if self.selection_method == 3:
            idx = [size - 1, size - 2]

        return idx
    def run(self):
        plot = []
        global_best = 0
        self.initialization()
        best_index = np.argsort(self.fitness)[0]
        global_best_fitness = self.fitness[best_index]
        global_best_ind = self.population[best_index, :]
        eva_times = self.population_shape[0]
        count = 0
        for it in range(1000):
            next_gene = []
            for n in range(int(self.population_shape[0]/2)):
                i, j = self.selection(2, self.fitness)
                individual_0, individual_1 = self.population[i],self.population[j]
                if np.random.rand() < self.crossover_prob:
                    individual_0, individual_1 = self.crossover(individual_0, individual_1)
                if np.random.rand() < self.mutation_prob:
                    individual_0 = self.mutation(individual_0)
                    individual_1 = self.mutation(individual_1)
                next_gene.append(individual_0)
                next_gene.append(individual_1)
            self.population = np.array(next_gene)
            self.fitness = self.evaluation(self.population)
            eva_times += self.population_shape[0]
            plot.append(global_best_fitness)
            if self.maximum:
                if np.max(self.fitness) > global_best_fitness:
                    best_index = np.argsort(self.fitness)[-1]
                    global_best_fitness = self.fitness[best_index]
                    global_best_ind = self.population[best_index, :]
                    count = 0
                else:
                    count += 1
                worst_index = np.argsort(self.fitness)[-1]
                self.population[worst_index, :] = global_best_ind
                self.fitness[worst_index] = global_best_fitness
            else:
                if np.min(self.fitness) < global_best_fitness:
                    best_index = np.argsort(self.fitness)[0]
                    global_best_fitness = self.fitness[best_index]
                    global_best_ind = self.population[best_index, :]
                    count = 0
                else:
                    count +=1

                worst_index = np.argsort(self.fitness)[-1]
                self.population[worst_index, :] = global_best_ind
                self.fitness[worst_index] = global_best_fitness
            print('\n Solution: {} \n Fitness: {} \n Evaluation times: {}'.format(global_best_ind, global_best_fitness, eva_times))

        return global_best_ind, global_best_fitness, plot
