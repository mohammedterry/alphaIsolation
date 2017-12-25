import random
import numpy as np 

class Game: #used as the fitness function for the genetic algorithm
    def __init__(self):
        self.dice = random.random()

    def play(self, player1, player2):
        if self.dice == .5:
            return (0, 0)
        elif self.dice > .5:
            return (1,-1)
        else:
            return (-1, 1)

class Synapses:  #these are the neural network's weights which are to be evolved
    def __init__(self, input_dimension, output_dimension):
        self.weights = 2 * np.random.random((input_dimension, output_dimension)) - 1    
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def io(self, x):
        return self.sigmoid(np.dot(self.weights, x))        

class Organism: #neural network
    def __init__(self, topology, probability):
        self.p = probability
        self.fitness = 0
        [self.input_dimension, self.hidden_dimension, self.output_dimension] = topology
        self.l0 = Synapses(self.input_dimension, self.hidden_dimension)
        self.l1 = Synapses(self.hidden_dimension, self.output_dimension)
    
    def mutate(self):
        if random.random() <= self.p:
            self.l0.weights[random.randint(0, len(self.l0.weights) - 1)] = random.random()  
        if random.random() <= self.p:
            self.l1.weights[random.randint(0, len(self.l1.weights) - 1)] = random.random()  

    def think(self, data_in):
        #input: position of 7x7 board game Isolation as bitstring of 49 elements & the current position of player as (x,y) coordinate
        #output: next move (x,y) coordinate
        data_out = self.l1.io( self.l0.io(data_in) )
        return data_out

class GA:  # genetic algorithm      
    def __init__(self):
        self.generation = 0
        self.fittest = 0

    def run(self, topology = [1,3,1], pop_size = 50, iterations = 100, elitism = .2, pairs = False, mutation = .1):
        self.pop_size = pop_size
        self.iterations = iterations
        self.elitism = elitism
        self.organisms = [Organism(topology = topology, probability = mutation) for i in range(self.pop_size)]
        for t in range(self.iterations):
            self.tournament(pairs)
            self.display_stats()
            self.evolve()

    def tournament(self, pairs):
        if pairs: #quicker - everyone paired off
            for i,j in zip(range(0,self.pop_size,2), range(1,self.pop_size,2)):
                f1, f2 = self.compete(self.organisms[i], self.organisms[j] )
                self.organisms[i].fitness += f1
                self.organisms[j].fitness += f2
        else: #thorough - everyone faces each other once
            for i in range(self.pop_size):
                for j in range(i, self.pop_size):
                    f1, f2 = self.compete(self.organisms[i], self.organisms[j] )
                    self.organisms[i].fitness += f1
                    self.organisms[j].fitness += f2
    
    def evolve(self):
        elite = self.get_elite()
        rest = self.reproduce(elite)
        self.organisms = elite + rest

    def compete(self, organism1, organism2):
        #fitness function based on outcome of a game of isolation
        #+1 for a win, -1 for a loss, 0 for a draw
        score = Game().play(organism1, organism2)
        return score[0], score[1]

    def display_stats(self):
        total = 0
        for organism in self.organisms:
            if organism.fitness > self.fittest:
                self.fittest = organism.fitness
            total += organism.fitness
        self.generation += 1
        print('> GEN:',self.generation,'BEST:',self.fittest,'AVG:',float(total / self.pop_size))

    def get_elite(self):
        #keep fittest percentage of organisms
        return sorted(self.organisms, key=lambda x: x.fitness, reverse=True)[:int(self.elitism * self.pop_size)]

    def reproduce(self, elite):
        new_organisms = []
        elite_size = int(self.elitism * self.pop_size)
        leftover = self.pop_size - elite_size
        for i in range(0,leftover,2):
            a = b = 0
            while a == b: a, b = random.randint(0, elite_size - 1), random.randint(0, elite_size - 1)
            child1, child2 = self.crossover(elite[a], elite[b])
            child1.mutate()
            child2.mutate()
            new_organisms.append(child1)
            new_organisms.append(child2)
        return new_organisms

    def crossover(self, parent1, parent2):
        topology = [parent1.input_dimension, parent1.hidden_dimension, parent1.output_dimension]
        child1, child2 = Organism(topology = topology, probability = parent1.p), Organism(topology = topology, probability = parent1.p)
        child1.l0.weights, child1.l1.weights = parent1.l0.weights, parent2.l1.weights 
        child2.l0.weights, child2.l1.weights = parent2.l0.weights, parent1.l1.weights
        return child1, child2


GA().run(topology = [51, 100, 2])
