import random
import numpy as np 

class Game: 
    #used as the fitness function for the genetic algorithm
    def __init__(self):
        self.dice = random.random()

    def play(self, player1, player2):
        if self.dice == .5:
            return (0, 0)
        elif self.dice > .5:
            return (1,-1)
        else:
            return (-1, 1)

class Synapses:
    def __init__(self, input_dimension, output_dimension):
        self.weights = 2 * np.random.random((input_dimension, output_dimension)) - 1    
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def io(self, x):
        return self.sigmoid(np.dot(self.weights, x))        

class Organism:        
    #neural network
    def __init__(self, topology = [49, 100, 2]):
        [self.input_dimension, self.hidden_dimension, self.output_dimension] = topology
        self.fitness = 0
        self.l0 = Synapses(self.input_dimension, self.hidden_dimension)
        self.l1 = Synapses(self.hidden_dimension, self.output_dimension)
    
    def mutate(self, probability):
        if random.random() <= probability:
            self.l0.weights[random.randint(0, len(self.l0.weights) - 1)] = random.random()  
        if random.random() <= probability:
            self.l1.weights[random.randint(0, len(self.l1.weights) - 1)] = random.random()  

    def think(self, board_position):
        #input: position of 7x7 board game Isolation as bitstring of 49 elements
        #output: next move (x,y) coordinate
        coordinates = self.l1.io( self.l0.io(board_position) )
        return coordinates

class GA:        
    def __init__(self):
        self.generation = 0
        self.fittest = 0
        self.total = 0

    def run(self, pop_size = 50, iterations = 100, elitism = .2, mutation = .1):
        self.elitism = elitism
        self.pop_size = pop_size
        self.mutation = mutation
        self.iterations = iterations
        self.organisms = self.populate()
        for t in range(self.iterations):
            i = j = 0
            while i == j: i, j = random.randint(0, self.pop_size - 1), random.randint(0, self.pop_size - 1)
            self.organisms[i].fitness, self.organisms[j].fitness = self.compete(self.organisms[i], self.organisms[j] )
            self.display_stats()
            self.evolve()

    def populate(self):
        return [Organism() for i in range(self.pop_size)]
    
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
        for organism in self.organisms:
            if organism.fitness > self.fittest:
                self.fittest = organism.fitness
            self.total += organism.fitness
        self.generation += 1
        print('> GEN:',self.generation,'BEST:',self.fittest,'AVG:',float(self.total / self.pop_size))

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
            child1.mutate(self.mutation)
            child2.mutate(self.mutation)
            new_organisms.append(child1)
            new_organisms.append(child2)
        return new_organisms

    def crossover(self, parent1, parent2):
        child1, child2 = Organism(), Organism()
        child1.l0.weights, child1.l1.weights = parent1.l0.weights, parent2.l1.weights 
        child2.l0.weights, child2.l1.weights = parent2.l0.weights, parent1.l1.weights
        return child1, child2


x = GA()
x.run()