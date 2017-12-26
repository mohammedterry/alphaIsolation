import random
import numpy as np 

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
        #input: a 1D array representing the current state of the 7x7 board  
        # & an int for the current position of player as a 1D embedded (x,y) coordinate 
        #output:   the next suggested position of the player as 1D embedded (x,y) coordinate
        data_out = self.l1.io( self.l0.io(data_in) )
        return data_out * 7 * 7

class GA:  # genetic algorithm      
    def __init__(self):
        self.generation = 0
        self.fittest = 0

    def run(self, topology = [1,3,1], pop_size = 50, iterations = 100, elitism = .2, pairs = False, mutation = .1):
        self.pop_size = pop_size
        self.iterations = iterations
        self.elitism = elitism
        self.organisms = [Organism(topology = topology, probability = mutation) for _ in range(self.pop_size)]
        for _ in range(self.iterations):
            self.tournament(pairs)
            self.display_stats()
            self.evolve()

    def tournament(self, pairs):
        if pairs: #quicker - everyone paired off
            for i,j in zip(range(0,self.pop_size,2), range(1,self.pop_size,2)):
                winner, loser = self.compete(self.organisms[i], self.organisms[j] )
                self.organisms[i] = winner
                self.organisms[j] = loser
        else: #thorough - everyone faces each other once
            for i in range(self.pop_size):
                for j in range(i, self.pop_size):
                    winner, loser = self.compete(self.organisms[i], self.organisms[j] )
                    self.organisms[i] = winner
                    self.organisms[j] = loser
    
    def evolve(self):
        elite = self.get_elite()
        rest = self.reproduce(elite)
        self.organisms = elite + rest

    def compete(self, player1, player2):
        winner, loser = Game(player1, player2).play()
        winner.fitness += 1
        loser.fitness -= 1
        return winner, loser
    
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

class Game: #game "isolation" used as the fitness function for the genetic algorithm
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]  #variant: players moves like a chess knight

    def __init__(self, player1, player2, board_dimensions = [7,7]):
        [self.width, self.height] = board_dimensions
        self.turn = 0
        self._active_player = player1
        self._inactive_player = player2
        self.board_state = [0] * self.width * self.height  #the 2D board represented as a 1D array [0000...] 
        self.p1_location = -1    # -1 indicates not on board yet
        self.p2_location = -1

    def legal_move(self, coordinate):
        embedded = coordinate[0] + coordinate[1] * self.height  #2D x,y --> 1D
        return (0 <= coordinate[0] < self.height and 0 <= coordinate[1] < self.width and self._board_state[embedded] == 0) #legal IF within board width, height & position is not occupeied (0)

    def possible_moves(self):
        if self.my_location() == -1: #not moved yet - player can be placed anywhere on board thats empty to begin with
            return [(i, j) for j in range(self.width) for i in range(self.height) if self._board_state[i + j * self.height] == 0]
        (r, c) = xy(self.my_location())
        valid_moves = [(r + dr, c + dc) for dr, dc in self.directions if self.legal_move((r + dr, c + dc))]
        random.shuffle(valid_moves)
        return valid_moves

    def apply_move(self, coordinate): #1D embedded coord 
        self._board_state[coordinate] = 1
        if self.turn % 2 == 0: 
            self._p1_location = coordinate
        else:
            self._p2_location = coordinate

    def my_location(self):
        if self.turn % 2 == 0: #even turn means its player1's turn
            return self.p1_location
        return self.p2_location

    def xy(self, embedded): #1D embedded coordinate --> 2D x,y coordinate
        return (embedded % self.height, embedded // self.height)

    def play(self): 
        while True: 
            data = self.board_state + [self.my_location()]
            coord = self._active_player.think(data) #neural network provides guess for next best move (x,y)
            
            if xy(coord) not in self.possible_moves(): #illegal move - you lose! 
                return self._inactive_player #return winning player (the other guy)

            self.apply_move(coord)
            self._active_player, self._inactive_player = self._inactive_player, self._active_player  #switch players
            self.turn += 1


GA().run(topology = [50, 100, 1])
