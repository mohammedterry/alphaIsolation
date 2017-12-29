import random
import numpy as np 

class Synapses:  #these are the neural network's weights which are to be evolved
    def __init__(self, input_dimension, output_dimension):
        self.weights = 2 * np.random.random((input_dimension, output_dimension)) - 1    
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def f(self, values):
        return self.sigmoid(np.dot(values, self.weights))        


class Organism: #neural network
    def __init__(self, topology, probability = .1):
        self.p = probability
        self.fitness = 0
        [self.input_dimension, self.hidden_dimension, self.output_dimension] = topology
        self.l0 = Synapses(self.input_dimension, self.hidden_dimension)
        self.l1 = Synapses(self.hidden_dimension, self.output_dimension)
    
    def mutate(self):
        if random.random() <= self.p:
            for i in range(len(self.l0.weights) - 1):
                if random.random() <= self.p:
                    self.l0.weights[i] = 2*random.random() - 1.
            for i in range(len(self.l1.weights) - 1):
                if random.random() <= self.p:
                    self.l1.weights[i] = 2*random.random() - 1.

    def think(self, data_in):
        data = np.array(data_in)
        data_out = self.l1.f( self.l0.f(data))
        return int(data_out[0]) + int(data_out[1]) * 7

    def save(self, file_name = 'evolved_nn.npy'):
        np.save(file_name, {'l0_weights':self.l0.weights, 'l1_weights':self.l1.weights}) 

    def load(self, file_name = 'evolved_nn.npy'):
        config = np.load(file_name).item()
        self.l0.weights = config['l0_weights']
        self.l1.weights = config['l1_weights']

class GA:  # genetic algorithm      
    def __init__(self):
        self.generation = 0

    def run(self, topology = [1,3,1], pop_size = 50, iterations = 100, elitism = .2, option = 0, mutation = .2):
        self.pop_size = pop_size
        self.iterations = iterations
        self.elitism = elitism
        self.organisms = [Organism(topology = topology, probability = mutation) for _ in range(self.pop_size)]
        self.fittest = random.choice(self.organisms) 
        for _ in range(self.iterations):
            self.tournament(option)
            self.display_stats()
            self.evolve()
        self.fittest.save()

    def tournament(self, option):
        for organism in self.organisms:
            organism.fitness = 0
        if option == 0: #each plays against a random moving bot
            for organism in self.organisms:
                winner, loser = self.compete(organism, organism)
                organism = loser
        elif option == 1: #quicker - everyone paired off
            for i,j in zip(range(0,self.pop_size,2), range(1,self.pop_size,2)):
                winner, loser = self.compete(self.organisms[i], self.organisms[j] )
                self.organisms[i] = winner
                self.organisms[j] = loser
        elif option == 2: #thorough - everyone faces each other once
            for i in range(self.pop_size):
                for j in range(i, self.pop_size):                 
                    winner, loser = self.compete(self.organisms[i], self.organisms[j] )
                    self.organisms[i] = winner
                    self.organisms[j] = loser
    
    def evolve(self):
        elite = sorted(self.organisms, key=lambda x: x.fitness, reverse = True)[:int(self.elitism * self.pop_size)]
        rest = self.reproduce(elite)
        self.organisms = elite + rest

    def compete(self, player1, player2):
        winner, loser = Game(player1, player2).play()
        winner.fitness += 1
        loser.fitness -= 1 
        return winner, loser
    
    def display_stats(self):
        for organism in self.organisms:
            if organism.fitness > self.fittest.fitness:
                self.fittest = organism
        self.generation += 1
        print('> GEN:',self.generation,'BEST:',self.fittest.fitness)

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

class Human: #to play a game against evolved A.I.
    def think(self, _):
        return (int(input('> row: ')) + int(input('> column: ')) * 7)

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
        return (0 <= coordinate[0] < self.height and 0 <= coordinate[1] < self.width and self.board_state[embedded] == 0) #legal IF within board width, height & position is not occupeied (0)

    def possible_moves(self):
        if self.my_location() == -1: #not moved yet - player can be placed anywhere on board thats empty to begin with
            return [(i, j) for j in range(self.width) for i in range(self.height) if self.board_state[i + j * self.height] == 0]
        (r, c) = self.xy(self.my_location())
        valid_moves = [(r + dr, c + dc) for dr, dc in self.directions if self.legal_move((r + dr, c + dc))]
        random.shuffle(valid_moves)
        return valid_moves

    def apply_move(self, coordinate): #1D embedded coord 
        self.board_state[coordinate] = 1
        if self.turn % 2 == 0: 
            self.p1_location = coordinate
        else:
            self.p2_location = coordinate

    def my_location(self):
        if self.turn % 2 == 0: #even turn means its player1's turn
            return self.p1_location
        return self.p2_location

    def xy(self, embedded): #1D embedded coordinate --> 2D x,y coordinate
        return (embedded % self.height, embedded // self.height)

    def play(self, history = False): 
        while True: 
            choice = self._active_player.think(self.board_state + [self.my_location()]) #neural network provides guess for next best move (x,y)        

            if self.turn %2 == 0:
                coord = random.choice(self.possible_moves())
                coord = coord[0] + coord[1] * self.height

            if self.xy(coord) not in self.possible_moves(): #illegal move - you lose! 
                return self._inactive_player, self._active_player #return winning player (the other guy) & losing player
            
            self.apply_move(coord)
            
            if history: 
                print(self.turn, self.xy(coord))
                print( self.display())

            self._active_player, self._inactive_player = self._inactive_player, self._active_player  #switch players
            self.turn += 1
    
    def display(self, symbols=['x', 'o']):
        col_margin = len(str(self.height - 1)) + 1
        prefix = "{:<" + "{}".format(col_margin) + "}"
        offset = " " * (col_margin + 3)
        out = offset + '   '.join(map(str, range(self.width))) + '\n\r'
        for i in range(self.height):
            out += prefix.format(i) + ' | '
            for j in range(self.width):
                idx = i + j * self.height
                if not self.board_state[idx]:
                    out += ' '
                elif self.p1_location == idx:
                    out += symbols[0]
                elif self.p2_location == idx:
                    out += symbols[1]
                else:
                    out += '-'
                out += ' | '
            out += '\n\r'
        return out

GA().run(iterations = 1000, topology = [50, 100, 2], option = 0, mutation = .4)
player1 = Organism([50,100,2])
player1.load()
player2 = Human()
Game(player2, player1).play(history = True)