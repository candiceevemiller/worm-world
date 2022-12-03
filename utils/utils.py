""" Defines Classes for Simulation """

from settings.settings import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

class Linear(layers.Layer):
    """ Extends the tf.keras.layers.Layer class to more easily access weights """
    def __init__(self, weights, activation='softmax', units=4, input_dim=(2,)):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = weights
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )
        self.activation = activation

    def call(self, inputs):
        """ performs matrix multiplication and calls activation on result """
        # TODO separate activation function out
        return self._activation(np.matmul(inputs, self.w) + self.b)

    def _activation(self, input_value):
        """ handles activation function for the layer """
        if self.activation == 'softmax':
            return softmax(input_value)
        elif self.activation == 'sigmoid':
            return sigmoid(input_value)

class Creature:
    """ Defines our base creature or 'worm' """
    # TODO refactor to extend from a SimulationObject base class
    def __init__(self,
                 genome_size: int = DEFAULT_GENOME_SIZE,
                 genome=None,
                 step_size: int = DEFAULT_STEP_SIZE,
                 mutation_rate: float = DEFAULT_MUTATION_RATE):
        """Each creature has a position - self.x and self.y they also have a
        genome and a phenotype. Step size determines how fast the creature moves
        mutation rate is stored for asexual reproduction"""
        # If passed a certain genome, use it, else create a random one
        if genome:
            self.genome=genome
        else:
            self.genome = get_random_genome(genome_size)
        
        # Precalculate & store for easy reference
        self.genome_size = genome_size
        self.phenotype = get_phenotype(self.genome)
        self.brain_weights = make_brain_weights(self.phenotype)
        self.make_brain()

        # Set Initial Position
        # TODO implement starting next to a parent
        self.x = np.random.randint(WORLD_SIZE)
        self.y = np.random.randint(WORLD_SIZE)
        self.position = (self.x, self.y)
        
        # How fast does it move
        # TODO encode within genome and allow to evolve
        self.step_size = step_size
        
        # Mutation rate for asexual reproduction
        # TODO add environmental influence and encode volatility in genome
        self.mutation_rate = DEFAULT_MUTATION_RATE

    # TODO implement collision detection
    def move_up(self):
        """ Moves up one cell """
        self.y -= 1
        self.y %= WORLD_SIZE
        self.update_position()

    def move_down(self):
        """ Moves up down cell """
        self.y += 1
        self.y %= WORLD_SIZE
        self.update_position()

    def move_left(self):
        """ Moves up left cell """
        self.x -= 1
        self.x %= WORLD_SIZE
        self.update_position()

    def move_right(self):
        """ Moves up right cell """
        self.x += 1
        self.x %= WORLD_SIZE
        self.update_position()

    def reproduce(self):
        """ Alias for asexual reproduction """
        return reproduce([self], self.mutation_rate)

    def action(self):
        """Queries neural net for choice of action at a given state"""
        probabilities = self.model.call([self.x, self.y])
        action = np.argmax(probabilities)
        if action == 0:
                self.move_left()
        elif action == 1:
                self.move_right()
        elif action == 2:
                self.move_up()
        elif action == 3:
                self.move_down()

    def make_brain(self):
        """ Constructs model based on weights from phenotype """
        # TODO Allow for more complex networks than just a single layer
        self.model = Linear(self.brain_weights)

    def update_position(self):
        self.position = (self.x, self.y)

    def __repr__(self):
        return self.genome

### Helper Functions ###
### Genome Processing Functions ###
def get_bin(hexa):
    """takes hex and turns into binary string, preserves leading zeros"""
    return bin(int('1'+hexa,16))[3:]

def get_hex(binary):
    """takes binary string and turns into hex, preserves leading zeros"""
    return '{:0{}X}'.format(int(binary, 2), len(binary) // 4)

def construct_connection(chromosome, neurons=NEURONS):
    """ translates chromosome into connection information """
    # TODO refactor - does too much
    # accepts both binary and hex but have to translate hex to bin
    if len(chromosome) == 8:
        chromosome = get_bin(chromosome)

    # extract genes from chromosome
    source_type_gene = chromosome[0]
    source_id_gene = chromosome[1:8]
    sink_type_gene = chromosome[8]
    sink_id_gene = chromosome[9:16]
    weight_gene = chromosome[17:]
    pos_neg = 1 if chromosome[16] == '0' else -1

    # set source type
    if source_type_gene == '0':
        source_type = 'sensory'
    else:
        source_type = 'hidden'

    # set source_id
    mod = len(neurons[source_type])  # get modulo operator
    source_id = int(source_id_gene, 2)+(1 << 7)
    source_id %= mod # so we don't select a source with ID too large
    # retrieve neuron
    source_neuron = neurons[source_type][source_id]

    # set sink type
    if sink_type_gene == '0':
        sink_type = 'action'
    else:
        sink_type = 'hidden'

    # set sink_id
    mod = len(neurons[sink_type]) # get modulo operator
    sink_id = int(sink_id_gene, 2)+(1 << 7)
    sink_id %= mod # so we don't select a sink with ID too large
    # retrieve sink
    sink_neuron = neurons[sink_type][sink_id]

    # convert weight gene from signed 16 bit bin to dec
    # and scale between -4 and 4
    weight = int(weight_gene, 2) / 8192 * pos_neg

    return source_neuron, sink_neuron, weight

def get_random_genome(num_chromosomes):
    """ constructs a random genome given a number of chromosomes in the genome
    chromosomes encode one connection so num_chromosomes will determine how
    connected the brain encoded by the genome will be"""
    genome = []
    for i in range(num_chromosomes):
        chrom = ''.join(np.random.randint(2, size=32).astype(str))
        chrom = get_hex(chrom)
        genome.append(chrom)
    genome = ' '.join(genome).lower()
    return genome

def get_phenotype(genome):
    """ Given a genome runs construct_connection() for each chromosome in the
    genome to describe all the links in the network"""
    chromosomes = genome.split(' ')
    return [construct_connection(chromosome) for chromosome in chromosomes]

def make_brain_weights(phenotype):
    # hardcoded for 2 inputs to 4 outputs
    # TODO expand capabilities to larger more complex nets

    # initialize weights
    weights = np.zeros((2,4))
    for connection in phenotype:
        # parse phenotype
        source = connection[0]
        sink = connection[1]
        weight = connection[2]
        source_index = neuron_lookup[source]
        sink_index = neuron_lookup[sink] - 5 # -5 because we're temporarily skipping h1-h3 and input neurons can't be sinks
        if (source == 'position_x' or source == 'position_y') and (sink == 'move_up' or sink == 'move_down' or sink == 'move_left' or sink == 'move_right'):
            weights[source_index][sink_index] = weight # write the weight for that connection
    return weights

### Activation Functions ###
def sigmoid(x):
    """ sigmoid activation function """
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """ softmax activation function """
    return np.exp(x) / np.sum(np.exp(x))

### REPRODUCTION FUNCTIONS ###
def reproduce(parents, mutation_rate: float=0.001):
    """Takes a list of parents and a mutation rate to generate a new"""
    #TODO
    num_parents = len(parents)
    len_genome = len(parents[0].genome)
    if num_parents == 1:
        child_genome = mutate(parents[0].genome, mutation_rate=mutation_rate)
        child = Creature(genome=child_genome)
    elif num_parents == 2 and parents[0].genome_size == parents[1].genome_size:
        slice_point = np.random.randint(1000) % len_genome
        parent_copies = [mutate(parent.genome, mutation_rate=mutation_rate) for parent in parents]
        child_genome = parent_copies[0][:slice_point] + parent_copies[1][slice_point:]
        child = Creature(parents[0].genome_size, genome=child_genome)
    return child

def mutate(genome: str, mutation_rate: float):
    """Mutates a genome with a certain mutation rate"""
    # below is kludged together an could be better refactored. The issue is
    # when you call np.random.randint(1) it always returns 0 as the randint
    # so there's a conditional thrown in to account for that discontinuity
    # mutation_rate of 0 turns off mutation calculations
    if mutation_rate != 0:
        die_roll = np.random.randint(int(1 / mutation_rate)) if mutation_rate != 1 else 1
        len_genome = len(genome)
        hexdigits = 'abcdef0123456789'
        if die_roll == 1:
            # there's a potential the pointer points to a space which will break the
            # genome so the below loops until it succesfully replaces a char safely
            while True:
                pointer = np.random.randint(len_genome-1)
                if genome[pointer] != ' ':
                    char = np.random.choice(list(hexdigits))
                    genome = genome[:pointer] + char + genome[pointer+1:]
                    break
    return genome

def select_parents(population, num_parents=2):
    """Selects a number of parents from the population"""
    return [np.random.choice(population) for x in range(num_parents)]

def generate_population(size, genome_size=DEFAULT_GENOME_SIZE, parents=None):
    """ Generates a population of a given size."""
    if parents:
        genome_size = parents[0].genome_size
        population = []
        while len(population) < size:
            population.append(reproduce(select_parents(parents)))
    else:
        population = [Creature(genome_size) for x in range(size)]
    return population

def fitness_test(population):
    survivors = []
    cX, cY = (WORLD_SIZE // 2, WORLD_SIZE // 2)
    r = 10
    for creature in population:
        x, y = creature.position
        if (((x-cX)**2) + ((y-cY)**2)) <= (r**2): # tests if the creature is within the circle
            survivors.append(creature)
    return survivors

### Simulation Functions ###
def save_video(history, epoch_number):
    """ Called after every generation so if the simulation breaks you keep the log of all
    of the generations that completed."""
    width = WORLD_SIZE
    height = WORLD_SIZE
    FPS = 24
    radius = 0

    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(
        f'./videos/generation{epoch_number}.avi',
        fourcc,
        float(FPS),
        (width, height)
    )

    for timestep, log in history.items():
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for point in log:
            cv2.circle(frame, point, radius, (0, 255, 0), -1)
        video.write(frame)

    video.release()

def run_epoch(population, time_steps):
    """ Handles the simulation of each generation. Iterates over
    all the creatures in the population, logs their position at
    the current timestep then calls their action to move them for the next.
    Returns a log and the next generation."""
    generation_size = len(population)
    simulation_log = {}
    for time_step in range(time_steps + 1):
        positions = []
        for creature in population:
            positions.append(creature.position)
            creature.action()
        simulation_log[time_step] = positions
    survivors = fitness_test(population)
    next_gen = generate_population(generation_size, DEFAULT_GENOME_SIZE, survivors)
    return simulation_log, next_gen

def run_simulation(population, epochs, steps_per_epoch):
    """ Takes an population and a number of epochs and feeds the previous
    output generation into the next epoch and saves the video for each epoch """
    generations = {}
    for epoch in range(epochs):
        log, survivors = run_epoch(population, steps_per_epoch)
        generations[epoch] = log
        save_video(log, epoch)
        population = survivors

    return generations
