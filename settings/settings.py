""" Contains global settings for program """

import numpy as np

DEFAULT_GENOME_SIZE = 20
DEFAULT_MUTATION_RATE = 0.01
DEFAULT_STEP_SIZE = 1
GENERATION_SIZE = 1000
WORLD_SIZE = 128
TIME_STEPS = 150
EPOCHS = 50

# make lookup dictionary
available_neurons = [
    'position_x',
    'position_y',
    'h1',
    'h2',
    'h3',
    'move_up',
    'move_down',
    'move_left',
    'move_right'
]
neuron_lookup = dict(zip(available_neurons, np.arange(len(available_neurons))))

# h1-h3 temporarily disabled, but due to gene code they have to exist for now
NEURONS = {
    'sensory': {
        0: 'position_x',
        1: 'position_y',
    },

    'hidden': {
        0: 'h1',
        1: 'h2',
        2: 'h3',
    },

    'action': {
        0: 'move_left',
        1: 'move_right',
        2: 'move_up',
        3: 'move_down',
    },
}
