# -*- coding: utf-8 -*-
"""
██╗    ██╗ ██████╗ ██████╗ ███╗   ███╗    ██╗    ██╗ ██████╗ ██████╗ ██╗     ██████╗
██║    ██║██╔═══██╗██╔══██╗████╗ ████║    ██║    ██║██╔═══██╗██╔══██╗██║     ██╔══██╗
██║ █╗ ██║██║   ██║██████╔╝██╔████╔██║    ██║ █╗ ██║██║   ██║██████╔╝██║     ██║  ██║
██║███╗██║██║   ██║██╔══██╗██║╚██╔╝██║    ██║███╗██║██║   ██║██╔══██╗██║     ██║  ██║
╚███╔███╔╝╚██████╔╝██║  ██║██║ ╚═╝ ██║    ╚███╔███╔╝╚██████╔╝██║  ██║███████╗██████╔╝
 ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝     ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝

 Worm World is an evolution simulator. The "worms" are single pixel creatures with a
 hexadecimal genome. The genome encodes the nonzero weights for the neural network
 that controls the worms. Currently there are configurations in the genome that can be
 specified yet are turned off. Different fitness functions can be tested and the worms
 can have larger or smaller brains by changing the default settings value. Worm World
 is a work in progress. There are plans to implement:
 * Simulation improvements
    * allow for in-generation reproduction and include an option to switch from epoch system to continuous simulation
    * mathematical optimizations
    * collision checking
    * new fitness functions
    * data analysis functions for post-mortem
    * create better graphical representation of worm-world
 * environmental factors
    * temperature
    * food
    * water
    * day/night cycle
    * obstacles
    * cover
    * mutagens/radiation
 * add more neurons of all types
    * sensory neurons
        * oscillator neuron (for time-keeping/circadian rhythms)
        * r & theta neurons i.e. directionality neurons
        * temperature sensors
        * population density sensors
    * hidden neurons
        * implement skip connections in order to accomodate hidden neurons
    * action neurons
            * fight/kill
            * eat
            * sleep
 * add more complexities to creatures & their genome
    * encode new features in the genome
        * size
        * food/water requirements (influenced by other features)
        * sleep requirements
        * species (limit sexual reproduction unless mutation)
        * poisonous/venomous
        * poison/venom tolerance
        * mutagen sensitivity
        * promiscuity/libido
 * DONE - split the script into more compartmentalized modules (though could still use improvement)
"""

from utils.utils import *

# senses_array = [x_pos, y_pos]
# can add more but lets go for minimum viable product
# actions_array = [move_left(), move_right(), move_up(), move_down()]

# chromosome
# 0|1011110|0|0011110|1010001000100110
# A|BBBBBBB|C|DDDDDDD|EEEEEEEEEEEEEEEE
# A = source type | 0 = hidden neuron; 1 = sensory neuron
# B = source id | mod number of possible source neurons
# C = sink type | 0 = hidden neuron; 1 = output neuron
# D = sink id | mod number of possible sink neurons
# E = weight | signed 16 bit weight to be divided by scaling factor
# test_chromosome = '000000001000000101101000000000000'

# TODO change epoch and generation naming so they're consistent
# TODO Add hidden layers to the neural nets
# TODO ADD UNIT TESTS

if __name__ == "__main__":
    initial_population = generate_population(GENERATION_SIZE)
    history = run_simulation(initial_population, EPOCHS, TIME_STEPS)
