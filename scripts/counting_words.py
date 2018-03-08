import sys
sys.path.append("./sequemem")
from neuron import *
from sequemem import *
from layer_output import LayerCount
import random
import numpy as np
import pandas as pd

layer = LayerCount()
layer.train_from_file('data/cortical_example1.1.txt')


print(layer.initialize_frequency_dict())

print(layer.get_counts_for_specific_key('eat'))

