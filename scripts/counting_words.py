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

for nrn in layer.columns["eat"]:
    nrn.propagate_up(layer.global_counter, 2)

print(layer.global_counter)

