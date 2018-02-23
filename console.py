import sys
sys.path.append("./sequemem")
from neuron import *
from layer import *
from layer_multi import *
import random

layer = LayerMulti()

layer.train_from_file('data/sent_cats_of_ulthar.txt')


