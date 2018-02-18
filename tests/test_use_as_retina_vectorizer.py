import sys
sys.path.append("./sequemem")
from neuron import *
from layer_multi import *
import random


def test_loading_a_full_file_of_sentences():
    layer = LayerMulti()

#    layer.train_from_file('data/sentfr.txt')

 # assert len(layer.columns) == 999
