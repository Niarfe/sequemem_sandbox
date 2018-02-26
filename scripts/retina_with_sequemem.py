import sys
sys.path.append("./sequemem")
from neuron import *
from layer_multi import *
import random
import numpy as np
import pandas as pd

layer = LayerMulti()

print("staring cats")
layer.train_from_file_group_line('data/sent_cats_of_ulthar')
print("starting alice in w")
layer.train_from_file_group_line('data/sent_alice_in_wonderland.txt')
print("starting andersen")
layer.train_from_file_group_line('data/sent_andersens_fairy_tales_pg1597.txt')
print("starting grimms")
layer.train_from_file_group_line('data/sent_grimms_fairy_tales_2591-0.txt')
print("starting jungle")
layer.train_from_file_group_line('data/sent_jungle_book_236-0.txt')
print("starting tao")
layer.train_from_file_group_line('data/sent_tao_te_king.txt')
print("starting the price")
layer.train_from_file_group_line('data/sent_the_prince.txt')
print("starting one thousand")
layer.train_from_file_group_line('data/sent_thousand_and_one.txt')
print("starting fairy tales")
layer.train_from_file_group_line('data/sent_fairy_tales.txt')









