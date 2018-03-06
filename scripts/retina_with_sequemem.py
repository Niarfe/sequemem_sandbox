import sys
sys.path.append("./sequemem")
from neuron import *
from sequemem import *
import random
import numpy as np
import pandas as pd

layer = Sequemem()

files = [
    'data/sent_cats_of_ulthar.txt',
    'data/sent_alice_in_wonderland.txt',
    'data/sent_andersens_fairy_tales_pg1597.txt',
    'data/sent_grimms_fairy_tales_2591-0.txt',
    'data/sent_jungle_book_236-0.txt',
    'data/sent_tao_te_king.txt',
    'data/sent_the_prince.txt',
    'data/sent_thousand_and_one.txt',
    'data/sent_fairy_tales.txt',
    'data/king-james-bible-30.txt.utf-8',
    'data/shakespear-complete-pg100.txt',
    'data/Oxford_English_Dictionary.txt',
    ]

for f in files:
    print("starting: ", f)
    layer.train_from_file_group_line(f)

layer.initialize_frequency_dict()
layer.comparison_frequencies('morgiana', 0.05, 15, True)
