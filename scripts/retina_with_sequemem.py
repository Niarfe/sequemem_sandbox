import sys
sys.path.append("./sequemem")
from neuron import *
from layer_output import LayerCount 
import random
import numpy as np
import pandas as pd

layer = LayerCount()

files = [
    'data/00_clean/sent_alice_in_wonderland.txt',
#    'data/00_clean/sent_andersens_fairy_tales_pg1597.txt',
#    'data/00_clean/sent_cats_of_ulthar.txt',
#    'data/00_clean/sent_fairy_tales.txt',
#    'data/00_clean/sent_grimms_fairy_tales.txt',
#    'data/00_clean/sent_iris_fairy_tales.txt',
#    'data/00_clean/sent_jungle_book_236-0.txt',
#    'data/00_clean/sent_king_james_bible.txt',
#    'data/00_clean/sent_shakespear.txt',
#    'data/00_clean/sent_tao_te_king.txt',
#    'data/00_clean/sent_the_prince.txt',
#    'data/00_clean/sent_thousand_and_one.txt',
#    'data/00_clean/sents_fables_la_fontaine.txt',
]

for f in files:
    print("starting: ", f)
    layer.train_from_file(f)

layer.get_frequency_dict()
layer.comparison_frequencies('Alice', 0.05, 15, True)
