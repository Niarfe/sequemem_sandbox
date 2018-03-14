import sys
sys.path.append("./sequemem")
from collections import Counter
from neuron import *
from layer_output import LayerCount

def test_layer_count_smoke_test():
    layer = LayerCount()
    layer.load_from_file('data/test_layercounter.txt')

    assert sorted(list(layer.columns.keys())) == ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    assert layer.get_frequency_dict() == (5, {'e': 1.0, 'd': 0.8, 'f': 0.8, 'c': 0.6, 'g': 0.6, 'b': 0.4, 'h': 0.4, 'a': 0.2, 'i': 0.2})
    assert layer.total_neurons == 25
    assert layer.total_sentences == 5
    assert layer.get_counts_for_specific_key('e') == Counter({'e': 5.0, 'f': 4, 'g': 3, 'd': 4, 'h': 2, 'i': 1, 'a': 1, 'b': 2, 'c': 3})
    assert layer.get_counts_for_specific_key('b') == Counter({'a': 1, 'b': 2, 'c': 2, 'd': 2, 'e': 2, 'f': 1})
    assert layer.get_counts_for_specific_key('f') == Counter({'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 4, 'g':3,'h':2, 'i':1})

    assert layer.get_counts_for_specific_key('e', 1) == Counter({'e': 5})
    assert layer.get_counts_for_specific_key('f', 1) == Counter({'f':4})
    assert layer.get_counts_for_specific_key('f', 2) == Counter({'e': 4, 'f': 4, 'g': 3})
    assert layer.get_frequency_dict_word('e') == { 'e': 1, 'd': 0.8, 'f':0.8, 'c':0.6, 'g':0.6, 'b':0.4,'h':0.4,'a':0.2, 'i':0.2}
