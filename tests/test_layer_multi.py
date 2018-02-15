import sys
sys.path.append("./sequemem")
from layer_multi import *


def tokenize(sentence):
    return [word.strip('\t\n\r .') for word in sentence.split(' ')]

def test_tokenize():
    assert tokenize("we are") == ["we", "are"]

def test_layer_multi_basic_two_step():
    layer = LayerMulti()

    layer.predict([['a','b'], ['A','B']])
    layer.show_status()
    assert layer.predict([['a','b']]) == ['A','B']

def test_layer_multi_triple():
    layer = LayerMulti()

    sequence = [
        ['a','b'],
        ['1','2'],
        ['a','c']
    ]

    layer.predict(sequence)

    assert layer.predict(sequence[:2]) == sequence[2]

def test_layer_multi_triple_imbalanced():
    layer = LayerMulti()

    sequence = [
        ['a','b'],
        ['1','2','3'],
        ['d']
    ]

    layer.predict(sequence)

    assert layer.predict(sequence[:2]) == sequence[2]

def test_different_sequence_mid_retrieval():
    layer = LayerMulti("multi")

    sequence1 = [
        ['a','b'],
        ['1','2','3'],
        ['d']
    ]

    layer.predict(sequence1)
    layer.show_status()

    sequence2 = [
        ['a','b'],
        ['1','2','4'],
        ['d']
    ]

    layer.predict(sequence2)
    layer.show_status()
    assert sorted(layer.predict(sequence1[:2])) == sorted(['1','2','3','4'])
    assert sorted(layer.predict(sequence2[:1])) == sorted(['1','2','3','4'])

# def test_different_sequence_diff_endpoint():
#     layer = LayerMulti("multi")

#     sequence1 = [
#         ['a','b'],
#         ['1','2','3'],
#         ['d']
#     ]

#     layer.predict(sequence1)
#     layer.show_status()

#     sequence2 = [
#         ['a','b'],
#         ['1','2','4'],
#         ['e']
#     ]

#     layer.predict(sequence2)
#     layer.show_status()
#     assert sorted(layer.predict(sequence1[:2])) == sorted(sequence1[2])
#     assert sorted(layer.predict(sequence2[:2])) == sorted(sequence2[2])