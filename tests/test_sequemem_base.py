import sys
sys.path.append("./sequemem")
from neuron import *
from sequemem import Sequemem
import random

def test_sequencer():
    layer = Sequemem()
    assert layer.sequencer("this is a string") == [["this"], ["is"],["a"],["string"]]

def test_sequencer_actually_works_with_layer():
    layer = Sequemem()
    layer.predict("this is a sequence")
    assert layer.predict("this is a") == ["sequence"]
    layer.predict("this is a foobar")
    assert layer.predict("this is a") == ["foobar", "sequence"]

def test_should_behave_as_it_was_sequenced():
    layer = Sequemem()
    sentence = layer.sequencer("this is a sequence")
    layer.predict(sentence)
    assert layer.predict("this is a") == ["sequence"]

    another_sentence = layer.sequencer("this is a foobar")
    layer.predict(another_sentence)
    assert layer.predict("this is a") == ["foobar", "sequence"]


def test_clarify_behavior_with_array_of_arryas():
    layer = Sequemem()
    sentence = [["this is"], ["a sequence"]]
    layer.predict(sentence)
    assert layer.predict([["this is"]]) == ["a sequence"]

    another_sentence = [["this is"],["a foobar"]]
    layer.predict(another_sentence)
    assert sorted(layer.predict([["this is"]])) == ["a foobar", "a sequence"]

def test_layer_multi_basic_two_step():
    layer = Sequemem()

    layer.predict([['a','b'], ['A','B']])
    layer.show_status()
    assert layer.predict([['a','b']]) == ['A','B']

def test_layer_multi_triple():
    layer = Sequemem()

    sequence = [
        ['a','b'],
        ['1','2'],
        ['a','c']
    ]

    layer.predict(sequence)

    assert layer.predict(sequence[:2]) == sequence[2]

def test_layer_multi_triple_imbalanced():
    layer = Sequemem()

    sequence = [
        ['a','b'],
        ['1','2','3'],
        ['d']
    ]

    layer.predict(sequence)

    assert layer.predict(sequence[:2]) == sequence[2]

def test_different_sequence_mid_retrieval():
    layer = Sequemem("multi")

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
    assert sorted(layer.predict(sequence1[:1])) == sorted(['1','2','3','4'])
    assert sorted(layer.predict(sequence2[:1])) == sorted(['1','2','3','4'])

    assert sorted(layer.predict(sequence1[:2])) == sorted(['d'])
    assert sorted(layer.predict(sequence2[:2])) == sorted(['d'])

def test_different_sequence_end_points():
    layer = Sequemem("multi")

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
        ['e']
    ]

    layer.predict(sequence2)
    layer.show_status()
    assert sorted(layer.predict(sequence1[:1])) == sorted(['1','2','3','4'])
    assert sorted(layer.predict(sequence2[:1])) == sorted(['1','2','3','4'])

    assert sorted(layer.predict(sequence1[:2])) == sorted(['d'])
    assert sorted(layer.predict(sequence2[:2])) == sorted(['e'])

def test_several_sentences():
    layer = Sequemem()
    layer.train_from_file('data/cortical_example1.1.txt')
    print("\n=============\n")
    sentence = [["fox"], ["eat"]]

    print("\nLAYER PREDICT: ", layer.predict(sentence))

    is_like_fox = layer.is_like(["fox"])
    print("\nIS LIKE FOX: ", is_like_fox)
    similar_to_fox = layer.is_like(is_like_fox)

    print("\nSIMILAR TO: ", similar_to_fox)
    layer.show_status()
    similar_to_fox.remove("fox")

    print("\nSimilar to fox: ", similar_to_fox)
    random_choice = random.choice(similar_to_fox)
    print("\nRandom Choice: ", random_choice)

    res = layer.predict([[ random_choice ], ["eat"]])
    print("Well pick this guy and see what he/she eats: ",res)


def test_prep_prediction_new():
    layer = Sequemem()
    layer.train_from_file('data/cortical_example1.1.txt')

    animal = "fox"
    verb = "eat"
    sentence = "{} {}".format(animal, verb)
    print(sentence)
    print("IS LIKE", layer.is_like([animal]))

    similar_to_fox = layer.is_like(layer.is_like([animal]))
    similar_to_fox.remove(animal)

    res = {}
    collected = []
    for simile in similar_to_fox:
        res[simile] = layer.predict([[simile], [verb]])
        collected.extend(layer.predict([[simile], [verb]]))

    finalcol = list(set(collected))
    for k, v in res.items():
        print(k, v)
    print(finalcol)

    assert sorted(finalcol) == sorted(['flies', 'squirrel', 'cow', 'salmon', 'rodent', 'rabbit', 'mice'])
