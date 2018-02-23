import sys
sys.path.append("./sequemem")

from neuron import Neuron
from layer_multi import LayerMulti as Layer
from layer_output import LayerOutput
from sequemem import Sequemem
def tokenize(sentence):
    return [word.strip('\t\n\r .') for word in sentence.split(' ')]

def test_tokenize():
    assert tokenize("we are") == ["we", "are"]

def test_simple_sequemem():
    brain = Sequemem("sequemem")

    brain.predict(["1", "1","1"], ["ones"])
    brain.predict(["0", "0","0"], ["zero"])

    print("######### BEGIN TEST ASSERTS ###########")
    print("########################################")
    assert brain.predict(["1", "1"], []) == ["1"]
    assert brain.predict(["0", "0"], []) == ["0"]

    brain.predict(["1", "1","0"], ["ones"])
    assert brain.predict(["1", "1"], []) == ["0", "1"]
    assert sorted(brain.get_output_layer_keys()) == sorted([[],['ones']])
    brain.predict(["0","0","0"], [])
    assert sorted(brain.get_output_layer_keys()) == sorted([[],['zero']])


def test_sequemem():
    brain = Sequemem("sequemem")

    brain.predict(["1", "1", "1"], ["and"])
    brain.predict(["1", "0", "0"], ["and"])
    brain.predict(["0", "1", "0"], ["and"])
    brain.predict(["0", "0", "0"], ["and"])
    brain.predict(["1", "1", "1"], ["or"])
    brain.predict(["1", "0", "1"], ["or"])
    brain.predict(["0", "1", "1"], ["or"])
    brain.predict(["0", "0", "0"], ["or"])
    brain.predict(["1", "1", "0"], ["xor"])
    brain.predict(["1", "0", "1"], ["xor"])
    brain.predict(["0", "1", "1"], ["xor"])
    brain.predict(["0", "0", "0"], ["xor"])
    print(" XXXXXXXXXXXXXXXX  CHECK IF LEARNED XXXXXXXXXXXXXX")
    assert brain.predict(["1", "1"], ["and"]) == ["1"]
    assert brain.predict(["1", "0"], ["and"]) == ["0"]
    assert brain.predict(["0", "1"], ["and"]) == ["0"]
    assert brain.predict(["0", "0"], ["and"]) == ["0"]
    assert brain.predict(["1", "1"], ["or"]) == ["1"]
    assert brain.predict(["1", "0"], ["or"]) == ["1"]
    assert brain.predict(["0", "1"], ["or"]) == ["1"]
    assert brain.predict(["0", "0"], ["or"]) == ["0"]
    assert brain.predict(["1", "1"], ["xor"]) == ["0"]
    assert brain.predict(["1", "0"], ["xor"]) == ["1"]
    assert brain.predict(["0", "1"], ["xor"]) == ["1"]
    assert brain.predict(["0", "0"], ["xor"]) == ["0"]
