import sys
sys.path.append("./sequemem")

from neuron import Neuron
from neuron import SimpleNeuron
from layer_multi import LayerMulti as Layer
from layer_output import LayerOutput
from layer_output import LayerSimpleOutput
from sequemem import Sequemem
def tokenize(sentence):
    return [word.strip('\t\n\r .') for word in sentence.split(' ')]

def test_tokenize():
    assert tokenize("we are") == ["we", "are"]

def test_simple_out_layer():
    layer = Layer()

    one = Neuron(layer)
    one.add_key("one")
    two = Neuron(layer)
    two.add_key("two")
    three = Neuron(layer)
    three.add_key("three")
    four = Neuron(layer)
    four.add_key("four")
    five = Neuron(layer)
    five.add_key("five")

    lfibo = [one, two, three, five]
    lsequ = [one, two, three, four]

    simpout = LayerSimpleOutput()

    simpout.add_new("sequ")
    simpout.add_new("fibo")

    nsequ = simpout.get_neuron("sequ")
    nfibo = simpout.get_neuron("fibo")
    simpout.reset()
    ## This is the part you embed into sequemem::hit LEARN
    for n in lsequ:
        n.add_upstream(nsequ)
        nsequ.add_upstream(n)
    for n in lfibo:
        n.add_upstream(nfibo)
        nfibo.add_upstream(n)

    ## After the reset the stuff should just happen naturally in sequemem::hit predict
    simpout.reset()
    for n in lsequ:
        n.set_active()
        n.set_inactive()

    ## Then after runnning it, you can get the counts of the highest, in this case 4
    # and query for the predicted set
    # which you can then overlap to get the right prediction.
    # In this case, three would predict both four and five...
    # but the sets from below differ depending on if it's fobo or sequ
    print(simpout.global_state)
    for n in simpout.global_state["predict"]:
        print(n.keys, n.predict_times)
    predicts = simpout.get_set_of_predicted(4)
    print("Predicts:", predicts)
    assert sorted(list(predicts)) == ["four", "one", "three", "two"]

    simpout.clear("predict")
    for n in lfibo:
        n.set_active()
        n.set_inactive()
    for n in simpout.global_state["predict"]:
        print(n.keys, n.predict_times)
    predicts = simpout.get_set_of_predicted(4)
    print("Predicts:", predicts)
    assert sorted(list(predicts)) == ["five", "one", "three", "two"]

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
    assert brain.predict(["1", "1"], [],["and"]) == ["1"]
    assert brain.predict(["1", "0"], [],["and"]) == ["0"]
    assert brain.predict(["0", "1"], [],["and"]) == ["0"]
    assert brain.predict(["0", "0"], [],["and"]) == ["0"]
    assert brain.predict(["1", "1"], [],["or"]) == ["1"]
    assert brain.predict(["1", "0"], [],["or"]) == ["1"]
    assert brain.predict(["0", "1"], [],["or"]) == ["1"]
    assert brain.predict(["0", "0"], [],["or"]) == ["0"]
    assert brain.predict(["1", "1"], [],["xor"]) == ["0"]
    assert brain.predict(["1", "0"], [],["xor"]) == ["1"]
    assert brain.predict(["0", "1"], [],["xor"]) == ["1"]
    assert brain.predict(["0", "0"], [],["xor"]) == ["0"]
