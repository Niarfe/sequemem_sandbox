import sys
sys.path.append("./sequemem")
from sequemem import *

def tokenize(sentence):
    return [word.strip('\t\n\r .') for word in sentence.split(' ')]

def test_tokenize():
    assert tokenize("we are") == ["we", "are"]


def test_ccn_layer():
    input_layer = Layer("input")
    conv_verb = Layer("verb")
    conv_agent = Layer("agent")
    conv_noun = Layer("noun")
    conv_agent_verb_noun = Layer("avn")

    conv_verb.predict("ate <verb>")
    conv_noun.predict("pizza <noun>")
    conv_agent.predict("Robby <agent>")
    CONV_LAYER = [conv_agent, conv_verb, conv_noun]
    for layer in CONV_LAYER:
        layer.set_learning(False)


    conv_agent_verb_noun.predict("<agent> <verb> <noun> <avn>")
    conv_agent_verb_noun.set_learning(False)


    for layer in CONV_LAYER:
        input_layer.add_upstream(layer)
        layer.add_upstream(conv_agent_verb_noun)

    for layer in CONV_LAYER:
        layer.reset()
    conv_agent_verb_noun.reset()

    input_layer.predict("Robby ate pizza")
    print("TRAINING DONE")
    input_layer.predict("Robby ate")
    print(conv_noun.predict("pizza"))
    conv_agent_verb_noun.set_learning(True)
    print(conv_agent_verb_noun.predict("<agent> <verb> <noun>"))


    for layer in CONV_LAYER:
        layer.show_status()
    conv_agent_verb_noun.show_status()
    assert False

def test_layer_inactive():
    layer = Layer()
    layer.predict("test")
    layer.set_learning(False)

    layer.predict("test one two")
    layer.show_status()
    assert layer.predict("test") == []

def test_layer_create():
    layer = Layer()
    assert layer.column_keys() == []
    prediction = layer.predict(["this"])
    layer.show_status()
    assert len(layer.columns["this"]) == 1
    assert layer.column_keys() == ["this"]
    assert prediction == []


def test_layer_create_two_words():
    layer = Layer()
    assert layer.column_keys() == []
    prediction = layer.predict(["are","we"])
    assert sorted(layer.column_keys()) == sorted(["are","we"])
    assert sorted(prediction) == sorted([])


def test_layer_create_two_words_predict_one():
    layer = Layer()
    _ = layer.predict(["are","we"])
    prediction = layer.predict(["are"])
    layer.show_status()
    assert prediction == ["we"]


def test_layer_create_two_words_predict_3gram():
    layer = Layer()
    _ = layer.predict(["are","we","there"])
    prediction = layer.predict(["are","we"])
    layer.show_status()
    assert prediction == ["there"]


def test_layer_create_new_second_sequence():
    layer = Layer()
    _ = layer.predict(["are","we","there"])
    prediction = layer.predict(["are","we","here"])
    layer.show_status()
    assert sorted(prediction) == sorted([])


def test_layer_create_two_words_predict_two():
    layer = Layer()
    _ = layer.predict(["are","we","there"])
    _ = layer.predict(["are","we","here"])
    prediction = layer.predict(["are","we"])
    layer.show_status()
    assert sorted(prediction) == sorted(["here","there"])

import random
def test_several_sentences():
    layer = Layer()
    layer.train_from_file('data/cortical_example1.1.txt')

    sentence = "fox eat"
    print(sentence)
    print(layer.predict(sentence))

    similar_to_fox = layer.is_like(layer.is_like("fox"))
    similar_to_fox.remove("fox")

    print("Similar to fox: ", similar_to_fox)
    random_choice = random.choice(similar_to_fox)
    print("Random Choice: ", random_choice)

    res = layer.predict([ random_choice ] + ["eat"])
    print("Well pick this guy and see what he/she eats: ",res)

def test_full_tri_gate():
    layer = Layer()
    layer.train_from_file('data/logic_gates_no_about.txt')
    neuron_count = 0
    for key, lst in layer.columns.items():
        print(type(key))
        neuron_count += len(lst)
    layer.show_status()
    assert neuron_count == 13

def test_sequence_layer():
    layer = Layer()
    layer.train_from_file('data/logic_gates_no_about.txt')
    layer.reset()
    layer.show_status()
    prediction = layer.predict(["0","1"])
    layer.show_status()
    assert sorted(prediction) == ["0","1"]

def test_prep_prediction_new():
    layer = Layer()
    layer.train_from_file('data/cortical_example1.1.txt')
    animal = "fox"
    verb = "eat"
    sentence = "{} {}".format(animal, verb)
    print(sentence)
    print("IS LIKE", layer.is_like(animal))

    similar_to_fox = layer.is_like(layer.is_like(animal))
    similar_to_fox.remove(animal)

    res = {}
    collected = []
    for simile in similar_to_fox:
        res[simile] = layer.predict([simile, verb])
        collected.extend(layer.predict([simile, verb]))

    finalcol = list(set(collected))
    for k, v in res.items():
        print(k, v)
    print(finalcol)

    assert sorted(finalcol) == sorted(['flies', 'squirrel', 'cow', 'salmon', 'rodent', 'rabbit', 'mice'])

def test_logic():
    layer = Layer()
    layer.predict("man is mortal")
    layer.predict("homer is man")
    logic = Logic(layer)
    assert logic.double_ism("homer") == ['mortal'], "Cheesy first try at logic"


def test_brain_create():
    brain = Brain()
    assert len(brain.layers) == 2, "We should have a cortex and a hypothalamus"

def test_multiple_inputs():
    layer = Layer()
    contx = Layer()

    context = "upper"
    sequence = ["A","B","C"]
    cumulative = []
    for letter in sequence:
        cumulative.append(letter)
        layer.predict(cumulative)
        contx.predict([context])
        lact = layer.get_active_neurons()
        cact = contx.get_active_neurons()
        for cn in cact:
            for ln in lact:
                cn.add_upstream(ln)
                ln.add_upstream(cn)

    assert layer.predict(["A"]) == ["B"]
    assert contx.get_all_actives() == []
    assert contx.get_predicted() == ["upper"]

    assert layer.predict(["A", "B"]) == ["C"]
    assert contx.get_all_actives() == []
    assert contx.get_predicted() == ["upper"]

    context = "lower"
    sequence = ["A","b","c"]
    cumulative = []
    for letter in sequence:
        cumulative.append(letter)
        layer.predict(cumulative)
        contx.predict([context])
        lact = layer.get_active_neurons()
        cact = contx.get_active_neurons()
        for cn in cact:
            for ln in lact:
                cn.add_upstream(ln)
                ln.add_upstream(cn)

    pred_layer = layer.predict(["A"])
    assert pred_layer == ["B","b"]
    layer.show_status()
    assert contx.get_all_actives() == []
    assert contx.get_predicted() == ["upper", "lower"]
    contx.predict("lower")
    pred_layer2 = layer.get_predicted()

    layer.show_status()
    contx.show_status()
    final_prediction = list(set(pred_layer) & set(pred_layer2))
    assert final_prediction == ["b"]


def test_classic_bass_example():
    layer = Layer()
    contx = Layer()

    context = "music"
    sequence = ["bass","is","instrument"]
    cumulative = []
    for letter in sequence:
        cumulative.append(letter)
        layer.predict(cumulative)
        contx.predict([context])
        lact = layer.get_active_neurons()
        cact = contx.get_active_neurons()
        for cn in cact:
            for ln in lact:
                cn.add_upstream(ln)
                ln.add_upstream(cn)

    assert layer.predict(["bass"]) == ["is"]
    assert contx.get_all_actives() == []
    assert contx.get_predicted() == ["music"]

    assert layer.predict(["bass", "is"]) == ["instrument"]
    assert contx.get_all_actives() == []
    assert contx.get_predicted() == ["music"]

    context = "fishing"
    sequence = ["bass","is","fish"]
    cumulative = []
    for letter in sequence:
        cumulative.append(letter)
        layer.predict(cumulative)
        contx.predict([context])
        lact = layer.get_active_neurons()
        cact = contx.get_active_neurons()
        for cn in cact:
            for ln in lact:
                cn.add_upstream(ln)
                ln.add_upstream(cn)

    pred_layer = layer.predict(["bass","is"])
    assert pred_layer == ["instrument", "fish"]
    layer.show_status()
    assert contx.get_all_actives() == []
    assert contx.get_predicted() == ["music", "fishing"]
    contx.predict("fishing")
    pred_layer2 = layer.get_predicted()

    layer.show_status()
    contx.show_status()
    final_prediction = list(set(pred_layer) & set(pred_layer2))
    assert final_prediction == ["fish"]


def test_brain_train():
    brain = Brain()
    brain.train_from_file('data/disambiguation.txt')
    prediction = brain.predict(["bass", "is"], "music")
    print(prediction)
    assert prediction  == ["instrument"], prediction

def test_logic_gates_with_brain():
    brain = Brain()
    brain.train_from_file('data/logic_gates.txt')
    prediction = brain.predict(["0", "1"], "or")
    print(prediction)
#    brain.show_status()
    assert prediction  == ["1"], prediction

def test_logic_gates_with_brain_full_monte():
    brain = Brain()
    brain.train_from_file('data/logic_gates.txt')
    assert brain.predict(["1", "1"], "and") == ["1"]
    assert brain.predict(["1", "0"], "and") == ["0"]
    assert brain.predict(["0", "1"], "and") == ["0"]
    assert brain.predict(["0", "0"], "and") == ["0"]
    assert brain.predict(["1", "1"], "or") == ["1"]
    assert brain.predict(["1", "0"], "or") == ["1"]
    assert brain.predict(["0", "1"], "or") == ["1"]
    assert brain.predict(["0", "0"], "or") == ["0"]
    assert brain.predict(["1", "1"], "xor") == ["0"]
    assert brain.predict(["1", "0"], "xor") == ["1"]
    assert brain.predict(["0", "1"], "xor") == ["1"]
    assert brain.predict(["0", "0"], "xor") == ["0"]

def test_triple_context():
    brain = Brain()
    brain.train_from_file('data/long_context_test.txt')
    assert brain.predict(["bass", "is"], "music") == ["instrument"]
    assert brain.predict(["viola", "is"], "music") == ["instrument"]
    assert brain.predict(["bass", "is"], "fishing") == ["fish"]
    assert brain.predict(["viola", "is"], "names") == ["name"]
    assert brain.predict(["salmon", "is"], "fishing") == ["fish"]
    assert brain.predict(["efrain", "is"], "names") == ["name"]

# import os
# def test_act_in_response():
#     inp_layer = Layer()
#     act_layer = Layer()

#     insentence = "hello tex"
#     response = "hi_this_is_the_expected_response"

#     inp_layer.predict(insentence)
#     inp_active = inp_layer.get_active_neurons()

#     act_layer.predict(response)
#     act_active = act_layer.get_active_neurons()
#     inp_active[0].add_upstream(act_active[0])

#     inp_layer.predict(insentence)
#     reflex = act_layer.get_predicted()

#     os.system("say {}".format(reflex[0].split('_')))
#     assert reflex[0] == response

# def test_async_memory():
#     layer = Layer()
#     layer.train_from_file('data/sent_cats_of_ulthar')

#     sentence = ["uttered"]
#     prediction = [""]
#     while True:
#         prediction = layer.predict(sentence, True)
#         print(prediction)
#         if len(prediction) == 1:
#             sentence.extend(prediction)
#         else:
#             break
#     assert " ".join(sentence) == "uttered his petition there seemed to form overhead the shadowy nebulous figures of exotic things of hybrid creatures crowned with horn flanked discs"



