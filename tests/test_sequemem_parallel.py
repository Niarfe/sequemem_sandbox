import sys
sys.path.append("./sequemem")
from neuron import Neuron
from sequemem import Sequemem


def test_ccn_layer():
    input_layer = Sequemem("input")
    conv_verb = Sequemem("verb")
    conv_agent = Sequemem("agent")
    conv_noun = Sequemem("noun")
    conv_agent_verb_noun = Sequemem("avn")

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
    assert True

def test_layer_inactive():
    layer = Sequemem()
    layer.predict("test")
    layer.set_learning(False)

    layer.predict("test one two")
    layer.show_status()
    assert layer.predict("test") == []

def test_layer_create():
    layer = Sequemem()
    assert len(layer.column_keys()) == len([])
    prediction = layer.predict([["this"]])
    layer.show_status()
    assert len(layer.columns["this"]) == 1
    assert layer.column_keys() == ["this"]
    assert prediction == []


def test_layer_create_two_words():
    layer = Sequemem()
    assert layer.column_keys() == []
    prediction = layer.predict([["are"],["we"]])
    assert sorted(layer.column_keys()) == sorted(["are","we"])
    assert sorted(prediction) == sorted([])


def test_layer_create_two_words_predict_one():
    layer = Sequemem()
    _ = layer.predict([["are"],["we"]])
    prediction = layer.predict([["are"]])
    layer.show_status()
    assert prediction == ["we"]


def test_layer_create_two_words_predict_3gram():
    layer = Sequemem()
    _ = layer.predict([["are"],["we"],["there"]])
    prediction = layer.predict([["are"],["we"]])
    layer.show_status()
    assert prediction == ["there"]


def test_layer_create_new_second_sequence():
    layer = Sequemem()
    _ = layer.predict([["are"],["we"],["there"]])
    prediction = layer.predict([["are"],["we"],["here"]])
    layer.show_status()
    assert sorted(prediction) == sorted([])


def test_layer_create_two_words_predict_two():
    layer = Sequemem()
    _ = layer.predict([["are"],["we"],["there"]])
    _ = layer.predict([["are"],["we"],["here"]])
    prediction = layer.predict([["are"],["we"]])
    layer.show_status()
    assert sorted(prediction) == sorted(["here","there"])


def test_full_tri_gate():
    layer = Sequemem()
    layer.train_from_file('data/logic_gates_no_about.txt')
    neuron_count = 0
    for key, lst in layer.columns.items():
        print(type(key))
        neuron_count += len(lst)
    layer.show_status()
    assert neuron_count == 13

def test_sequence_layer():
    layer = Sequemem()
    layer.train_from_file('data/logic_gates_no_about.txt')
    layer.reset()
    layer.show_status()
    prediction = layer.predict(["0","1"])
    layer.show_status()
    assert sorted(prediction) == ["0","1"]

# def test_multiple_inputs():
#     layer = Sequemem("layer")
#     contx = Sequemem("context")

#     context = "upper"
#     sequence = [["A"],["B"],["C"]]
#     cumulative = []
#     for letter in sequence:
#         cumulative.append(letter)
#         layer.predict(cumulative)
#         contx.predict([[context]])
#         lact = layer.get_active_neurons()
#         cact = contx.get_active_neurons()
#         for cn in cact:
#             for ln in lact:
#                 cn.add_upstream(ln)
#                 ln.add_upstream(cn)

#     assert layer.predict(["A"]) == ["B"]
#     assert contx.get_all_actives() == []
#     assert contx.get_predicted() == ["upper"]

#     assert layer.predict([["A"],["B"]]) == ["C"]
#     assert contx.get_all_actives() == []
#     assert contx.get_predicted() == ["upper"]

#     context = "lower"
#     sequence = [["A"],["b"],["c"]]
#     cumulative = []
#     for letter in sequence:
#         cumulative.append(letter)
#         layer.predict(cumulative)
#         contx.predict([[context]])
#         lact = layer.get_active_neurons()
#         cact = contx.get_active_neurons()
#         for cn in cact:
#             for ln in lact:
#                 cn.add_upstream(ln)
#                 ln.add_upstream(cn)

#     pred_layer = layer.predict(["A"])
#     assert pred_layer == ["B","b"]
#     layer.show_status()
#     assert contx.get_all_actives() == []
#     assert contx.get_predicted() == ["upper", "lower"]
#     contx.predict([[context]])
#     pred_layer2 = layer.get_predicted()

#     layer.show_status()
#     contx.show_status()
#     final_prediction = list(set(pred_layer) & set(pred_layer2))
#     assert final_prediction == ["b"]


# def test_classic_bass_example():
#     layer = Sequemem()
#     contx = Sequemem()

#     context = ["music"]
#     sequence = [["bass"],["is"],["instrument"]]
#     cumulative = []
#     for letter in sequence:
#         cumulative.append(letter)
#         layer.predict(cumulative)
#         contx.predict([context])
#         lact = layer.get_active_neurons()
#         cact = contx.get_active_neurons()
#         for cn in cact:
#             for ln in lact:
#                 cn.add_upstream(ln)
#                 ln.add_upstream(cn)

#     assert layer.predict([["bass"]]) == ["is"]
#     assert contx.get_all_actives() == []
#     assert contx.get_predicted() == ["music"]

#     assert layer.predict([["bass"],["is"]]) == ["instrument"]
#     assert contx.get_all_actives() == []
#     assert contx.get_predicted() == ["music"]

#     context = ["fishing"]
#     sequence = [["bass"],["is"],["fish"]]
#     cumulative = []
#     for letter in sequence:
#         cumulative.append(letter)
#         layer.predict(cumulative)
#         contx.predict([context])
#         lact = layer.get_active_neurons()
#         cact = contx.get_active_neurons()
#         for cn in cact:
#             for ln in lact:
#                 cn.add_upstream(ln)
#                 ln.add_upstream(cn)

#     pred_layer = layer.predict([["bass"],["is"]])
#     assert pred_layer == ["instrument", "fish"]
#     layer.show_status()
#     assert contx.get_all_actives() == []
#     assert sorted(contx.get_predicted()) == sorted(["music", "fishing"])
#     contx.predict([["fishing"]])
#     pred_layer2 = layer.get_predicted()

#     layer.show_status()
#     contx.show_status()
#     final_prediction = list(set(pred_layer) & set(pred_layer2))
#     assert final_prediction == ["fish"]







# import os
# def test_act_in_response():
#     inp_layer = Sequemem()
#     act_layer = Sequemem()

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



