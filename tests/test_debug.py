import sys
sys.path.append("./sequemem")
#from layer import *
from neuron import Neuron
from layer_multi import LayerMulti as Layer
# def tokenize(sentence):
#     return [word.strip('\t\n\r .') for word in sentence.split(' ')]

# def test_tokenize():
#     assert tokenize("we are") == ["we", "are"]


# def test_ccn_layer():
#     input_layer = Layer("input")
#     conv_verb = Layer("verb")
#     conv_agent = Layer("agent")
#     conv_noun = Layer("noun")
#     conv_agent_verb_noun = Layer("avn")

#     conv_verb.predict("ate <verb>")
#     conv_noun.predict("pizza <noun>")
#     conv_agent.predict("Robby <agent>")
#     CONV_LAYER = [conv_agent, conv_verb, conv_noun]
#     for layer in CONV_LAYER:
#         layer.set_learning(False)


#     conv_agent_verb_noun.predict("<agent> <verb> <noun> <avn>")
#     conv_agent_verb_noun.set_learning(False)


#     for layer in CONV_LAYER:
#         input_layer.add_upstream(layer)
#         layer.add_upstream(conv_agent_verb_noun)

#     for layer in CONV_LAYER:
#         layer.reset()
#     conv_agent_verb_noun.reset()

#     input_layer.predict("Robby ate pizza")
#     print("TRAINING DONE")
#     input_layer.predict("Robby ate")
#     print(conv_noun.predict("pizza"))
#     conv_agent_verb_noun.set_learning(True)
#     print(conv_agent_verb_noun.predict("<agent> <verb> <noun>"))


#     for layer in CONV_LAYER:
#         layer.show_status()
#     conv_agent_verb_noun.show_status()
#     assert True

# def test_layer_inactive():
#     layer = Layer()
#     layer.predict("test")
#     layer.set_learning(False)

#     layer.predict("test one two")
#     layer.show_status()
#     assert layer.predict("test") == []

# def test_layer_create():
#     layer = Layer()
#     assert layer.column_keys() == []
#     prediction = layer.predict([["this"]])
#     layer.show_status()
#     assert len(layer.columns["this"]) == 1
#     assert layer.column_keys() == ["this"]
#     assert prediction == []


# def test_layer_create_two_words():
#     layer = Layer()
#     assert layer.column_keys() == []
#     prediction = layer.predict([["are"],["we"]])
#     assert sorted(layer.column_keys()) == sorted(["are","we"])
#     assert sorted(prediction) == sorted([])


# def test_layer_create_two_words_predict_one():
#     layer = Layer()
#     _ = layer.predict([["are"],["we"]])
#     prediction = layer.predict([["are"]])
#     layer.show_status()
#     assert prediction == ["we"]


# def test_layer_create_two_words_predict_3gram():
#     layer = Layer()
#     _ = layer.predict([["are"],["we"],["there"]])
#     prediction = layer.predict([["are"],["we"]])
#     layer.show_status()
#     assert prediction == ["there"]


# def test_layer_create_new_second_sequence():
#     layer = Layer()
#     _ = layer.predict([["are"],["we"],["there"]])
#     prediction = layer.predict([["are"],["we"],["here"]])
#     #layer.show_status()
#     assert sorted(prediction) == sorted([])


# def test_layer_create_two_words_predict_two():
#     layer = Layer()
#     _ = layer.predict([["are"],["we"],["there"]])
#     _ = layer.predict([["are"],["we"],["here"]])
#     prediction = layer.predict([["are"],["we"]])
#     #layer.show_status()
#     assert sorted(prediction) == sorted(["here","there"])




# def test_full_tri_gate():
#     layer = Layer()
#     layer.train_from_file('data/logic_gates_no_about.txt')
#     neuron_count = 0
#     for key, lst in layer.columns.items():
#         print(type(key))
#         neuron_count += len(lst)
#     layer.show_status()
#     assert neuron_count == 13

# def test_sequence_layer():
#     layer = Layer()
#     layer.train_from_file('data/logic_gates_no_about.txt')
#     layer.reset()
#     layer.show_status()
#     prediction = layer.predict(["0","1"])
#     layer.show_status()
#     assert sorted(prediction) == ["0","1"]

# def test_multiple_inputs():
#     layer = Layer("layer")
#     contx = Layer("context")

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
#     layer = Layer()
#     contx = Layer()

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
