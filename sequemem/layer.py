# from collections import defaultdict
# from neuron import *
# class Layer:
#     def __init__(self, name="anon"):
#         self.columns = defaultdict(list)
#         self.panic_neuron = Neuron()
#         self.activation_neuron = Neuron()
#         self.is_learning = True
#         self.upstream_layers = []
#         self.name = name

#     def add_upstream(self, layer):
#         self.upstream_layers.append(layer)

#     def set_learning(self, bool_state):
#         self.is_learning = bool_state
#         self.activation_neuron.set_inactive()
#         self.panic_neuron.set_inactive()

#     def tokenize(self, sentence):
#         return [str(word.strip().strip('\r\n\t.')) for word in sentence.strip().split(' ')]

#     def train_from_file(self, filepath):
#         with open(filepath,'r') as source:
#             for sentence in source:
#                 tokens = self.tokenize(sentence)
#                 self.predict(tokens)

#     def is_like(self, word):
#         if type(word) == type([]):
#             assert len(word) == 1, "Multiple is_like not supported yet"
#             word = word[0]
#         return self.predict("{} is".format(word))

#     def predict(self, sequence, any_match=False):
#         if any_match:
#             self.reset_any_match()
#         else:
#             self.reset()
#         if type(sequence) == type(""):
#             sequence = self.tokenize(sequence)

#         for input in sequence:
#             self.hit(str(input))

#         prediction = list(self.get_predicted())
#         for layer in self.upstream_layers:
#             print(self.name, " got ", sequence, "and hitting upward with ", prediction)
#             layer.predict(prediction)

#         return list(prediction)

#     def initialize_with_single_column_lit(self, word):
#         assert type(sentence) != type(""), "Input to predict with light column is single word"
#         self.full_reset()
#         self.light_column(word)
#         actives = self.get_active_neurons()
#         predicted = self.get_predicted()
#         return active, predicted


#     def predict_clear_then_light_column_on_sentence(self, sentence):
#         if type(sentence) == type(""):
#             sentence = self.tokenize(sentence)
#         # FORGET UPDATE PREDICT
#         self.full_reset()       # Forget
#         self.light_column(word)
#         predicted = self.get_predicted()

#     def full_reset(self):
#         for key, neurons in self.columns.items():
#             for neuron in neurons:
#                 neuron.set_inactive()

#     def light_column(self, key):
#         for neuron in self.columns[key]:
#             neuron.set_active()


#     def reset(self):
#         for key, neurons in self.columns.items():
#             for neuron in neurons:
#                 neuron.set_inactive()
#         self.activation_neuron.set_active()

#     def reset_any_match(self):
#         self.full_reset()
#         self.activation_neuron.set_active()
#         for _, col_neurons in self.columns.items():
#             for col_neuron in col_neurons:
#                 down_neuron = next((nrn for nrn in col_neuron.ns_downstream if nrn in col_neurons), None)
#                 if down_neuron:
#                     continue
#                 else:
#                     col_neuron.set_predict()

#     def hit(self, column_key):
#         # Gather neurons that will be set active
#         prd_nrns = self._column_get('P', column_key)
#         act_nrns = self.get_all_actives()

#         # FORGET
#         self.full_reset()

#         # UPDATE
#         if len(prd_nrns) > 0:
#             self.panic_neuron.set_inactive()
#             # UPDATE set previous chosen predicts to active
#             for prd_nrn in prd_nrns:
#                 prd_nrn.set_active()
#         else:
#             if not self.is_learning:
#                 print("{} hit witn {} returning".format(self.name, column_key))
#                 return
#             print("{} adding {} neuron".format(self.name, column_key))
#             self.panic_neuron.set_inactive()
#             nw_nrn = Neuron()
#             for act_nrn in act_nrns:
#                 act_nrn.add_upstream(nw_nrn)
#             self.columns[column_key].append(nw_nrn)
#             nw_nrn.set_active()


#     def get_all_actives(self):
#         """Get ALL active neurons, even if it's the activation neuron.
#             The other function gets all actives in columns only.
#         """
#         actives = []
#         if self.activation_neuron.state == 'A':
#             actives.append(self.activation_neuron)
#         for k in self.columns.keys():
#             colactives = self._column_get('A',k)
#             actives.extend(colactives)
#         return actives


#     def _column_get(self, state, column_key):
#         if not self.is_learning:
#             return []
#         return [neuron
#             for neuron in self.columns[column_key]
#             if neuron.state == state
#             ]

#     def get_predicted(self):
#         predicted = [k
#             for k in self.columns.keys()
#             if len(self._column_get('P', k)) > 0
#             ]
#         if len(predicted) > 0:
#             return predicted
#         else:
#             return []
#             #return self.columns.keys()


#     def column_keys(self):
#         return [key for key in self.columns.keys()]

#     def show_status(self):
#         print("STATUS {}: ".format(self.name), self.activation_neuron.state)
#         for key, neurons in self.columns.items():
#             print(
#                 "{}:\t{}".format(
#                     key,
#                     str([neuron.state for neuron in neurons])
#                     )
#             )
#     def _get_neurons(self, state):
#         return [neuron
#             for key in self.columns.keys()
#             for neuron in self._column_get(state, key)
#             ]
#     def get_active_neurons(self):
#         return self._get_neurons('A')
#     def get_predicted_neurons(self):
#         return self._get_neurons('P')



