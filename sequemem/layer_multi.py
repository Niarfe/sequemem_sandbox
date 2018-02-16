from collections import defaultdict
import sys
from neuron import *
def debug(out_str):
    print("[{}]: {}".format(sys._getframe(1).f_code.co_name, out_str))
class LayerMulti:
    def __init__(self, name="anon"):
        self.columns = defaultdict(list)
        self.panic_neuron = Neuron()
        self.activation_neuron = Neuron()
        self.is_learning = True
        self.upstream_layers = []
        self.name = name

    def add_upstream(self, layer):
        self.upstream_layers.append(layer)

    def set_learning(self, bool_state):
        self.is_learning = bool_state
        self.activation_neuron.set_inactive()
        self.panic_neuron.set_inactive()

    def tokenize(self, sentence):
        return [str(word.strip().strip('\r\n\t.')) for word in sentence.strip().split(' ')]

    def train_from_file(self, filepath):
        with open(filepath,'r') as source:
            for sentence in source:
                tokens = self.tokenize(sentence)
                self.predict(tokens)

    def is_like(self, word):
        if type(word) == type([]):
            assert len(word) == 1, "Multiple is_like not supported yet"
            word = word[0]
        return self.predict("{} is".format(word))

    def predict(self, sequence, any_match=False):
        if any_match:
            self.reset_any_match()
        else:
            self.reset()

        if type(sequence) == type(""):
            sequence = self.tokenize(sequence)
        print(sequence, "in predict")
        for input in sequence:
            self.hit(input)

        prediction = list(self.get_predicted())
        for layer in self.upstream_layers:
            print(self.name, " got ", sequence, "and hitting upward with ", prediction)
            layer.predict(prediction)

        return list(prediction)


    def hit(self, column_key):
        column_keys = [column_key] if type(column_key) == type("") else column_key
        assert type(column_keys) == type([])
        print("in hit", column_keys)

        
        pred_keys = self.get_predicted()
        # Gather neurons that will be set active
        prd_nrns = [neuron for _key in column_key for neuron in self._column_get('P', _key)]
        if set(column_keys) == set(pred_keys):
            print(column_keys, "IS EQUAL ", pred_keys)
            is_new = False
        else:
            print(column_keys, "is NOT EQUAL ", pred_keys)
            is_new = True

        act_nrns = self.get_all_actives()

        def process_one_key(_key):
            # UPDATE
            if not is_new:
                self.panic_neuron.set_inactive()
                debug("\tUPDATE set previous chosen predicts to active")
                for prd_nrn in prd_nrns:
                    prd_nrn.set_active()
            else:
                if not self.is_learning:
                    return
                self.panic_neuron.set_inactive()
                nw_nrn = Neuron()
                debug("\tabout to add neuron to column {}".format(_key))
                self.columns[_key].append(nw_nrn)
                for act_nrn in act_nrns:
                    debug("\t\tadding new neuron to active one")
                    act_nrn.add_upstream(nw_nrn)

                debug("\tsetting new neuron active")
                nw_nrn.set_active()
        def process_all_keys(lst_keys):
            # UPDATE
            if not is_new:
                self.panic_neuron.set_inactive()
                debug("\tUPDATE set previous chosen predicts to active")
                for prd_nrn in prd_nrns:
                    prd_nrn.set_active()
            else:
                if not self.is_learning:
                    return
                self.panic_neuron.set_inactive()
                nw_nrn = Neuron()
                
                for _key in lst_keys:
                    debug("\t\tadding neuron to column {}".format(_key))
                    self.columns[_key].append(nw_nrn)
                for act_nrn in act_nrns:
                    debug("\t\tadding new neuron upstream to active")
                    act_nrn.add_upstream(nw_nrn)

                debug("\tsetting new neuron active")
                nw_nrn.set_active()
        
        process_all_keys(column_keys)
        # for key in column_keys:
        #     process_one_key(key)


    def initialize_with_single_column_lit(self, word):
        assert type(sentence) != type(""), "Input to predict with light column is single word"
        self.full_reset()
        self.light_column(word)
        actives = self.get_active_neurons()
        predicted = self.get_predicted()
        return active, predicted


    def predict_clear_then_light_column_on_sentence(self, sentence):
        if type(sentence) == type(""):
            sentence = self.tokenize(sentence)
        # FORGET UPDATE PREDICT
        self.full_reset()       # Forget
        self.light_column(word)
        predicted = self.get_predicted()

    def full_reset(self):
        for key, neurons in self.columns.items():
            for neuron in neurons:
                neuron.set_inactive()

    def light_column(self, key):
        for neuron in self.columns[key]:
            neuron.set_active()


    def reset(self):
        for key, neurons in self.columns.items():
            for neuron in neurons:
                neuron.set_inactive()
        self.activation_neuron.set_active()

    def reset_any_match(self):
        self.full_reset()
        self.activation_neuron.set_active()
        for _, col_neurons in self.columns.items():
            for col_neuron in col_neurons:
                down_neuron = next((nrn for nrn in col_neuron.ns_downstream if nrn in col_neurons), None)
                if down_neuron:
                    continue
                else:
                    col_neuron.set_predict()



    def get_all_actives(self):
        """Get ALL active neurons, even if it's the activation neuron.
            The other function gets all actives in columns only.
        """
        actives = []
        if self.activation_neuron.state == 'A':
            actives.append(self.activation_neuron)
        for k in self.columns.keys():
            colactives = self._column_get('A',k)
            actives.extend(colactives)
        return actives


    def _column_get(self, state, column_key):
        assert type(column_key) == type("")
        if not self.is_learning:
            return []
        return [neuron
            for neuron in self.columns[column_key]
            if neuron.state == state
            ]

    def get_predicted(self):
        predicted = [k
            for k in self.columns.keys()
            if len(self._column_get('P', k)) > 0
            ]
        if len(predicted) > 0:
            return predicted
        else:
            return []
            #return self.columns.keys()


    def column_keys(self):
        return [key for key in self.columns.keys()]

    def show_status(self):
        print("STATUS {}: ".format(self.name), self.activation_neuron.state)
        for key, neurons in self.columns.items():
            print(
                "{}:\t{}".format(
                    key,
                    str([neuron.state for neuron in neurons])
                    )
            )
    def _get_neurons(self, state):
        return [neuron
            for key in self.columns.keys()
            for neuron in self._column_get(state, key)
            ]
    def get_active_neurons(self):
        return self._get_neurons('A')
    def get_predicted_neurons(self):
        return self._get_neurons('P')



