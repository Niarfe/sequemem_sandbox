from collections import defaultdict
import sys
from neuron import *
from layer_output import LayerOutput

from collections import Counter

def debug(out_str):
    return
    print("[{}]: {}".format(sys._getframe(1).f_code.co_name, out_str))

class Sequemem:

    def __init__(self, name="anon"):
        self.columns = defaultdict(set)
        self.output_layer = LayerOutput()
        self.output = None
        self.global_state = {
            "active": set(),
            "predict": set(),
            "inactive": set()
            }
        self.panic_neuron = Neuron(self)
        self.activation_neuron = Neuron(self)
        self.is_learning = True
        self.upstream_layers = []
        self.name = name

        self.act_nrns = []
        self.all_prd_nrns = []
        self.inactives = []

    def get_output_layer_keys(self):
        actives = [key
            for nrn in self.output_layer.global_state["active"]
            for key in nrn.keys]
        predict = [key
            for nrn in self.output_layer.global_state["predict"]
            for key in nrn.keys]
        return [actives, predict]

    def global_keys(self, group):
        return list(set([key for neuron in self.global_state[group] for key in neuron.keys]))

    def sequencer(self, str_rep):
        assert type(str_rep) == type(""), "sequencer type must be string"
        return [[word.strip()] for word in str_rep.split(' ')]


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
            for idl, sentence in enumerate(source):
                self.predict([[word] for word in self.tokenize(sentence)])

    def train_from_file_group_line(self, filepath):
        with open(filepath, 'r') as source:

            for idx, line in enumerate(source):
                if idx % 1000 == 0:
                    print(idx)
                self.predict([[word.strip() for word in line.split(' ')]])

    def is_like(self, word):
        if type(word) == type([]):
            assert len(word) == 1, "Multiple is_like not supported yet"

        return self.predict([word, ["is"]])

    def predict(self, sequence, output=[], as_context=[]):
        print("\n##### PREDICT START NEW SEQUENCE ######")
        self.output = output
        assert type(self.output) == type([]), "Passed in output must be a list"
        if len(output) > 0:
            for out in output:
                self.output_layer.add_new(out)



        if type(sequence) == type(""):
            sequence = self.sequencer(sequence)

        self.output_layer.reset()
        self.reset()
        debug(sequence)

        for input in sequence:
            self.hit(input)
            print("post-hit", self.output_layer.predicted_keys(), self.output_layer.active_keys())

        ## FINAL BIT!

        prediction = self.global_keys("predict")
        pred_nns = set(list(self.global_state["predict"]))
        print("==============")
        print("before", self.output_layer.predicted_keys(), self.output_layer.active_keys(), pred_nns)
        out_preds_keys = self.output_layer.global_keys("predict")
        if len(out_preds_keys) == 0:
            return sorted(prediction)

        self.full_reset()
        assert len(self.global_state["predict"]) == 0
        print("reset",self.output_layer.predicted_keys(), self.output_layer.active_keys(),self.global_keys("predict"))
        if len(as_context) > 0:
            self.output_layer.set_active(as_context)
        else:
            self.output_layer.set_active(out_preds_keys)
        print("activated",self.output_layer.predicted_keys(), self.output_layer.active_keys(),self.global_keys("predict"))

        prediction2 = self.global_keys("predict")
        pred_nns2 = set(list(self.global_state["predict"]))
        print("final",self.output_layer.predicted_keys(), self.output_layer.active_keys(),pred_nns2)
        print("FINALLY:", pred_nns.intersection(pred_nns2))
        final_prediction = pred_nns.intersection(pred_nns2)
        prediction = [key for neuron in final_prediction for key in neuron.keys]
        return sorted(prediction)

    def hit(self, sequence):
        sequence = [sequence] if type(sequence) == type("") else sequence
        assert type(sequence) == type([])
        print("\nNEW HIT: {}".format(sequence))

        # print("\n***** HOW WE FOUND HIT for {} *****".format(sequence))
        # self.show_status()
        # print(" ***** END HOW WE FOUND IT ********")
        # Gather neurons that will be turned off this cycle
        act_nrns = self.global_state["active"]
        assert len(act_nrns) > 0, "A There should always be at least 1 active nrn"

        # Gather neurons that will be set active
        all_prd_nrns = self.global_state["predict"]

        copy_act_nrns = [nrn for nrn in act_nrns]
        copy_all_prd_nrns = [nrn for nrn in all_prd_nrns]

        is_new = True
        prd_nrns = []
        for prd_neuron in all_prd_nrns:
            if prd_neuron.get_keys() == set(sequence):
                for nrn in copy_act_nrns:
                    nrn.set_inactive()
                debug("pattern {} is a match!".format(prd_neuron.get_keys()))
                prd_neuron.set_active()
                self.panic_neuron.set_inactive()
                is_new = False
                break

        # UPDATE
        if is_new and self.is_learning:
            debug("pattern match not found for {}".format(sequence))
            for nrn in copy_act_nrns:
                nrn.set_inactive()
            assert len(self.global_state["active"]) == 0, "Everything should be off here"

            nw_nrn = Neuron(self)

            for _key in sequence:
                debug("\t\tadding neuron to column {}".format(_key))
                nw_nrn.add_key(_key)
                self.columns[_key].add(nw_nrn)

            assert len(copy_act_nrns) > 0, "There should always be at least 1 active nrn"

            for act_nrn in copy_act_nrns:
                debug("\t\tadding new neuron upstream to active")
                act_nrn.add_upstream(nw_nrn)

            debug("\tsetting new neuron active")
            nw_nrn.set_active()
            assert nw_nrn.state == "active"

            # output connecting section
            out_nrns = self.output_layer.get_neurons(self.output)
            debug("OUTPUT NEURONS HERE->{}".format(out_nrns))
            for onrn in out_nrns:
                debug("\t\tadding output neurons upstream to new neuron")
                onrn.add_upstream(nw_nrn)
                nw_nrn.add_upstream(onrn)

        # print("-------\tHOW WE LEFT IT")
        # self.show_status()
        # print("\tOUTPUTS: {}".format(self.output))
        # print("\tEXT OUTPUTS LAYER: {}".format(self.output_layer))
        # print("\tEND HOW WE LEFT IT")
        assert len(self.global_state["active"]) > 0, "Do not leave hit with no actives"


    def initialize_with_single_column_lit(self, word):
        assert type(word) != type(""), "Input to predict with light column is single word"
        self.full_reset()
        self.light_column(word)
        active = self.global_state["active"]
        predicted = self.global_state["predict"]
        return active, predicted


    def predict_clear_then_light_column_on_word(self, word):
        assert type(word) == type(""), "Input must be a string <key>"
        self.full_reset()
        self.light_column(word)
        predicts =  self.global_state["predict"]
        return list(set([key for neuron in predicts for key in neuron.keys]))

    def reset(self):
        self.full_reset()
        self.activation_neuron.set_active()

    def get_number_neurons_per_key(self):
        d_nums={}
        for key, neurons in self.columns.items():
            d_nums[key] = len(neurons)
        return Counter(d_nums)

    def get_counts_for_specific_key(self, key):
        ct = Counter()
        for neuron in self.columns[key]:
            ct += Counter(neuron.keys)
        return ct

    def get_predicted_counts_from_lighting_columns(self, lst_keys):
        self.full_reset()
        for key in lst_keys:
            self.light_column(key)
        predicted_neurons = self.global_state["predict"]
        lst = []
        for neuron in predicted_neurons:
            [lst.append(key) for key in neuron.keys]

        return Counter(lst)

    def compare_two_words(self, w1, w2, nhits=100):
        common_w1 = self.get_counts_for_specific_key(w1).most_common(nhits)
        common_w2 = self.get_counts_for_specific_key(w2).most_common(nhits)
        set_w1 = set([k for k, v in common_w1])
        set_w2 = set([k for k, v in common_w2])
        return set_w1 - set_w2

    def related_to_word(self, w1, remove_common=True, nhits=100, nstops=100):
        common_w1 = self.get_counts_for_specific_key(w1).most_common(nhits)
        set_w1 = set([k for k, v in common_w1])
        if remove_common:
            stop_words = self.get_number_neurons_per_key().most_common(nstops)
            set_stop_words = set([k for k, v in stop_words])
            return set_w1 - set_stop_words
        else:
            return set_w1



    def get_top_x_words_in_layer(self, x):
        """Return list of the top x words with the highest count in the layer"""
        return [word for word, _ in layer.get_number_neurons_per_key().most_common()[:x]]


    def full_reset(self):
        # Get them in a list first because the set changes during operation
        actives = list(self.global_state["active"])
        predicts = list(self.global_state["predict"])
        for neuron in actives:
            neuron.set_inactive()
        for neuron in predicts:
            neuron.set_inactive()
        assert len(self.global_state["active"]) == 0, "Full rest failed on actives"
        assert len(self.global_state["predict"]) == 0, "Full reset failed on predicted"

    def light_column(self, key):
        for neuron in self.columns[key]:
            neuron.set_active()

    def reset_any_match(self):
        self.reset()

        for _, col_neurons in self.columns.items():
            for col_neuron in col_neurons:
                down_neuron = next((nrn for nrn in col_neuron.ns_downstream if nrn in col_neurons), None)
                if down_neuron:
                    continue
                else:
                    col_neuron.set_predict()


    def get_current_neurons(self):
        """Get all the states at once, maybe we can save time this way"""
        actives = []
        predict = []

        for key, neurons in self.columns.items():
            for neuron in neurons:
                if neuron.state == 'A':
                    actives.append(neuron)
                elif neuron.state == 'P':
                    predict.append(neuron)

        state_activation = self.activation_neuron.state
        if state_activation == 'A':
            actives.append(self.activation_neuron)
        elif state_activation == 'P':
            predict.append(self.activation_neuron)

        return actives, predict

    def show_status(self):
        print("\tSTATUS {}: ".format(self.name), self.activation_neuron.state)
        for key, neurons in self.columns.items():
            print(
                "\t{}:\t{}".format(
                    key,
                    str([neuron.state for neuron in neurons])
                    )
            )

    def column_keys(self):
        return list(self.columns.keys())

    # def get_all_actives(self):
    #     """Get ALL active neurons, even if it's the activation neuron.
    #         The other function gets all actives in columns only.
    #     """
    #     assert False, "Who's calling me?"
    #     actives = []
    #     if self.activation_neuron.state == 'A':
    #         actives.append(self.activation_neuron)
    #     for k in self.columns.keys():
    #         colactives = self._column_get('A',k)
    #         actives.extend(colactives)
    #     return actives



    # def _column_get(self, state, column_key):
    #     assert False, "_column_get is deprecated"
    #     assert type(column_key) == type("")
    #     if not self.is_learning:
    #         return []
    #     return [neuron
    #         for neuron in self.columns[column_key]
    #         if neuron.state == state
    #         ]

    # def get_predicted(self):
    #     assert False, "get_predicted is deprecated"
    #     predicted = [k
    #         for k in self.columns.keys()
    #         if len(self._column_get('P', k)) > 0
    #         ]
    #     if len(predicted) > 0:
    #         return predicted
    #     else:
    #         return []
    #         #return self.columns.keys()





    # def _get_neurons(self, state):
    #     assert False, "deprecated"
    #     return [neuron
    #         for key in self.columns.keys()
    #         for neuron in self._column_get(state, key)
    #         ]
    # def get_active_neurons(self):
    #     assert False, "deprecated"
    #     return self._get_neurons('A')
    # def get_predicted_neurons(self):
    #     assert False, "deprecated"
    #     return self._get_neurons('P')



