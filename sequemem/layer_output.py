from neuron import Neuron
from neuron import SimpleNeuron
from neuron import CountingNeuron

class LayerSimpleOutput:
    def __init__(self):
        self.neurons = {}
        self.global_state = {
            "inactive": set(),
            "active": set(),
            "predict": set()
        }

    def is_empty(self):
        count= 0
        for _, set_state in self.global_state.items():
            count += len(set_state)
        if count == 0:
            return True
        else:
            return False

    def global_keys(self, group):
        return list(set([key for neuron in self.global_state[group] for key in neuron.keys]))

    def get_with_threshold(self, threshold):
        lwinners = []
        for key, neuron in self.neurons.items():
            if neuron.predict_times >= threshold:
                lwinners.append(neuron)
        return lwinners

    def get_set_of_predicted(self, threshold, nsb=False):
        predicts = set()
        ns = set()
        winners = self.get_with_threshold(threshold)
        for winner in winners:
           for npreds in winner.ns_upstream:
               ns.add(npreds)
               for key in npreds.keys:
                   predicts.add(key)
        if nsb:
            return ns
        else:
            return predicts

    def set_outputs_active(self, outputs):
        for key in outputs:
            if key not in self.neurons.keys():
                self.add_new(key)
            self.neurons[key].set_active()


    def activate_predicted(self):
        pred_nns = self.global_state["predict"]
        lst_nns = [nn for nn in pred_nns]
        for nn in lst_nns:
            nn.set_active()

    def reset(self):
        states = ["active", "predict"]
        for state in states:
            self.clear(state)

    def clear(self, state):
        lst_nrns = [nrn for nrn in self.global_state[state]]
        for nrn in lst_nrns:
            nrn.set_inactive(True)
            nrn.reset_count()
        assert len(self.global_state[state]) == 0, "LayerOutput clear failed!"

    def transition(self, neuron, state_from, state_to):
        self.global_state[state_from].remove(neuron)
        self.global_state[state_to].add(neuron)

    def add_new(self, str_key):
        if str_key in self.neurons.keys():
            #print("{} already exists".format(str_key))
            return
        else:
            neuron = SimpleNeuron(self)
            neuron.add_key(str_key)
            self.neurons[str_key] = neuron
            self.global_state["inactive"].add(neuron)

    def get_neuron(self, str_key):
        return self.neurons[str_key]
    def get_neurons(self, lst_keys):
        return [self.neurons[str_key] for str_key in lst_keys]

    def set_active(self, lst_keys):
        for key in lst_keys:
            n = self.get_neuron(key)
            n.set_active()
    def active_keys(self):
        return list(set([k for neuron in self.global_state["active"] for k in neuron.keys]))
    def predicted_keys(self):
        return list(set([k for neuron in self.global_state["predict"] for k in neuron.keys]))
    def __repr__(self):
        return "OutputLayer: {}".format(self.global_state)
    def __str__(self):
        return "OutputLayer: {}".format(self.global_state)

from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
def debug(out_str):
    return
    print("[{}]: {}".format(sys._getframe(1).f_code.co_name, out_str))
class LayerCount:
    def __init__(self, name="anon"):
        self.columns = defaultdict(set)
        self.global_counter = Counter()
        self.output = None
        self.word_counter = Counter()
        self.window_size = 5
        self.name = name
        self.total_neurons = 0
        self.last_touched_neuron = None
        self.d_w_uber_freq = {}

    def reset(self):
        self.clear_global_counter()
        self.clear_word_counter()

    def get_global_counter(self):
        return self.global_counter
    def get_word_counter(self):
        return self.word_counter
    def set_window(self, window):
        self.window_size = window
    def clear_word_counter(self):
        self.word_counter = Counter()
    def clear_global_counter(self):
        self.global_counter = Counter()

    def sequencer(self, str_rep):
        assert type(str_rep) == type(""), "sequencer type must be string"
        return [[word.strip()] for word in str_rep.split(' ')]

    def tokenize(self, sentence):
        return [str(word.strip().strip('\r\n\t.')) for word in sentence.strip().split(' ')]

    def train_from_file(self, filepath):
        print("Loading file {}".format(filepath))
        with open(filepath,'r') as source:
            for idx, sentence in enumerate(source):
                if idx % 10000 == 0: print(idx)
                self.last_touched_neuron = None
                self.simple_hit([word for word in self.tokenize(sentence)])

    def simple_hit(self, sequence):
        debug(sequence)
        for _key in sequence:
            debug("\t\tadding neuron to column {}".format(_key))
            nw_nrn = CountingNeuron(_key)
            self.columns[_key].add(nw_nrn)

            if self.last_touched_neuron != None:
                self.last_touched_neuron.add_upstream(nw_nrn)
                nw_nrn.add_downstream(self.last_touched_neuron)
            self.last_touched_neuron = nw_nrn

    def initialize_frequency_dict(self):
        total_neurons = len(self.get_number_neurons_per_key())
        d_w_uber_freq = {}
        for word, count in self.get_number_neurons_per_key().most_common():
            d_w_uber_freq[word] = float(count)/total_neurons
        self.total_neurons = total_neurons
        self.d_w_uber_freq = d_w_uber_freq
        return total_neurons, d_w_uber_freq

    def comparison_frequencies(self, the_WORD, ratio=0.05, cutoff=15, visualize_it=False):
        word_test, total_spec_w = self.get_counts_for_specific_key(the_WORD).most_common(1)[0] # should be itself
        print("Count for ", word_test," is ", total_spec_w)
        arr_the_word = []
        arr_global_f = []
        arr_spec_f   = []

        for word, count in self.get_counts_for_specific_key(the_WORD).most_common():
            this_freq = float(count/(total_spec_w + 0.01))
            if float(self.d_w_uber_freq[word]/this_freq) <= ratio:
                arr_the_word.append(word)
                arr_global_f.append(self.d_w_uber_freq[word])
                arr_spec_f.append(this_freq)
                if len(arr_the_word) > cutoff:
                    break

        print("Going to start visual with {}".format(arr_the_word))
        if visualize_it:
            fig, ax = plt.subplots()
            ax.scatter(arr_spec_f[1:cutoff], arr_global_f[1:cutoff])

            for i, txt in enumerate(arr_the_word[1:cutoff]):
                ax.annotate(txt, (arr_spec_f[i+1],arr_global_f[i+1]))
            arr_the_word[1:cutoff]
            plt.show()

        return arr_global_f[:cutoff], arr_spec_f[:cutoff], arr_the_word[:cutoff]

    def get_counts_for_specific_key(self, key):
        self.word_counter.clear()
        for neuron in self.columns[key]:
            neuron.propagate_up(self.word_counter, self.window_size)
            neuron.propagate_dn(self.word_counter, self.window_size)
        self.word_counter[key] = self.word_counter[key]/2
        return self.word_counter

    def get_number_neurons_per_key(self):
        d_nums={}
        for key, neurons in self.columns.items():
            d_nums[key] = len(neurons)
        return Counter(d_nums)



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

