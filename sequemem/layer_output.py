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
        self.name = name
        self.columns = defaultdict(set)
        self.global_counter = Counter()
        self.word_counter = Counter()
        self.window_size = 5
        self.total_neurons = 0
        self.total_sentences = 0
        self.last_touched_neuron = None
        self.d_w_uber_freq = {}

    def reset(self):
        self.clear_global_counter()
        self.clear_word_counter()
    def clear_word_counter(self):
        self.word_counter = Counter()
    def clear_global_counter(self):
        self.global_counter = Counter()

    def set_window(self, window):
        self.window_size = window
    def sequencer(self, str_rep):
        assert type(str_rep) == type(""), "sequencer type must be string"
        return [[word.strip()] for word in str_rep.split(' ')]

    def tokenize(self, sentence):
        return [str(word.strip().strip('\r\n\t.')) for word in sentence.strip().split(' ')]

    def load_from_file(self, filepath):
        print("Loading file {}".format(filepath))
        with open(filepath,'r') as source:
            for idx, sentence in enumerate(source):
                if idx % 10000 == 0: print(idx)
                self.last_touched_neuron = None
                self.add([word for word in self.tokenize(sentence)])

    def add(self, arr_sentence):
        debug(arr_sentence)
        for word in arr_sentence:
            debug("\t\tadding neuron to column {}".format(word))
            nw_nrn = CountingNeuron(word)
            self.columns[word].add(nw_nrn)

            if self.last_touched_neuron != None:
                self.last_touched_neuron.add_upstream(nw_nrn)
                nw_nrn.add_downstream(self.last_touched_neuron)
            self.last_touched_neuron = nw_nrn
        self.total_sentences += 1

    def get_counter_for_sequence(self, sequence, window_size=2, direction=0):
        """Get counts for n-grams (treating sequence as n-gram)
        Go out window_size steps, includes end words of seqence.
        Args:
            sequence: list<words> list of words to use for n-gram
            window_size: how far to go forward or backwards, includes endpoints
            direction: -1 back only, 0 both ways, 1 forward only
        """
        self.word_counter.clear()
        self.window_size = window_size
        nrn_group = []
        nrns_group1 = self.columns[sequence[0]]
        for nrn in nrns_group1:
            nrns_group2 = [nrn for nrn in nrns_group1
                            for upnrn in nrn.ns_upstream
                            if upnrn.word == sequence[1]]

        for neuron in nrns_group2:
            if direction != -1:
                neuron.propagate_up(self.word_counter, self.window_size, sequence)
            if direction != 1:
                neuron.propagate_dn(self.word_counter, self.window_size, sequence)
        self.word_counter[key] = self.word_counter[key]/2
        return self.word_counter

    def get_counts_for_specific_key(self, key, window_size=5, direction=0):
        """Get the neighborhood count of words for given key
        Args:
            key: get every neuron found in this column
            window_size: Counting 1 from given column, take up to that many steps
            direction: 0 go both up and down.  -1 down only and 1 up only.
        """

        self.word_counter.clear()
        self.window_size = window_size
        for neuron in self.columns[key]:
            if direction != -1:
                neuron.propagate_up(self.word_counter, self.window_size)
            if direction != 1:
                neuron.propagate_dn(self.word_counter, self.window_size)
        self.word_counter[key] = self.word_counter[key]/2
        return self.word_counter

    def get_number_neurons_per_key(self):
        d_nums={}
        for key, neurons in self.columns.items():
            d_nums[key] = len(neurons)
        return Counter(d_nums)

    def get_frequency_dict(self, force_init=False):
        """Get a count of all words but in frequency terms, for global counts"""
        if len(self.d_w_uber_freq) == 0 or force_init == True:
            print("Re-iitializing dictionary")
            self.d_w_uber_freq = {}


            for word, count in self.get_number_neurons_per_key().most_common():
                self.d_w_uber_freq[word] = float(count)/self.total_sentences

        return self.total_sentences, self.d_w_uber_freq

    def __repr__(self):
        val  =   "LayerCounter:     {}".format(self.name)
        val += "\nTotal Neurons:    {}".format(self.total_neurons)
        val += "\nNumber of cols:   {}".format(len(self.columns))
        val += "\nTop 10:           {}".format(self.global_counter.most_common(10))
        val += "\nFreq dict sample  {}".format(self.d_w_uber_freq)

        return val

#########################################################################################
#  PRISM
#########################################################################################
class Prism:
    def __init__(self, layerCounter):
        self.layer_counter = layerCounter
        self.d_w_uber_freq = {}



    def comparison_frequencies(self, the_WORD, ratio=0.05, cutoff=15, visualize_it=False):
        word_test, total_spec_w = self.layer_counter.get_counts_for_specific_key(the_WORD).most_common(1)[0] # should be itself
        print("Count for ", word_test," is ", total_spec_w)
        arr_the_word = []
        arr_global_f = []
        arr_spec_f   = []

        for word, count in self.layer_counter.get_counts_for_specific_key(the_WORD).most_common():
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

