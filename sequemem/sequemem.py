

class Neuron:
    def __init__(self):
        self.ns_downstream = []
        self.state = 'I'
        self.ns_upstream = []

    def set_active(self):
        if self.state == 'A' or self.state == 'I' or self.state == 'P':
            for neuron in self.ns_downstream:
                neuron.set_inactive()
            for neuron in self.ns_upstream:
                neuron.set_predict()

        self.state = 'A'

    def set_hard_state(self, state):
        self.state = state

    def set_predict(self):
        if self.state == 'P':
            return
        elif self.state == 'A':
            raise "active neuron can't be set to predict"
        elif self.state == 'I':
            self.state = 'P'

        self.state = 'P'

    def set_inactive(self):
        if self.state == 'I':
            return
        elif self.state == 'P':
            pass
        elif self.state == 'A':
            self.state = 'I'
            for neuron in self.ns_upstream:
                    neuron.set_inactive()

        self.state = 'I'

    def add_upstream(self, neuron):
        neuron.add_downstream(self)
        self.ns_upstream.append(neuron)

    def add_downstream(self, neuron):
        self.ns_downstream.append(neuron)

    def is_unused(self):
        return not self.ns_upstream and not self.ns_downstream

from collections import defaultdict

class Layer:
    def __init__(self):
        self.columns = defaultdict(list)
        self.panic_neuron = Neuron()
        self.activation_neuron = Neuron()
        self.is_learning = True

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

        for input in sequence:
            self.hit(str(input))

        return list(self.get_predicted())

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

    def hit(self, column_key):
        # Gather neurons that will be set active
        prd_nrns = self._column_get('P', column_key)
        act_nrns = self.get_all_actives()

        # FORGET
        self.full_reset()

        # UPDATE
        if len(prd_nrns) > 0:
            self.panic_neuron.set_inactive()
            # UPDATE set previous chosen predicts to active
            for prd_nrn in prd_nrns:
                prd_nrn.set_active()
        else:
            if not self.is_learning:
                return
            self.panic_neuron.set_inactive()
            nw_nrn = Neuron()
            for act_nrn in act_nrns:
                act_nrn.add_upstream(nw_nrn)
            self.columns[column_key].append(nw_nrn)
            nw_nrn.set_active()


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
        print("LAYER STATUS:", self.activation_neuron.state)
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



class Logic:
    def __init__(self, layer):
        self.layer = layer

    def double_ism(self, word):
        assert type(word) == type(""), "Input to double_ism is a string"

        return self.layer.predict(self.layer.predict([word, "is"] ) + ["is"])


class Brain:
    def __init__(self):
        self.layer = Layer()
        self.contx = Layer()
        self.layers = [
            self.layer,
            self.contx
        ]
        self.current_context = None

    def train_from_file(self, filepath):
        with open(filepath,'r') as source:
            for sentence in source:
                tokens = self.tokenize(sentence)
                if tokens[0] == 'about':
                    self.current_context = tokens[1]
                else:
                    cumulative = []
                    for word in tokens:
                        cumulative.append(word)
                        self.layer.predict(cumulative)
                        lact = self.layer.get_active_neurons()
                        self.contx.predict([self.current_context])
                        cact = self.contx.get_active_neurons()
                        for cn in cact:
                            for ln in lact:
                                cn.add_upstream(ln)
                                ln.add_upstream(cn)


    def tokenize(self, sentence): # is dup with layer tokenize!
        return [word.strip('\t\n\r .') for word in sentence.split(' ')]

    def predict(self, sentence, context=None):
        self.layer.reset()
        self.contx.full_reset()

        self.current_context = "neutral" if not context else context
        self.layer.predict(sentence)
        act_layer = self.layer.get_active_neurons()
        pred_layer = self.layer.get_predicted_neurons()

        self.layer.reset()
        self.layer.show_status()
        self.contx.predict(self.current_context)
        pred_layer2 = self.layer.get_predicted_neurons()
        self.layer.show_status()
        final_preds = list(set(pred_layer) & set(pred_layer2))

        self.layer.full_reset()
        self.layer.show_status()
        for active in act_layer:
            active.set_hard_state('A')
        for pred in final_preds:
            pred.set_hard_state('P')
        self.layer.show_status()
        return self.layer.get_predicted()

    def show_status(self):
        print("")
        print("Context: ", self.current_context)
        print("CONTEXT: ")
        self.contx.show_status()
        print("LAYER: ")
        self.layer.show_status()
