

class Neuron:
    def __init__(self):
        self.ns_downstream = []
        self.state = 'I'
        self.ns_upstream = []

    def set_active(self):
        if self.state == 'A':
            return
        elif self.state == 'I' or self.state == 'P':
            for neuron in self.ns_downstream:
                neuron.set_inactive()
            for neuron in self.ns_upstream:
                print("Setting upstreams to predict")
                neuron.set_predict()

        self.state = 'A'

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
        ups = len(self.ns_upstream)
        downs = len(self.ns_downstream)
        if (ups == 0 and downs == 0):
            return True
        return False

from collections import defaultdict

class Layer:
    def __init__(self):
        self.columns = defaultdict(list)
        self.is_learning = False
        self.activation_neuron = Neuron()

    def tokenize(self, sentence):
        return [word.strip('\t\n\r .') for word in sentence.split(' ')]

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

    def predict(self, sequence):
        print("layer caleld to predict on: ", sequence)
        self.reset()
        if type(sequence) == type(""):
            sequence = self.tokenize(sequence)

        for input in sequence:
            self.hit(input)
        return list(self.get_predicted())

    def full_reset(self):
        for key, neurons in self.columns.items():
            for neuron in neurons:
                neuron.set_inactive()

    def reset(self):
        for key, neurons in self.columns.items():
            for neuron in neurons:
                neuron.set_inactive()
        self.activation_neuron.set_active()

    def hit(self, column_key):
        npred = self._column_get('P', column_key)
        if len(npred) > 0:
            for neuron in npred:
                neuron.set_active()
            self.is_learning = False
        else:
            neuron = Neuron()
            actives = self.get_all_actives()
            for active in actives:
                active.add_upstream(neuron)
            self.columns[column_key].append(neuron)
            neuron.set_active()
            self.is_learning = True

    def get_all_actives(self):
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
        if self.is_learning:
            return self.columns.keys()
        else:
            return [k
                for k in self.columns.keys()
                if self._column_get('P', k)
                ]

    def column_keys(self):
        return [key for key in self.columns.keys()]

    def show_status(self):
        print("LAYER STATUS:")
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
                print(tokens)
                if tokens[0] == 'about':
                    print("Setting Context to", tokens[1])
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
        print("#####")
        self.show_status()
        print("    SENTENCE: ", sentence)
        pred_layer = self.layer.predict(sentence)
        print("    layer predicted: ", pred_layer) 
        self.show_status()
        
        self.contx.predict(self.current_context)
        
        self.show_status()
        pred_layer2 = self.layer.get_predicted()
        return list(set(pred_layer) & set(pred_layer2))

    def show_status(self):
        print("")
        print("Context: ", self.current_context)
        print("CONTEXT: ")
        self.contx.show_status()
        print("LAYER: ")
        self.layer.show_status()
