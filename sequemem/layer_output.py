from neuron import Neuron
from neuron import SimpleNeuron

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