
from neuron import Neuron

class LayerOutput:
    def __init__(self):
        self.neurons = {}
        self.global_state = {
            "inactive": set(),
            "active": set(),
            "predict": set()
        }

    def set_outputs_active(self, outputs):
        for key in outputs:
            if key not in self.neurons.keys():
                self.add_new(Neuron(self))
                self.neurons[key].set_active()

    def transition(self, neuron, state_from, state_to):
        self.global_state[state_from].remove(neuron)
        self.global_state[state_to].add(neuron)

    def add_new(self, str_key):
        if str_key in self.neurons.keys():
            print("{} already exists".format(str_key))
            return
        else:
            neuron = Neuron(self)
            self.neurons[str_key] = neuron
            self.global_state["inactive"].add(neuron)

    def get_neuron(self, str_key):
        return self.neurons[str_key]

    def set_active(self, lst_keys):
        for key in lst_keys:
            n = self.get_neuron(key)
            n.set_active()
    def active_keys(self):
        return list(set([k for neuron in self.global_state["active"] for k in neuron.keys]))