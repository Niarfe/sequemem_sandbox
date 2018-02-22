
class Neuron:
    def __init__(self, layer):
        self.layer = layer

        self.ns_downstream = []
        self.ns_upstream = []
        self.keys = set()
        print(self.layer)
        self.layer.global_state["inactive"].add(self)
        self.state = 'inactive'

    def transition(self, s_from, s_to):
        self.layer.global_state[s_from].remove(self)
        self.layer.global_state[s_to].add(self)
        self.state = s_to

    def add_key(self, key):
        self.keys.add(key)

    def get_keys(self):
        return self.keys

    def set_active(self):
        for neuron in self.ns_downstream:
            neuron.set_inactive()
        for neuron in self.ns_upstream:
            neuron.set_predict()

        if self.state == 'active':
            return
        elif self.state == 'inactive':
            self.transition("inactive", "active")
            return
        elif self.state == 'predict':
            self.transition("predict", "active")
            return
        assert False, "set_active failed"


    def set_predict(self):
        if self.state == 'predict':
            return
        elif self.state == 'active':
            raise "active neuron can't be set to predict"
        elif self.state == 'inactive':
            self.transition("inactive", "predict")
            return
        assert False, "set_predict failed"


    def set_inactive(self):
        if self.state == 'inactive':
            return
        elif self.state == 'predict':
            self.transition("predict", "inactive")
            return
        elif self.state == 'active':
            self.transition("active", "inactive")
            for neuron in self.ns_upstream:
                    neuron.set_inactive()
            return
        assert False, "set_inactive failed"


    def add_upstream(self, neuron):
        neuron.add_downstream(self)
        self.ns_upstream.append(neuron)

    def add_downstream(self, neuron):
        self.ns_downstream.append(neuron)

    def __repr__(self):
        return "{} state: {} keys: {}".format(self.state, self.keys)
    def __str__(self):
        return "{} state: {} keys: {}".format(self.state, self.keys)

