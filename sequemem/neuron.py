

class Neuron:
    global_state = {
        "active": set(),
        "predict": set(),
        "inactive": set()
        }
    def __init__(self):
        self.ns_downstream = []
        self.state = 'I'
        Neuron.global_state["inactive"].add(self)
        self.ns_upstream = []
        self.keys = set()

    def add_key(self, key):
        self.keys.add(key)

    def get_keys(self):
        return self.keys

    def set_active(self):
        for neuron in self.ns_downstream:
            neuron.set_inactive()
        for neuron in self.ns_upstream:
            neuron.set_predict()

        if self.state == 'A':
            return

        elif self.state == 'I':
            Neuron.global_state["inactive"].remove(self)
            Neuron.global_state["active"].add(self)
            self.state = 'A'
            return
        elif self.state == 'P':
            Neuron.global_state["predict"].remove(self)
            Neuron.global_state["active"].add(self)
            self.state == 'A'
            return

        assert False, "set_active failed"

    def set_hard_state(self, state):
        assert False
        self.state = state

    def set_predict(self):
        if self.state == 'P':
            return
        elif self.state == 'A':
            return
            #raise "active neuron can't be set to predict"
        elif self.state == 'I':
            Neuron.global_state["inactive"].remove(self)
            Neuron.global_state["predict"].add(self)
            self.state = 'P'
            return

        assert False, "set_predict failed"

    def set_inactive(self):
        if self.state == 'I':
            return
        elif self.state == 'P':
            Neuron.global_state["predict"].remove(self)
            Neuron.global_state["inactive"].add(self)
            self.state = 'I'
            return
        elif self.state == 'A':
            Neuron.global_state["active"].remove(self)
            Neuron.global_state["inactive"].add(self)
            self.state = 'I'
            for neuron in self.ns_upstream:
                    neuron.set_inactive()
            return

        assert False, "set_inactive failed"

    def add_upstream(self, neuron):
        neuron.add_downstream(self)
        self.ns_upstream.append(neuron)

    def add_downstream(self, neuron):
        self.ns_downstream.append(neuron)

    def is_unused(self):
        return not self.ns_upstream and not self.ns_downstream

