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

