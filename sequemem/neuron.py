class BaseNeuron(object):
    def __init__(self, layer):
        self.layer = layer
        self.predict_times = 0
        self.ns_downstream = []
        self.ns_upstream = []
        self.keys = set()
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

    def add_upstream(self, neuron):
        self.ns_upstream.append(neuron)

    def add_downstream(self, neuron):
        self.ns_downstream.append(neuron)

    def __repr__(self):
        return "<nrn keys: {}>".format([key for key in self.keys])
    def __str__(self):
        return "<nrn keys: {}>".format([key for key in self.keys])

    def set_active(self):
        raise NotImplementedError()
    def set_predict(self):
        raise NotImplementedError()
    def set_inactive(self):
        raise NotImplementedError()



class SimpleNeuron(BaseNeuron):
    def __init__(self, layer):
        super(SimpleNeuron, self).__init__(layer)

    def reset_count(self):
        self.predict_times = 0

    def set_active(self):
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
        self.predict_times += 1
        if self.state == 'predict':
            return
        elif self.state == 'active':
            return
            raise "active neuron can't be set to predict"
        elif self.state == 'inactive':
            self.transition("inactive", "predict")
            return
        assert False, "set_predict failed"

    def set_inactive(self, really=False):
        if not really:
            return
        if self.state == 'inactive':
            return
        elif self.state == 'predict':
            self.transition("predict", "inactive")
            return
        elif self.state == 'active':
            self.transition("active", "inactive")
            return
        assert False, "set_inactive failed"


class Neuron(BaseNeuron):
    def __init__(self, layer):
        super(Neuron, self).__init__(layer)

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
            return
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

class CountingNeuron(object):
    def __init__(self, key):
        self.predict_times = 0
        self.ns_downstream = []
        self.ns_upstream = []
        self.word = key

    def get_word(self):
        return self.keys

    def add_upstream(self, neuron):
        self.ns_upstream.append(neuron)

    def add_downstream(self, neuron):
        self.ns_downstream.append(neuron)

    def __repr__(self):
        return "<nrn: {}>".format(self.word)
    def __str__(self):
        return "<nrn: {}>".format(self.word)

    def propagate_up(self, cntr, ntimes, sequence=None):
        if sequence != None:
            if sequence[0] != self.key:
                return
            else:
                sequence = sequence[1:]
        cntr[self.word] += 1
        ntimes -= 1
        if ntimes > 0:
            for nrn in self.ns_upstream:
                nrn.propagate_up(cntr, ntimes, sequence)
        else:
            return
    def propagate_dn(self, cntr, ntimes):
        cntr[self.word] += 1
        ntimes -= 1
        if ntimes > 0:
            for nrn in self.ns_downstream:
                nrn.propagate_up(cntr, ntimes)
        else:
            return