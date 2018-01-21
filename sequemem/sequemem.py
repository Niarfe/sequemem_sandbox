

class Neuron:
    def __init__(self):
        self.ns_downstream = []
        self.state = 'inactive'
        self.ns_upstream = []

    def set_active(self):
        if self.state == 'active':
            return
        elif self.state == 'inactive':
            for neuron in self.ns_downstream:
                neuron.set_inactive()
            for neuron in self.ns_upstream:
                neuron.set_predict()
            self.state = 'active'
        elif self.state == 'predict':
            for neuron in self.ns_downstream:
                neuron.set_inactive()
            for neuron in self.ns_upstream:
                neuron.set_predict()
            self.state = 'active'

    def set_predict(self):
        if self.state == 'predict':
            return
        elif self.state == 'active':
            raise "active neuron can't be set to predict"
        elif self.state == 'inactive':
            self.state = 'predict'

    def set_inactive(self):
        if self.state == 'inactive':
            return
        elif self.state == 'predict':
            for neuron in self.ns_upstream:
                neuron.set_inactive()
            self.state = 'inactive'
        elif self.state == 'active':
            for neuron in self.ns_upstream:
                neuron.set_inactive()
            self.state = 'inactive'

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


class Column:
    def __init__(self, key):
        self.key = key
        self.neurons = []
        self.state = 'inactive'

    def is_predicted(self):
        is_active = None
        for neuron in self.neurons:
            if neuron.state == 'predict':
                return True
        return False

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def hit(self):
        for neuron in self.neurons:
            if neuron.state == 'predict':
                neuron.set_active()
                return True
        return False

    def get_first_unused(self):
        for neuron in self.neurons:
            if neuron.is_unused():
                return neuron
        return None


class Bundle():
    def __init__(self, vocab=[], ncols=2, ndepth=3):
        self.ncols = ncols if len(vocab) == 0 else len(vocab)
        self.columns = []
        self.learning = False
        self.activation_neuron = Neuron()
        init_keys = range(self.ncols) if len(vocab) == 0 else vocab
        # Initialize bundle of columns
        for col_idx in init_keys:
            column = Column(col_idx)
            for dep_idx in range(ndepth):
                column.add_neuron(Neuron())
            self.columns.append(column)
        self.activation_neuron.set_active()
        self.reset_all()

    def predict(self, sequence):
        self.activation_neuron.set_active()
        self.reset_all()
        self.predict_sequence(sequence)
        prediction = final_prediction(self.predict_state())
        return prediction

    def show_status(self):
        """Print out the status of the columns"""
        print("BUNDLE STATUS:")
        for column in self.columns:
            print(column.key, [neuron.state for neuron in column.neurons])

    def set_activation_neuron(self, neuron):
        self.activation_neuron = neuron
        for column in self.columns:
            neuron.add_upstream(column.neurons[0])

    def predict_state(self):
        """lincoming is list of positions to hit"""
        result = [[],[]]
        for column in self.columns:
            result[0].append(column.key)
            if self.learning == False:
                result[1].append(column.is_predicted())
            else:
                result[1].append(True)
        return result

    def get_active_neurons(self):
        actives = []
        for column in self.columns:
            for neuron in column.neurons:
                if neuron.state == 'active':
                    actives.append(neuron)
        return actives

    def set_learning(self, bFlag):
        self.learning = bFlag

    def hit(self, lhits):
        """Input a list or array of active pixels, this function delivers the hits
            To each appropriate column, and lets it take over to do i'ts thing"""
        for hit in lhits:
            for column in self.columns:
                if hit == column.key:
                    if column.is_predicted():
                        column.hit()
                        if not True in self.predict_state()[1]:
                            self.set_learning(True)
                        else:
                            self.set_learning(False)
                    else:
                        actives = self.get_active_neurons()
                        next_unused = column.get_first_unused()
                        next_unused.set_predict()
                        for neuron in actives:
                            neuron.add_upstream(next_unused)
                        column.hit()
                        self.set_learning(True)

    def reset_all(self):
        self.learning = False
        self.activation_neuron.set_inactive()
        for column in self.columns:
            for neuron in column.neurons:
                neuron.set_inactive()

    def predict_sequence(self, lst_sequence):
        self.reset_all()
        group_neuron = Neuron()
        self.set_activation_neuron(group_neuron)
        group_neuron.set_active()

        for word in lst_sequence:
            self.hit([word])

        predictions = []
        predict_state = self.predict_state()
        for idx, prediction in enumerate(self.predict_state()[1]):
            if prediction == True:
                predictions.append(self.predict_state()[0][idx])
        return predictions

    def process_sequence(self, sequence):
        self.reset_all()
        self.activation_neuron.set_active()
        for word in sequence:
            self.hit([word])

def final_prediction(predict_state):
    predictions = []
    for idx, prediction in enumerate(predict_state[1]):
        if prediction == True:
            predictions.append(predict_state[0][idx])
    return predictions