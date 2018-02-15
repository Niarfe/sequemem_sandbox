from collections import defaultdict

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
