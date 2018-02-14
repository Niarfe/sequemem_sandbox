from collections import defaultdict
from layer import Layer



class Logic:
    def __init__(self, layer):
        self.layer = layer

    def double_ism(self, word):
        assert type(word) == type(""), "Input to double_ism is a string"

        return self.layer.predict(self.layer.predict([word, "is"] ) + ["is"])