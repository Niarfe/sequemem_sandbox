import sys
sys.path.append("./sequemem")
from layer import Layer
from logic import Logic

def tokenize(sentence):
    return [word.strip('\t\n\r .') for word in sentence.split(' ')]

def test_tokenize():
    assert tokenize("we are") == ["we", "are"]


def test_logic():
    layer = Layer()
    layer.predict("man is mortal")
    layer.predict("homer is man")
    logic = Logic(layer)
    assert logic.double_ism("homer") == ['mortal'], "Cheesy first try at logic"

