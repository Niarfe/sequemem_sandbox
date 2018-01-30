import sys
sys.path.append("./sequemem")
from sequemem import *

def tokenize(sentence):
    return [word.strip('\t\n\r .') for word in sentence.split(' ')]

def test_tokenize():
    assert tokenize("we are") == ["we", "are"]


def test_layer_create():
    layer = Layer()
    assert layer.column_keys() == []
    prediction = layer.predict(["this"])
    layer.show_status()
    assert len(layer.columns["this"]) == 1
    assert layer.column_keys() == ["this"]
    assert prediction == ["this"]


def test_layer_create_two_words():
    layer = Layer()
    assert layer.column_keys() == []
    prediction = layer.predict(["are","we"])
    assert sorted(layer.column_keys()) == sorted(["are","we"])
    assert sorted(prediction) == sorted(["are","we"])


def test_layer_create_two_words_predict_one():
    layer = Layer()
    _ = layer.predict(["are","we"])
    prediction = layer.predict(["are"])
    layer.show_status()
    assert prediction == ["we"]


def test_layer_create_two_words_predict_3gram():
    layer = Layer()
    _ = layer.predict(["are","we","there"])
    prediction = layer.predict(["are","we"])
    layer.show_status()
    assert prediction == ["there"]


def test_layer_create_new_second_sequence():
    layer = Layer()
    _ = layer.predict(["are","we","there"])
    prediction = layer.predict(["are","we","here"])
    layer.show_status()
    assert sorted(prediction) == sorted(["are","here","there","we"])


def test_layer_create_two_words_predict_two():
    layer = Layer()
    _ = layer.predict(["are","we","there"])
    _ = layer.predict(["are","we","here"])
    prediction = layer.predict(["are","we"])
    layer.show_status()
    assert sorted(prediction) == sorted(["here","there"])

import random
def test_several_sentences():
    layer = Layer()
    layer.train_from_file('data/cortical_example1.1.txt')

    sentence = "fox eat"
    print(sentence)
    print(layer.predict(sentence))

    similar_to_fox = layer.is_like(layer.is_like("fox"))
    similar_to_fox.remove("fox")

    print("Similar to fox: ", similar_to_fox)
    random_choice = random.choice(similar_to_fox)
    print("Random Choice: ", random_choice)

    res = layer.predict([ random_choice ] + ["eat"])
    print("Well pick this guy and see what he/she eats: ",res)

def test_prep_prediction_new():
    layer = Layer()
    layer.train_from_file('data/cortical_example1.1.txt')
    animal = "fox"
    verb = "eat"
    sentence = "{} {}".format(animal, verb)
    print(sentence)
    print("IS LIKE", layer.is_like(animal))

    similar_to_fox = layer.is_like(layer.is_like(animal))
    similar_to_fox.remove(animal)

    res = {}
    collected = []
    for simile in similar_to_fox:
        res[simile] = layer.predict([simile, verb])
        collected.extend(layer.predict([simile, verb]))

    finalcol = list(set(collected))
    for k, v in res.items():
        print(k, v)
    print(finalcol)

    assert sorted(finalcol) == sorted(['flies', 'squirrel', 'cow', 'salmon', 'rodent', 'rabbit', 'mice'])

def test_logic():
    layer = Layer()
    layer.predict("man is mortal")
    layer.predict("homer is man")
    logic = Logic(layer)
    assert logic.double_ism("homer") == ['mortal'], "Cheesy first try at logic"


def test_brain_create():
    brain = Brain()
    assert len(brain.layers) == 2, "We should have a cortex and a hypothalamus"

def test_multiple_inputs():
    layer = Layer()
    contx = Layer()

    context = "upper"
    sequence = ["A","B","C"]
    cumulative = []
    for letter in sequence:
        cumulative.append(letter)
        layer.predict(cumulative)
        contx.predict([context])
        lact = layer.get_active_neurons()
        cact = contx.get_active_neurons()
        for cn in cact:
            for ln in lact:
                cn.add_upstream(ln)
                ln.add_upstream(cn)

    assert layer.predict(["A"]) == ["B"]
    assert contx.get_all_actives() == []
    assert contx.get_predicted() == ["upper"]

    assert layer.predict(["A", "B"]) == ["C"]
    assert contx.get_all_actives() == []
    assert contx.get_predicted() == ["upper"]
    
    context = "lower"
    sequence = ["A","b","c"]
    cumulative = []
    for letter in sequence:
        cumulative.append(letter)
        layer.predict(cumulative)
        contx.predict([context])
        lact = layer.get_active_neurons()
        cact = contx.get_active_neurons()
        for cn in cact:
            for ln in lact:
                cn.add_upstream(ln)
                ln.add_upstream(cn)

    pred_layer = layer.predict(["A"])
    assert pred_layer == ["B","b"]
    layer.show_status()
    assert contx.get_all_actives() == []
    assert contx.get_predicted() == ["upper", "lower"]
    contx.predict("lower")
    pred_layer2 = layer.get_predicted()

    layer.show_status()
    contx.show_status()
    final_prediction = list(set(pred_layer) & set(pred_layer2))
    assert final_prediction == ["b"]


    # assert layer.predict(["A", "b"]) == ["C", "c"]
    # assert contx.get_all_actives() == []
    # assert contx.get_predicted() == ["upper", "lower"]
# def test_brain_train_basic():
#     brain = Brain()
#     brain.train_from_file('data/disambiguation.txt')
#     brain.hypo.reset()
#     brain.hypo.show_status()
#     brain.cortex.reset()
#     brain.cortex.predict("violin")
#     brain.hypo.show_status()
#     pews = brain.hypo.get_predicted_neurons()
#     for p in pews:
#         print("@@@@@@hypo pred neurons", p)


#     print("################")
#     corepred, hypopred = brain.predict("violin is")
#     brain.hypo.show_status()
#     brain.cortex.show_status()
#     assert corepred == ['instrument']
#     assert hypopred == ['music']
#     corepred, hypopred = brain.predict("bass is")
#     #brain.hypo.show_status()
#     #brain.cortex.show_status()
#     print("THE corepred:  ", corepred)
#     print("THE hypopred:  ", hypopred)
#     assert corepred == ['instrument', 'fish']
#     assert hypopred == ['about','music','fishing']
#     raise