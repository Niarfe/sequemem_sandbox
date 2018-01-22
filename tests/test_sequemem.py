from sequemem import *

vocab = ["are", "we", "there", "here", "question", "statement"]
are_we = ["are","we"]
are_we_there = ["are","we","there"]
are_we_here = ["are","we","here"]


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

