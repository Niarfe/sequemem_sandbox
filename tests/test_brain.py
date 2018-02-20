import sys
sys.path.append("./sequemem")
from brain import *



# def tokenize(sentence):
#     return [word.strip('\t\n\r .') for word in sentence.split(' ')]

# def test_tokenize():
#     assert tokenize("we are") == ["we", "are"]


# def test_brain_train():
#     brain = Brain()
#     brain.train_from_file('data/disambiguation.txt')
#     prediction = brain.predict(["bass", "is"], "music")
#     print(prediction)
#     assert prediction  == ["instrument"], prediction

# def test_logic_gates_with_brain():
#     brain = Brain()
#     brain.train_from_file('data/logic_gates.txt')
#     prediction = brain.predict(["0", "1"], "or")
#     print(prediction)
# #    brain.show_status()
#     assert prediction  == ["1"], prediction

# def test_logic_gates_with_brain_full_monte():
#     brain = Brain()
#     brain.train_from_file('data/logic_gates.txt')
#     assert brain.predict(["1", "1"], "and") == ["1"]
#     assert brain.predict(["1", "0"], "and") == ["0"]
#     assert brain.predict(["0", "1"], "and") == ["0"]
#     assert brain.predict(["0", "0"], "and") == ["0"]
#     assert brain.predict(["1", "1"], "or") == ["1"]
#     assert brain.predict(["1", "0"], "or") == ["1"]
#     assert brain.predict(["0", "1"], "or") == ["1"]
#     assert brain.predict(["0", "0"], "or") == ["0"]
#     assert brain.predict(["1", "1"], "xor") == ["0"]
#     assert brain.predict(["1", "0"], "xor") == ["1"]
#     assert brain.predict(["0", "1"], "xor") == ["1"]
#     assert brain.predict(["0", "0"], "xor") == ["0"]

# def test_triple_context():
#     brain = Brain()
#     brain.train_from_file('data/long_context_test.txt')
#     assert brain.predict(["bass", "is"], "music") == ["instrument"]
#     assert brain.predict(["viola", "is"], "music") == ["instrument"]
#     assert brain.predict(["bass", "is"], "fishing") == ["fish"]
#     assert brain.predict(["viola", "is"], "names") == ["name"]
#     assert brain.predict(["salmon", "is"], "fishing") == ["fish"]
#     assert brain.predict(["efrain", "is"], "names") == ["name"]

