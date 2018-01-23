import sequemem as sq
import os

def tokenize(sentence):
        return [word.strip('\t\n\r .') for word in sentence.split(' ')]

layer = sq.Layer()
with open('data/cortical_example1.1.txt','r') as source:
    for sentence in source:
        tokens = tokenize(sentence)
        layer.predict(tokens)

layer.show_status()
sentence = "fox eat"
