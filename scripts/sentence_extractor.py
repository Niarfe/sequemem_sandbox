import nltk.data
import nltk as nltk
import sys
import re

"""
Installing nltk.
1. pip install nltk
2. python <enter console>
> import nltk
> nltk.download() # this opens a download ui panel
## select last tab with packages, select punkt, that will get the english tokenizer
"""
def load_file(f_path):
    return open(f_path).read().decode('utf-8', 'ignore')

def get_sentences(txt):
    txt = prep_data(txt)
    sents =  nltk.sent_tokenize(txt)
    return sents

def load_sentences(f_path, lower=False):
    with open(f_path, 'r') as source:
        txt = source.read() #.decode('utf-8', 'ignore')
        sentences = get_sentences(txt)
        clean_sentences = clean_all_sentences(sentences, lower)
        return clean_sentences

def prep_data(txt):
    txt = remove_carriage_returns(txt)
    txt = replace_empty_lines_with_dot(txt)
    txt = flatten_compress_white_space(txt)
    return txt

def replace_empty_lines_with_dot(txt):
    txt = re.sub(r'\n\n', '\n.\n', txt)
    return txt

def remove_carriage_returns(txt):
    txt = txt.replace('\r', '')
    return txt

def flatten_compress_white_space(txt):
    """Replace all white space sequences with a single white space"""
    return ' '.join(txt.split())

def clean_a_sentence(arr_sen, lower=True):
    arr_words = [word.lower() if lower else word for word in nltk.word_tokenize(arr_sen) if word.isalpha()]
    return ' '.join(arr_words).strip().encode('utf-8')

def clean_all_sentences(sents, lower=True):
    return [clean_a_sentence(sen, lower) for sen in sents]

def write_sentences_file(filename, arr_sentences):
    with open(filename, 'w') as target:
        for sentence in arr_sentences:
            target.write(sentence + "\n")

def remove_stop_words(sents_str):
    assert False, "We don't want to use this right now"
    #print sents_str
    #words = [word for sent in sents_str for word in sent.split(' ') if word not in stopwords.words('english')]
    # create English stop words list
    en_stop = get_stop_words('en')
    new_sents_str = []
    words = [word for word in sents_str if not word in en_stop]
    # for sent in sents_str:
    #     words = [word for word in sent if not word in en_stop]
    #     new_sents_str.append(words)
    return words


##########################################################################
# TESTS
##########################################################################

# def test_replace_empty_lines_with_dot():
#     txt0 = 'FAIRY TALES EVERY CHILD\r\n\r\nSHOULD KNOW'
#     txtR = 'FAIRY TALES EVERY CHILD.\r\nSHOULD KNOW'
#     assert txtR == replace_empty_lines_with_dot(txt0)

def test_flatten_compress_white_space():
    txt1 = "This is \r\n \n        single white space    "
    assert "This is single white space" == flatten_compress_white_space(txt1), "example 1"

def test_get_sentences():
    txt0 = load_file('testsuite/data/test0.txt')
    sents0 = get_sentences(txt0)
    assert len(sents0) == 5, "test0 should have 5 sentences"

    txt1 = load_file('testsuite/data/test1.txt')
    sentences = get_sentences(txt1)
    assert len(sentences) == 40, "Extract 40 sentences for test1"

    txt3 = load_file('testsuite/data/test2.txt')
    sentences = get_sentences(txt3)
    assert len(sentences) == 2, "There should be 2 sentences"

def test_clean_all_sentences():
    txt1 = load_file('testsuite/data/test1.txt')
    sentences = get_sentences(txt1)
    clean_sentences = clean_all_sentences(sentences)
    assert len(sentences) == 40, "Assert that original set is complete with 106"
    assert len(clean_sentences) == 40, "Count of clean sentes should go down to 91"

    txt2 = ["This is a comma, and a period."]
    clean_sentences = clean_all_sentences(txt2, False)
    assert "This is a comma and a period" == clean_sentences[0], "should keep case and remove punctuation"
    clean_sentences = clean_all_sentences(txt2)
    assert "this is a comma and a period" == clean_sentences[0], "should lower case and remove puctuation"

def test_remove_stop_words():
    sent0 = ["a", "test"]
    expect = remove_stop_words(sent0)
    assert ["test"] == expect
    sents = ["this", "to", "the", "max", "and", "onward", "to", "the", "rest"]
    expect = remove_stop_words(sents)
    assert ["max", "onward", "rest"] == expect, "only remove the stop words"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", dest="source", required=True, help="name of file to extract sentences from")
    parser.add_argument("--remove_stop", dest="remove_stop", required=False, help="Remove stop Words")
    args = parser.parse_args()
    ######### END INPUT ############

    clean_sentences = load_sentences(args.source, False)

    for sent in clean_sentences:
        if args.remove_stop:
            print(' '.join(remove_stop_words(sent.split(' '))))
        else:
            print(sent.decode('utf-8'))