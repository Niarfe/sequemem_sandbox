from sequemem import *

vocab = ["are", "we", "there", "here", "question", "statement"]
are_we = ["are","we"]
are_we_there = ["are","we","there"]
are_we_here = ["are","we","here"]


def test_basic_sequence():
    layer = Bundle(vocab)
    layer.process_sequence(["are","we","there","question"])
    prediction = layer.predict(["are","we"])
    assert prediction == ["there"]

def test_basic_sequence_():
    layer = Bundle()
    prediction = layer.predict(["are","we"])
    assert prediction == ["are","we"]


def test_create_neuron():
    neuron = Neuron()
    assert neuron.state == 'inactive'

def test_add_neuron():
    neuron1 = Neuron()
    neuron2 = Neuron()

    assert neuron1.state == 'inactive'

    neuron1.add_upstream(neuron2)

    assert len(neuron1.ns_upstream) == 1

    neuron1.set_active()

    assert neuron1.state == 'active'
    assert neuron2.state == 'predict'

def test_two_layers():
    n1 = Neuron()
    n2 = Neuron()
    n3 = Neuron()
    assert n1.state == 'inactive'

    n1.add_upstream(n2)
    n2.add_upstream(n3)
    assert len(n1.ns_upstream) == 1
    for neuron in [n1, n2, n3]:
        assert neuron.state == 'inactive'

    n1.set_active()
    assert n1.state == 'active'
    assert n2.state == 'predict'
    assert n3.state == 'inactive'

    n2.set_active()
    assert n1.state == 'inactive'
    assert n2.state == 'active'
    assert n3.state == 'predict'

def test_column_create():
    column = Column('0')

    assert column.key == '0'
    n1 = Neuron()

    column.add_neuron(n1)
    assert column.is_predicted() == False

    n2 = Neuron()
    n1.add_upstream(n2)
    column.add_neuron(n2)
    assert column.is_predicted() == False

    n1.set_active()
    assert column.is_predicted() == True

def test_bundle_000():
    column0 = Column('0')
    column1 = Column('1')

    print column0.is_predicted()

    bundle = [
        column0,
        column1
    ]

    # let's load the 000 case first
    neuron_0__ = Neuron()
    neuron_0__.set_predict()
    column0.add_neuron(neuron_0__)
    neuron_00_ = Neuron()
    neuron_0__.add_upstream(neuron_00_)
    column0.add_neuron(neuron_00_)
    neuron_000 = Neuron()
    neuron_00_.add_upstream(neuron_000)
    column0.add_neuron(neuron_000)

    column0.hit()
    result =  bundle[0].is_predicted()
    assert result == True
    column0.hit()
    result =  bundle[0].is_predicted()
    assert result == True

def test_auto_add_new_connection():
    column0 = Column('0')
    column1 = Column('1')

    bundle = [column0, column1]

    for idx in range(10):
        for column in bundle:
            column.add_neuron(Neuron())

    activation_neuron = Neuron()
    for column in bundle:
        activation_neuron.add_upstream(column.neurons[0])

    assert bundle[0].is_predicted() == False
    assert bundle[1].is_predicted() == False

    activation_neuron.set_active()
    assert bundle[0].is_predicted() == True
    assert bundle[1].is_predicted() == True

    bundle[0].hit()
    result0 = bundle[0].is_predicted()
    result1 = bundle[1].is_predicted()
    assert result0 == False
    assert result1 == False


def test_bundle_create():
    bundle = Bundle([], 2,3)
    bundle.show_status()

    group_neuron = Neuron()
    bundle.set_activation_neuron(group_neuron)

    group_neuron.set_active()

    assert bundle.predict_state() == [[0, 1],[True, True]]
    bundle.show_status()

def test_bundle_hit_1():
    bundle = Bundle([], 2,3)
    bundle.show_status()

    group_neuron = Neuron()
    bundle.set_activation_neuron(group_neuron)

    group_neuron.set_active()
    bundle.show_status()

    assert bundle.predict_state() == [[0, 1],[True, True]]

    bundle.hit([1])
    bundle.show_status()
    assert bundle.predict_state() == [[0,1],[True,True]]

def test_bundle_hit_2():
    bundle = Bundle([], 2,3)
    bundle.show_status()

    group_neuron = Neuron()
    bundle.set_activation_neuron(group_neuron)

    group_neuron.set_active()
    bundle.show_status()

    assert bundle.predict_state() == [[0, 1],[True, True]]

    bundle.hit([1])
    bundle.show_status()
    assert bundle.predict_state() == [[0,1],[True,True]]

    bundle.hit([1])
    bundle.show_status()
    assert bundle.predict_state() == [[0,1],[True,True]]

def test_bundle_hit_full():
    bundle = Bundle([],2,3)
    bundle.show_status()

    group_neuron = Neuron()
    bundle.set_activation_neuron(group_neuron)

    group_neuron.set_active()
    bundle.show_status()

    assert bundle.predict_state() == [[0, 1],[True, True]]

    bundle.hit([1])
    bundle.show_status()
    assert bundle.predict_state() == [[0,1],[True,True]]

    bundle.hit([1])
    bundle.show_status()
    assert bundle.predict_state() == [[0,1],[True,True]]

    bundle.hit([1])

    bundle.reset_all()

    group_neuron.set_active()
    bundle.show_status()

    assert bundle.predict_state() == [[0, 1],[True, True]]

    bundle.hit([1])
    bundle.show_status()
    assert bundle.predict_state() == [[0,1],[False,True]]

    bundle.hit([1])
    bundle.show_status()
    assert bundle.predict_state() == [[0,1],[False,True]]

def test_any_key():
    bundle = Bundle([],6,4)
    bundle.show_status()

    group_neuron = Neuron()
    bundle.set_activation_neuron(group_neuron)
    group_neuron.set_active()
    bundle.show_status()

    sequence = [0, 1, 2, 3]

    for word in sequence:
        bundle.hit([word])
        bundle.show_status()
    print bundle.predict_state()

    print "REEEEBOOOOOOOT!"
    bundle.reset_all()
    group_neuron.set_active()
    short_sequence = [0,1,2]
    for word in short_sequence:
        bundle.hit([word])
        bundle.show_status()
    print "FINAL PRED: ", final_prediction(bundle.predict_state())
    print bundle.predict_state()

    sequence = [1,0,2,4]

    print "REEEEBOOOOOOOT!"
    bundle.reset_all()
    group_neuron.set_active()

    for word in sequence:
        bundle.hit([word])
        bundle.show_status()
    print bundle.predict_state()

    group_neuron.set_active()
    short_sequence = [1,0,2]
    for word in short_sequence:
        bundle.hit([word])
        bundle.show_status()
    print "FINAL PRED: ", final_prediction(bundle.predict_state())
    print bundle.predict_state()

    bundle.reset_all()
    print("FINAL PREDICTION! ", bundle.predict_sequence([0,1,2]))
    bundle.reset_all()

    print("FINAL PREDICTION! ", bundle.predict_sequence([1,0,2]))
    bundle.show_status()

def test_compact_bundle():
    bundle = Bundle(["are","we","there","here", "question", "statement"])

    group_neuron = Neuron()
    bundle.set_activation_neuron(group_neuron)
    group_neuron.set_active()
    bundle.reset_all()

    sequences = [
        ["are", "we", "there", "question"],
        ["we", "are", "there", "statement"],
        ["are", "we", "here", "question"],
        ["we", "are", "here", "statement"],
        ["are", "we", ["there", "here"]]
    ]


    # Learning loop
    for sequence in sequences:
        print "Learning: ", sequence

        bundle.reset_all()
        group_neuron.set_active()
        bundle.show_status()
        bundle.process_sequence(sequence)
        print "FINAL PRED: ", final_prediction(bundle.predict_state())

    # predict loop
    for sequence in sequences:
        leadup = sequence[:-1]
        prediction = sequence[-1:]
        print
        print "Processing: ", leadup

        bundle.reset_all()
        group_neuron.set_active()
        bundle.process_sequence(leadup)
        if type(prediction[0]) == type([]): # If we have multiple, flatten them out
            prediction = prediction[0]
        assert final_prediction(bundle.predict_state()) == prediction, "Result equals prediction"
        print bundle.predict_state()

