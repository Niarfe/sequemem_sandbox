import sys
sys.path.append("./sequemem")
from seqds import *

def test_seqds():
    seq = Seqds('adz')

    assert seq.predict("The quick brown fox").sdr_predicted        == []
    assert seq.predict("The quick").sdr_predicted                  == ['brown']
    assert seq.predict("The quick").sdr_predicted                  == ['brown']
    
    assert seq.predict("The quick").sdr_predicted                  == ['brown']
    assert seq.sdr_active                                          == ['The#quick']
    assert seq.sdr_predicted                                       == ['brown'] 
    
    assert seq.predict("The quick brown").sdr_predicted            == ['fox']
    assert seq.predict("The quick brown fox").sdr_predicted        == []
    assert seq.sdr_active                                          == ['The#quick#brown#fox']
    
    assert seq.predict("The quick brown").sdr_predicted            == ['fox']
    assert seq.predict("The quick brown").sdr_predicted            == ['fox']
    
    assert seq.predict("Every good boy does fine").sdr_predicted   == []
    assert seq.predict("Every").sdr_predicted                      == ['good']

    assert seq.predict("The quick red dog ran away").sdr_predicted == []
    assert seq.predict("The quick red fox ran away").sdr_predicted == []

    assert seq.predict("The quick red").sdr_predicted              == ['dog', 'fox']
    assert seq.sdr_active                                          == ['The#quick#red']
    assert seq.get_sdrs()                                          == [['The#quick#red'], ['dog', 'fox']]
