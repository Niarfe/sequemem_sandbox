"""
I want to check if speed can be increased for the purposes
of collecting total sums, detecting most occuring words
which help detect stop words, and to get 'related words'
The use with sequemem takes about 20 minutes because of
all the neuron activations etc.  This can run in seconds
if we skip all that.  Also this uses a Counter not a Set.
"""
from collections import Counter
from collections import defaultdict
import pandas as pd
df = pd.DataFrame()
lines = []
bigkeys = defaultdict(list)
with open('../data/sent_grimms_fairy_tales.txt', 'r') as source:
    for idx, line in enumerate(source):
        c = Counter(line.split(' '))
        [bigkeys[k].append(c) for k, _ in c.items()]
        if idx % 1000 == 0:
            print idx

print len(bigkeys['feather'])

cf = Counter()

for item in bigkeys['feather']:
    cf += item

print cf
