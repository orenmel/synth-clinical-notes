'''
Aggregate results from the output of diff_script.py
'''

import sys


max_diff1 = None
max_diff2 = None

diffs1 = []
diffs2 = []
for line in sys.stdin:
    toks = line.split()
    diff1 = float(toks[3])
    diffs1.append(diff1)
    diff2 = float(toks[7])
    diffs2.append(diff2)

print('max_mean_note_measure:\t%.2f\tmax_max_note_measure:\t%.2f\tmean_mean_note_measure:\t%.2f\tmean_max_note_measure:\t%.2f' % (max(diffs1), max(diffs2), sum(diffs1)/len(diffs1), sum(diffs2)/len(diffs2)))

