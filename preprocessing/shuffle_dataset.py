'''
Shuffles a file containing notes
'''

import sys
import random

RANDOM_SEED = 34628743

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: %s <dataset-filename>" % sys.argv[0])
        sys.exit(1)

    filename = sys.argv[1]
    notes = []

    with open(filename, 'r') as fin:
        note_lines = []
        for line in fin:
            if len(line.strip()) == 0:
                if len(note_lines) > 0:
                    notes.append(''.join(note_lines))
                    note_lines = []
            else:
                note_lines.append(line)
        if len(note_lines) > 0:
            notes.append(''.join(note_lines))

    random.Random(RANDOM_SEED).shuffle(notes)
    
    with open(filename+'.shuffle', 'w') as fout:
        for note in notes:
            fout.write(note+'\n')
