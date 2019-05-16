'''
Splits a file containing notes
'''

import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: %s <input-filename> <output1-filename> <output2-filename> <output3-filename> <split-size>" % sys.argv[0])
        sys.exit(1)

    input_filename = sys.argv[1]
    output1_filename = sys.argv[2]
    output2_filename = sys.argv[3]
    output3_filename = sys.argv[4]
    split_size = int(sys.argv[5])

    notes = []

    with open(input_filename, 'r') as fin, open(output1_filename, 'w') as fout1, open(output2_filename, 'w') as fout2, open(output3_filename, 'w') as fout3:
        n = 0
        for line in fin:
            if len(line.strip()) == 0:
                n += 1
            if n > 2*split_size:
                fout = fout3
            elif n > split_size:
                fout = fout2
            else:
                fout = fout1
            fout.write(line)



