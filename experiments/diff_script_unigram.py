'''
Input:
[list of dataset dirs] - list of directories containing an lm model (full path).
<diff-command> - command to run the predictions diff
<ref-vocab-count-file> - the vocab counts on the full train set

Output: diff results in each directory
'''

import sys
import os
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: %s <diff-command> <ref-vocab-count-file> < [list of heldout dirs]" % sys.argv[0])
        sys.exit(1)

    base_command = sys.argv[1]
    vocab_filename = sys.argv[2]
    alpha = "1.0"

    for line in sys.stdin:
        ho_model_dir = line.strip()
        ho_note = ho_model_dir + '/' + 'heldout_note.txt'
        diff_result = ho_model_dir + '/' + 'unigram.alpha'+ alpha + '.diff_result'

        command = base_command + ' ' + ho_note + ' ' + vocab_filename + ' > ' + diff_result
        print('\nrunning command:\n' + command)
        os.system(command)