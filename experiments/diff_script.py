'''
Input:
[list of dataset dirs] - list of directories containing an lm model (full path).
<diff-command> - command to run the predictions diff
<ref-dirname> - directory of reference model trained on the full train set (no heldout)
<model-name> - name of model being diff'ed

Output: diff results in each directory
'''

import sys
import os
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: %s <diff-command> <ref-dirname> <model-name> < [list of heldout model dirs]" % sys.argv[0])
        sys.exit(1)

    base_command = sys.argv[1]
    ref_dir = sys.argv[2]
    modelname = sys.argv[3]
    ref_model = ref_dir+'/'+modelname

    for line in sys.stdin:
        ho_model_dir = line.strip()
        ho_model = ho_model_dir + '/' + modelname
        ho_note = ho_model_dir + '/' + 'heldout_note.txt'
        diff_result = ho_model + '.diff_result.debug'

        # if os.path.exists(diff_result) and os.stat(diff_result).st_size > 0:
        #     print('\nNOTE: skipping existing results: ' + diff_result + '\n')
        # else:
        command = base_command + ' --corpus_eval ' + ho_note + ' --model1 ' + ref_model + '  --model2 ' + ho_model + ' > ' + diff_result
        print('\nrunning command:\n' + command)
        os.system(command)