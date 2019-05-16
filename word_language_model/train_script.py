'''
Input:
[list of dataset dirs] - list of directories containing an lm dataset each (not full path, relative to <datasets-root-dirname>).
<lm-command> - command to run the lm training
<datasets-root-dirname> - directory under which all dataset dirs are
<output-model-name> - name to use when saving the learned lm model

Output: trained lm model in each directory
'''

import sys
import os
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: %s <lm-command> <heldout-root-dirname> <output-model-name> < [list of dataset dirs]" % sys.argv[0])
        sys.exit(1)

    base_command = sys.argv[1]
    rootdir = sys.argv[2]
    modelname = sys.argv[3]

    for line in sys.stdin:
        datadir = rootdir+'/'+line.strip()

        model_full_path = datadir+'/'+modelname

        if os.path.exists(model_full_path) and os.stat(model_full_path).st_size > 0:
            print('\nNOTE: skipping existing trained model: ' + model_full_path + '\n')
        else:
            command = base_command + ' --data ' + datadir + '  --save ' + model_full_path
            print('\nrunning command:\n' + command)
            os.system(command)
