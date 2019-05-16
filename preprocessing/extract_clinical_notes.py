''' 
Extracts individual notes from a preprocessed multi-note file (not from MIMIC format) 
and saves them in an output directory. 
'''

import sys
import os


def read_note(f):

    note_lines = []
    while True:
        line = f.readline()
        if len(line) == 0: #EOF
            break
        if len(line.strip()) == 0 and len(note_lines) > 0:
            note_lines.append(line) # empty line denotes end-of-note
            break
        else:
            note_lines.append(line)

    return ''.join(note_lines)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: %s <input-filename> <output-dirname> " % sys.argv[0])
        sys.exit(1)

    input_filename = sys.argv[1]
    output_dirname = sys.argv[2]

    print("Extracting clinical notes from %s to dir %s" % (input_filename, output_dirname))

    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    i = 0
    with open(input_filename, 'r') as f:
        while True:
            note = read_note(f)
            if len(note) > 0:
                i += 1
                filename = 'note.'+str(i)+'.txt'
                with open(os.path.join(output_dirname, filename), 'w') as note_file:
                    note_file.write(note)
            else:
                break

    print("Extracted %d notes" % i)

