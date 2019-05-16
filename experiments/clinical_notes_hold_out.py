''' 
This module takes a preprocessed multi-note file and generates N output dirs,
where each output dir has a notes file that is missing one random note 
and the missing note is included in a different file.
'''

import sys
import os
import random


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


def read_all_notes(input_filename):
    notes = []
    with open(input_filename, 'r') as f:
        while True:
            note = read_note(f)
            if len(note) > 0:
                notes.append(note)
            else:
                break

    print("Read %d notes" % len(notes))
    return notes


def create_held_out_dir(output_dirname, notes, note_id, valid_filename):
    held_out_dirname = output_dirname+'/heldout.'+str(note_id)
    if os.path.exists(held_out_dirname):
        return False
    else:
        os.makedirs(held_out_dirname)
        note_filename = held_out_dirname+'/heldout_note.txt'
        with open(note_filename, 'w') as f:
            f.write(notes[note_id])

        rest_filename = held_out_dirname+'/train.txt'
        with open(rest_filename, 'w') as f:
            for i in range(len(notes)):
                if i != note_id:
                    f.write(notes[i])

        valid_copy_filename = held_out_dirname+'/valid.txt'
        os.system("cp %s %s" % (valid_filename, valid_copy_filename))
        return True


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: %s <input-dirname> <output-root-dirname> <N>" % sys.argv[0])
        sys.exit(1)

    train_filename = sys.argv[1]+'/train.txt'
    valid_filename = sys.argv[1] + '/valid.txt'

    output_dirname = sys.argv[2]
    n = int(sys.argv[3])

    print("Creating %d held-out clinical notes from %s under output root dir %s" % (n, train_filename, output_dirname))

    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)

    notes = read_all_notes(train_filename)
    sampled_note_ids = set()
    for i in range(n):
        while True:
            note_id = random.randint(0, len(notes)-1)
            if note_id not in sampled_note_ids and create_held_out_dir(output_dirname, notes, note_id, valid_filename):
                sampled_note_ids.add(note_id)
                break


