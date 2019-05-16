'''
Loads notes from MIMIC, splits to sentences and tokens and dumps output to file
'''

import csv
import re
import spacy
import sys


SPECIAL_IDENTIFIER = 'TSPECIALID'
MIN_COLUMN_WIDTH = 42

# temporarily using the left-hand side tokens to make it safely as a single token through the spacy tokenizer
# plus forcing newline after colon
special_tokens = {SPECIAL_IDENTIFIER:'<deidentified>', ':':':\n'}


def decorate(tok):
    return special_tokens[tok.strip()] if tok in special_tokens else tok.strip()


def load_notes(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        notes = {}
        for record in reader:
            text = record['TEXT']
            category = record['CATEGORY']
            if category not in notes:
                notes[category] = []
            notes[category].append(text)

    return notes


def tokenize(lines):

    line = ' '.join(lines)
    line = re.sub('>[\.]+<', ' ', line)  # spacy tokenizer can't handle >...<

    is_bullet = re.search('^[1-9][0-9]{0,1}\. ', lines[0])

    if is_bullet: # spacy has issues with handling bullet sentences
        prefix = line[:line.find('.')]+' . '
        line = line[len(prefix)-2:]
    else:
        prefix = ''

    single_sent = (len(lines) == 1 and is_bullet)  # prevent sentence splitting in one-line bullets
    
    try:
        return prefix + tokenize_imp(line, single_sent=single_sent)
    except AssertionError:
        line = re.sub('\.[\.]+', ' ', line)
    try:
        return prefix + tokenize_imp(line, single_sent=single_sent)
    except AssertionError:
        print("\nFatal parse error. Skipping line: \n" + line + "\n")
        return '\n'


def tokenize_imp(line, single_sent=False):
    sents = []
    doc = nlp(str(line))
    toks = []
    for sent in doc.sents:
        toks.extend([decorate(tok.text) for tok in sent if len(tok.text.strip()) > 0])
        # conservative sentence splitting (but forcing sent split on ':' in the decorate method)
        if len(toks) > 0 and (toks[-1] == '.' or toks[-1] == '?' or toks[-1] == '!'):
            toks_str = ' '.join(toks)
            sub_sents = [s.strip() for s in toks_str.split('\n')]
            sents.extend(sub_sents)
            toks = []
    if len(toks) > 0:
        toks_str = ' '.join(toks)
        sub_sents = [s.strip() for s in toks_str.split('\n') if len(s.strip()) > 0]
        sents.extend(sub_sents)

    if single_sent:
        return ' '.join(sents)
    else:
        return '\n'.join(sents)


if __name__ == '__main__':

    import sys

    print('Loading spacy en...')
    nlp = spacy.load('en')

    if len(sys.argv) < 2:
        print("Usage: %s <noteevents-filename> <output-filename> [<note-type>]" % sys.argv[0])
        sys.exit(1)

    filename = sys.argv[1]
    notes = load_notes(filename)



    output = sys.argv[2]

    if len(sys.argv) > 3:
        note_type = sys.argv[3]
    else:
        note_type = None

    for category, cat_notes in notes.items():

        category = re.sub(r'/', '_', category)
        suffix = '_'.join(category.split())

        if note_type != None and suffix != note_type:
            continue

        # with open(output+'.'+suffix+'.txt', 'w') as f:
        with open(output, 'w') as f:

            for i, n in enumerate(cat_notes):

                print('\r', end='')
                print('Processed %d notes' % i, end='')

                # if i == 8:
                #     break

                n = re.sub(r'Discharge Date:', '\nDischarge Date:', n)

                paragraph_lines = []
                last_short_lines = 0
                # eol sometimes breaks a sentence in the middle, so we're trying to concat lines when appropriate
                for l in n.split('\n'):
                    l = l.strip()
                    l = re.sub(r'\(?\'?\[\*{2}[\w \-\(\)\\\/]+\*{2}\]\)?\:?', ' '+SPECIAL_IDENTIFIER+' ', l)

                    if len(l) == 0:
                        last_short_lines += 1
                    else:
                        if last_short_lines > 0:
                            if not(last_short_lines == 1 and l[0].islower()): # handles accidental single space lines in the middle of a sentence
                                if len(paragraph_lines) > 0:
                                    f.write(tokenize(paragraph_lines) + '\n')
                                    paragraph_lines = []
                        if len(l) >= MIN_COLUMN_WIDTH:
                            last_short_lines = 0
                        else:
                            last_short_lines = 1

                        is_bullet = re.search('^[1-9][0-9]{0,1}\. ', l)
                        if is_bullet: # bullet starts a new sentence
                            if len(paragraph_lines) > 0:
                                f.write(tokenize(paragraph_lines) + '\n')
                                paragraph_lines = []
                        if l.endswith(':'):# ':' ends a sentence
                            paragraph_lines.append(l)
                            f.write(tokenize(paragraph_lines) + '\n')
                            paragraph_lines = []
                        else:
                            assert(len(l) != 0)
                            paragraph_lines.append(l)


                if len(paragraph_lines) > 0:
                    f.write(tokenize(paragraph_lines) + '\n\n')
                else:
                    f.write('\n\n')

            print()
