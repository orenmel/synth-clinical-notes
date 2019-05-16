#!/bin/bash
# This script generates MedText-2 (small) and MedText-103 (large) datasets from the MIMIC NOTEEVENT.csv file

display_usage() {
	echo -e "\nFirst cd into the directory of the script. Then run:"
	echo -e "./generate_lm_benchmark.sh <input_file> <output_dir> \n"
}

if [  $# -le 1 ]
	then
		display_usage
		exit 1
fi

DEBUG=false

INPUT_FILE=$1
OUTPUT_DIR=$2
NOTE_TYPE="Discharge_summary"
TEST_RECORDS_NUM=128
SMALL_TRAIN_RECORDS_NUM=1280

echo "INPUT_FILE:" $INPUT_FILE
echo "OUTPUT_DIR:" $OUTPUT_DIR

TMP_DIR=$OUTPUT_DIR"/tmp/"
LARGE_DIR=$OUTPUT_DIR"/large/"
SMALL_DIR=$OUTPUT_DIR"/small/"
COMMAND="mkdir -p "$TMP_DIR" && mkdir -p "$LARGE_DIR" && mkdir -p "$SMALL_DIR
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi


ALL_TEXT_FILE=$TMP_DIR"/all.txt"
COMMAND="python mimic3_extract_and_tokenize.py "$INPUT_FILE" "$ALL_TEXT_FILE" "$NOTE_TYPE
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi

COMMAND="python shuffle_dataset.py "$ALL_TEXT_FILE
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi

SHUFFLED_TEXT_FILE=$ALL_TEXT_FILE".shuffle"
TEST_FILE=$SHUFFLED_TEXT_FILE".test"
VALIDATION_FILE=$SHUFFLED_TEXT_FILE".valid"
TRAIN_LARGE_FILE=$SHUFFLED_TEXT_FILE".large.train"
TRAIN_SMALL_FILE=$SHUFFLED_TEXT_FILE".small.train"
ALL_SMALL_FILE=$SHUFFLED_TEXT_FILE".small.all"
ALL_LARGE_FILE=$SHUFFLED_TEXT_FILE

COMMAND="python split_dataset.py "$SHUFFLED_TEXT_FILE" "$TEST_FILE" "$VALIDATION_FILE" "$TRAIN_LARGE_FILE" "$TEST_RECORDS_NUM
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi


COMMAND="python split_dataset.py "$TRAIN_LARGE_FILE" "$TRAIN_SMALL_FILE" /dev/null /dev/null "$SMALL_TRAIN_RECORDS_NUM
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi


COMMAND="cat "$TRAIN_SMALL_FILE" "$VALIDATION_FILE" "$TEST_FILE" > "$ALL_SMALL_FILE
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    eval $COMMAND
fi

ALL_SMALL_FILE_VOCAB=$OUTPUT_DIR"/"$NOTE_TYPE".small.vocab"
ALL_LARGE_FILE_VOCAB=$OUTPUT_DIR"/"$NOTE_TYPE".large.vocab"

COMMAND="python vocab_dataset.py "$ALL_SMALL_FILE" "$ALL_SMALL_FILE_VOCAB
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi


COMMAND="python vocab_dataset.py "$ALL_LARGE_FILE" "$ALL_LARGE_FILE_VOCAB
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi


TEST_FILE_FINAL_LARGE=$OUTPUT_DIR"/"$NOTE_TYPE".large.test.txt"
VALIDATION_FILE_FINAL_LARGE=$OUTPUT_DIR"/"$NOTE_TYPE".large.valid.txt"
TRAIN_FILE_FINAL_LARGE=$OUTPUT_DIR"/"$NOTE_TYPE".large.train.txt"
ALL_FILE_FINAL_LARGE=$OUTPUT_DIR"/"$NOTE_TYPE".large.all.txt"

TEST_FILE_FINAL_SMALL=$OUTPUT_DIR"/"$NOTE_TYPE".small.test.txt"
VALIDATION_FILE_FINAL_SMALL=$OUTPUT_DIR"/"$NOTE_TYPE".small.valid.txt"
TRAIN_FILE_FINAL_SMALL=$OUTPUT_DIR"/"$NOTE_TYPE".small.train.txt"
ALL_FILE_FINAL_SMALL=$OUTPUT_DIR"/"$NOTE_TYPE".small.all.txt"

COMMAND="python unknown_dataset.py "$TEST_FILE" "$ALL_LARGE_FILE_VOCAB" "$TEST_FILE_FINAL_LARGE
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi
COMMAND="python unknown_dataset.py "$VALIDATION_FILE" "$ALL_LARGE_FILE_VOCAB" "$VALIDATION_FILE_FINAL_LARGE
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi
COMMAND="python unknown_dataset.py "$TRAIN_LARGE_FILE" "$ALL_LARGE_FILE_VOCAB" "$TRAIN_FILE_FINAL_LARGE
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi

COMMAND="cat "$TRAIN_FILE_FINAL_LARGE" "$VALIDATION_FILE_FINAL_LARGE" "$TEST_FILE_FINAL_LARGE" > "$ALL_FILE_FINAL_LARGE
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    eval $COMMAND
fi



COMMAND="python unknown_dataset.py "$TEST_FILE" "$ALL_SMALL_FILE_VOCAB" "$TEST_FILE_FINAL_SMALL
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi
COMMAND="python unknown_dataset.py "$VALIDATION_FILE" "$ALL_SMALL_FILE_VOCAB" "$VALIDATION_FILE_FINAL_SMALL
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi
COMMAND="python unknown_dataset.py "$TRAIN_SMALL_FILE" "$ALL_SMALL_FILE_VOCAB" "$TRAIN_FILE_FINAL_SMALL
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi

COMMAND="cat "$TRAIN_FILE_FINAL_SMALL" "$VALIDATION_FILE_FINAL_SMALL" "$TEST_FILE_FINAL_SMALL" > "$ALL_FILE_FINAL_SMALL
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    eval $COMMAND
fi


COMMAND="python ../word_language_model/data.py "$ALL_FILE_FINAL_SMALL
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi

COMMAND="python ../word_language_model/data.py "$ALL_FILE_FINAL_LARGE
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi


COMMAND="python vocab_dataset.py "$TRAIN_FILE_FINAL_SMALL" "$TRAIN_FILE_FINAL_SMALL".eo.vocab EO "$ALL_FILE_FINAL_SMALL".vocab"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi

COMMAND="python vocab_dataset.py "$TRAIN_FILE_FINAL_SMALL" "$TRAIN_FILE_FINAL_SMALL".count.vocab "$ALL_FILE_FINAL_SMALL".vocab"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi


COMMAND="python vocab_dataset.py "$TRAIN_FILE_FINAL_LARGE" "$TRAIN_FILE_FINAL_LARGE".eo.vocab EO "$ALL_FILE_FINAL_LARGE".vocab"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi

COMMAND="python vocab_dataset.py "$TRAIN_FILE_FINAL_LARGE" "$TRAIN_FILE_FINAL_LARGE".count.vocab "$ALL_FILE_FINAL_LARGE".vocab"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi


COMMAND="ln -s "$TRAIN_FILE_FINAL_SMALL" $SMALL_DIR/train.txt"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi
COMMAND="ln -s "$VALIDATION_FILE_FINAL_SMALL" $SMALL_DIR/valid.txt"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi
COMMAND="ln -s "$TEST_FILE_FINAL_SMALL" $SMALL_DIR/test.txt"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi
COMMAND="ln -s "$TRAIN_FILE_FINAL_LARGE" $LARGE_DIR/train.txt"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi
COMMAND="ln -s "$VALIDATION_FILE_FINAL_LARGE" $LARGE_DIR/valid.txt"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi
COMMAND="ln -s "$TEST_FILE_FINAL_LARGE" $LARGE_DIR/test.txt"
echo "Running cmd: "$COMMAND
if [ "$DEBUG" = false ] ; then
    $COMMAND
fi



