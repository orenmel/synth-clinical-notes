# Synthetic Clinical Notes
This repository contains the code used in the paper:

**Towards Automatic Generation of Shareable Synthetic Clinical Notes Using Neural Language Models.**   
Oren Melamud and Chaitanya Shivade. The Clinical NLP Workshop at NAACL (2019).

## Requirements

* Python 3.6
* Pytorch 0.3.1
* Spacy 2.0.7
* `python -m spacy download en`

It is recommended to run the neural training code on a GPU-enabled platform.   
If running on CPU, remove param --cuda where relevant.

## License

The code is this repository is available under the **Apache 2.0 License** with the following exceptions:

The code under **word_language_model** is mostly copied from this [repository](https://github.com/pytorch/examples/tree/master/word_language_model).      
It is available under the [BSD 3-Clause license](https://github.com/pytorch/examples/blob/master/LICENSE)   

The code under **BioNLP-2016-master** is mostly copied from this [repository](https://github.com/cambridgeltl/BioNLP-2016).   
It is available under the **Creative Commons Attribution (CC BY) license**.

## Setting up the environment

You can use the environment.yml file to ensure that your environment is compatible with the one used in the paper. Run the following commands: 

```
conda env create -f environment.yml
source activate medlm
python -m spacy download en
```

## Obtaining the data. 

Follow these steps to generate MedText-2 (small) and MedText-103 (large):
* Obtain MIMIC-III version v1.4 from https://mimic.physionet.org/ (contact MIMIC for authorization)  
```
md5sum NOTEEVENTS.csv = df33ab9764256b34bfc146828f440c2b
```

Run the command below to generate the train/valid/test splits. Note that this might take a day to run.
* `cd /PATH-TO-CODE/preprocessing/`
* `./generate_lm_benchmark.sh /PATH-TO-DATA/NOTEEVENTS.csv /PATH-TO-DATA/MED-TEXT-DIR/`

Ensure that the md5sum for the generated files inside `/PATH-TO-DATA/MED-TEXT-DIR/` match as follows:

```
8a9ef62b91aa44c8fa01aebeb65cab62  tmp/all.txt
927a10bcf1effee89833f8a3206925e2  tmp/all.txt.shuffle
00f96b4150f9b0353ffcfe2fad0a9aef  Discharge_summary.small.train.txt
5c9c103fb677e2021fd06ab42ae7503b  Discharge_summary.small.valid.txt
4e6b93b088c6c66cb30bb211acbd72b9  Discharge_summary.small.test.txt
9e60c1e6b646f307668c0a8c2c93a244  Discharge_summary.large.train.txt
9a559ead5aae7e5b8518a7ceed113b83  Discharge_summary.large.valid.txt
669e0fe2e0d4c3e75525bcac8b97dbf6  Discharge_summary.large.test.txt
```


## Training and evaluating perplexity of language models

### Training neural LMs and evaluating on validation

```
python /PATH-TO-CODE/word_language_model/main.py --epochs 20 --cuda --dropout DROP --emsize 650 --nhid 650 --vocab /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.small.all.txt.vocab --lr 20.0 --log-interval 2000 --data /PATH-TO-DATA/MED-TEXT-DIR/small/  --save /PATH-TO-MODELS/MODEL_NAME.pt
```

```
python /PATH-TO-CODE/word_language_model/main.py --epochs 80 --epoch_size 2860000 --cuda --dropout DROP --emsize 650 --nhid 650 --vocab /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.large.all.txt.vocab --lr 20.0 --lr_backoff 1.2 --batch_size 20 --log-interval 2000 --data /PATH-TO-DATA/MED-TEXT-DIR/large --save /PATH-TO-MODELS/OUTPUT-MODEL_NAME.pt
```

(e.g. OUTPUT-MODEL-NAME.pt = rnn_model.e20.d650.drop0.0.pt)

Note that 80 epochs of size 2860000 tokens (including artificial EOS tokens) is equivalent to doing 2 full epochs

### Evaluating neural LMs on test 

```
python /PATH-TO-CODE/word_language_model/main.py --cuda --data /PATH-TO-DATA/MED-TEXT-DIR/SIZE/ --load /PATH-TO-MODELS/MODEL-NAME.pt --test
```

SIZE stands for either 'small' or 'large'


### Computing perplexity of unigram models

```
python /PATH-TO-CODE/experiments/unigram_perplexity.py /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.SIZE.TYPE.txt /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.SIZE.train.txt.eo.vocab 1.0
```

TYPE stands for 'valid' or 'test'


## Generating synthetic notes

Generate notes using neural LMs:   
```
python /PATH-TO-CODE/word_language_model/generate.py --data /PATH-TO-DATA/MED-TEXT-DIR/SIZE/ --checkpoint /PATH-TO-MODELS/MODEL_NAME.pt --outf /PATH-TO-DATA/MED-TEXT-M-DIR/SIZE/SYNTH-NOTES-FILENAME.txt --words NUMBER-OF-WORDS-TO-GENERATE --cuda
```

Generate notes using unigram model:    
```
python /PATH-TO-CODE/experiments/generate_notes_from_unigram.py /PATH-TO-DATA/MED-TEXT-M-DIR/SIZE/SYNTH-NOTES-FILENAME.txt /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.SIZE.train.txt.eo.vocab NUMBER-OF-WORDS-TO-GENERATE
```

## Computing the privacy measure

### Generating held-out datasets (T\\{t}) and training models on them

Generate held-out datasets:

```
python /PATH-TO-CODE/experiments/clinical_notes_hold_out.py /PATH-TO-DATA/MED-TEXT-DIR/SIZE/ /PATH-TO-DATA/MED-TEXT-DIR/heldout/SIZE/ 30
```

Train heldout lm models:

```
cd /PATH-TO-DATA/MED-TEXT-DIR/heldout/SIZE/
```
```
ls -d heldout.* | python /PATH-TO-CODE/word_language_model/train_script.py "python /PATH-TO-CODE/word_language_model/main.py --epochs EPOCHS_NUM --epoch_size EPOCH_SIZE --cuda --dropout DROP --emsize 650 --nhid 650 --vocab /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.SIZE.all.txt.vocab --lr 20.0 --lr_backoff 1.2 --batch_size 20 --log-interval 2000" /PATH-TO-DATA/MED-TEXT-DIR/heldout/SIZE/ OUTPUT-MODEL-NAME.pt
```

Use OUTPUT-MODEL-NAME.pt that corresponds with the model trained with the same data and parameters in the previous steps
(e.g. OUTPUT-MODEL-NAME.pt = rnn_model.e20.d650.drop0.0.pt)

Use the same parameters (i.e. EPOCHS_NUM, EPOCH_SIZE) as used in the respective SIZE=small/large models trained in previous steps.


### Using held-out dataset models to compute privacy measure (prediction diffs)


Compute predictions diff per every word in the heldout note and dump it to "diff_result.debug" files:

```
ls -d /PATH-TO-DATA/MED-TEXT-DIR/heldout/SIZE/* | python /PATH-TO-CODE/word_language_model/experiments/diff_script.py "python /PATH-TO-CODE/word_language_model/model_predictions_diff.py --cuda --corpus_vocab /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.SIZE.all.txt.vocab"  /PATH-TO-DATA/MED-TEXT-DIR/SIZE/ INPUT-MODEL-NAME.pt
```

(INPUT-MODEL-NAME corresponds to previously learned models e.g. MODEL-NAME.pt = rnn_model.e20.d650.drop0.0.pt)

Aggregate results from the 30 different runs into the "privacy.debug" file (we chose the mean-max metric)

```
cat `find /PATH-TO-DATA/MED-TEXT-DIR/heldout/SIZE/ -name "INPUT-MODEL-NAME.pt.diff_result.debug"` | grep "mean diff metric" | python /PATH-TO-CODE/experiments/max_diff_script.py > /PATH-TO-DATA/MED-TEXT-DIR/heldout/SIZE/INPUT-MODEL-NAME.privacy
```

In the paper, the 'mean_max_note_measure' was used.

Compute the same for unigram diff privacy

```
ls -d /PATH-TO-DATA/MED-TEXT-DIR/heldout/SIZE/heldout* | python /PATH-TO-CODE/experiments/diff_script_unigram.py "python /PATH-TO-CODE/experiments/unigram_diff_privacy.py" /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.SIZE.train.txt.count.vocab
```
```
cat `find /PATH-TO-DATA/MED-TEXT-DIR/heldout/SIZE/ -name "unigram*diff_result"` | awk '{ total += $0; count++ } END { print total/count }'
```

## Training word embeddings 

Word embeddings were trained using the word2vec [package](https://code.google.com/archive/p/word2vec/) from [Mikolov et al. (2013)](http://arxiv.org/pdf/1301.3781.pdf). The command below specifies the parameters used in the paper to train word embeddings on both real and synthetic texts.

```
word2vec -train <text_file> -output <output_text> -cbow 0 -size 300 -window 5 -negative 10 -threads 20 -binary 0 -iter 10
```

## Word similarity evaluation

```
cd /PATH-TO-CODE/BioNLP-2016-master/
```
```
python ./evaluate.py -w /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.large.train.txt.count.vocab  -i 
/PATH-TO-EMBEDDINGS/EMBEDDING-FILE.large.txt -m 30 > /PATH-TO-EMBEDDINGS/EMBEDDING-FILE.large.txt.sim
```
```
python ./evaluate.py -w /PATH-TO-DATA/MED-TEXT-DIR/Discharge_summary.small.train.txt.count.vocab  -i 
/PATH-TO-EMBEDDINGS/EMBEDDING-FILE.small.txt -m 20 > /PATH-TO-EMBEDDINGS/EMBEDDING-FILE.small.txt.sim
```

## NLI evaluation
Follow instructions from the MedNLI [github repository](https://github.com/jgc128/mednli_baseline) from [Romanov and Shivade (2018)](). Specify the bag of words model and word embeddings specific to each experiment as a parameter. 


## True-casing evaluation
Follow instructions from the [github repository](https://github.com/raymondhs/char-rnn-truecase) from [Susanto et al. 2016](https://www.aclweb.org/anthology/D16-1225). LSTM with the 'small'/'large' default parameters were used in the paper for MedText-2/MedText-103 experiments, respectively.


## Reference
The [paper](https://www.aclweb.org/anthology/W19-1905.pdf) was accepted to the ClinicalNLP Workshop at NAACL 2019

Oren Melamud, and Chaitanya Shivade. __Towards Automatic Generation of Shareable Synthetic Clinical Notes Using Neural Language Models.__ Proceedings of the 2nd Clinical Natural Language Processing Workshop. 2019.

```
@inproceedings{melamud2019towards,
  title={Towards Automatic Generation of Shareable Synthetic Clinical Notes Using Neural Language Models},
  author={Melamud, Oren and Shivade, Chaitanya},
  booktitle={Proceedings of the 2nd Clinical Natural Language Processing Workshop at NAACL},
  url={https://www.aclweb.org/anthology/W19-1905.pdf},
  pages={35--45},
  year={2019}
}
```
