# Nordic Neural Net

This project contains a neural network for machine translation between English and Swedish.

Please also check our other github repository. We train on models on other repository first and push it to here in a cleaner version.
https://github.com/Razbolt/nmt-small

## Description

The Nordic Neural Net project aims to develop a machine translation system that can accurately translate text between English and Swedish. The neural network model used in this project is designed to handle the complexities of language translation and provide high-quality translations.

## Directory Structure

- `sv-en/`: This directory contains two text files, one for each language (English and Swedish). These files serve as the input data for training and testing the neural network model.

## Files

- `dataset.py`: This file contains a `TranslationDataset` class that is used for text cleaning and creating a vocabulary for each language. The class takes in two files (one for each language) and performs several preprocessing steps, including tokenization, stopword removal, punctuation removal, and lowercasing. It also has the ability to build a vocabulary from scratch or load a pre-existing vocabulary from a pickle file.

The important things in here is that TranslationDataset may take different parameters to build a vocabulary. Main objective of this dataset to create a `vocabulary`, `word2index` and `index2word` mapping.
Inside `TranslationDataset` there is `MAX_LENGTH= 50` parameters takes the maximum number of sequence length. If it less then given parameters it will be truncated. Feel free to increase or decrease based on computational resources.

- `config.yaml`: This file contains the setting of couple of things.
First it contains `paths` for folders and some parameters in order to create a vocabulary. 2 important parameters are `max_vocab_size` and `num_lines`. Former is limitting the vocabulary size and latter is used for creating a vocabulary. In original there are around 2 millions sentences. So, `num_lines` used for limitting number of rows to read and therefore build vocabulary.

Secondly, it contains `model_settings` is for easily adjust and train the model. 

- `models.py`: In this folder there will be different models in order to feed machine translation. But now for basic apporach we have `Encoder` `Decoder` and `Seq2Seq` that contains both models. 
In future it will be updated as attention layers and transformers also. It is important to check this file and play around `main()` to see how this models are working before pre-processing.

- `train.py`: This file contains the combination of every steps. 

## Usage

To use the `TextCleaner` class, instantiate it with the paths to the English and Swedish text files:

```python
from data import TextCleaner

# Instantiate the TextCleaner class with the paths to the English and Swedish text files
cleaner = TextCleaner('sv-en/europarl-v7.sv-en.en', 'sv-en/europarl-v7.sv-en.sv')

# Access the cleaned tokens for English and Swedish
english_tokens = cleaner.tokens_en
swedish_tokens = cleaner.tokens_sv

# Access the vocabulary
vocabulary = cleaner.vocabulary

# Print the first 10 English tokens
print(english_tokens[:10])

# Print the first 10 Swedish tokens
print(swedish_tokens[:10])
