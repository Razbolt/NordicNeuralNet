# Nordic Neural Net

This project contains a neural network for machine translation between English and Swedish.

## Description

The Nordic Neural Net project aims to develop a machine translation system that can accurately translate text between English and Swedish. The neural network model used in this project is designed to handle the complexities of language translation and provide high-quality translations.

## Directory Structure

- `sv-en/`: This directory contains two text files, one for each language (English and Swedish). These files serve as the input data for training and testing the neural network model.

## Files

- `data.py`: This file contains a `TextCleaner` class that is used for text cleaning and creating a vocabulary for each language. The class takes in two files (one for each language) and performs several preprocessing steps, including tokenization, stopword removal, punctuation removal, and lowercasing. It also has the ability to build a vocabulary from scratch or load a pre-existing vocabulary from a pickle file.

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
