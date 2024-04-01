import os, sys, re
import numpy as np
import nltk
from nltk import tokenize
import torch
import nltk.lm as lm
from nltk.corpus import stopwords, gutenberg
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import ngrams as ngrams
import pickle
import time
from nltk.tokenize import word_tokenize

from collections import defaultdict, Counter

from utils import parse_arguments, read_settings
import nltk
from nltk.corpus import stopwords

from logger import Logger

from torch.utils.data import random_split, DataLoader, Dataset

import spacy
import pickle


device  = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')

#print ('Device set to {0}'.format(device))

# Spacy model is used to tokenize the data
nlp_sv = spacy.load('sv_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')

#Open the data from sv-en and en-sv
#Return the data in a list of tuples

nlp_sv.max_length = 10_000_000  # Adjust based on your text size
nlp_en.max_length = 10_000_000  # Adjust based on your text size




class TranslationDataset:
    MAX_LENGTH =50 # Maximum length of the sentence
    
    

    def __init__(self, file_en, file_sv, stopword_remove=True, punctuation_remove=True, use_lower=True, rebuild_vocabulary=True,min_freq=2,max_vocab_size=10000,num_lines=1000):
        self.file_en = file_en
        self.file_sv = file_sv
        self.num_lines = num_lines
        self.stop_words_en = set(stopwords.words('english'))
        self.stop_words_sv = set(stopwords.words('swedish'))
        self.vocabulary = {}
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2count = {}


        self.word2_idx = {}
        selfindex2word = {}

        sv,en = self.open_data()
        self.data = list(zip(en,sv))


        # Example
        self.index2word_en = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.index2word_sv = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.word2idx_en = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.word2idx_sv = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.n_words_en = 4
        self.n_words_sv = 4

        #self.n_words = 2  # Count SOS and EOS

        if rebuild_vocabulary:
            self.tokenize()
            #self.remove_stopwords()
            self.to_lower()
            #self.remove_punctuation()
            self.build_vocabulary()

            for lang in ['en', 'sv']:
                with open(f"vocabulary_{lang}.pkl", "wb") as file:
                    pickle.dump(getattr(self, f'word2idx_{lang}'), file)
        else:
            for lang in ['en', 'sv']:
                path_vocab = os.path.join(os.getcwd(), f'vocabulary_{lang}.pkl')
                if os.path.exists(path_vocab):
                    print(f"Loading {lang} word2idx from {path_vocab}")
                    with open(path_vocab, "rb") as file:
                        loaded_dict = pickle.load(file)
                        if isinstance(loaded_dict, dict):
                            setattr(self, f'word2idx_{lang}', loaded_dict)
                            setattr(self, f'vocabulary_{lang}', list(loaded_dict.keys()))  # Set vocabulary based on loaded word2idx

                            # Create idx2word dictionary by reversing word2idx
                            idx2word = {idx: word for word, idx in loaded_dict.items()}     #Another way to have idx2word 
                            setattr(self, f'idx2word_{lang}', idx2word)                     #Double check it !!
                        else:
                            print(f"Error: Expected a dictionary in {path_vocab}, but got a {type(loaded_dict)}")
    def __len__(self):
        return len(self.data)

    def open_data(self): # Reading the first n lines of the data 
        with open(self.file_sv, 'r', encoding='utf-8') as f:
            sv = [next(f) for _ in range(self.num_lines)]
        with open(self.file_en, 'r', encoding='utf-8') as f:
            en = [next(f) for _ in range(self.num_lines)]

        return sv, en

    def tokenize(self): # Important part read the data and tokenize it 
        sv, en = self.open_data()
        
        print('Now Tokenizing Swedish')
        self.tokens_sv = [token.text for sentence in sv for token in nlp_sv(sentence[:nlp_sv.max_length]) if token.text.isalpha() ] # Tokenize it based on the max length of the model
        print('Now Tokenizing English')
        self.tokens_en = [token.text for sentence in en for token in nlp_en(sentence[:nlp_en.max_length])]
        return self.tokens_sv, self.tokens_en
    

    def to_lower(self):
        print('Converting to Lowercase')
        self.tokens_en = [word.lower() for word in self.tokens_en]
        self.tokens_sv = [word.lower() for word in self.tokens_sv]

    def build_vocabulary(self):
        print('Building Vocabulary')
        self.vocabulary_en = Counter(self.tokens_en)
        self.vocabulary_sv = Counter(self.tokens_sv)

        # Filtering by minimum frequency
        self.vocabulary_en = Counter({word: count for word, count in self.vocabulary_en.items() if count >= self.min_freq})
        self.vocabulary_sv = Counter({word: count for word, count in self.vocabulary_sv.items() if count >= self.min_freq})

        # Limiting vocabulary size (if applicable)
        if self.max_vocab_size is not None:
            self.vocabulary_en = Counter(dict(self.vocabulary_en.most_common(self.max_vocab_size)))  # Now using Counter
            self.vocabulary_sv = Counter(dict(self.vocabulary_sv.most_common(self.max_vocab_size)))  # Now using Counter

        self.word2count = {'en': self.vocabulary_en, 'sv': self.vocabulary_sv}

       
        self.vocabulary = {'en': list(self.vocabulary_en), 'sv': list(self.vocabulary_sv)} ##Check this !!
        print('Vocabulary built for both languages')
        #print(self.vocabulary['en'])
        self.create_mapping()


    def create_mapping(self):
        for word in self.vocabulary['en']:
            self.word2idx_en[word] = self.n_words_en
            self.index2word_en[self.n_words_en] = word
            self.n_words_en += 1

        #print('Mapping finished for English here is the mapping for the first 10 words')
        #print(self.word2idx_en)
        

        for word in self.vocabulary['sv']:
            self.word2idx_sv[word] = self.n_words_sv
            self.index2word_sv[self.n_words_sv] = word
            self.n_words_sv += 1

        #print('Mapping finished for Swedish here is the mapping for the first 10 words')
        #print(self.word2idx_sv)

        self.word2_idx = {'en': self.word2idx_en, 'sv': self.word2idx_sv}
        self.index2word = {'en': self.index2word_en, 'sv': self.index2word_sv}

        #print('Mapping finished for both languages')
        #print(self.word2_idx['en'])
        #print(self.word2_idx['sv'])


    def sentences_to_sequences(self, input_sentence, output_sentence):
        input_words = [self.word2idx_en['SOS']] + [self.word2idx_en.get(word, self.word2idx_en['UNK']) for word in input_sentence.split()]
        output_words = [self.word2idx_sv['SOS']] + [self.word2idx_sv.get(word, self.word2idx_sv['UNK']) for word in output_sentence.split()]

        # Truncate the sequences if they're too long
        input_words = input_words[:self.MAX_LENGTH-1]
        output_words = output_words[:self.MAX_LENGTH-1]

        # Add the 'EOS' token at the end
        input_words.append(self.word2idx_en['EOS'])
        output_words.append(self.word2idx_sv['EOS'])

        # Pad the sequences to have a fixed length
        input_tensor = input_words + [self.word2idx_en['PAD']] * (self.MAX_LENGTH - len(input_words))
        output_tensor = output_words + [self.word2idx_sv['PAD']] * (self.MAX_LENGTH - len(output_words))

        return input_tensor, output_tensor

            
    def __getitem__(self, idx):
        input_sentence, output_sentence = self.data[idx]
        input_tensor, output_tensor = self.sentences_to_sequences(input_sentence, output_sentence)
        return torch.tensor(input_tensor).view(-1), torch.tensor(output_tensor).view(-1)
  
    

def main():

    args = parse_arguments()

    # Read the settings from the YAML file
    settings = read_settings(args.config)
    # Create an instance of TextCleaner

    dataset = TranslationDataset(**settings['paths'])


if __name__ == '__main__':

    main()


   