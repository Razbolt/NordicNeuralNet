import os, sys, re
import numpy as np
import nltk
from nltk import tokenize

import nltk.lm as lm
from nltk.corpus import stopwords, gutenberg
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import ngrams as ngrams
import pickle
import time
from nltk.tokenize import word_tokenize

from collections import defaultdict, Counter

import nltk
from nltk.corpus import stopwords


import spacy
import pickle

nlp_sv = spacy.load('sv_core_news_sm')
nlp_en = spacy.load('en_core_web_sm')

#Open the data from sv-en and en-sv
#Return the data in a list of tuples

nlp_sv.max_length = 10_000_000  # Adjust based on your text size
nlp_en.max_length = 10_000_000  # Adjust based on your text size


class TextCleaner:
    def __init__(self, file_en, file_sv, stopword_remove=True, punctuation_remove=True, use_lower=True, rebuild_vocabulary=True):
        self.file_en = file_en
        self.file_sv = file_sv
        self.stop_words_en = set(stopwords.words('english'))
        self.stop_words_sv = set(stopwords.words('swedish'))
        self.vocabulary = {}

        self.tokens_sv, self.tokens_en = self.tokenize()

        if stopword_remove:
            self.remove_stopwords()

        if use_lower:
            self.to_lower()

        if punctuation_remove:
            self.remove_punctuation()

        if rebuild_vocabulary:
            self.build_vocabulary()
            with open("vocabulary.pkl", "wb") as file:
                pickle.dump(self.vocabulary, file)
        else:
            path_vocab = os.path.join(os.getcwd(), 'vocabulary.pkl')
            if os.path.exists(path_vocab):
                # Load the object back from the file
                with open(path_vocab, "rb") as file:
                    self.vocabulary = pickle.load(file)


    def open_data(self,n = 10000):
        with open(self.file_sv, 'r', encoding='utf-8') as f:
            sv = [next(f) for _ in range(n)]
        with open(self.file_en, 'r', encoding='utf-8') as f:
            en = [next(f) for _ in range(n)]

        return sv, en

    def tokenize(self):
        sv, en = self.open_data()
        
        print('Now Tokenizing Swedish')
        self.tokens_sv = [token.text for sentence in sv for token in nlp_sv(sentence[:nlp_sv.max_length])]
        print('Now Tokenizing English')
        self.tokens_en = [token.text for sentence in en for token in nlp_en(sentence[:nlp_en.max_length])]
        return self.tokens_sv, self.tokens_en

    def remove_stopwords(self):
        print('Removing Stopwords')
        self.tokens_en = [word for word in self.tokens_en if word.lower() not in self.stop_words_en]
        self.tokens_sv = [word for word in self.tokens_sv if word.lower() not in self.stop_words_sv]

    def to_lower(self):
        print('Converting to Lowercase')
        self.tokens_en = [word.lower() for word in self.tokens_en]
        self.tokens_sv = [word.lower() for word in self.tokens_sv]

    def remove_punctuation(self):
        print('Removing Punctuation')
        self.tokens_en = [word for word in self.tokens_en if word.isalpha()]
        self.tokens_sv = [word for word in self.tokens_sv if word.isalpha()]

    def build_vocabulary(self):
        print('Building Vocabulary')
        self.vocabulary_en = Counter(self.tokens_en)
        self.vocabulary_sv = Counter(self.tokens_sv)
        self.vocabulary = {'en': self.vocabulary_en, 'sv': self.vocabulary_sv}

if __name__ == '__main__':
    cleaner = TextCleaner(file_en='sv-en/europarl-v7.sv-en.en',file_sv='sv-en/europarl-v7.sv-en.sv')
    
    print(cleaner.vocabulary_en.most_common(10))  # Print the 10 most common words in English corpus
    print(cleaner.vocabulary_sv.most_common(10))

   


