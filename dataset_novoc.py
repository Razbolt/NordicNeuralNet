# dataset_novoc.py

import os
import torch
import pickle
import nltk
import spacy
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from utils import parse_arguments, read_settings


class TranslationDataset(Dataset):
    MAX_LENGTH = 10  # Maximum length of the sentence

    def __init__(self, file_en, file_sv, num_lines=30000):
        self.nlp_en = spacy.load('en_core_web_sm')
        self.nlp_sv = spacy.load('sv_core_news_sm')

        self.file_en = file_en
        self.file_sv = file_sv
        self.num_lines = num_lines

        # Load vocabularies
        self.load_vocabularies()

        # Load data
        sv, en = self.open_data()
        self.data = [(self.tokenize(eng, 'en'), self.tokenize(swe, 'sv')) for eng, swe in zip(en, sv)]

        '''
                print("First 10 sentence pairs:")
        for i, (eng, swe) in enumerate(self.data[:10]):
            print(f"{i+1}: English: {eng} | Swedish: {swe}")
        '''


    def load_vocabularies(self):
        self.word2idx_en = self.load_vocabulary('en')
        self.word2idx_sv = self.load_vocabulary('sv')
        self.index2word_en = {idx: word for word, idx in self.word2idx_en.items()}
        self.index2word_sv = {idx: word for word, idx in self.word2idx_sv.items()}

        # Log special tokens for debugging
        #print("Special tokens in English vocabulary:",  self.word2idx_en['what'] )
        #print("Special tokens in Swedish vocabulary:",  self.word2idx_sv['<SOS>'])

    def load_vocabulary(self, lang):
        vocab_path = f'vocabulary_{lang}.pkl'
        if os.path.exists(vocab_path):
            print(f"Loading {lang} vocabulary from {vocab_path}")
            with open(vocab_path, 'rb') as file:
                vocab = pickle.load(file)
                #return pickle.load(file)

                #Add special characters to 
                special_words = ['PAD', 'SOS', 'EOS', 'UNK']
                for word in special_words:
                    if word in vocab:
                        index = vocab[word]
                        del vocab[word]
                        vocab['<' + word + '>'] = index

                return vocab

        else:
            raise FileNotFoundError(f"No vocabulary file found for {lang}.")
        
        ''''
                                # Define special tokens with fixed indices      
        special_tokens = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }

        # Start adding vocabulary words after the special tokens
        word2idx = {**special_tokens}
        start_index = len(special_tokens)

        # Add words to the dictionary with their corresponding index
        for index, word in enumerate(vocab, start=start_index):
            if word not in special_tokens:
                word2idx[word] = index

        return word2idx
        
        
        '''


        
        
        
        

        
    def open_data(self):
        # Reading the first n lines of the data
        with open(self.file_sv, 'r', encoding='utf-8') as f:
            sv = [next(f).strip() for _ in range(self.num_lines)]
        with open(self.file_en, 'r', encoding='utf-8') as f:
            en = [next(f).strip() for _ in range(self.num_lines)]
        return sv, en
    
    def tokenize(self, sentence, language):
        nlp = self.nlp_en if language == 'en' else self.nlp_sv
        doc = nlp(sentence.lower())
        words = [token.text for token in doc if token.is_alpha]
        return words

    def sentences_to_sequences(self, input_sentence, output_sentence):
        # Convert sentences to sequences of indices
        input_words = [self.word2idx_en.get(word, self.word2idx_en['<UNK>']) for word in input_sentence]
        output_words = [self.word2idx_sv.get(word, self.word2idx_sv['<UNK>']) for word in output_sentence]
        
        # Truncate and add special tokens
        input_words = [self.word2idx_en['<SOS>']] + input_words[:self.MAX_LENGTH-2] + [self.word2idx_en['<EOS>']]
        output_words = [self.word2idx_sv['<SOS>']] + output_words[:self.MAX_LENGTH-2] + [self.word2idx_sv['<EOS>']]
        
        # Pad sequences to fixed length
        input_tensor = input_words + [self.word2idx_en['<PAD>']] * (self.MAX_LENGTH - len(input_words))
        output_tensor = output_words + [self.word2idx_sv['<PAD>']] * (self.MAX_LENGTH - len(output_words))

        return input_tensor, output_tensor
    
    def __getitem__(self, idx):
        input_sentence, output_sentence = self.data[idx]
        input_tensor, output_tensor = self.sentences_to_sequences(input_sentence, output_sentence)
        return torch.tensor(input_tensor), torch.tensor(output_tensor)
    
    def __len__(self):
        return len(self.data)
    
def main():
    args = parse_arguments()
    # Read the settings from the YAML file
    settings = read_settings(args.config)
    # Create an instance of TextCleaner
    dataset = TranslationDataset(**settings['paths'])
    
    #Print the 10th sentence pair
    print(f"10th sentence pair: {dataset[14]}")
    # Sort the dictionaries by values (indices)
    sorted_word2idx_sv = dict(sorted(dataset.word2idx_sv.items(), key=lambda item: item[1]))
    sorted_word2idx_en = dict(sorted(dataset.word2idx_en.items(), key=lambda item: item[1]))

    # Print the first 10 words of sorted_word2idx_sv and sorted_word2idx_en
    print(f"First 10 words of sorted_word2idx_sv: {list(sorted_word2idx_sv.items())[-10:]}")
    print(f"First 10 words of sorted_word2idx_en: {list(sorted_word2idx_en.items())[-10:]}")

if __name__ == '__main__':

    main()