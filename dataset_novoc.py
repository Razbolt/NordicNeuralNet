# dataset_novoc.py

import os
import torch
import pickle
from torch.utils.data import Dataset
from utils import parse_arguments, read_settings

class TranslationDataset(Dataset):
    MAX_LENGTH = 50  # Maximum length of the sentence

    def __init__(self, file_en, file_sv, num_lines=1000):
        self.file_en = file_en
        self.file_sv = file_sv
        self.num_lines = num_lines

        # Load vocabularies
        self.load_vocabularies()

        # Load data
        sv, en = self.open_data()
        self.data = list(zip(en, sv))

        # Print the first 10 pairs of the dataset
        print("First 10 sentence pairs:")
        for i, (eng, swe) in enumerate(self.data[:10]):
            print(f"{i+1}: English: {eng} | Swedish: {swe}")

    def load_vocabularies(self):
        self.word2idx_en = self.load_vocabulary('en')
        self.word2idx_sv = self.load_vocabulary('sv')
        self.index2word_en = {idx: word for word, idx in self.word2idx_en.items()}
        self.index2word_sv = {idx: word for word, idx in self.word2idx_sv.items()}

    def load_vocabulary(self, lang):
        vocab_path = f'vocabulary_{lang}.pkl'
        if os.path.exists(vocab_path):
            print(f"Loading {lang} vocabulary from {vocab_path}")
            with open(vocab_path, 'rb') as file:
                return pickle.load(file)
        else:
            raise FileNotFoundError(f"No vocabulary file found for {lang}.")
        
    def open_data(self):
        # Reading the first n lines of the data
        with open(self.file_sv, 'r', encoding='utf-8') as f:
            sv = [next(f) for _ in range(self.num_lines)]
        with open(self.file_en, 'r', encoding='utf-8') as f:
            en = [next(f) for _ in range(self.num_lines)]
        return sv, en

    def sentences_to_sequences(self, input_sentence, output_sentence):
        # Convert sentences to sequences of indices
        input_words = [self.word2idx_en.get(word, self.word2idx_en['UNK']) for word in input_sentence.split()]
        output_words = [self.word2idx_sv.get(word, self.word2idx_sv['UNK']) for word in output_sentence.split()]
        
        # Truncate and add special tokens
        input_words = [self.word2idx_en['SOS']] + input_words[:self.MAX_LENGTH-2] + [self.word2idx_en['EOS']]
        output_words = [self.word2idx_sv['SOS']] + output_words[:self.MAX_LENGTH-2] + [self.word2idx_sv['EOS']]
        
        # Pad sequences to fixed length
        input_tensor = input_words + [self.word2idx_en['PAD']] * (self.MAX_LENGTH - len(input_words))
        output_tensor = output_words + [self.word2idx_sv['PAD']] * (self.MAX_LENGTH - len(output_words))

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

if __name__ == '__main__':

    main()