import sentencepiece as spm
import re
import torch 
from torch.utils.data import Dataset

sp = spm.SentencePieceProcessor()
sp.load('m_translation.model')


class SentencePairDataset(Dataset):
    def __init__(self, data_path, sp, max_length, num_lines= 600000):
        self.sp = sp
        self.max_length = max_length
        self.num_lines = num_lines
        self.pairs = self.create_sentence_pairs(data_path,self.num_lines)
        

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        english, swedish = self.pairs[idx]
        
        # Encode sentences
        encoded_input = self.encode_sentence(english)
        encoded_output = self.encode_sentence(swedish)
        
        # Pad sequences
        encoded_input = self.pad_sequence(encoded_input)
        encoded_output = self.pad_sequence(encoded_output)
        
        return torch.tensor(encoded_input), torch.tensor(encoded_output)

    def create_sentence_pairs(self, data_path, num_lines):
        with open(data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()[:num_lines]  # Only read the first num_lines lines

        # Find the midpoint of the list of lines
        midpoint = len(lines) // 2

        # The first half of the lines are English, the second half are Swedish
        # Use str.strip() to remove trailing newline characters
        english_sentences = [line.strip() for line in lines[:midpoint]]
        swedish_sentences = [line.strip() for line in lines[midpoint:]]

        # Create pairs
        pairs = list(zip(english_sentences, swedish_sentences))

        return pairs

    def encode_sentence(self, sentence):
        # Encode the sentence into IDs
        ids = self.sp.encode_as_ids(sentence)
    
        # Add <bos> and <eos> tokens
        ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
    
        return ids

    def pad_sequence(self, sequence):
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        else:
            return sequence + [self.sp.pad_id()] * (self.max_length - len(sequence))



def main():


  data_path = 'cleaned_combined_data.txt'  
  max_length = 25
  dataset = SentencePairDataset(data_path, sp, max_length)
  print("Number of sentence pairs:", len(dataset))
  print("First pair of the dataset:", dataset[105345])

if __name__ == '__main__':

    main()

  










































