
import torch
from nltk import ngrams as ngrams
from utils import parse_arguments, read_settings
from dataset import TranslationDataset
from logger import Logger
from torch.utils.data import random_split, DataLoader, Dataset

from models import Encoder, Decoder, Seq2Seq




def main():

    args = parse_arguments()

    # Read the settings from the YAML file
    settings = read_settings(args.config)
    # Create an instance of TextCleaner

    dataset = TranslationDataset(**settings['paths'])
    print( len(dataset.vocabulary_en))
    print(len(dataset.word2idx_en))
    print(len(dataset.word2idx_sv))
    print( len(dataset.vocabulary_sv))


    vocabulary_sv = dataset.vocabulary
    total_size= len(dataset)

    #print(f"Vocabulary size of English: {len(vocabulary_en)}")
    #print(f"Vocabulary size of Swedish: {len(vocabulary_sv)}")

    train_size = int(0.8 * total_size)
    val_size =  int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train, val, test = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train, settings['batch_size'], shuffle=True)
    val_loader = DataLoader(val, settings['batch_size'], shuffle=True)
    test_loader = DataLoader(test, settings['batch_size'], shuffle=True)
    

    #Show the first two elements of  train_loader input and output
    for i, (input_tensor, output_tensor) in enumerate(train_loader):
        #print(f"Input tensor: {input_tensor}")
        #print(f"Output tensor: {output_tensor}")

        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Output tensor shape: {output_tensor.shape}")
        break




    


        

    

if __name__ == '__main__':

    main()
