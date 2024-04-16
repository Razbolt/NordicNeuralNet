# main.py

import torch
from nltk import ngrams as ngrams
from utils import parse_arguments, read_settings
from dataset_novoc import TranslationDataset
#from dataset import TranslationDataset
from logger import Logger
from torch.utils.data import random_split, DataLoader, Dataset
from models import Encoder, Decoder, Seq2Seq
from utils import calculate_bleu_score

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
#print(f"Using device: {device}")


def evaluate_model(model, val_loader, criterion, device, dataset):
    model.eval()
    total_val_loss = 0
    references = []
    hypotheses = []

    with torch.no_grad():
        print('---Validation---')
        
        for input_tensor, target_tensor in val_loader:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            output = model(input_tensor, target_tensor)
            output = output.view(-1, output.shape[2])
            target_tensor = target_tensor.view(-1)

            val_loss = criterion(output, target_tensor)
            total_val_loss += val_loss.item()

            predicted_indices = output.argmax(1).cpu().tolist()
            predicted_words = [dataset.index2word_sv.get(idx, '<UNK>') for idx in predicted_indices]
            target_words_list = target_tensor.cpu().tolist()
            target_words = [[dataset.index2word_en.get(idx, '<UNK>') for idx in target_words_list]]

            hypotheses.append(predicted_words)
            references.append(target_words)

        # Debugging output
            #if '<UNK>' in predicted_words or '<UNK>' in target_words[0]:
            #    print("Unknown index in predictions or targets:", predicted_indices, target_words_list)


    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss, references, hypotheses     
    


def train_model(model, train_loader, val_loader, model_settings, my_logger, dataset):
    # Ensure model is on the right device
    model = model.to(device)
    #Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_settings['learning_rate'])

    for epoch in range(model_settings['num_epochs']):
        print('---Training loop---')
        model.train()
        total_loss = 0
        for input_tensor, output_tensor in train_loader:
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)

            #Debugging to try and find the error
            #print("Input tensor shape:", input_tensor.shape)
            #print("Output tensor shape:", output_tensor.shape)
            #print("Input device:", input_tensor.device)
            #print("Output device:", output_tensor.device)
            #print("Model device:", next(model.parameters()).device)

            optimizer.zero_grad()
            output = model(input_tensor, output_tensor)
            #print(f"Model output shape: {output.shape}")
            output_reshaped = output.view(-1, output.shape[-1])
            target_reshaped = output_tensor.view(-1)
            #print(f"Output reshaped shape: {output_reshaped.shape}, Target reshaped shape: {target_reshaped.shape}")
            loss = criterion(output_reshaped, target_reshaped)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()            
            
        avg_train_loss = total_loss / len(train_loader)
        val_loss, references, hypotheses = evaluate_model(model, val_loader, criterion, device, dataset)
        bleu_score = calculate_bleu_score(references, hypotheses)

        # Logging
        my_logger.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'bleu_score': bleu_score
        })
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss}, Val Loss: {val_loss}, BLEU Score: {bleu_score}")

def main():
    args = parse_arguments()

    # Read the settings from the YAML file
    settings = read_settings(args.config)

    config = settings['model_settings']
   
    # Initialize logger
    logger_settings = settings['logger']
    experiment_name = logger_settings['experiment_name']
    project = logger_settings['project']
    entity = logger_settings['entity']
    

    my_logger = Logger(experiment_name, project, entity)
    #my_logger.login()
    my_logger.start(settings)

    dataset = TranslationDataset(**settings['paths'])

    #print(f'Vocabulary length of English dataset', len(dataset.vocabulary_en))
    print(f'Vocabulary length for English', len(dataset.word2idx_en))
    print(f'Vocabulary length of Swedish',len(dataset.word2idx_sv))
    #print(len(dataset.vocabulary_sv))
   
    total_size= len(dataset)
    print(f'Total size of the dataset used is {total_size}')
    train_size = int(0.8 * total_size)
    val_size =  int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train, val, test = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val, config['batch_size'], shuffle=True)
    test_loader = DataLoader(test, config['batch_size'], shuffle=True)

    encoder = Encoder(len(dataset.word2idx_en), embedding_size=300, hidden_size=1024, num_layers=5, dropout=0.5)
    decoder = Decoder(len(dataset.word2idx_sv), embedding_size=300, hidden_size=1024, num_layers=5, dropout=0.5)

    model = Seq2Seq(encoder, decoder)

    #Initialize the logger with the model settings as project of Machine Translation


    train_model(model, train_loader, val_loader, settings['model_settings'], my_logger, dataset)

if __name__ == '__main__':
    main()
