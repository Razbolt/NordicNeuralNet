# main.py

import torch
from nltk import ngrams as ngrams
from utils import parse_arguments, read_settings
from dataset import TranslationDataset
from logger import Logger
from torch.utils.data import random_split, DataLoader, Dataset

from models import Encoder, Decoder, Seq2Seq

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
#print('Device set to {0}'.format(device))


def evaluate_model(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        print('---Evaluation has begun---')
        total_val_loss = 0
        for i, (input_tensor, output_tensor) in enumerate(val_loader):
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)

            output = model(input_tensor, output_tensor)

            output = output.view(-1, output.shape[2])
            output_tensor = output_tensor.view(-1)

            val_loss = criterion(output, output_tensor)

            total_val_loss += val_loss.item()

        return (total_val_loss / len(val_loader))
    


def train_model(model, train_loader, val_loader, model_settings, my_logger):
    #Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_settings['learning_rate'])

    model = model.to(device)

    for epoch in range(model_settings['num_epochs']):
        print('---Training has begun---')
        model.train()
        total_loss = 0
        for i, (input_tensor, output_tensor) in enumerate(train_loader):
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)

            optimizer.zero_grad()

            output = model(input_tensor, output_tensor)

            output = output.view(-1, output.shape[2])
            output_tensor = output_tensor.view(-1)

            loss = criterion(output, output_tensor)
            total_loss += loss.item() #Include the loss in total loss before backpropagation

            loss.backward()

            optimizer.step()

            
            
            #Print the loss in given batch size
            print(f'Epoch {epoch+1}, Batch {i}, Loss {loss.item()}')

        val_loss = evaluate_model(model, val_loader, criterion)
        my_logger.log({
                'avg_train_loss': total_loss / len(train_loader), 
                'total_train_loss': total_loss, 
                'val_loss': val_loss
                })

        print(f"Epoch {epoch + 1}/{model_settings['num_epochs']}, Avg Training Loss: {total_loss / len(train_loader)}, Total Training Loss: {total_loss}, Val Loss: {val_loss}")
    

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
    my_logger.login()
    my_logger.start(settings)

    dataset = TranslationDataset(**settings['paths'])

    print(f'Vocabulary length of English dataset', len(dataset.vocabulary_en))
    #print(len(dataset.word2idx_en))
    print(f'Vocabulary length of Swedish',len(dataset.word2idx_sv))
    #print( len(dataset.vocabulary_sv))
   
    total_size= len(dataset)
    print(f'Total size of the dataset is {total_size}')
    train_size = int(0.8 * total_size)
    val_size =  int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train, val, test = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val, config['batch_size'], shuffle=True)
    test_loader = DataLoader(test, config['batch_size'], shuffle=True)

    encoder = Encoder(len(dataset.vocabulary_en), embedding_size=256, hidden_size=512, num_layers=5, dropout=0.5)
    decoder = Decoder(len(dataset.vocabulary_sv), embedding_size=256, hidden_size=512, num_layers=5, dropout=0.5)

    model = Seq2Seq(encoder, decoder)

    #Initialize the logger with the model settings as project of Machine Translation


    train_model(model, train_loader, val_loader, settings['model_settings'], my_logger)

if __name__ == '__main__':
    main()
