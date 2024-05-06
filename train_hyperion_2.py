
import torch
from nltk import ngrams as ngrams
from utils import parse_arguments, read_settings
from dataset_novoc import TranslationDataset
from logger import Logger, Logger2                              #Logger 2 imported from other projects to logg everything into wandb
from torch.utils.data import random_split, DataLoader, Dataset
import sentencepiece as spm

from piece_dataset import SentencePairDataset
from models import Encoder, Decoder, Seq2Seq



device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
print('Device set to {0}'.format(device))


def indices_to_sentence(indices, sp): #Converting indices to sentences 
    return ' '.join([sp.id_to_piece(idx) for idx in indices if idx not in (0, 1, 2)])  # Ignoring pad, sos, eos


def evaluate_model(model, val_loader, criterion, sp):
    model.eval()

    total_val_loss = 0
    num_examples = 10
    example_counter = 0
    with torch.no_grad():
        print('---Evaluation has begun---')
        
        for i, (input_tensor, output_tensor) in enumerate(val_loader):
            input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)

            output = model(input_tensor, output_tensor)
            _, predicted_indices = torch.max(output, dim=2)  # Correctly targeting the last dimension

            for j in range(input_tensor.size(0)):
                if example_counter >= num_examples:
                    break
                print_example(predicted_indices[j], output_tensor[j], sp, example_counter)
                example_counter += 1


            output = output.view(-1, output.shape[2]) # Flatten the output for CrossEntropyLoss
            output_tensor = output_tensor.view(-1)  #Flatten the  ground truth labels 
            val_loss = criterion(output, output_tensor)
            total_val_loss += val_loss.item()


        avg_val_loss = total_val_loss / len(val_loader)
        print(f"---Evaluation completed. Average Validation Loss: {avg_val_loss}---")
        return avg_val_loss
    
def print_example(predicted_indices, actual_indices, sp, example_counter):
    predicted_sentence = indices_to_sentence(predicted_indices.cpu().numpy(), sp)
    actual_sentence = indices_to_sentence(actual_indices.cpu().numpy(), sp)
    print(f"Example {example_counter + 1}")
    print("Predicted:", predicted_sentence)
    print("Actual:", actual_sentence)

def train_model(model, train_loader, val_loader, model_settings):
    

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0 ) #Define the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=model_settings['learning_rate']) #Define the optimizer
    model = model.to(device)

    for epoch in range(model_settings['num_epochs']):
        print('---Training has begun---')
        model.train()
        total_loss = 0
        print(f'---Epoch {epoch+1} Training Start---')
        for i, (input_tensor, output_tensor) in enumerate(train_loader): # i is the batch number

            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)
            optimizer.zero_grad()

            output = model(input_tensor, output_tensor)
            output = output.view(-1, output.shape[2])
            output_tensor = output_tensor.view(-1)

            #Print the output and output tensor
            #print("Actual Output",output)
            #print("Predicted:", output_tensor)

            loss = criterion(output, output_tensor)
            total_loss += loss.item() #Include the loss in total loss before backpropagation
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}, Batch {i}, Loss {loss.item()}') #Print the loss in given batch size

            

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, criterion, sp)

        logger.log({'epoch': epoch + 1,  
                'avg_train_loss': avg_train_loss, 
                'val_loss': val_loss
                })

        print(f"Epoch {epoch + 1}/{model_settings['num_epochs']},  Summary: Avg Training Loss: {avg_train_loss}, Val Loss: {val_loss}")

    #Save  the model at the end
    torch.save({'epochs':epoch+1,'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss.item()}, 'models/local_seq2seq.pth')
        
    print('Finished Training and saved the model')
    

def main():

    return

if __name__ == '__main__':
    args = parse_arguments()
    settings = read_settings(args.config)  # Read the settings from the YAML file

    sp = spm.SentencePieceProcessor()
    sp.load('m_translation.model')

    data_path = 'cleaned_combined_data.txt' 
    max_length = 25
    dataset = SentencePairDataset(data_path=data_path,sp =sp, max_length=max_length,num_lines=2000000)


    total_size= len(dataset)
    print(f'Total size of the dataset is {total_size}')
    train_size = int(0.8 * total_size)
    val_size =  int(0.1 * total_size)
    test_size = total_size - train_size - val_size

  
    gen = torch.Generator() # This is working as random seed
    gen.manual_seed(0)
    train, val, test = random_split(dataset, [train_size, val_size, test_size], generator=gen)
    torch.save(test,'test_data/test_2.pt') #Save test dataset to a file
    print('Test data saved to test_data/test_2.pt')

    train_loader = DataLoader(train, settings['model_settings']['batch_size'], shuffle=True)
    val_loader = DataLoader(val, settings['model_settings']['batch_size'], shuffle=True)

    
    vocab_size = sp.get_piece_size()
    encoder = Encoder(vocab_size, embedding_size=256, hidden_size=512, num_layers=5, dropout=0.5)
    decoder = Decoder(vocab_size, embedding_size=256, hidden_size=512, num_layers=5, dropout=0.5)
    model = Seq2Seq(encoder, decoder)


    #Take learning rate and number of epochs from the model settings
    num_epochs = settings['model_settings']['num_epochs']
    learning_rate = settings['model_settings']['learning_rate']
    batch_size = settings['model_settings']['batch_size']
    #Initialize the logger with the model settings as project of Machine Translation
    wandb_logger = Logger2(f'LocalNMT_BaseModel_Seq2Seq_epochs{num_epochs}_b{batch_size}_lr{learning_rate}',project='Machine Translation') # Logger Changed to Logger2
    logger = wandb_logger.get_logger()

    train_model(model, train_loader, val_loader, settings['model_settings'])
    


