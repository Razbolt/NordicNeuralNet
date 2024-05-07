from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import pickle
import torch 
from  models import Encoder, Decoder, Seq2Seq
from utils import parse_arguments, read_settings

from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device set to {0}'.format(device))


#Load the vocabularies for each language
with open('vocabulary_en.pkl', 'rb') as file:
    vocab_en = pickle.load(file)
with open('vocabulary_sv.pkl', 'rb') as file:
    vocab_sv = pickle.load(file)

'''
special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
updated_vocab_en = {}
updated_vocab_sv = {}

for i, token in enumerate(special_tokens): # Add special tokens to the updated vocabularies
    updated_vocab_en[token] = i
    updated_vocab_sv[token] = i

for word, index in vocab_en.items(): # Add the rest of the words to the updated vocabularies
    updated_vocab_en[word] = index + len(special_tokens)

for word, index in vocab_sv.items():
    updated_vocab_sv[word] = index + len(special_tokens)

vocab_en = updated_vocab_en # Updating the vocabulary to have same number of tokens as the original
vocab_sv = updated_vocab_sv 

'''



'''
# Print the indices of the special tokens
for vocab in [vocab_en, vocab_sv]:
    for token in special_tokens:
        print(f"{token}: {vocab[token]}")

'''

#Double check the lengths of each vocabulary 
print(f"length of vocab_en: {len(vocab_en)}")
print(f"length of vocab_sv: {len(vocab_sv)}")





#Build the inverse of the vocabularies
def build_inverse_vocab(vocab):
    return {idx: word for word, idx in vocab.items()}

#Build the mapping function between vocabulary and indices
def indices_to_word(indices, vocab):
    return [[vocab[i.item()] for i in sequence] for sequence in indices]

def remove_special_tokens(sentence, special_tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']):
    return [word for word in sentence if word not in special_tokens]


#Function to calculate the BLEU score
def calculate_bleu_score(predictions, targets):
    # Calculate BLEU score
    bleu_scores = []
    for prediction, target in zip(predictions, targets):
        prediction = remove_special_tokens(prediction)
        target = remove_special_tokens(target)
        bleu_score = sentence_bleu([target], prediction)
        bleu_scores.append(bleu_score)

        print(f'Prediction: {prediction}')
        print(f'Target: {target}')

    return bleu_scores


if __name__ == '__main__':
    args = parse_arguments()
    settings = read_settings(args.config)

    encoder = Encoder(len(vocab_en), embedding_size=256, hidden_size=512, num_layers=4, dropout=0.4)
    decoder = Decoder(len(vocab_sv), embedding_size=256, hidden_size=512, num_layers=4, dropout=0.4)
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(torch.load('models/base-hyperion-1.22.pth',  map_location=torch.device('cpu'))['model_state_dict'])
    model.to(device)
    model.eval()



    test_data = torch.load('test_data/test_2.pt')
    test_loader = DataLoader(test_data,settings['model_settings']['batch_size'], shuffle=False)

    english_inverse_volacb = build_inverse_vocab(vocab_en)
    swedish_inverse_vocab = build_inverse_vocab(vocab_sv)

    predictions = []
    targets = []
    for i, (source, target) in enumerate(test_loader):
        
        print(source.max()) 
        #trg = target.unsqueeze(0)
        src = source.to(device)
        trg = target.to(device)
        #print(f'Target size',trg.shape)

        #Create a hypothesis from the source sentence
        hypothesis = model(src,trg,0) # Turn off the teacher
        #print(f'Hypothesis size',hypothesis.shape)


        predicted_ids = torch.argmax(hypothesis,dim =-1)
        #print(f'Predicted IDs shape',predicted_ids.shape)

        target_sentences = indices_to_word(trg, swedish_inverse_vocab)
        predicted_sentences = indices_to_word(predicted_ids, swedish_inverse_vocab)

        predictions.extend(predicted_sentences)
        targets.extend(target_sentences)

    bleu_scores = calculate_bleu_score(predictions, targets)
    print(f'Average BLEU Score: {sum(bleu_scores)/len(bleu_scores)}')
  




