# models.py

import torch
import torch.nn as nn

'''
This architecture influenced by the following tutorial: https://www.youtube.com/watch?v=EoGUlvhRYpk&t=663s
And modified architecture from LSTM to GRU
The changes applied in seq2seq model others encoder and decoder are straightforward as in all examples
'''

#TODO:
 #1. Apply attention mechanism to the model
    #1.2. Implement BLEU score calculation for the model

#2. Apply transformer model to the model
    #2.1. Implement BLEU score calculation for the model



class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        
    def forward(self, x):
        #print('----Encoder----')
        #Shape of x: (batch_size, seq_length)
        #print(f' Before starting shape:',x.shape)
        embedded = self.dropout(self.embedding(x))
        #Shape of embedded: (batch_size, seq_length, embedding_size)
        #print(f'Embedding shape:',embedded.shape)

        output, hidden = self.rnn(embedded)
        # output, (hidden, cell) = self.rnn(embedded)   ???

        #Shape of output: (batch_size, seq_length, hidden_size)
        #print(f'Output layer shape:',output.shape)
        #Shape of hidden: (num_layers, batch_size, hidden_size)
        #print(f'Hidden layer shape:',hidden.shape)
        return  output,hidden


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        #print('----Decoder----')
        #print(f' Before starting shape:',x.shape)
        #Shape of x: (batch_size)

        x = x.unsqueeze(1) 
        #Shape of x: (batch_size, 1)

        embedded = self.dropout(self.embedding(x))
        #Shape of embedded: (batch_size, 1, embedding_size)

        output, hidden = self.rnn(embedded, hidden)
        #Shape of output: (batch_size, 1, hidden_size)
        #Shape of hidden: (num_layers, batch_size, hidden_size)

        output = output.squeeze(1)
        #Shape of output: (batch_size, hidden_size)
        predictions = self.fc(output)
        #Shape of predictions: (batch_size, output_size)
        return predictions, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        print(f"Source shape: {source.shape}, Target shape: {target.shape}")
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        print(f"Target Vocabulary Size: {target_vocab_size}")
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        encoder_output, hidden = self.encoder(source)
        print(f"Encoder output shape: {encoder_output.shape}, Hidden state shape: {hidden.shape}")
        
        decoder_input = target[:, 0]
        print(f"Initial decoder input shape: {decoder_input.shape}, Device: {decoder_input.device}")

        for t in range(1, target_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            #print(f"Decoder output at time {t} shape: {decoder_output.shape}")
            outputs[:, t] = decoder_output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1
            #print(f"Next decoder input shape: {decoder_input.shape}")
        
        return outputs


def main():
    # Generate a random input tensor
    input_tensor = torch.randint(0, 100, (16, 5))  # 16 is batch size, 50 is sequence length, 100 is vocab size
    # Generate a random output tensor
    output_tensor = torch.randint(0, 100, (16, 5)) # 16 is batch size, 50 is sequence length, 100 is vocab size

    # Instantiate the encoder
    encoder = Encoder(input_size=5000, embedding_size=300, hidden_size=1024, num_layers=5, dropout=0.5)

    # Instantiate the decoder
    decoder = Decoder(output_size=5000, embedding_size=300, hidden_size=1024, num_layers=5, dropout=0.5)

    # Instantiate the seq2seq model
    model = Seq2Seq(encoder, decoder)

    # Pass the input tensor through the model
    output = model(input_tensor, output_tensor)

    #print(output)

if __name__ == '__main__':
    main()