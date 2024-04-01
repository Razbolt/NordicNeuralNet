import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        print('----Encoder----')
        #Shape of x: (batch_size, seq_length)
        print(f' Before starting shape:',x.shape)
        embedded = self.dropout(self.embedding(x))
        #Shape of embedded: (batch_size, seq_length, embedding_size)
        print(f'Embedding shape:',embedded.shape)

        output, hidden = self.rnn(embedded)
        #Shape of output: (batch_size, seq_length, hidden_size)
        print(f'Output layer shape:',output.shape)
        #Shape of hidden: (num_layers, batch_size, hidden_size)
        print(f'Hidden layer shape:',hidden.shape)
        return  output,hidden


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        print('----Decoder----')
        print(f' Before starting shape:',x.shape)
        #Shape of x: (batch_size)

        x = x.unsqueeze(1)
        #Shape of x: (batch_size, 1)
        print(f' After unsqueeze shape:',x.shape)

        embedded = self.dropout(self.embedding(x))

        output, hidden = self.rnn(embedded, hidden)
        output = output.squeeze(1)
        output = self.fc(output)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        
        encoder_output, hidden = self.encoder(source)
        
        decoder_input = target[:, 0]
        
        for t in range(1, target_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t] = decoder_output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1
        
        return outputs




def main():
    # Generate a random input tensor
    input_tensor = torch.randint(0, 100, (2, 50))  # 2 is batch size, 50 is sequence length, 100 is vocab size
    # Generate a random output tensor
    output_tensor = torch.randint(0, 100, (2, 50))

    # Instantiate the encoder
    encoder = Encoder(input_size=100, embedding_size=300, hidden_size=512, num_layers=10, dropout=0.5)

    # Instantiate the decoder
    decoder = Decoder(output_size=100, embedding_size=300, hidden_size=512, num_layers=10, dropout=0.5)

    # Instantiate the seq2seq model
    model = Seq2Seq(encoder, decoder)

    # Pass the input tensor through the model
    output = model(input_tensor, output_tensor)

    #print(output)

if __name__ == '__main__':
    main()