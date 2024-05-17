# models.py

import torch
import torch.nn as nn

#Setting the device 
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print('Device in models.py set to {0}'.format(device))

'''
This architecture influenced by the following tutorial: https://www.youtube.com/watch?v=EoGUlvhRYpk&t=663s
The changes applied in seq2seq model others encoder and decoder are straightforward as in all examples
Minor changes are applied to the model to not include <bos> token in the decoder input.
'''

SRC_VOCAB_SIZE = 1000  # source vocabulary size
TRG_VOCAB_SIZE = 1000  # target vocabulary size
EMB_DIM = 512          # embedding dimension
ENC_HID_DIM = 512      # hidden dimension of LSTM for encoder
DEC_HID_DIM = 512      # hidden dimension of LSTM for decoder
ATTN_DIM = 256         # attention dimension
N_LAYERS = 4           # number of LSTM layers
DROPOUT = 0.5          # dropout rate
MAX_LENGTH = 25

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True,bidirectional=True)
        
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden,cell) = self.rnn(embedded)

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)

        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, enc_hidden_size, dec_hidden_size, attn_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.dec_hidden_size = dec_hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size + enc_hidden_size, dec_hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attention = BahdanauAttention(enc_hidden_size, dec_hidden_size, attn_dim)
        self.fc = nn.Linear(enc_hidden_size + dec_hidden_size + embedding_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.hid_transform = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        
    def forward(self, x, hidden, cell, encoder_outputs):
        hidden = self.hid_transform(hidden)
        cell = self.hid_transform(cell)
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        
        attention_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        output = output.squeeze(1)
        
        prediction = self.fc(torch.cat((output, context, embedded.squeeze(1)), dim=1))
        
        return prediction, hidden.squeeze(0), cell.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target=None, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1] if target is not None else MAX_LENGTH
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(source)
        
        decoder_input = target[:, 0]
        
        for t in range(1, target_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = decoder_output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = target[:, t] if teacher_force and target is not None else top1
        
        return outputs


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attn_dim = attn_dim

        self.attn = nn.Linear(2048, attn_dim)
        self.v = nn.Parameter(torch.rand(attn_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        enc_seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, enc_seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        energy = torch.bmm(v, energy.permute(0, 2, 1)).squeeze(1)
        attention = torch.softmax(energy, dim=1)

        return attention


def main():


    encoder = Encoder(SRC_VOCAB_SIZE, EMB_DIM, ENC_HID_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(TRG_VOCAB_SIZE, EMB_DIM, ENC_HID_DIM * 2, DEC_HID_DIM, ATTN_DIM, N_LAYERS, DROPOUT)


    # Create a batch of source and target sentences
    batch_size = 16
    src_len = 10  # length of source sentences
    trg_len = 12  # length of target sentences

    src = torch.randint(0, SRC_VOCAB_SIZE, (batch_size, src_len)).to(device)  # source sequences
    trg = torch.randint(0, TRG_VOCAB_SIZE, (batch_size, trg_len)).to(device)  # target sequences

    model = Seq2Seq(encoder, decoder, device).to(device)

    output = model(src, trg, teacher_forcing_ratio=0.5)
    print("Output shape:", output.shape)  # Expected shape: [batch_size, trg_len, TRG_VOCAB_SIZE]

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output.view(-1, TRG_VOCAB_SIZE), trg.view(-1))
    print("Output shape:", output.shape)
    print("Mock loss:", loss.item())
    


if __name__ == '__main__':
    main()