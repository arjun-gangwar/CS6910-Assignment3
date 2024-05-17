import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

# constants
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
UNK_TOKEN = 3
    
class EncoderRNN(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 in_embed_dims,
                 cell_type,
                 max_length,
                 n_layers=1,
                 bidirectional=False,
                 dropout_p=0.1,):
        super(EncoderRNN, self).__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, in_embed_dims)
        if cell_type=="gru":
            self.cell = nn.GRU(in_embed_dims, self.hidden_size, batch_first=True, num_layers=n_layers, bidirectional=bidirectional)
        elif cell_type=="rnn":
            self.cell = nn.RNN(in_embed_dims, self.hidden_size, batch_first=True, num_layers=n_layers, bidirectional=bidirectional)
        elif cell_type=="lstm":
            self.cell = nn.LSTM(in_embed_dims, self.hidden_size, batch_first=True, num_layers=n_layers, bidirectional=bidirectional)
        else:
            raise Exception("cell type not supported.")
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        if self.cell_type == "gru" or self.cell_type == "rnn":
            output, hidden = self.cell(embedded)
        else:
            output, (hidden, state) = self.cell(embedded)
        if self.bidirectional:
            hidden = hidden.view(self.n_layers, 2, -1, self.hidden_size)
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=-1)
            if self.cell_type == "lstm":
                state = state.view(self.n_layers, 2, -1, self.hidden_size)
                state = torch.cat((state[:, 0, :, :], state[:, 1, :, :]), dim=-1)
        return (output, hidden, state) if self.cell_type == "lstm" else (output, hidden)
    
class DecoderRNN(nn.Module):
    def __init__(self,
                 output_size, 
                 hidden_size, 
                 in_embed_dims,
                 cell_type,
                 max_length,
                 n_layers=1,
                 bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.max_length = max_length
        self.n_layers = n_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.hidden_size = hidden_size * 2
        else:
            self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, in_embed_dims)
        if cell_type=="gru":
            self.cell = nn.GRU(in_embed_dims, self.hidden_size, batch_first=True, num_layers=n_layers)
        elif cell_type=="rnn":
            self.cell = nn.RNN(in_embed_dims, self.hidden_size, batch_first=True, num_layers=n_layers)
        elif cell_type=="lstm":
            self.cell = nn.LSTM(in_embed_dims, self.hidden_size, batch_first=True, num_layers=n_layers)
        else:
            raise Exception("cell type not supported.")
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, encoder_hidden, encoder_outputs, encoder_state=None, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        # initial values of decoder input and decoder hidden state
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        if self.cell_type == "lstm":
            decoder_state = encoder_state
        decoder_outputs = []

        for i in range(self.max_length):
            if self.cell_type == "gru" or self.cell_type == "rnn":
                decoder_output, decoder_hidden = self.forward_step_rnn(decoder_input, decoder_hidden)
            else:
                decoder_output, decoder_hidden, decoder_state = self.forward_step_lstm(decoder_input, decoder_hidden, decoder_state)
            decoder_outputs.append(decoder_output)
            if target_tensor is not None: # Teacher Forcing
                decoder_input = target_tensor[:,i].unsqueeze(1)  # adding batch dimension
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden

    def forward_step_rnn(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.cell(embedded, hidden)
        output = self.out(output)
        return output, hidden
    
    def forward_step_lstm(self, input, hidden, state):
        embedded = self.embedding(input)
        output, (hidden, state) = self.cell(embedded, (hidden, state))
        output = self.out(output)
        return output, hidden, state