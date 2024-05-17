import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

nirm = './Nirmala.ttf'
hindi_font = FontProperties(fname=nirm)

train_df = pd.read_csv("/speech/arjun/1study/CS6910-Assignment3/dataset/aksharantar_sampled/hin/hin_train.csv", header=None)
valid_df = pd.read_csv("/speech/arjun/1study/CS6910-Assignment3/dataset/aksharantar_sampled/hin/hin_valid.csv", header=None)
test_df = pd.read_csv("/speech/arjun/1study/CS6910-Assignment3/dataset/aksharantar_sampled/hin/hin_test.csv", header=None)

train_df.dropna(inplace=True)

# constants
MAX_LENGTH = 25
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
UNK_TOKEN = 3

class Vocab:
    def __init__(self, data):
        self.char2int = {c:i+4 for i,c in enumerate(sorted(list(set("".join(data)))))}
        self.char2int["<sos>"] = SOS_TOKEN
        self.char2int["<eos>"] = EOS_TOKEN
        self.char2int["<pad>"] = PAD_TOKEN
        self.char2int["<unk>"] = UNK_TOKEN
        self.int2char = {v:k for k, v in self.char2int.items()}

    def tokenize(self, word):
        return [self.char2int[c] if c in self.char2int.keys() else self.char2int['<unk>'] for c in word]
    
    def detokenize(self, idx):
        return [self.int2char[i] for i in idx]

class TextDataset(Dataset):
    def __init__(self, data, transform=None):
        self.source_data = data[0]
        self.target_data = data[1]
        self.transform = transform
        self.source_vocab = Vocab(self.source_data)
        self.target_vocab = Vocab(self.target_data)
    def __len__(self):
        return len(self.source_data)
    def __getitem__(self, index):
        source_text = [c for c in self.source_data[index]]
        target_text = [c for c in self.target_data[index]]
        
        source_text = ['<sos>'] + source_text + ['<eos>']
        target_text = ['<sos>'] + target_text + ['<eos>']

        tokenized_source = self.source_vocab.tokenize(source_text)
        tokenized_target = self.target_vocab.tokenize(target_text)

        return torch.tensor(tokenized_source), torch.tensor(tokenized_target) 

class CollateFunc():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        source = [x[0] for x in batch]
        target = [x[1] for x in batch]
        source[0] = nn.ConstantPad1d((0, MAX_LENGTH - source[0].shape[0]), self.pad_idx)(source[0])
        target[0] = nn.ConstantPad1d((0, MAX_LENGTH - target[0].shape[0]), self.pad_idx)(target[0])
        source = pad_sequence(source, batch_first=True, padding_value=self.pad_idx)
        target = pad_sequence(target, batch_first=True, padding_value=self.pad_idx)
        return source, target

trainDataset = TextDataset(train_df)
validDataset = TextDataset(valid_df)
testDataset = TextDataset(test_df)

trainDataLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=CollateFunc(PAD_TOKEN))
validDataLoader = DataLoader(validDataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=CollateFunc(PAD_TOKEN))
testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=CollateFunc(PAD_TOKEN))

class EncoderRNN(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 dropout_p=0.1,
                 bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.num_layers = 1
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=bidirectional, num_layers=self.num_layers)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.num_layers = 1
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True, num_layers=self.num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_hidden, encoder_outputs, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

def calculate_accuracy(decoder_outputs, target_tensor):
    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()

    comp = ((decoded_ids == target_tensor).to(torch.float).mean(dim=-1)).to(torch.long)

    return sum(comp)/len(comp)

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    total_acc = 0
    for data in dataloader:
        input_tensor, target_tensor = data[0].to(device), data[1].to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, attn = decoder(encoder_hidden, encoder_outputs, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        acc = calculate_accuracy(decoder_outputs, target_tensor)

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)

@torch.no_grad()
def valid_epoch(dataloader, encoder, decoder, criterion):
    total_loss = 0
    total_acc = 0
    for data in dataloader:
        input_tensor, target_tensor = data[0].to(device), data[1].to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        # add teacher forcing using if statement
        decoder_outputs, _, attn = decoder(encoder_hidden, encoder_outputs, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        acc = calculate_accuracy(decoder_outputs, target_tensor)

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001):
    train_loss_hist = []
    valid_loss_hist = []
    train_acc_hist = []
    valid_acc_hist = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        valid_loss, valid_acc = valid_epoch(train_dataloader, encoder, decoder, criterion)
        
        # print losses and main histories
        print(f"epoch: {epoch}, avg_train_loss: {train_loss}, avg_valid_loss: {valid_loss}, avg_train_acc: {train_acc}, avg_valid_acc: {valid_acc}")
        train_loss_hist.append(train_loss)
        valid_loss_hist.append(valid_loss)
        train_acc_hist.append(train_acc)
        valid_acc_hist.append(valid_acc)
        
@torch.no_grad()
def evaluate(dataloader, target_vocab, encoder, decoder):
    all_words = []
    for data in dataloader:
        input_tensor = data[0].to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, attn = decoder(encoder_hidden, encoder_outputs)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for i in range(len(input_tensor)):
            word = []
            for idx in decoded_ids[i, 1:]:
                if idx.item() == EOS_TOKEN:
                    break
                word.append(idx.item())
            decoded_words.append(word)
        
        detokenized_words = []
        for word in decoded_words:
            detokenized_words.append(target_vocab.detokenize(word))
        
        for word in detokenized_words:
            all_words.append("".join(word))
            
    return all_words, attn

# training

HIDDEN_SIZE = 256
SOURCE_VOCAB_SIZE = len(trainDataset.source_vocab.char2int)
TARGET_VOCAB_SIZE = len(trainDataset.target_vocab.char2int)

encoder = EncoderRNN(SOURCE_VOCAB_SIZE, HIDDEN_SIZE).to(device)
decoder = AttnDecoderRNN(HIDDEN_SIZE, TARGET_VOCAB_SIZE).to(device)

train(trainDataLoader, encoder, decoder, 10)

# evaluations
encoder.eval()
decoder.eval()
decoded_words, attn = evaluate(testDataloader, trainDataset.target_vocab, encoder, decoder)

comparisions = [hypo == ref for hypo, ref in zip(decoded_words,test_df[1].tolist())]
print(f"Test Error: {sum(comparisions) / len(comparisions)}")

rows=3
cols=3
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
idx = 0
for i in range(rows):
    for j in range(cols):
        input_sequence = test_df[0].tolist()[-32+idx]
        output_sequence = decoded_words[-32+idx]
        ax[i,j].imshow(attn[idx, :len(input_sequence), :len(output_sequence)].cpu().detach(), cmap='plasma', interpolation='nearest')
        ax[i,j].set_xticks(np.arange(len(output_sequence)), list(output_sequence), fontproperties=hindi_font)
        ax[i,j].set_yticks(np.arange(len(input_sequence)), list(input_sequence), fontproperties=hindi_font)
        idx += 1
plt.savefig("attention_map.png")

out_path = "predictions_attention"
if not os.path.isdir(out_path):
    os.mkdir(out_path)
else:
    os.system(f"rm -r {out_path}/*")
with open(os.path.join(out_path, "predictions.txt"), "w") as f:
    f.writelines([word+"\n" for word in decoded_words])


