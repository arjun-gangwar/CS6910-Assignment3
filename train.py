import os
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 

from model import EncoderRNN, DecoderRNN

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

# constants
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
    
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data[0].to(device), data[1].to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def valid_epoch(dataloader, encoder, decoder, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data[0].to(device), data[1].to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        # add teacher forcing using if statement
        decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001):
    train_loss_hist = []
    valid_loss_hist = []
    train_acc_hist = []
    valid_acc_hist = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        valid_loss = valid_epoch(train_dataloader, encoder, decoder, criterion)
        
        # print losses and main histories
        print(f"epoch: {epoch}, avg_train_loss: {train_loss}, avg_valid_loss: {valid_loss}")
        train_loss_hist.append(train_loss)
        valid_loss_hist.append(valid_loss)


def main(args: argparse.Namespace):
    if args.use_wandb:
        pass
    else:
        encoder = EncoderRNN(SOURCE_VOCAB_SIZE, 
                             HIDDEN_SIZE).to(device)
        decoder = DecoderRNN(HIDDEN_SIZE, 
                             TARGET_VOCAB_SIZE, 
                             MAX_LENGTH).to(device)
        train(trainDataLoader, encoder, decoder, 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training file for assignment 3")
    parser.add_argument("-uw",
                        "--use_wandb",
                        default=False,
                        action="store_true",
                        help="Use Weights and Biases or not")
    parser.add_argument("-wp", 
                        "--wandb_project",
                        type=str, 
                        default="CS6910-Assignment2",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we",
                        "--wandb_entity", 
                        type=str,
                        default="arjungangwar",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-ie", 
                        "--in_embed_dims", 
                        type=int, 
                        default=256,
                        help="Input Embedding Dimension")
    parser.add_argument("-el", 
                        "--n_encoder_layers", 
                        type=int, 
                        default=1,
                        help="Number of Encoder Layers")
    parser.add_argument("-dl", 
                        "--n_decoder_layers", 
                        type=int, 
                        default=1,
                        help="Number of Decoder Layers")
    parser.add_argument("-hs", 
                        "--hidden_layer_size", 
                        type=int, 
                        default=128,
                        help="Hidden Layer Size")
    parser.add_argument("-ct", 
                        "--cell_type", 
                        type=str, 
                        default="gru",
                        help="Cell Type: rnn, lstm, gru")
    parser.add_argument("-bi", 
                        "--bidirectional", 
                        type=int, 
                        default=0,
                        help="Bidirectional (0: False, 1: True)")
    parser.add_argument("-do", 
                        "--dropout", 
                        type=float, 
                        default=0.2,
                        help="Dropout Percentage")
    parser.add_argument("-ne", 
                        "--n_epochs", 
                        type=int, 
                        default=30,
                        help="Number of Epochs")
    parser.add_argument("-lr", 
                        "--learning_rate", 
                        type=float, 
                        default=1e-4,
                        help="Learning Rate")
    parser.add_argument("-ml", 
                        "--max_length", 
                        type=int, 
                        default=25,
                        help="Max Sequence Length")
    args = parser.parse_args()
    logging.info(args)

    # constants
    HIDDEN_SIZE = args.hidden_layer_size
    MAX_LENGTH = args.max_length

    train_df = pd.read_csv("/speech/arjun/1study/CS6910-Assignment3/dataset/aksharantar_sampled/hin/hin_train.csv", header=None)
    valid_df = pd.read_csv("/speech/arjun/1study/CS6910-Assignment3/dataset/aksharantar_sampled/hin/hin_valid.csv", header=None)
    test_df = pd.read_csv("/speech/arjun/1study/CS6910-Assignment3/dataset/aksharantar_sampled/hin/hin_test.csv", header=None)
    
    train_df.dropna(inplace=True)

    trainDataset = TextDataset(train_df)
    validDataset = TextDataset(valid_df)
    testDataset = TextDataset(test_df)

    trainDataLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=CollateFunc(PAD_TOKEN))
    validDataLoader = DataLoader(validDataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=CollateFunc(PAD_TOKEN))
    testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=CollateFunc(PAD_TOKEN))

    SOURCE_VOCAB_SIZE = len(trainDataset.source_vocab.char2int)
    TARGET_VOCAB_SIZE = len(trainDataset.target_vocab.char2int)

    main(args)


