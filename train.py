import os
import wandb
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
    
def calculate_accuracy(decoder_outputs, target_tensor):
    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()

    comp = ((decoded_ids == target_tensor).to(torch.float).mean(dim=-1)).to(torch.long)

    return sum(comp)/len(comp)
    
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio):
    total_loss = 0
    total_acc = 0
    for data in dataloader:
        input_tensor, target_tensor = data[0].to(device), data[1].to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        if torch.rand(1).item() > teacher_forcing_ratio:  # teacher forcing
            if encoder.cell_type == "lstm":
                encoder_outputs, encoder_hidden, encoder_state = encoder(input_tensor)
                decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs, encoder_state, target_tensor)
            else:
                encoder_outputs, encoder_hidden = encoder(input_tensor)
                decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs, target_tensor)
        else:
            if encoder.cell_type == "lstm":
                encoder_outputs, encoder_hidden, encoder_state = encoder(input_tensor)
                decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs, encoder_state)
            else:
                encoder_outputs, encoder_hidden = encoder(input_tensor)
                decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs)

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

        if encoder.cell_type == "lstm":
            encoder_outputs, encoder_hidden, encoder_state = encoder(input_tensor)
            decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs, encoder_state, target_tensor)
        else:
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            # add teacher forcing using if statement
            decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        acc = calculate_accuracy(decoder_outputs, target_tensor)

        total_loss += loss.item()
        total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, use_wandb=False):
    train_loss_hist = []
    valid_loss_hist = []
    train_acc_hist = []
    valid_acc_hist = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    initial_teacher_forcing_ratio = 0.9
    final_teacher_forcing_ratio = 0.5


    for epoch in range(1, n_epochs + 1):
        teacher_forcing_ratio = initial_teacher_forcing_ratio - (initial_teacher_forcing_ratio - final_teacher_forcing_ratio) * (epoch / n_epochs)
        train_loss, train_acc = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio)
        valid_loss, valid_acc = valid_epoch(train_dataloader, encoder, decoder, criterion)
        
        # print losses and main histories
        print(f"epoch: {epoch}, avg_train_loss: {train_loss}, avg_valid_loss: {valid_loss}, avg_train_acc: {train_acc}, avg_valid_acc: {valid_acc}")
        print(f"teacher forcing ratio: {teacher_forcing_ratio}")
        train_loss_hist.append(train_loss)
        valid_loss_hist.append(valid_loss)
        train_acc_hist.append(train_acc)
        valid_acc_hist.append(valid_acc)

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc': train_acc*100,
                'valid_acc': valid_acc*100
            })

# evaluate on test set
@torch.no_grad()
def evaluate(dataloader, target_vocab, encoder, decoder):  
    all_words = []
    for data in dataloader:
        input_tensor = data[0].to(device)

        if encoder.cell_type == "lstm":
            encoder_outputs, encoder_hidden, encoder_state = encoder(input_tensor)
            decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs, encoder_state)
        else:
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _ = decoder(encoder_hidden, encoder_outputs)

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
            
    return all_words

def wandb_sweep():
    with wandb.init() as run:
        config = wandb.config
        in_embed_dims = config.in_embed_dims
        learning_rate = config.learning_rate
        batch_size = config.batch_size
        n_layers = config.n_layers
        hidden_layer_size = config.hidden_layer_size
        cell_type = config.cell_type
        dropout = config.dropout
        n_epochs = config.n_epochs
        bidirectional = config.bidirectional
        max_length = config.max_length

        run_name=f"bs_{batch_size}_ie_{in_embed_dims}_lr_{learning_rate}_hl_{hidden_layer_size}_cl_{cell_type}_bi_{bidirectional}"
        wandb.run.name=run_name

        encoder = EncoderRNN(input_size=SOURCE_VOCAB_SIZE, 
                             hidden_size=hidden_layer_size,
                             in_embed_dims=in_embed_dims,
                             cell_type=cell_type,
                             max_length=max_length,
                             n_layers=n_layers,
                             bidirectional=bidirectional,
                             dropout_p=dropout).to(device)
        decoder = DecoderRNN(output_size=TARGET_VOCAB_SIZE,
                             hidden_size=hidden_layer_size,
                             in_embed_dims=in_embed_dims,
                             cell_type=cell_type,
                             max_length=max_length,
                             n_layers=n_layers,
                             bidirectional=bidirectional).to(device)
        train(trainDataLoader, encoder, decoder, n_epochs=n_epochs, learning_rate=learning_rate, use_wandb=True)

def main(args: argparse.Namespace):
    if args.use_wandb:
        wandb.login()
        sweep_config = {
            'method': 'bayes',
            'name' : 'RNN sweeps May 17th',
            'metric': {
                'name': 'valid_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'in_embed_dims': {
                    'values': [64, 128, 256]
                },'learning_rate': {
                    'values': [1e-3, 1e-4]
                },'batch_size':{
                    'values': [32,64,128]
                },'n_layers':{
                    'values': [1,2,3]
                },'hidden_layer_size':{
                    'values': [64, 128, 256]
                },'cell_type':{
                    'values': ["rnn", "gru", "lstm"]
                },'dropout':{
                    'values': [0.2, 0.5]
                },'n_epochs':{
                    'values': [10]
                },'bidirectional':{
                    'values': [True, False]
                },'max_length':{
                    'values': [25]
                }
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=wandb_sweep, count=50)
        wandb.finish()
    else:
        encoder = EncoderRNN(input_size=SOURCE_VOCAB_SIZE, 
                             hidden_size=HIDDEN_SIZE,
                             in_embed_dims=args.in_embed_dims,
                             cell_type=args.cell_type,
                             max_length=MAX_LENGTH,
                             n_layers=args.n_layers,
                             bidirectional=args.bidirectional).to(device)
        decoder = DecoderRNN(output_size=TARGET_VOCAB_SIZE,
                             hidden_size=HIDDEN_SIZE,
                             in_embed_dims=args.in_embed_dims,
                             cell_type=args.cell_type,
                             max_length=MAX_LENGTH,
                             n_layers=args.n_layers,
                             bidirectional=args.bidirectional).to(device)
        train(trainDataLoader, encoder, decoder, n_epochs=args.n_epochs, learning_rate=args.learning_rate)
        encoder.eval()
        decoder.eval()
        decoded_words = evaluate(testDataloader, trainDataset.target_vocab, encoder, decoder)
        comparisions = [hypo == ref for hypo, ref in zip(decoded_words,test_df[1].tolist())]
        print(f"Test Error: {sum(comparisions) / len(comparisions)}")

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
    parser.add_argument("-dl", 
                        "--n_layers", 
                        type=int, 
                        default=1,
                        help="Number of Layers in Encoder and Decoder")
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
                        default=False,
                        action="store_true",
                        help="Bidirectional (False, True)")
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
    parser.add_argument("-bs", 
                        "--batch_size", 
                        type=int, 
                        default=32,
                        help="Batch Size")
    args = parser.parse_args()
    logging.info(args)

    # constants
    HIDDEN_SIZE = args.hidden_layer_size
    MAX_LENGTH = args.max_length
    BATCH_SIZE = args.batch_size

    train_df = pd.read_csv("/speech/arjun/1study/CS6910-Assignment3/dataset/aksharantar_sampled/hin/hin_train.csv", header=None)
    valid_df = pd.read_csv("/speech/arjun/1study/CS6910-Assignment3/dataset/aksharantar_sampled/hin/hin_valid.csv", header=None)
    test_df = pd.read_csv("/speech/arjun/1study/CS6910-Assignment3/dataset/aksharantar_sampled/hin/hin_test.csv", header=None)
    
    train_df.dropna(inplace=True)

    trainDataset = TextDataset(train_df)
    validDataset = TextDataset(valid_df)
    testDataset = TextDataset(test_df)

    trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=CollateFunc(PAD_TOKEN))
    validDataLoader = DataLoader(validDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=CollateFunc(PAD_TOKEN))
    testDataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=CollateFunc(PAD_TOKEN))

    SOURCE_VOCAB_SIZE = len(trainDataset.source_vocab.char2int)
    TARGET_VOCAB_SIZE = len(trainDataset.target_vocab.char2int)

    main(args)


