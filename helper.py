

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