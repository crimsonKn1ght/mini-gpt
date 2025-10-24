import torch


with open('dataset/text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chrs = sorted(list(set(text)))
vocab_size = len(chrs)

stoi = {ch:i for i, ch in enumerate(chrs)}
itos = {i:ch for i, ch in enumerate(chrs)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda s: [itos[ch] for ch in s]

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]