import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Data Preparation
df = pd.read_csv("dataset.csv")

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

def build_vocab(tokens):
    vocab = {tok: i+4 for i, tok in enumerate(tokens)}
    vocab["<pad>"] = 0
    vocab["<sos>"] = 1
    vocab["<eos>"] = 2
    vocab["<unk>"] = 3
    return vocab

source_tokens = [t.lower() for t in df['source'].str.split().explode().unique()]
target_tokens = [t.lower() for t in df['target'].str.split().explode().unique()]
SRC_VOCAB = build_vocab(source_tokens)
TGT_VOCAB = build_vocab(target_tokens)
SRC_IVOCAB = {i: t for t, i in SRC_VOCAB.items()}
TGT_IVOCAB = {i: t for t, i in TGT_VOCAB.items()}

def encode(tokens, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

class Seq2SeqDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab):
        self.pairs = []
        for _, row in df.iterrows():
            src = encode(row['source'].split(), src_vocab)
            tgt = [tgt_vocab["<sos>"]] + encode(row['target'].split(), tgt_vocab) + [tgt_vocab["<eos>"]]
            self.pairs.append((src, tgt))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    src_pad = torch.zeros(len(srcs), max(src_lens), dtype=torch.long)
    tgt_pad = torch.zeros(len(tgts), max(tgt_lens), dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :len(s)] = torch.tensor(s)
        tgt_pad[i, :len(t)] = torch.tensor(t)
    return src_pad, tgt_pad, src_lens, tgt_lens

# Use train_df and test_df to create datasets and loaders
train_dataset = Seq2SeqDataset(train_df, SRC_VOCAB, TGT_VOCAB)
test_dataset = Seq2SeqDataset(test_df, SRC_VOCAB, TGT_VOCAB)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 2. Model Definition
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=SRC_VOCAB["<pad>"])
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=False, batch_first=True)
    def forward(self, src, src_lens):
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens, batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=TGT_VOCAB["<pad>"])
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        emb = self.embedding(input)
        output, (hidden, cell) = self.lstm(emb, (hidden, cell))
        output = output.squeeze(1)
        pred = self.fc_out(output)
        return pred, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.decoder.embedding.num_embeddings).to(self.device)
        encoder_outputs, (h, c) = self.encoder(src, src_lens)
        h_dec = h
        c_dec = c
        input = tgt[:,0]
        for t in range(1, tgt_len):
            output, h_dec, c_dec = self.decoder(input, h_dec, c_dec)
            outputs[:,t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:,t] if teacher_force else top1
        return outputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = len(SRC_VOCAB)
OUTPUT_DIM = len(TGT_VOCAB)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM)
model = Seq2Seq(enc, dec, SRC_VOCAB["<pad>"], device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TGT_VOCAB["<pad>"])

# Metric evaluation functions
def levenshtein(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def compute_cer(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(ref, hyp) / len(ref)

def compute_wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)

# Use train_loader for training
for epoch in range(10):
    model.train()
    total_loss = 0
    for src, tgt, src_lens, tgt_lens in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, src_lens, tgt)
        output_dim = output.shape[-1]
        output = output[:,1:].reshape(-1, output_dim)
        tgt = tgt[:,1:].reshape(-1)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Metric evaluation on the test set (unseen data)
    model.eval()
    total_cer = 0
    total_wer = 0
    n_samples = 0
    with torch.no_grad():
        example_printed = 0
        for src, tgt, src_lens, tgt_lens in test_loader:
            src = src.to(device)
            for i in range(src.size(0)):
                src_sentence = ' '.join([SRC_IVOCAB[idx.item()] for idx in src[i] if idx.item() != SRC_VOCAB["<pad>"]])
                tgt_sentence = ' '.join([TGT_IVOCAB[idx.item()] for idx in tgt[i] if idx.item() not in [TGT_VOCAB["<pad>"], TGT_VOCAB["<sos>"], TGT_VOCAB["<eos>"]]])
                pred_sentence = translate(model, src_sentence)
                cer = compute_cer(tgt_sentence, pred_sentence)
                wer = compute_wer(tgt_sentence, pred_sentence)
                total_cer += cer
                total_wer += wer
                n_samples += 1
                # Print a few examples
                if example_printed < 5:
                    print(f"Example {example_printed+1}:")
                    print(f"  Source:    {src_sentence}")
                    print(f"  Target:    {tgt_sentence}")
                    print(f"  Predicted: {pred_sentence}")
                    print(f"  CER: {cer:.2%}, WER: {wer:.2%}\n")
                    example_printed += 1
    print(f"CER: {total_cer/n_samples:.2%}, WER: {total_wer/n_samples:.2%}")

def translate(model, src_sentence, max_len=30):
    model.eval()
    src_tokens = encode(src_sentence.lower().split(), SRC_VOCAB)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
    src_lens = [len(src_tokens)]
    with torch.no_grad():
        encoder_outputs, (h, c) = model.encoder(src_tensor, src_lens)
        h_dec = h
        c_dec = c
        input = torch.tensor([TGT_VOCAB["<sos>"]]).to(device)
        result = []
        for _ in range(max_len):
            output, h_dec, c_dec = model.decoder(input, h_dec, c_dec)
            top1 = output.argmax(1)
            if top1.item() == TGT_VOCAB["<eos>"]:
                break
            result.append(TGT_IVOCAB[top1.item()])
            input = top1
    return " ".join(result)

# Example usage:
print(translate(model, "ka na pa")) 