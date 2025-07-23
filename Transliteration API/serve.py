from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def build_vocab(tokens):
    vocab = {tok: i+4 for i, tok in enumerate(tokens)}
    vocab["<pad>"] = 0
    vocab["<sos>"] = 1
    vocab["<eos>"] = 2
    vocab["<unk>"] = 3
    return vocab

import pandas as pd
df = pd.read_csv("dataset.csv")
source_tokens = [t.lower() for t in df['source'].str.split().explode().unique()]
target_tokens = [t.lower() for t in df['target'].str.split().explode().unique()]
SRC_VOCAB = build_vocab(source_tokens)
TGT_VOCAB = build_vocab(target_tokens)
SRC_IVOCAB = {i: t for t, i in SRC_VOCAB.items()}
TGT_IVOCAB = {i: t for t, i in TGT_VOCAB.items()}

def encode(tokens, vocab):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=SRC_VOCAB["<pad>"])
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, batch_first=True)
    def forward(self, src, src_lens):
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens, batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, (h, c)

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
    def forward(self, hidden, encoder_outputs, mask):
        src_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy @ self.v
        energy = energy.masked_fill(mask == 0, -1e10)
        return F.softmax(energy, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, attention, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=TGT_VOCAB["<pad>"])
        self.lstm = nn.LSTM(emb_dim + enc_hid_dim*2, dec_hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim*2 + emb_dim, vocab_size)
        self.attention = attention
    def forward(self, input, hidden, cell, encoder_outputs, mask):
        input = input.unsqueeze(1)
        emb = self.embedding(input)
        attn_weights = self.attention(hidden[-1], encoder_outputs, mask)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((emb, attn_applied), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)
        attn_applied = attn_applied.squeeze(1)
        emb = emb.squeeze(1)
        pred = self.fc_out(torch.cat((output, attn_applied, emb), dim=1))
        return pred, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.fc_hidden = nn.Linear(encoder.lstm.hidden_size * 2, decoder.lstm.hidden_size)
        self.fc_cell = nn.Linear(encoder.lstm.hidden_size * 2, decoder.lstm.hidden_size)
    def create_mask(self, src):
        return (src != self.src_pad_idx)

def translate(model, src_sentence, max_len=30):
    model.eval()
    src_tokens = encode(src_sentence.lower().split(), SRC_VOCAB)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
    src_lens = [len(src_tokens)]
    with torch.no_grad():
        encoder_outputs, (h, c) = model.encoder(src_tensor, src_lens)
        h_cat = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        c_cat = torch.cat((c[-2,:,:], c[-1,:,:]), dim=1)
        h_dec = torch.tanh(model.fc_hidden(h_cat)).unsqueeze(0)
        c_dec = torch.tanh(model.fc_cell(c_cat)).unsqueeze(0)
        input = torch.tensor([TGT_VOCAB["<sos>"]]).to(device)
        mask = model.create_mask(src_tensor)
        result = []
        for _ in range(max_len):
            output, h_dec, c_dec, _ = model.decoder(input, h_dec, c_dec, encoder_outputs, mask)
            top1 = output.argmax(1)
            if top1.item() == TGT_VOCAB["<eos>"]:
                break
            result.append(TGT_IVOCAB[top1.item()])
            input = top1
    return " ".join(result)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = len(SRC_VOCAB)
OUTPUT_DIM = len(TGT_VOCAB)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
attn = BahdanauAttention(HID_DIM, HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, attn)
model = Seq2Seq(enc, dec, SRC_VOCAB["<pad>"], device).to(device)

if os.path.exists("Model/seq2seq_model_bi_attention_10k.pt"):
    model.load_state_dict(torch.load("Model/seq2seq_model_bi_attention_10k.pt", map_location=device))
    model.eval()
else:
    raise FileNotFoundError("Model weights not found.")

app = Flask(__name__)

@app.route("/transliterate", methods=["POST"])
def transliterate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request."}), 400
    input_text = data["text"]
    output = translate(model, input_text)
    return jsonify({"input": input_text, "output": output})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000) 
