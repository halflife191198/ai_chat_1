import torch
import torch.nn as nn
import json
import pickle
import argparse
import math
from torch.utils.data import DataLoader, TensorDataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerRetriever(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.encoder(src.transpose(0, 1))
        return self.pool(output.transpose(1, 2)).squeeze(-1)

def main(chunks_file, tokenizer_file, vocab_file, output_embeds, d_model, nhead, num_layers, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)['chunks']
    
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)['vocab']
    vocab_size = len(vocab)

    tokenized_chunks = []
    max_len = 512
    for chunk in chunks:
        tokens = tokenizer.encode(chunk)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            pad_id = vocab.get('<PAD>', 0)
            tokens += [pad_id] * (max_len - len(tokens))
        tokenized_chunks.append(tokens)
    
    tensor_chunks = torch.tensor(tokenized_chunks, dtype=torch.long).to(device)
    dataset = TensorDataset(tensor_chunks)
    loader = DataLoader(dataset, batch_size=batch_size)

    model = TransformerRetriever(vocab_size, d_model, nhead, num_layers).to(device)
    model.eval()

    embeds = []
    with torch.no_grad():
        for batch in loader:
            src = batch[0]
            embed = model(src)
            embeds.append(embed.cpu())
    
    all_embeds = torch.cat(embeds, dim=0)
    
    with open(output_embeds, 'wb') as f:
        pickle.dump(all_embeds, f)
    
    print(f"Эмбеддинги для {len(chunks)} чанков сохранены в {output_embeds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retriever: вычисление эмбеддингов чанков")
    parser.add_argument("--chunks", default="chunks.json", help="JSON с чанками")
    parser.add_argument("--tokenizer", default="tokenizer.pkl", help="Файл токенизатора")
    parser.add_argument("--vocab", default="vocab.json", help="JSON с vocab")
    parser.add_argument("--output_embeds", default="chunk_embeds.pkl", help="Выходной pickle с эмбеддингами")
    parser.add_argument("--d_model", type=int, default=256, help="Размер эмбеддинга")
    parser.add_argument("--nhead", type=int, default=4, help="Количество attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Количество слоёв")
    args = parser.parse_args()
    main(args.chunks, args.tokenizer, args.vocab, args.output_embeds, args.d_model, args.nhead, args.num_layers)