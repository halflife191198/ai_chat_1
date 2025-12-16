import json
import tiktoken
import argparse
import os

def build_tokenizer(chunks_file, output_file):
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = ' '.join(data['chunks']) 
    
    enc = tiktoken.get_encoding("gpt2")
    enc = enc.train(text) 
    
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(enc, f)
    
    print(f"Токенизатор построен и сохранён в {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение токенизатора")
    parser.add_argument("--chunks", default="chunks.json", help="JSON с чанками")
    parser.add_argument("--output", default="tokenizer.pkl", help="Выходной файл токенизатора")
    args = parser.parse_args()
    build_tokenizer(args.chunks, args.output)