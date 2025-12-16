import json
import pickle
import argparse

def main(tokenizer_file, output_file):
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
    
    vocab = tokenizer.token_to_id
    special_tokens = {
        "<PAD>": len(vocab),
        "<EOS>": len(vocab) + 1,
        "<BOS>": len(vocab) + 2,
        "<UNK>": len(vocab) + 3
    }
    
    full_vocab = {**{token: id for token, id in enumerate(tokenizer.token_to_id.values())}, **special_tokens}
    id_to_token = {id: token for token, id in full_vocab.items()}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"vocab": full_vocab, "id_to_token": id_to_token}, f, ensure_ascii=False, indent=4)
    
    print(f"vocab размером массивчика {len(full_vocab)} сохранён в {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение vocab")
    parser.add_argument("--tokenizer", default="tokenizer.pkl", help="Файл токенизатора")
    parser.add_argument("--output", default="vocab.json", help="js в нём vocab")
    args = parser.parse_args()
    main(args.tokenizer, args.output)