
import json
import argparse

BIO_TEXT = "потом добавлю"

def main(input_file, output_file, chunk_size=512):
    with open(input_file, 'r', encoding='utf-8') as f:
        book_text = f.read().strip()
    
    if "Биография Джулии Кэмерон" not in book_text:
        full_text = BIO_TEXT + "\n\n" + book_text
    else:
        full_text = book_text
    
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=4)
    
    print(f"Создано {len(chunks)} чанков. Сохранено в {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разбиение текста на чанки")
    parser.add_argument("--input", default="book_full.txt", help="Входной TXT файл")
    parser.add_argument("--output", default="chunks.json", help="Выходной JSON файл")
    parser.add_argument("--chunk_size", type=int, default=512, help="Размер чанка в символах")
    args = parser.parse_args()
    main(args.input, args.output, args.chunk_size)