# preprocess.py
import json
import spacy
import tqdm
import sys
import subprocess
import re
from pathlib import Path

# MODEL constant and automatic download of spaCy model
MODEL = "de_core_news_lg"

def install_model(name: str):
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", name],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

# Проверяем, доступна ли модель
try:
    NLP = spacy.load(MODEL)
except OSError:
    print(f"Модель {MODEL} не найдена. Устанавливаем…")
    install_model(MODEL)
    NLP = spacy.load(MODEL)

def _iter_records(f):
    """Возвращает генератор словарей из файла в формате JSONL или JSON-массив."""
    first_chars = ''
    # Буфер первых 2048 байт чтобы распознать формат
    pos = f.tell()
    first_chars = f.read(2048)
    f.seek(pos)
    first_non_ws = next((c for c in first_chars if not c.isspace()), '')

    if first_non_ws == '[':
        # обычный JSON массив
        data = json.load(f)
        for obj in data:
            yield obj
    else:
        # JSON Lines
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def preprocess(path_in: str | Path, path_out: str | Path):
    with open(path_in, encoding="utf-8") as f_in, \
         open(path_out, "w", encoding="utf-8") as f_out:

        for doc_raw in tqdm.tqdm(_iter_records(f_in)):
            # Разбиваем исходный текст по одному или более переводам строки
            for segment in re.split(r"\n+", doc_raw.get("text", "").strip()):
                segment = segment.strip()
                if not segment:
                    continue  # пропускаем пустые фрагменты

                doc = NLP(segment.lower())
                tokens = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]

                out_item = {
                    "id": doc_raw.get("id"),
                    "name": f"{doc_raw.get('name', '').strip()} {doc_raw.get('surname', '').strip()}".strip(),
                    "text": segment,
                    "clean": " ".join(tokens),
                }
                f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2])
