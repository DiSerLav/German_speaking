# predict_emotion.py
import json
import torch
import tqdm
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from colorama import Fore, Style

GERMAN_MODEL = "oliverguhr/german-sentiment-bert"

# Подгружаем токенизатор и модель. Если скачивание не удалось,
# можно fallback на прежний чекпойнт.
try:
    tok = AutoTokenizer.from_pretrained(GERMAN_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(GERMAN_MODEL).eval()
    id2label = mdl.config.id2label
    print(Fore.GREEN + "Немецкий чекпойнт загружен успешно" + Style.RESET_ALL)
except Exception as e:
    print(Fore.RED + "Не удалось загрузить немецкий чекпойнт, fallback на bert-emotion" + Style.RESET_ALL, e)
    fallback = "bert-emotion"
    tok = AutoTokenizer.from_pretrained(fallback)
    mdl = AutoModelForSequenceClassification.from_pretrained(fallback).eval()
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

LABELS = [id2label[i] for i in sorted(id2label)]

def predict(path_in: str, path_out: str):
    with open(path_in, encoding="utf-8") as f_in, \
         open(path_out, "w", encoding="utf-8") as f_out, \
         torch.no_grad():
        for line in tqdm.tqdm(f_in):
            item = json.loads(line)
            enc = tok(item["text"], return_tensors="pt", truncation=True)
            logits = mdl(**enc).logits
            item["pred_sentiment"] = LABELS[logits.argmax().item()]
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])
