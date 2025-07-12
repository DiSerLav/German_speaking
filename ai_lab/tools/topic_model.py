# topic_model.py
import json
import sys
import os
from typing import Optional
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN


def topic_model(path_in: str, model_out: str, topics_out: Optional[str] = None) -> str:
    """Обучает BERTopic-модель и сохраняет результаты.

    Args:
        path_in: Путь к .jsonl файлу с полем "text" в каждом документе.
        model_out: Директория/файл для сохранения модели BERTopic.
        topics_out: Путь для вывода документов с полями topic и prob.
                    Если None, путь формируется автоматически как
                    f"{model_out}_topics.jsonl".
    Returns:
        Путь к файлу topics_out.
    """
    embed = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    docs, meta = [], []
    with open(path_in, encoding="utf-8") as f:
        for line in tqdm(f, desc="Чтение документов"):
            j = json.loads(line)
            docs.append(j["text"])
            meta.append(j)

    # Настройка гибкого HDBSCAN
    min_cluster_size = max(2, min(50, len(docs) // 2))
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_method="leaf",
        prediction_data=True,
    )

    mdl = BERTopic(
        embedding_model=embed,
        language="german",
        hdbscan_model=hdb,
    ).fit(docs)

    mdl.save(model_out)

    topics, probs = mdl.transform(docs)

    if topics_out is None:
        root, _ = os.path.splitext(model_out)
        topics_out = f"{root}_topics.jsonl"

    with open(topics_out, "w", encoding="utf-8") as f_out:
        for m, t, p in zip(meta, topics, probs):
            m["topic"], m["prob"] = int(t), float(p)
            f_out.write(json.dumps(m, ensure_ascii=False) + "\n")

    return topics_out


if __name__ == "__main__":
    # Пример использования из командной строки:
    # python topic_model.py data/preprocessed.json bertopic_model
    if len(sys.argv) < 3:
        print("Usage: python topic_model.py <input_jsonl> <model_out>")
        sys.exit(1)

    topic_model(sys.argv[1], sys.argv[2])
