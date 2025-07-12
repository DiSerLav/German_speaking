# analyse.py
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional


def analyse(path_in: str, csv_out: str, heatmap_out: Optional[str] = None) -> str:
    """Строит матрицу частот эмоций по темам и сохраняет визуализацию.

    Args:
        path_in: jsonl файл с полями "topic" и "pred_sentiment".
        csv_out: Путь для сохранения csv-файла кросс-таблицы.
        heatmap_out: Путь для сохранения png-файла с тепловой картой.
                     Если None, формируется автоматически как
                     f"{Path(csv_out).stem}_heatmap.png" рядом с csv.
    Returns:
        Путь к сохранённому изображению heatmap_out.
    """
    df = pd.read_json(path_in, lines=True)

    pivot = pd.crosstab(df["topic"], df["pred_sentiment"], normalize="index")
    pivot.to_csv(csv_out)

    if heatmap_out is None:
        root, _ = os.path.splitext(csv_out)
        heatmap_out = f"{root}_heatmap.png"

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="coolwarm", annot=True)
    plt.title("Эмоциональный профиль тем")
    plt.ylabel("Topic")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.savefig(heatmap_out, dpi=200)
    plt.close()

    return heatmap_out


if __name__ == "__main__":
    # Пример использования из командной строки:
    # python analyse.py data/predicted.json analyse.csv
    if len(sys.argv) < 3:
        print("Usage: python analyse.py <input_jsonl> <csv_out>")
        sys.exit(1)

    analyse(sys.argv[1], sys.argv[2])
