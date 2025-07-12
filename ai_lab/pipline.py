from pathlib import Path

from colorama import Fore, Style, init

from ai_lab.tools.analyse import analyse
from ai_lab.tools.predict_emotion import predict
from ai_lab.tools.preprocess import preprocess
from ai_lab.tools.topic_model import topic_model
from ai_lab.tools.visualize import visualize_clusters

init(autoreset=True)
CUR_DIR = Path(__file__).parent


# --------------------------------------------------------------------------------------
# Основная функция
# --------------------------------------------------------------------------------------
def main(begin_step: int = 0):
    print(Fore.GREEN + "Старт программы..." + Style.RESET_ALL)

    # === Шаг 0: Обработка данных ===
    if begin_step <= 0:
        print(Fore.YELLOW + "Обработка данных..." + Style.RESET_ALL)
        preprocess(CUR_DIR / "input.json", CUR_DIR / "data" / "preprocessed.jsonl")

    # === Шаг 1: Обучение модели тематик ===
    if begin_step <= 1:
        print(Fore.YELLOW + "Обучение модели тематик..." + Style.RESET_ALL)
        topics_path = topic_model(
            CUR_DIR / "data" / "preprocessed.jsonl",
            CUR_DIR / "data" / "topic_model.pkl",
        )

    # === Шаг 2: Предсказание эмоций ===
    if begin_step <= 2:
        print(Fore.YELLOW + "Предсказание эмоций..." + Style.RESET_ALL)
        predict(Path(topics_path), CUR_DIR / "data" / "predicted.jsonl")

    # === Шаг 3: Анализ данных ===
    if begin_step <= 3:
        print(Fore.YELLOW + "Анализ данных..." + Style.RESET_ALL)
        analyse(CUR_DIR / "data" / "predicted.jsonl", CUR_DIR / "data" / "analyse.csv")

    # === Шаг 4: Визуализация кластеров (UMAP) ===
    if begin_step <= 4:
        print(Fore.YELLOW + "Визуализация кластеров (UMAP)..." + Style.RESET_ALL)
        visualize_clusters(
            CUR_DIR / "data" / "predicted.jsonl",
            CUR_DIR / "data" / "clusters_umap.png",
            method="umap",
        )

    # === Шаг 5: Визуализация кластеров (t-SNE) ===
    if begin_step <= 5:
        print(Fore.YELLOW + "Визуализация кластеров (t-SNE)..." + Style.RESET_ALL)
        visualize_clusters(
            CUR_DIR / "data" / "predicted.jsonl",
            CUR_DIR / "data" / "clusters_tsne.png",
            method="tsne",
        )
    print(Fore.GREEN + "Программа завершена успешно!" + Style.RESET_ALL)

if __name__ == "__main__":
    main(0)
