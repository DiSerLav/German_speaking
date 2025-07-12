"""Визуализация тематических кластеров.

Файл предоставляет функцию `visualize_clusters` и CLI-обёртку.  Код разбит на
малые функции, покрыт типами и использует `logging` вместо `print`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP

# --------------------------------------------------------------------------------------
# Константы и глобальные объекты
# --------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Цветовая карта для >20 кластеров
_HSV_CMAP = plt.cm.get_cmap("hsv")

# Маркеры (используются, если diff_markers = True)
_MARKERS = [
    "o",
    "s",
    "^",
    "v",
    "P",
    "D",
    "*",
    "X",
    "<",
    ">",
    "h",
    "H",
    "8",
    "p",
]


# --------------------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------------------


def _read_dataset(path: Path) -> Tuple[List[str], List[int]]:
    """Читает JSONL-файл и возвращает список документов и их топиков."""

    docs, topics = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            docs.append(obj["clean"])
            topics.append(obj["topic"])
    return docs, topics


def _compute_embeddings(docs: List[str]) -> np.ndarray:
    """Возвращает векторные представления документов.

    Сначала пытается использовать SentenceTransformer; при неудаче
    автоматически переключается на TF-IDF + SVD.
    """

    try:
        logger.info("Пробуем загрузить модель SentenceTransformer (%s)…", EMBED_MODEL)
        embedder = SentenceTransformer(EMBED_MODEL)
        return embedder.encode(docs, show_progress_bar=True)
    except Exception as exc:  # noqa: BLE001 – внешние библиотеки могут кидать всё подряд
        logger.warning(
            "Не удалось использовать SentenceTransformer (%s). Переключаемся на TF-IDF + SVD.",
            exc,
        )
        # Ленивая импортировка тяжёлых зависимостей
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf_vec = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2))
        tfidf = tfidf_vec.fit_transform(docs)

        svd = TruncatedSVD(n_components=300, random_state=42)
        return svd.fit_transform(tfidf)


def _reduce_dimensions(
    embs: np.ndarray,
    method: str,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: int = 40,
    early_exaggeration: float = 12.0,
) -> np.ndarray:
    """Снижает размерность эмбеддингов до 2-D."""

    method = method.lower()
    if method == "umap":
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
    elif method in {"tsne", "t-sne"}:
        from sklearn.manifold import TSNE  # lazy import

        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            init="pca",
        )
    else:
        raise ValueError("method должен быть 'umap' или 'tsne', получено: %s" % method)

    logger.info("Снижение размерности методом %s…", method.upper())
    return reducer.fit_transform(embs)


def _plot_clusters(
    coords: np.ndarray,
    topics: np.ndarray,
    *,
    out_path: Path,
    method: str,
    alpha: float,
    annotate: bool,
    diff_markers: bool,
) -> None:
    """Строит и сохраняет scatter-диаграмму."""

    plt.figure(figsize=(8, 6))

    uniq = sorted(set(topics))
    cmap = (
        plt.cm.get_cmap("tab20", len(uniq)) if len(uniq) <= 20 else _HSV_CMAP(len(uniq))
    )

    for idx, t in enumerate(uniq):
        mask = topics == t
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=60,
            color=cmap(idx),
            marker=_MARKERS[idx % len(_MARKERS)] if diff_markers else "o",
            alpha=alpha,
            edgecolors="k",
            linewidths=0.4,
            label=str(t),
        )

        if annotate:
            cx, cy = coords[mask].mean(axis=0)
            plt.text(
                cx,
                cy,
                str(t),
                fontsize=10,
                weight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
            )

    plt.title(f"Документы в 2D ({method.upper()})")
    plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------------------
# Публичная функция
# --------------------------------------------------------------------------------------


def visualize_clusters(
    path_in: Union[str, Path],
    out_png: Union[str, Path] = "clusters.png",
    *,
    method: str = "umap",
    # UMAP
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    # t-SNE
    perplexity: int = 40,
    early_exaggeration: float = 12.0,
    # Plot
    alpha: float = 0.85,
    annotate: bool = False,
    diff_markers: bool = True,
) -> Path:
    """Готовит и сохраняет визуализацию кластеров.

    Parameters
    ----------
    path_in: Union[str, Path]
        JSONL-файл с полями ``clean`` и ``topic``.
    out_png: Union[str, Path]
        Куда сохранить PNG.
    method: str
        Способ снижения размерности: ``umap`` | ``tsne``.
    Returns
    -------
    Path
        Путь к сохранённому файлу.
    """

    path_in = Path(path_in)
    out_path = Path(out_png)

    docs, topics = _read_dataset(path_in)
    embs = _compute_embeddings(docs)
    coords = _reduce_dimensions(
        embs,
        method,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
    )

    _plot_clusters(
        coords,
        np.array(topics),
        out_path=out_path,
        method=method,
        alpha=alpha,
        annotate=annotate,
        diff_markers=diff_markers,
    )

    logger.info("Сохранено в %s", out_path)
    return out_path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Визуализация тематических кластеров")
    parser.add_argument("input", type=Path, help="Файл predicted.jsonl")
    parser.add_argument("output", type=Path, nargs="?", default=Path("clusters.png"))
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap")

    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)

    parser.add_argument("--perplexity", type=int, default=40)
    parser.add_argument("--early_exaggeration", type=float, default=12.0)

    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--diff_markers", action="store_true")

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = _build_cli().parse_args()

    visualize_clusters(
        args.input,
        args.output,
        method=args.method,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        alpha=args.alpha,
        annotate=args.annotate,
        diff_markers=args.diff_markers,
    )


if __name__ == "__main__":
    main() 