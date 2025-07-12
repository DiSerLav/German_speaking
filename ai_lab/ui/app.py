import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Внутренние инструменты проекта
from ai_lab.tools import predict_emotion as pe
from ai_lab.tools.visualize import _reduce_dimensions

#####################################################################################
# КЭШИРОВАНИЕ ТЯЖЁЛЫХ РЕСУРСОВ
#####################################################################################

@st.cache_resource(show_spinner=False)
def load_sentiment_model() -> Tuple[pe.AutoTokenizer, pe.AutoModelForSequenceClassification, List[str]]:  # type: ignore
    return pe.tok, pe.mdl, pe.LABELS

@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def find_data_file(filename: str) -> Path | None:
    """Ищет файл в ai_lab/data и ai_lab."""
    for folder in [Path(__file__).parent.parent / "data", Path(__file__).parent.parent]:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return None

@st.cache_data(show_spinner=False)
def compute_cluster_coordinates(texts: List[str], topics: List[int], method: str = "umap") -> np.ndarray:
    """Вычисляет 2-D координаты документов для интерактивного отображения.
    
    Параметр *topics* используется только для участия в кешировании – чтобы при изменении
    набора документов/тем кэш пересчитывался.
    """
    embedder = get_embedder()
    embs = embedder.encode(texts, show_progress_bar=False)
    coords = _reduce_dimensions(embs, method)
    return coords

#####################################################################################
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
#####################################################################################

def sentiment_page() -> None:
    st.header("Эмоциональный анализ")

    pred_path = find_data_file("predicted.jsonl")
    if pred_path:
        data = load_jsonl(pred_path)
        st.success(f"Используется файл: {pred_path.name}")
    else:
        preproc_path = find_data_file("preprocessed.jsonl")
        if preproc_path:
            data = load_jsonl(preproc_path)
            st.warning("Файл predicted.jsonl не найден, используются только тексты из preprocessed.jsonl")
        else:
            st.error("Нет подходящих файлов для отображения!")
            return

    # Показываем таблицу
    df = pd.DataFrame(data)
    if "pred_sentiment" in df.columns:
        st.dataframe(df[["text", "pred_sentiment"]])
    elif "sentiment" in df.columns:
        st.dataframe(df[["text", "sentiment"]])
    else:
        st.dataframe(df[["text"]])
    
    # Добавляем тепловую карту из analyse.csv
    analyse_path = find_data_file("analyse.csv")
    if analyse_path:
        st.subheader("Тепловая карта эмоционального профиля тем")
        
        # Загружаем данные
        pivot_df = load_csv(analyse_path)
        
        # Преобразуем данные для Altair (melt для long format)
        heatmap_data = pivot_df.melt(
            id_vars=['topic'], 
            var_name='emotion', 
            value_name='percentage'
        )
        
        # Создаем интерактивную тепловую карту с Altair
        base = alt.Chart(heatmap_data).encode(
            x=alt.X('emotion:N', title='Эмоция', sort=['negative', 'neutral', 'positive']),
            y=alt.Y('topic:N', title='Топик', sort=alt.EncodingSortField('topic', order='ascending'))
        )
        
        # Слой с прямоугольниками (цвета)
        heatmap_rects = base.mark_rect().encode(
            color=alt.Color('percentage:Q', 
                          scale=alt.Scale(scheme='redblue', domain=[0, 1]),
                          title='Доля эмоций'),
            tooltip=[
                alt.Tooltip('topic:N', title='Топик'),
                alt.Tooltip('emotion:N', title='Эмоция'),
                alt.Tooltip('percentage:Q', title='Доля', format='.1%')
            ]
        )
        
        # Слой с текстом (значения)
        heatmap_text = base.mark_text(
            align='center',
            baseline='middle',
            fontSize=11,
            fontWeight='bold'
        ).encode(
            text=alt.Text('percentage:Q', format='.1%'),
            color=alt.condition(
                alt.datum.percentage > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )
        
        # Объединяем слои
        heatmap = (heatmap_rects + heatmap_text).properties(
            width=600,
            height=400,
            title='Эмоциональный профиль тем'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16
        )
        
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("Файл analyse.csv не найден. Тепловая карта недоступна.")

#####################################################################################
def clustering_page() -> None:
    st.header("Визуализация кластеризации")
    st.markdown("Данные берутся из predicted.jsonl (если есть) или preprocessed.jsonl. Визуализация интерактивная с цветовой идентификацией по топикам.")

    pred_path = find_data_file("predicted.jsonl")
    if pred_path:
        data = load_jsonl(pred_path)
    else:
        preproc_path = find_data_file("preprocessed.jsonl")
        if preproc_path:
            data = load_jsonl(preproc_path)
        else:
            st.error("Нет подходящих файлов для отображения!")
            return

    texts = [item["text"] for item in data if item.get("text", "").strip()]
    method = st.selectbox("Метод снижения размерности", ["umap", "tsne"], index=1)
    
    if st.button("Построить кластеризацию"):
        with st.spinner("Вычисляем координаты..."):
            coords = compute_cluster_coordinates(texts, [], method)
        
        # Определяем цветовую индикацию
        color_field = None
        color_title = "Цветовая индикация"
        
        # Приоритет: топики > эмоции
        if any(item.get("topic") is not None for item in data):
            color_field = "topic_id"
            color_title = "Топики"
        elif any(item.get("pred_sentiment") is not None for item in data):
            color_field = "sentiment"
            color_title = "Эмоции"
        
        # Создаем DataFrame для Altair
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "text": [text[:197] + "…" if len(text) > 200 else text for text in texts],
            "topic_id": [item.get("topic", "?") for item in data if item.get("text", "").strip()],
            "sentiment": [item.get("pred_sentiment", "?") for item in data if item.get("text", "").strip()]
        })
        
        # Создаем Altair chart
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('x:Q', title='X координата'),
            y=alt.Y('y:Q', title='Y координата'),
            tooltip=[
                alt.Tooltip('text:N', title='Текст'),
                alt.Tooltip('topic_id:N', title='Топик'),
                alt.Tooltip('sentiment:N', title='Эмоция')
            ],
            color=alt.Color(color_field + ':N', title=color_title) if color_field else alt.value('steelblue')
        ).properties(
            width=700,
            height=500,
            title=f"Кластеризация документов ({method.upper()})"
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        if not color_field:
            st.info("Цветовая индикация недоступна - в данных нет полей topic или pred_sentiment")

#####################################################################################
def samples_page() -> None:
    st.header("Случайные фразы из данных")
    st.markdown("Данные берутся из predicted.jsonl (если есть) или preprocessed.jsonl. Показываются случайные фразы из каждого топика.")
    pred_path = find_data_file("predicted.jsonl")
    if pred_path:
        data = load_jsonl(pred_path)
    else:
        preproc_path = find_data_file("preprocessed.jsonl")
        if preproc_path:
            data = load_jsonl(preproc_path)
        else:
            st.error("Нет подходящих файлов для отображения!")
            return
    
    # Группируем по топикам (как в show_samples.py)
    buckets: dict[int, list[dict]] = defaultdict(list)
    for item in data:
        if item.get("text", "").strip():
            topic = item.get("topic", -1)
            buckets[topic].append(item)
    
    n = st.number_input("Количество фраз на топик", min_value=1, max_value=20, value=3, step=1)
    
    if st.button("Показать случайные фразы по топикам"):
        for topic in sorted(buckets):
            items = buckets[topic]
            sample = items if len(items) <= n else random.sample(items, n)
            
            st.subheader(f"Топик {topic} (документов: {len(items)})")
            
            for i, item in enumerate(sample, 1):
                text = item["text"]
                short = text if len(text) <= 500 else text[:497] + "…"
                info_parts = []
                if item.get("name"): info_parts.append(f"{item['name']}")
                if item.get("surname"): info_parts.append(f"{item['surname']}")
                if item.get("sentiment"): info_parts.append(f"({item['sentiment']})")
                if item.get("pred_sentiment"): info_parts.append(f"({item['pred_sentiment']})")
                info = " ".join(info_parts) if info_parts else ""
                
                with st.container(border=True):
                    st.markdown(f"{i}. **{info}**" if info else f"{i}", unsafe_allow_html=True)
                    st.markdown(f"{short}", unsafe_allow_html=True)
#####################################################################################
# НАВИГАЦИЯ
#####################################################################################
PAGES = {
    "Эмоциональный анализ": sentiment_page,
    "Визуализация кластеризации": clustering_page,
    "Случайные фразы": samples_page,
}
def main() -> None:
    st.set_page_config(page_title="AI-Lab Demo")
    page_name = st.sidebar.radio("Навигация", list(PAGES.keys()))
    PAGES[page_name]()
    
if __name__ == "__main__":
    main() 