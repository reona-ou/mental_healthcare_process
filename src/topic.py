"""
トピックモデリングモジュール
BERTopic + KMeans による話題モデリング
- 短文テキストフィルタ（tokenized < 2語 → -1）
- 統計距離フィルタ（mean + 1.5 * std）
- ruri-v3 プレフィックス: 「トピック: 」を使用（分類・クラスタリング用）
"""
import sys
import warnings
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import config
from fugashi import Tagger
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from umap import UMAP

warnings.filterwarnings("ignore")

# 埋め込みモデル（ローカル優先）
LOCAL_MODEL_PATH = config.MODELS_DIR / "ruri-v3-310m"
MODEL_NAME = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "cl-nagoya/ruri-v3-310m"
RANDOM_SEED = config.TOPIC_RANDOM_SEED

# デバイス選択
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {DEVICE}")

# fugashi タガーの初期化
tagger = Tagger()

# 品詞フィルタ（config.py から取得）
KEEP_POS = config.KEEP_POS
PUNCTUATION_POS = config.PUNCTUATION_POS

# ストップワード（config.py から取得）
STOPWORDS = config.STOPWORDS


def tokenize_with_fugashi(text: str) -> str:
    """fugashiで形態素解析し、名詞・動詞・形容詞の原形を返す"""
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = []
    for word in tagger(text):
        pos = word.feature.pos1
        if pos in config.PUNCTUATION_POS:
            continue
        if pos in config.KEEP_POS:
            lemma = word.feature.lemma if word.feature.lemma else word.surface
            if "-" in lemma:
                lemma = lemma.split("-")[0]
            if lemma not in config.STOPWORDS:
                tokens.append(lemma)
    return " ".join(tokens)


def verify_and_reassign(
    topics: list[int],
    embeddings: np.ndarray,
    topic_model,
    z_threshold: float = 2.0,
) -> list[int]:
    """統計距離に基づき異常文書をフィルタ（mean + z_threshold * std）"""
    from sklearn.metrics.pairwise import cosine_distances

    new_topics = list(topics)
    marked_outlier = 0

    topic_centroids = {}
    for tid in set(topics):
        if tid == -1:
            continue
        indices = [i for i, t in enumerate(topics) if t == tid]
        if indices:
            topic_centroids[tid] = np.mean(embeddings[indices], axis=0)

    topic_stats = {}
    for tid, centroid in topic_centroids.items():
        indices = [i for i, t in enumerate(topics) if t == tid]
        if len(indices) < 2:
            continue
        dists = cosine_distances(embeddings[indices], [centroid]).flatten()
        topic_stats[tid] = {"mean": np.mean(dists), "std": np.std(dists)}

    for i, (tid, emb) in enumerate(zip(topics, embeddings)):
        if tid == -1 or tid not in topic_stats:
            continue
        dist = cosine_distances([emb], [topic_centroids[tid]])[0][0]
        mean = topic_stats[tid]["mean"]
        std = topic_stats[tid]["std"]
        if std > 0 and dist > mean + z_threshold * std:
            new_topics[i] = -1
            marked_outlier += 1

    print(f"  验证：标记为 -1 {marked_outlier} 篇")
    return new_topics


def run_topic_modeling(

    texts: list[str],
    dataset_name: str,
    text_type: str,
    output_dir: Path,
    source_df: pd.DataFrame | None = None,
):
    """与えられたテキストリストに対してトピックモデリングを実行"""
    print(f"  トピックモデリング: {dataset_name} / {text_type}")
    print(f"  テキスト数: {len(texts)}")

    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(valid_texts) < 3:
        print(f"  有効テキストが{len(valid_texts)}件のみ、スキップ")
        return

    print(f"  有効テキスト数: {len(valid_texts)}")

    print("fugashi で形態素解析中...")
    tokenized_texts = [tokenize_with_fugashi(t) for t in valid_texts]

    # 短文テキストの識別（分詞後2語未満はクラスタリング対象外）
    short_threshold = 2
    short_indices = [i for i, t in enumerate(tokenized_texts) if len(t.split()) < short_threshold]
    long_indices = [i for i in range(len(valid_texts)) if i not in short_indices]
    print(f"  短文: {len(short_indices)} 件, クラスタリング対象: {len(long_indices)} 件")

    long_texts = [valid_texts[i] for i in long_indices]
    long_tokenized = [tokenized_texts[i] for i in long_indices]

    print(f"正在生成 BERT 嵌入向量（{MODEL_NAME}）...")
    print(f"  プレフィックス: 「{config.TOPIC_EMBEDDING_PREFIX}」")
    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    prefixed_texts = [f"{config.TOPIC_EMBEDDING_PREFIX}{t}" for t in long_texts]
    embeddings = embedding_model.encode(
        prefixed_texts,
        show_progress_bar=True,
        batch_size=config.TOPIC_EMBEDDING_BATCH_SIZE_CUDA if DEVICE == "cuda" else config.TOPIC_EMBEDDING_BATCH_SIZE_CPU,
    )
    print(f"  埋め込み次元: {embeddings.shape}")

    print("  BERTopic トピックモデリング実行中...")

    umap_model = UMAP(
        n_neighbors=config.TOPIC_UMAP_N_NEIGHBORS,
        n_components=config.TOPIC_UMAP_N_COMPONENTS,
        min_dist=config.TOPIC_UMAP_MIN_DIST,
        metric=config.TOPIC_UMAP_METRIC,
        random_state=RANDOM_SEED,
    )

    vectorizer = CountVectorizer(
        max_df=config.TOPIC_VECTORIZER_MAX_DF,
        min_df=config.TOPIC_VECTORIZER_MIN_DF,
        ngram_range=(1, 2),
    )

    from bertopic.representation import MaximalMarginalRelevance
    representation_model = MaximalMarginalRelevance(diversity=0.5)

    n_clusters = min(7, len(long_texts) // 10)
    n_clusters = max(3, n_clusters)
    km_model = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=km_model,
        vectorizer_model=vectorizer,
        nr_topics=None,
        verbose=True,
        language="japanese",
        seed_topic_list=config.TOPIC_SEED,
        representation_model=representation_model,
    )

    topics_long, probs_long = topic_model.fit_transform(
        long_tokenized,
        embeddings=embeddings,
    )

    topics_long = verify_and_reassign(topics_long, embeddings, topic_model, z_threshold=1.5)

    # UMAP 2次元座標を計算
    print("  UMAP 2次元座標を計算中...")
    umap_coords = topic_model.umap_model.transform(embeddings)

    topics = [-1] * len(valid_texts)
    probs = None
    umap_all = np.zeros((len(valid_texts), 2))
    for i, topic_idx in enumerate(long_indices):
        topics[topic_idx] = topics_long[i]
        umap_all[topic_idx] = umap_coords[i]
    if probs_long is not None:
        if probs_long.ndim == 1:
            probs_long = probs_long.reshape(-1, 1)
        probs = np.zeros((len(valid_texts), probs_long.shape[1]))
        for i, topic_idx in enumerate(long_indices):
            probs[topic_idx] = probs_long[i]

    # 結果保存
    print("結果保存中...")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_name}_{text_type}"

    topic_counts = Counter(topics)
    n_outliers = topic_counts.get(-1, 0)
    n_valid_topics = len([t for t in topic_counts.keys() if t != -1])
    print(f"\n  トピック数: {n_valid_topics}, 外れ値: {n_outliers}/{len(topics)} ({(len(topics)-n_outliers)/len(topics)*100:.1f}%)")

    doc_topics = pd.DataFrame({
        "document_index": range(len(valid_texts)),
        "original_text": valid_texts,
        "tokenized_text": tokenized_texts,
        "topic_id": topics,
        "umap_0": umap_all[:, 0],
        "umap_1": umap_all[:, 1],
    })
    if probs is not None:
        doc_topics["topic_probability"] = probs.max(axis=1) if probs.ndim > 1 else probs
    doc_topics_path = output_dir / f"{prefix}_doc_topics.csv"
    doc_topics.to_csv(doc_topics_path, index=False, encoding="utf-8-sig")
    print(f"  → {doc_topics_path}")

    topic_info_list = []
    for topic_id in sorted(topic_counts.keys()):
        if topic_id == -1:
            name = f"{topic_id}_outlier"
        else:
            kws = topic_model.get_topic(topic_id)
            name = f"{topic_id}_{kws[0][0]}_{kws[1][0]}_{kws[2][0]}" if kws else f"{topic_id}_unknown"
        topic_info_list.append({"Topic": topic_id, "Count": topic_counts[topic_id], "Name": name})
    topic_info_df = pd.DataFrame(topic_info_list)
    topic_info_path = output_dir / f"{prefix}_topic_info.csv"
    topic_info_df.to_csv(topic_info_path, index=False, encoding="utf-8-sig")
    print(f"  → {topic_info_path}")

    all_keywords = []
    for topic_id in sorted(topic_counts.keys()):
        if topic_id == -1:
            continue
        kws = topic_model.get_topic(topic_id)
        if kws:
            for word, score in kws:
                all_keywords.append({"topic_id": topic_id, "keyword": word, "score": score})
    keywords_df = pd.DataFrame(all_keywords)
    keywords_path = output_dir / f"{prefix}_topic_keywords.csv"
    keywords_df.to_csv(keywords_path, index=False, encoding="utf-8-sig")
    print(f"  → {keywords_path}")

    # トピック一覧
    print(f"\n  トピック一覧:")
    for topic_id in sorted(topic_counts.keys()):
        if topic_id == -1:
            continue
        print(f"    Topic {topic_id} ({topic_counts[topic_id]}件)")

    return topic_model


if __name__ == "__main__":
    output_dir = config.DATA_DIR / "topic_modeling"

    print("正在加载数据...")
    df_with_id = pd.read_csv(config.DATA_DIR / "data_with_id.csv")
    print(f"data_with_id: {len(df_with_id)} 行")

    all_input_texts = df_with_id["userInput"].fillna("").tolist()
    run_topic_modeling(
        texts=all_input_texts,
        dataset_name="combined",
        text_type="userInput",
        output_dir=output_dir,
        source_df=df_with_id,
    )

    print(f"  话题建模完成！输出目录: {output_dir}")
