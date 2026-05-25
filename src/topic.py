import sys
import os
import warnings
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加 / 将项目根目录添加到路径
sys.path.insert(0, str(Path(__file__).parent))

import config
from fugashi import Tagger
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

warnings.filterwarnings("ignore")

# 設定 / 设置
MODEL_NAME = "tohoku-nlp/bert-base-japanese-v3"
RANDOM_SEED = 42
MIN_TOPIC_SIZE = 2  # 小さいデータセット用に小さめに設定 / 针对小型数据集设为较小值

# CUDA が利用可能なら GPU を使用 / 如果 CUDA 可用则使用 GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス / 使用设备: {DEVICE}")

# fugashi タガーの初期化 / 初始化 fugashi 分词器
tagger = Tagger()

# 品詞フィルタ：名詞・動詞・形容詞のみを残す / 词性过滤：仅保留名词、动词、形容词
KEEP_POS = {"名詞", "動詞", "形容詞"}
PUNCTUATION_POS = {"補助記号", "記号", "助詞", "助動詞", "接続詞", "感動詞", "接頭詞", "接尾詞"}


def tokenize_with_fugashi(text: str) -> str:
    """
    fugashi を使って形態素解析し、
    名詞・動詞・形容詞の原形をスペース区切りで返す。
    BERTopic の CountVectorizer に渡すための前処理。

    使用 fugashi 进行形态素解析，
    将名词、动词、形容词的原形以空格分隔返回。
    作为 BERTopic 的 CountVectorizer 的预处理。
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = []
    for word in tagger(text):
        pos = word.feature.pos1
        if pos in PUNCTUATION_POS:
            continue
        if pos in KEEP_POS:
            # 原形（lemma）があれば原形を、なければ表層形を使用 / 如果有原形(lemma)则使用原形，否则使用表层形
            lemma = word.feature.lemma if word.feature.lemma else word.surface
            tokens.append(lemma)
    return " ".join(tokens)


def run_topic_modeling(
    texts: list[str],
    dataset_name: str,
    text_type: str,
    output_dir: Path,
    n_topics: int | None = None,
):
    """
    与えられたテキストリストに対してトピックモデリングを実行する。
    对给定的文本列表执行话题建模。

    Args:
        texts: 分析対象のテキストリスト / 分析对象的文本列表
        dataset_name: データセット名（例: 'mochiko'） / 数据集名称（例：'mochiko'）
        text_type: テキスト種別（例: 'userInput' or 'replyText'） / 文本类型（例：'userInput' 或 'replyText'）
        output_dir: 結果出力ディレクトリ / 结果输出目录
        n_topics: トピック数の指定（None なら自動決定） / 指定话题数（None 则自动决定）
    """
    print(f"\n{'='*60}")
    print(f"  トピックモデリング: {dataset_name} / {text_type}")
    print(f"  テキスト数: {len(texts)} / 文本数: {len(texts)}")
    print(f"{'='*60}")

    # 空テキストを除外 / 排除空文本
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(valid_texts) < 3:
        print(f"  ⚠ 有効テキストが{len(valid_texts)}件しかないためスキップ / 有效文本仅有{len(valid_texts)}条，跳过处理")
        return

    print(f"  有効テキスト数: {len(valid_texts)} / 有效文本数: {len(valid_texts)}")

    # fugashi で前処理（BERTopic の CountVectorizer 用）/ 使用 fugashi 预处理（用于 BERTopic 的 CountVectorizer）
    print("  [1/4] fugashi による形態素解析・前処理中... / 正在使用 fugashi 进行形态素解析与预处理...")
    tokenized_texts = [tokenize_with_fugashi(t) for t in valid_texts]

    # BERT埋め込みの生成 / 生成 BERT 嵌入向量
    print(f"  [2/4] BERT埋め込み生成中（{MODEL_NAME}）... / 正在生成 BERT 嵌入向量...")
    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embeddings = embedding_model.encode(
        valid_texts,
        show_progress_bar=True,
        batch_size=64 if DEVICE == "cuda" else 16,
    )
    print(f"  埋め込み次元: {embeddings.shape} / 嵌入维度: {embeddings.shape}")

    # BERTopic によるトピックモデリング / 使用 BERTopic 执行话题建模
    print("  [3/4] BERTopic によるトピックモデリング実行中... / 正在执行 BERTopic 话题建模...")

    # UMAP の再現性設定 / UMAP 的可重复性设置
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    # CountVectorizer: fugashi で前処理済みのテキストを使用 / 使用 fugashi 预处理后的文本
    vectorizer = CountVectorizer()

    topic_model = BERTopic(
        embedding_model=None,  # 事前計算済み埋め込みを使用 / 使用预先计算的嵌入向量
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics=n_topics,
        verbose=True,
        language="japanese",
    )

    topics, probs = topic_model.fit_transform(
        tokenized_texts,
        embeddings=embeddings,
    )

    # 結果の保存 / 保存结果
    print("  [4/4] 結果保存中... / 正在保存结果...")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_name}_{text_type}"

    # トピック情報の保存 / 保存话题信息
    topic_info = topic_model.get_topic_info()
    topic_info_path = output_dir / f"{prefix}_topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False, encoding="utf-8-sig")
    print(f"  → トピック情報: {topic_info_path}")

    # 各トピックのキーワード保存 / 保存各话题的关键词
    all_keywords = []
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue  # アウトライアートピックはスキップ / 跳过离群话题
        keywords = topic_model.get_topic(topic_id)
        for word, score in keywords:
            all_keywords.append({
                "topic_id": topic_id,
                "keyword": word,
                "score": score,
            })
    keywords_df = pd.DataFrame(all_keywords)
    keywords_path = output_dir / f"{prefix}_topic_keywords.csv"
    keywords_df.to_csv(keywords_path, index=False, encoding="utf-8-sig")
    print(f"  → トピックキーワード: {keywords_path}")

    # ドキュメント-トピック割り当ての保存 / 保存文档-话题分配结果
    doc_topics = pd.DataFrame({
        "document_index": range(len(valid_texts)),
        "original_text": valid_texts,
        "tokenized_text": tokenized_texts,
        "topic_id": topics,
    })
    if probs is not None:
        doc_topics["topic_probability"] = probs.max(axis=1) if probs.ndim > 1 else probs
    doc_topics_path = output_dir / f"{prefix}_doc_topics.csv"
    doc_topics.to_csv(doc_topics_path, index=False, encoding="utf-8-sig")
    print(f"  → ドキュメント-トピック割り当て: {doc_topics_path}")

    # トピック数のサマリー表示 / 显示话题数摘要
    n_valid_topics = len([t for t in topic_info["Topic"].values if t != -1])
    n_outliers = len([t for t in topics if t == -1])
    print(f"\n  ■ 結果サマリー / 结果摘要:")
    print(f"    トピック数: {n_valid_topics} / 话题数: {n_valid_topics}")
    print(f"    アウトライアードキュメント数: {n_outliers} / {len(topics)} / 离群文档数: {n_outliers} / {len(topics)}")
    print(f"    トピック割合: {(len(topics) - n_outliers) / len(topics) * 100:.1f}% / 话题占比: {(len(topics) - n_outliers) / len(topics) * 100:.1f}%")

    # トピック一覧の表示 / 显示话题一览
    print(f"\n  ■ トピック一覧 / 话题一览:")
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue
        count = topic_info.loc[topic_info["Topic"] == topic_id, "Count"].values[0]
        name = topic_info.loc[topic_info["Topic"] == topic_id, "Name"].values[0]
        print(f"    Topic {topic_id} ({count}件): {name}")

    return topic_model


# メイン処理 / 主处理逻辑
if __name__ == "__main__":
    output_dir = config.DATA_DIR / "topic_modeling"

    # データ読み込み / 加载数据
    print("データ読み込み中... / 正在加载数据...")
    df_mochiko = pd.read_csv(config.DATA_DIR / "data_mochiko.csv")
    df_pen_sensei = pd.read_csv(config.DATA_DIR / "data_pen_sensei.csv")

    print(f"mochiko: {len(df_mochiko)} 行")
    print(f"pen_sensei: {len(df_pen_sensei)} 行")

    # mochiko のトピックモデリング / mochiko 的话题建模
    # ユーザー入力 / 用户输入
    mochiko_input_texts = df_mochiko["userInput"].fillna("").tolist()
    run_topic_modeling(
        texts=mochiko_input_texts,
        dataset_name="mochiko",
        text_type="userInput",
        output_dir=output_dir,
    )

    # ボット応答 / 机器人回复
    mochiko_reply_texts = df_mochiko["replyText"].fillna("").tolist()
    run_topic_modeling(
        texts=mochiko_reply_texts,
        dataset_name="mochiko",
        text_type="replyText",
        output_dir=output_dir,
    )

    # pen_sensei のトピックモデリング / pen_sensei 的话题建模
    # ユーザー入力 / 用户输入
    pen_input_texts = df_pen_sensei["userInput"].fillna("").tolist()
    run_topic_modeling(
        texts=pen_input_texts,
        dataset_name="pen_sensei",
        text_type="userInput",
        output_dir=output_dir,
    )

    # ボット応答 / 机器人回复
    pen_reply_texts = df_pen_sensei["replyText"].fillna("").tolist()
    run_topic_modeling(
        texts=pen_reply_texts,
        dataset_name="pen_sensei",
        text_type="replyText",
        output_dir=output_dir,
    )

    # 両者を結合したユーザー入力のトピックモデリング / 合并两者用户输入进行话题建模
    print(f"\n{'='*60}")
    print(f"  両者結合ユーザー入力のトピックモデリング / 合并两者的用户输入进行话题建模")
    print(f"{'='*60}")
    all_input_texts = mochiko_input_texts + pen_input_texts
    run_topic_modeling(
        texts=all_input_texts,
        dataset_name="combined",
        text_type="userInput",
        output_dir=output_dir,
    )

    print(f"  トピックモデリング完了！ / 话题建模完成！")
    print(f"  結果出力先: {output_dir} / 结果输出目录: {output_dir}")
