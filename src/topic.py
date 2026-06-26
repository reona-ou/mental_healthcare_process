import sys
import os
import warnings
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# 将项目根目录添加到路径 / プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

import config
from fugashi import Tagger
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

warnings.filterwarnings("ignore")

# 设置 — 本地模型优先 / 設定 — ローカルモデル優先
LOCAL_MODEL_PATH = config.MODELS_DIR / "bert-base-japanese-v3"
MODEL_NAME = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "tohoku-nlp/bert-base-japanese-v3"
RANDOM_SEED = config.TOPIC_RANDOM_SEED
MIN_TOPIC_SIZE = config.TOPIC_MIN_TOPIC_SIZE

# 如果 CUDA 可用则使用 GPU / CUDA が利用可能なら GPU を使用
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备 / 使用デバイス: {DEVICE}")

# 初始化 fugashi 分词器 / fugashi タガーの初期化
tagger = Tagger()

# 词性过滤：仅保留名词、动词、形容词 / 品詞フィルタ：名詞・動詞・形容詞のみを残す
KEEP_POS = {"名詞", "動詞", "形容詞"}
PUNCTUATION_POS = {"補助記号", "記号", "助詞", "助動詞", "接続詞", "感動詞", "接頭詞", "接尾詞"}

# シードトピック（BERTopic 半監督モード用）
# 負面カテゴリ: 詐欺・浮気・DV・離婚・自殺等の极端ケースのみ
SEED_TOPICS = [
    # 離婚・別離
    ["離婚", "別れたい", "別居", "親権", "離婚届", "離婚を考え", "結婚を終わら"],
    # 浮気・不倫
    ["浮気", "不倫", "愛人", "二股", "浮気相手", "不倫相手"],
    # 流産・妊娠問題
    ["流産", "死産", "妊娠中絶"],
    # 詐欺・被害
    ["詐欺", "フィッシング", "騙されて", "騙す"],
    # DV・暴力
    ["DV", "暴力", "暴行", "傷害", "殴", "蹴", "脅", "叩", "暴言", "怒鳴"],
    # モラハラ
    ["モラハラ", "モラルハラスメント", "パワハラ"],
    # 自殺・自傷
    ["死にたい", "消えたい", "自殺", "死のう", "生きる意味", "いらない", "終わらせたい", "死ねない", "死なない", "死にきれ"],
]

# パターンベース検出用のキーワード組み合わせ（2カテゴリ分類用）
# 負面カテゴリ: 詐欺・浮気・DV・離婚等の极端ケースのみ
PATTERN_COMBOS = [
    # 離婚別離パターン
    {
        "name": "離婚別離",
        "required": [["離婚", "別居", "親権", "離婚届", "調停", "弁護士"]],
        "context": ["夫", "妻", "旦那", "主人", "パートナー"],
    },
    # 浮気不倫パターン
    {
        "name": "浮気不倫",
        "required": [["浮気", "不倫", "愛人", "二股", "不貞"]],
        "context": ["夫", "妻", "旦那", "主人", "彼", "パートナー"],
    },
    # 流産妊娠問題パターン
    {
        "name": "流産妊娠問題",
        "required": [["流産", "死産", "中絶", "妊娠中絶"]],
        "context": [],
    },
    # ロマンス詐欺パターン（SNS + 海外 + お金）
    {
        "name": "ロマンス詐欺",
        "required": [["Facebook", "facebook", "フェイスブック", "SNS", "LINE", "インスタ", "Instagram", "Twitter", "ツイッター", "マッチング", "matching"]],
        "context": ["海外", "外国", "貸", "送金", "お金", "投資", "返す"],
    },
    # 海外関係 + お金のやり取り（SNSなしでも検出）
    {
        "name": "海外金銭",
        "required": [["海外", "外国"]],
        "context": ["貸", "送金", "お金", "投資", "返す", "資金", "融資", "知り合", "出会"],
    },
    # 詐欺パターン（登録だけでお金、ベビーモデル、掲載料等）
    {
        "name": "詐欺",
        "required": [["詐欺", "フィッシング", "登録するだけで", "月々数万", "掲載料", "ベビーモデル"]],
        "context": [],
    },
    # DV暴力パターン
    {
        "name": "DV暴力",
        "required": [["DV", "暴力", "暴行", "傷害", "殴", "蹴", "脅", "叩", "暴言", "怒鳴"]],
        "context": ["夫", "妻", "旦那", "彼", "パートナー", "家族"],
    },
    # モラハラパターン
    {
        "name": "モラハラ",
        "required": [["モラハラ", "モラルハラスメント", "パワハラ"]],
        "context": [],
    },
    # 自殺・自傷パターン
    {
        "name": "自殺自傷",
        "required": [["死にたい", "消えたい", "自殺", "死のう", "生きる意味", "いらない", "終わらせたい", "死ねない", "死なない", "死にきれ"]],
        "context": [],
    },
]


def tokenize_with_fugashi(text: str) -> str:
    """
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
            lemma = word.feature.lemma if word.feature.lemma else word.surface
            tokens.append(lemma)
    return " ".join(tokens)


def run_topic_modeling(
    texts: list[str],
    dataset_name: str,
    text_type: str,
    output_dir: Path,
    n_topics: int | None = None,
    source_df: pd.DataFrame | None = None,
):
    """
    对给定的文本列表执行话题建模。
    """
    print(f"  话题建模: {dataset_name} / {text_type}")
    print(f"  文本数: {len(texts)} / テキスト数: {len(texts)}")

    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(valid_texts) < 3:
        print(f"  ⚠ 有效文本仅有{len(valid_texts)}条，跳过处理")
        return

    print(f"  有效文本数: {len(valid_texts)}")

    print("正在使用 fugashi 进行形态素解析与预处理...")
    tokenized_texts = [tokenize_with_fugashi(t) for t in valid_texts]

    print(f"正在生成 BERT 嵌入向量（{MODEL_NAME}）...")
    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embeddings = embedding_model.encode(
        valid_texts,
        show_progress_bar=True,
        batch_size=config.TOPIC_EMBEDDING_BATCH_SIZE_CUDA if DEVICE == "cuda" else config.TOPIC_EMBEDDING_BATCH_SIZE_CPU,
    )
    print(f"  嵌入维度: {embeddings.shape}")

    print("  正在执行 BERTopic 话题建模...")

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
    )

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics=None,
        verbose=True,
        language="japanese",
    )

    topics, probs = topic_model.fit_transform(
        tokenized_texts,
        embeddings=embeddings,
    )

    print("正在保存结果...")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_name}_{text_type}"

    topic_info = topic_model.get_topic_info()
    topic_info_path = output_dir / f"{prefix}_topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False, encoding="utf-8-sig")
    print(f"  → トピック情報: {topic_info_path}")

    all_keywords = []
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue
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
    print(f"  → 话题关键词: {keywords_path}")

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
    print(f"  → 文档-话题分配结果: {doc_topics_path}")

    n_valid_topics = len([t for t in topic_info["Topic"].values if t != -1])
    n_outliers = len([t for t in topics if t == -1])
    print(f"\n  ■ 结果摘要:")
    print(f"    话题数: {n_valid_topics}")
    print(f"    离群文档数: {n_outliers} / {len(topics)}")
    print(f"    话题占比: {(len(topics) - n_outliers) / len(topics) * 100:.1f}%")

    print(f"\n  ■ 话题一览:")
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue
        count = topic_info.loc[topic_info["Topic"] == topic_id, "Count"].values[0]
        name = topic_info.loc[topic_info["Topic"] == topic_id, "Name"].values[0]
        print(f"    Topic {topic_id} ({count}件): {name}")

    if source_df is not None:
        print(f"\n  [追加] 2カテゴリ分類を作成中...")
        classify_by_keywords(topic_model, valid_texts, topics, source_df, output_dir, prefix)

    return topic_model


def check_pattern_combos(text: str) -> tuple[bool, str | None]:
    """
    テキスト内でパターンベースの組み合わせを検出する。
    """
    if not isinstance(text, str) or not text.strip():
        return False, None

    for pattern in PATTERN_COMBOS:
        required_match = False
        for req_group in pattern["required"]:
            if any(kw in text for kw in req_group):
                required_match = True
                break
        if not required_match:
            continue

        if not pattern["context"]:
            return True, pattern["name"]

        context_match = any(kw in text for kw in pattern["context"])
        if not context_match:
            continue

        return True, pattern["name"]

    return False, None


def classify_by_keywords(
    topic_model,
    valid_texts,
    topics,
    source_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    topic_match_threshold: int = 2,
):
    """
    2カテゴリ分類（BERTopic 半監督 + パターンベース検出）。

    category 0: 負面（詐欺・浮気・DV 等のトピック）
    category 1: 非負面
    """
    result_df = source_df.copy()
    topic_id_list = list(topics)

    valid_mask = result_df["userInput"].fillna("").str.strip().str.len() > 0
    valid_indices = result_df.index[valid_mask].tolist()

    result_df["topic_id"] = None
    for idx, tid in zip(valid_indices[:len(topic_id_list)], topic_id_list):
        result_df.at[idx, "topic_id"] = tid

    all_seed_keywords = set()
    for seed_group in SEED_TOPICS:
        all_seed_keywords.update(seed_group)

    negative_topic_ids = set()
    topic_info = topic_model.get_topic_info()
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue
        topic_keywords = topic_model.get_topic(topic_id)
        if topic_keywords:
            keyword_list = [kw for kw, _ in topic_keywords]
            for seed_group in SEED_TOPICS:
                matches = sum(1 for kw in seed_group if kw in keyword_list)
                if matches >= topic_match_threshold:
                    negative_topic_ids.add(topic_id)
                    print(f"    負面トピック特定: Topic {topic_id} ({matches}キーワード一致)")
                    break

    negative_user_ids = set()
    for idx, row in result_df.iterrows():
        user_text = str(row.get("userInput", ""))
        if not user_text.strip():
            continue
        is_match, _ = check_pattern_combos(user_text)
        if is_match:
            negative_user_ids.add(row.get("userId"))
        if any(kw in user_text for kw in all_seed_keywords):
            negative_user_ids.add(row.get("userId"))

    negative_indices = set()
    all_rows = list(result_df.iterrows())
    for i, (idx, row) in enumerate(all_rows):
        user_text = str(row.get("userInput", ""))
        if not user_text.strip():
            continue

        context_texts = [user_text]
        for j in range(max(0, i-2), i):
            prev_text = str(all_rows[j][1].get("userInput", ""))
            if prev_text.strip():
                context_texts.append(prev_text)

        combined = " ".join(context_texts)
        is_match, pattern_name = check_pattern_combos(combined)
        if is_match and pattern_name in ("ロマンス詐欺", "海外金銭"):
            trigger_keywords = ["お金", "貸", "送金", "投資", "融資"]
            if any(kw in user_text for kw in trigger_keywords):
                negative_indices.add(idx)

    def assign_category(row):
        user_text = str(row.get("userInput", ""))

        if not user_text.strip():
            return 1

        if row.name in negative_indices:
            return 0

        is_pattern_match, pattern_name = check_pattern_combos(user_text)
        if is_pattern_match:
            return 0

        if any(kw in user_text for kw in all_seed_keywords):
            return 0

        tid = row.get("topic_id")
        if tid is not None and tid in negative_topic_ids:
            return 0

        return 1

    result_df["category"] = result_df.apply(assign_category, axis=1)

    data_dir = config.DATA_DIR

    all_path = data_dir / "2category_all.csv"
    result_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"\n  → 2カテゴリ（全件）: {all_path}  ({len(result_df)}件)")

    print(f"\n  ■ category 0 のセッション例:")
    matched = result_df[result_df["category"] == 0].head(15)
    for _, row in matched.iterrows():
        text = str(row["userInput"])[:60]
        print(f"    [Topic {row['topic_id']}] {text}...")

    cat_counts = result_df["category"].value_counts()
    print(f"\n  ■ サマリー:")
    print(f"    category 0（負面）:   {cat_counts.get(0, 0)}件")
    print(f"    category 1（非負面）: {cat_counts.get(1, 0)}件")


# 主处理逻辑 / メイン処理
if __name__ == "__main__":
    output_dir = config.DATA_DIR / "topic_modeling"
    temp_dir = Path(__file__).parent.parent / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("正在加载数据...")
    df_mochiko = pd.read_csv(config.DATA_DIR / "data_mochiko.csv")
    df_pen_sensei = pd.read_csv(config.DATA_DIR / "data_pen_sensei.csv")

    print(f"mochiko: {len(df_mochiko)} 行")
    print(f"pen_sensei: {len(df_pen_sensei)} 行")

    print(f"  合并全部用户输入进行话题建模")
    combined_df = pd.concat([df_mochiko, df_pen_sensei], ignore_index=True)
    all_input_texts = combined_df["userInput"].fillna("").tolist()
    run_topic_modeling(
        texts=all_input_texts,
        dataset_name="combined",
        text_type="userInput",
        output_dir=output_dir,
        source_df=combined_df,
    )

    print(f"  话题建模完成！")
    print(f"  结果输出目录: {output_dir}")
