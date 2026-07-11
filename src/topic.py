import sys
import os
import json
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
from sklearn.cluster import KMeans
from umap import UMAP

warnings.filterwarnings("ignore")

# 加载种子话题配置 / シードトピック設定の読み込み
SEED_TOPICS_PATH = Path(__file__).parent / "seed_topics.json"
with open(SEED_TOPICS_PATH, "r", encoding="utf-8") as f:
    SEED_CONFIG = json.load(f)

# BERTopic 用的种子话题（用于引导聚类）
BERTOPIC_SEED_TOPICS = [
    ["離婚", "別れたい", "別居", "親権", "離婚届"],
    ["浮気", "不倫", "愛人", "二股", "浮気相手"],
    ["詐欺", "騙す", "騙されて", "フィッシング", "退役軍人"],
    ["DV", "暴力", "殴", "暴言", "無視", "冷たい"],
    ["死にたい", "消えたい", "自殺", "死のう", "終わらせたい"],
    ["流産", "死産", "妊娠中絶", "中絶"],
    ["赤ちゃん", "授乳", "母乳", "おっぱい", "育児"],
    ["眠れない", "食欲", "頭痛", "疲労", "産後"],
]

# 二分类用的负面关键词
NEGATIVE_KEYWORDS = SEED_CONFIG["negative_keywords"]

# 设置 — 本地模型优先 / 設定 — ローカルモデル優先
LOCAL_MODEL_PATH = config.MODELS_DIR / "ruri-v3-310m"
MODEL_NAME = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "cl-nagoya/ruri-v3-310m"
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

# 停用词表：过滤极其常见的泛化词 / ストップワード：一般的すぎる単語を除外
STOPWORDS = {
    # 极其常见的泛化动词/形容词
    "御座る", "分かる", "言う", "無い", "有る", "居る", "為る", "呉れる",
    "思う", "出る", "来る", "行く", "見る", "良い", "悪い",
    # 敬语形式
    "下さる", "頂く", "致す",
    # 助词/助动词（虽然已通过 POS 过滤，但某些形式可能漏过）
    "ます", "です", "た", "て", "で",
    # 其他高频泛化词
    "成る", "置く", "付く", "持つ", "入る", "使う", "知る", "話す",
    # fugashi 分词伪影词
    "仕舞う", "れる", "られる", "ある", "いる", "する", "なる",
    "える", "やすい", "にくい", "方", "遣る", "凄い", "旨い", "レン",
}

# パターンベース検出用のキーワード組み合わせ（2カテゴリ分類用）
# 負面カテゴリ: 詐欺・浮気・DV・離婚等の极端ケースのみ
PATTERN_COMBOS = [
    # 離婚別離パターン（より严格的な条件）
    {
        "name": "離婚別離",
        "required": [["離婚", "別居", "親権", "離婚届", "調停", "弁護士", "離婚したい", "離婚する"]],
        "context": ["夫", "妻", "旦那", "主人", "パートナー", "彼氏", "彼女"],
    },
    # 浮気不倫パターン（より严格的な条件）
    {
        "name": "浮気不倫",
        "required": [["浮気", "不倫", "愛人", "二股", "不貞", "浮気された", "不貞行為"]],
        "context": ["夫", "妻", "旦那", "主人", "彼", "パートナー", "彼氏", "彼女"],
    },
    # 夫以外の幸せパターン（夫以外の人との関係性を示唆）
    {
        "name": "夫以外の幸せ",
        "required": [["夫以外", "他の人", "別の男性", "別の人の"]],
        "context": ["幸せ", "恋", "気持ち", "好き", "愛"],
    },
    # 夫隠しパターン（夫に隠す・見つかると怖い → 浮気・秘密の示唆）
    {
        "name": "夫隠し",
        "required": [["夫に見つかったら怖い", "見つかったら怖い", "夫に内緒", "夫にバレると怖い", "隠してる", "隠してた"]],
        "context": [],
    },
    # 外国人詐欺パターン（外国の友人・知り合いで夫より信頼 → ロマンス詐欺示唆）
    {
        "name": "外国人詐欺",
        "required": [["外国", "外国人", "海外の人", "フィリピン", "アメリカ", "イギリス"]],
        "context": ["信頼", "夫より", "優しい", "理解", "相談", "援助"],
    },
    # 退役軍人詐欺パターン（SNSなしでも検出）
    {
        "name": "退役軍人詐欺",
        "required": [["退役軍人", "軍人"]],
        "context": ["貸", "お金", "返す", "会いに来る", "送金"],
    },
    # 流産妊娠問題パターン（より严格的な条件）
    {
        "name": "流産妊娠問題",
        "required": [["流産", "死産", "中絶", "妊娠中絶", "流産した", "流産の心配"]],
        "context": [],
    },
    # ロマンス詐欺パターン（SNS + 海外 + お金、もしくは退役軍人+お金）
    {
        "name": "ロマンス詐欺",
        "required": [
            ["Facebook", "facebook", "フェイスブック", "SNS", "LINE", "インスタ", "Instagram", "Twitter", "ツイッター", "マッチング", "matching"],
            ["退役軍人", "軍人"],
        ],
        "context": ["海外", "外国", "貸", "送金", "お金", "投資", "返す", "詐欺", "騙された", "会いに来る"],
    },
    # 海外関係 + お金のやり取り（SNSなしでも検出）
    {
        "name": "海外金銭",
        "required": [["海外", "外国"]],
        "context": ["貸", "送金", "お金", "投資", "返す", "資金", "融資", "知り合", "出会", "詐欺", "騙された"],
    },
    # 詐欺パターン（登録だけでお金、ベビーモデル、掲載料等）
    {
        "name": "詐欺",
        "required": [["詐欺", "フィッシング", "登録するだけで", "月々数万", "掲載料", "ベビーモデル", "詐欺に遭った", "詐欺被害"]],
        "context": [],
    },
    # 金銭トラブルパターン（融資・投資・借金などの詐欺的勧誘）
    {
        "name": "金銭トラブル",
        "required": [["融資", "資金を持ちかけ", "投資", "借りる", "貸す", "借金"]],
        "context": ["多額", "大金", "ほとんど", "消える", "怪しい", "詐欺", "騙", "不安"],
    },
    # DV暴力パターン（より严格的な条件）
    {
        "name": "DV暴力",
        "required": [["DV", "暴力", "暴行", "傷害", "殴", "蹴", "脅", "叩", "暴言", "怒鳴", "ドメスティックバイオレンス", "警察沙汰"]],
        "context": ["夫", "妻", "旦那", "彼", "パートナー", "家族", "彼氏", "彼女"],
    },
    # モラハラパターン（より严格的な条件）
    {
        "name": "モラハラ",
        "required": [["モラハラ", "モラルハラスメント", "パワハラ", "精神的な虐待"]],
        "context": [],
    },
    # 自殺・自傷パターン（より严格的な条件）
    {
        "name": "自殺自傷",
        "required": [["死にたい", "消えたい", "自殺", "死のう", "生きる意味", "いらない", "終わらせたい", "死ねない", "死なない", "死にきれ", "自殺したい", "命を絶ちたい"]],
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
            # 清理 lemma：去除英文后缀（如 toilet-toilet -> toilet）
            if "-" in lemma:
                lemma = lemma.split("-")[0]
            # 过滤停用词
            if lemma not in STOPWORDS:
                tokens.append(lemma)
    return " ".join(tokens)


# 后处理函数：基于统计距离过滤异常文档
def verify_and_reassign(
    topics: list[int],
    embeddings: np.ndarray,
    valid_texts: list[str],
    topic_model,
    z_threshold: float = 2.0,
) -> list[int]:
    """
    基于统计距离过滤异常文档。
    对每个 topic，计算文档到中心点的距离。
    如果距离超过 mean + z_threshold * std，标记为 -1。
    """
    from sklearn.metrics.pairwise import cosine_distances

    new_topics = list(topics)
    marked_outlier = 0

    # 计算每个 topic 的中心点
    topic_centroids = {}
    for tid in set(topics):
        if tid == -1:
            continue
        indices = [i for i, t in enumerate(topics) if t == tid]
        if indices:
            topic_centroids[tid] = np.mean(embeddings[indices], axis=0)

    # 计算每个 topic 的距离统计
    topic_stats = {}
    for tid, centroid in topic_centroids.items():
        indices = [i for i, t in enumerate(topics) if t == tid]
        if len(indices) < 2:
            continue
        dists = cosine_distances(embeddings[indices], [centroid]).flatten()
        topic_stats[tid] = {
            "mean": np.mean(dists),
            "std": np.std(dists),
        }

    for i, (tid, emb) in enumerate(zip(topics, embeddings)):
        if tid == -1:
            continue

        if tid not in topic_stats:
            continue

        dist = cosine_distances([emb], [topic_centroids[tid]])[0][0]
        mean = topic_stats[tid]["mean"]
        std = topic_stats[tid]["std"]

        # 如果距离超过 mean + z_threshold * std，标记为 -1
        if std > 0 and dist > mean + z_threshold * std:
            new_topics[i] = -1
            marked_outlier += 1

    print(f"  验证：标记为 -1 {marked_outlier} 篇（距离 > mean + {z_threshold} * std）")
    return new_topics


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

    # 识别短文本（tokenized 后少于 2 个词），这些文本不参与聚类
    short_threshold = 2
    short_indices = [i for i, t in enumerate(tokenized_texts) if len(t.split()) < short_threshold]
    long_indices = [i for i in range(len(valid_texts)) if i not in short_indices]
    print(f"  短文本（< {short_threshold} 词）: {len(short_indices)} 篇，不参与聚类")
    print(f"  参与聚类文本: {len(long_indices)} 篇")

    # 只对长文本进行聚类
    long_texts = [valid_texts[i] for i in long_indices]
    long_tokenized = [tokenized_texts[i] for i in long_indices]

    print(f"正在生成 BERT 嵌入向量（{MODEL_NAME}）...")
    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embeddings = embedding_model.encode(
        long_texts,
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
        max_df=0.85,
        min_df=2,
        ngram_range=(1, 2),
    )

    from bertopic.representation import MaximalMarginalRelevance

    representation_model = MaximalMarginalRelevance(diversity=0.5)

    # 使用 KMeans 聚类
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
        seed_topic_list=BERTOPIC_SEED_TOPICS,
        representation_model=representation_model,
    )

    topics_long, probs_long = topic_model.fit_transform(
        long_tokenized,
        embeddings=embeddings,
    )

    # 后处理：基于 embedding 相似度验证并重新分配
    topics_long = verify_and_reassign(
        topics_long,
        embeddings,
        long_texts,
        topic_model,
        z_threshold=1.5,
    )

    # 合并结果：短文本标记为 -1，长文本使用聚类结果
    topics = [-1] * len(valid_texts)
    probs = None
    for i, topic_idx in enumerate(long_indices):
        topics[topic_idx] = topics_long[i]
    if probs_long is not None:
        if probs_long.ndim == 1:
            probs_long = probs_long.reshape(-1, 1)
        probs = np.zeros((len(valid_texts), probs_long.shape[1]))
        for i, topic_idx in enumerate(long_indices):
            probs[topic_idx] = probs_long[i]

    print("正在保存结果...")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_name}_{text_type}"

    # 获取 BERTopic 原始话题信息
    topic_info = topic_model.get_topic_info()

    n_valid_topics = len([t for t in topic_info["Topic"].values if t != -1])
    n_outliers = len([t for t in topics if t == -1])
    print(f"\n  ■ 结果摘要:")
    print(f"    话题数: {n_valid_topics}")
    print(f"    离群文档数: {n_outliers} / {len(topics)}")
    print(f"    话题占比: {(len(topics) - n_outliers) / len(topics) * 100:.1f}%")

    # 保存 doc_topics.csv
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

    # 保存话题信息
    from collections import Counter
    topic_counts = Counter(topics)
    topic_info_list = []
    for topic_id in sorted(topic_counts.keys()):
        if topic_id == -1:
            name = f"{topic_id}_outlier"
        else:
            kws = topic_model.get_topic(topic_id)
            name = f"{topic_id}_{kws[0][0]}_{kws[1][0]}_{kws[2][0]}" if kws else f"{topic_id}_unknown"
        topic_info_list.append({
            "Topic": topic_id,
            "Count": topic_counts[topic_id],
            "Name": name,
        })
    topic_info_df = pd.DataFrame(topic_info_list)
    topic_info_path = output_dir / f"{prefix}_topic_info.csv"
    topic_info_df.to_csv(topic_info_path, index=False, encoding="utf-8-sig")
    print(f"  → トピック情報: {topic_info_path}")

    # 保存话题关键词
    all_keywords = []
    for topic_id in sorted(topic_counts.keys()):
        if topic_id == -1:
            continue
        kws = topic_model.get_topic(topic_id)
        if kws:
            for word, score in kws:
                all_keywords.append({
                    "topic_id": topic_id,
                    "keyword": word,
                    "score": score,
                })
    keywords_df = pd.DataFrame(all_keywords)
    keywords_path = output_dir / f"{prefix}_topic_keywords.csv"
    keywords_df.to_csv(keywords_path, index=False, encoding="utf-8-sig")
    print(f"  → 话题关键词: {keywords_path}")

    # 话题统计
    from collections import Counter
    topic_counts = Counter(topics)
    n_outliers_after = topic_counts.get(-1, 1)
    print(f"\n  ■ 结果摘要:")
    print(f"    话题数: {n_valid_topics}")
    print(f"    离群文档数: {n_outliers_after} / {len(topics)}")
    print(f"    话题占比: {(len(topics) - n_outliers_after) / len(topics) * 100:.1f}%")

    print(f"\n  ■ 话题一览:")
    for topic_id in sorted(topic_counts.keys()):
        if topic_id == -1:
            continue
        print(f"    Topic {topic_id} ({topic_counts[topic_id]}件)")

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
):
    """
    2カテゴリ分類（フルパイプライン）。

    分類ロジック（優先度順）：
    1. スライディングウインドウ（現在+過去2メッセージ）によるパターンマッチ
    2. 1メッセージ単位のキーワード組み合わせマッチ（PATTERN_COMBOS）
    3. 負面キーワードの一致（NEGATIVE_KEYWORDS）
    4. 上記に該当しない場合 → category 1
    """
    result_df = source_df.copy()
    topic_id_list = list(topics)

    valid_mask = result_df["userInput"].fillna("").str.strip().str.len() > 0
    valid_indices = result_df.index[valid_mask].tolist()

    result_df["topic_id"] = None
    for idx, tid in zip(valid_indices[:len(topic_id_list)], topic_id_list):
        result_df.at[idx, "topic_id"] = tid

    # 负面キーワードの集合
    all_negative_keywords = set(NEGATIVE_KEYWORDS)

    # パターンベース検出（スライディングウインドウ）
    negative_pattern_indices = set()
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
            trigger_keywords = ["お金", "貸", "送金", "投資", "融資", "返す", "詐欺", "騙"]
            if any(kw in user_text for kw in trigger_keywords):
                negative_pattern_indices.add(idx)

    def assign_category(row):
        user_text = str(row.get("userInput", ""))

        if not user_text.strip():
            return 1

        # 1. パターンベース直接マッチ（強シグナル）
        is_pattern_match, pattern_name = check_pattern_combos(user_text)
        if is_pattern_match:
            return 0

        # 2. コンテキスト検出済みパターン（ロマンス詐欺・海外金銭）
        if row.name in negative_pattern_indices:
            return 0

        # 3. 負面キーワード直接マッチ（テキストにキーワードがあれば分類）
        if any(kw in user_text for kw in all_negative_keywords):
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

    print(f"  话题建模完成！")
    print(f"  结果输出目录: {output_dir}")
