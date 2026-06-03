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
RANDOM_SEED = 42
MIN_TOPIC_SIZE = 2  # 针对小型数据集设为较小值 / 小さいデータセット用に小さめに設定

# 如果 CUDA 可用则使用 GPU / CUDA が利用可能なら GPU を使用
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备 / 使用デバイス: {DEVICE}")

# 初始化 fugashi 分词器 / fugashi タガーの初期化
tagger = Tagger()

# 词性过滤：仅保留名词、动词、形容词 / 品詞フィルタ：名詞・動詞・形容詞のみを残す
KEEP_POS = {"名詞", "動詞", "形容詞"}
PUNCTUATION_POS = {"補助記号", "記号", "助詞", "助動詞", "接続詞", "感動詞", "接頭詞", "接尾詞"}

# シードトピック（BERTopic 半監督モード用）
SEED_TOPICS = [
    # 離婚・別離
    ["離婚", "別れ", "別れたい", "別居", "親権"],
    # 浮気・不倫
    ["浮気", "不倫", "愛人", "二股", "浮気相手", "不倫相手", "裏切り", "裏切られた"],
    # 流産・妊娠問題
    ["流産", "妊娠中絶", "死産"],
    # 詐欺・被害
    ["詐欺", "被害", "騙す", "騙された", "フィッシング"],
    # DV・暴力
    ["DV", "暴力", "殴る", "威圧", "脅す", "脅された"],
    # その他の困難
    ["虐待", "育児ノイローゼ", "育児放棄", "夫が嫌い", "もう無理", "限界", "死にたい"],
]

# パターンベース検出用のキーワード組み合わせ（2カテゴリ分類用）
# 複数キーワードが同時出現する場合に負面と判定
# required: 必須キーワード（少なくとも1つマッチ）
# context: 文脈キーワード（required と同時出現で負面判定）
# risk: リスクキーワード（required + context + risk で最終判定）
PATTERN_COMBOS = [
    # 離婚別離パターン
    {
        "name": "離婚別離",
        "required": [["離婚", "別れ", "別居", "親権", "離縁", "離婚届", "調停", "弁護士"]],
        "context": ["夫", "妻", "旦那", "主人", "パートナー", "相手", "もう無理", "限界"],
        "risk": ["耐え", "したい", "考え", "決意", "覚悟", "本当", "気持ち"],
    },
    # 浮気不倫パターン
    {
        "name": "浮気不倫",
        "required": [["浮気", "不倫", "愛人", "二股", "不貞"]],
        "context": ["夫", "妻", "旦那", "主人", "彼", "パートナー"],
        "risk": ["疑わ", "怪しい", "バレ", "証拠", "見つけ", "気づ", "確信", "本当"],
    },
    # 流産妊娠問題パターン
    {
        "name": "流産妊娠問題",
        "required": [["流産", "死産", "中絶", "妊娠中絶", "人工妊娠中絶"]],
        "context": ["辛い", "悲しい", "ショック", "失う", "赤ちゃん", "胎児"],
        "risk": ["妊娠", "出産", "子供", "体", "心", "病院", "処置", "痛い", "怖い", "涙", "落ち込"],
    },
    # ロマンス詐欺パターン
    {
        "name": "ロマンス詐欺",
        "required": [["Facebook", "facebook", "フェイスブック", "SNS", "LINE", "インスタ", "Instagram", "Twitter", "ツイッター", "sns"]],
        "context": ["海外", "外国", "知り合", "出会い", "メッセージ"],
        "risk": ["融資", "送金", "お金", "投資", "貯金", "ローン", "借金", "資金", "送っ", "振込"],
    },
    # 投資詐欺パターン
    {
        "name": "投資詐欺",
        "required": [["ベンチャー", "起業", "投資", "ビジネス", "儲け", "儲かる", "副業", "FX", "仮想通貨", "暗号資産", "資金", "出資"]],
        "context": ["誘わ", "持ちかけ", "話", "紹介", "頼ま", "言われ", "誘い", "勧め"],
        "risk": ["融資", "貯金", "多額", "借入", "ローン", "借金", "消える", "減る", "危険", "怪しい", "不安", "心配", "お金", "金", "額", "万円"],
    },
    # DV暴力パターン
    {
        "name": "DV暴力",
        "required": [["DV", "暴力", "暴行", "傷害", "殴", "蹴", "脅", "叩", "暴言", "怒鳴", "つつか", "ひっぱた"]],
        "context": ["夫", "妻", "旦那", "彼", "パートナー", "家族"],
        "risk": ["怖い", "危険", "逃げ", "助け", "警察", "痛い", "怪我", "病院"],
    },
    # モラハラパターン
    {
        "name": "モラハラ",
        "required": [["モラハラ", "モラルハラスメント", "パワハラ", "精神的虐待", "嫌味", "陰口", "無視"]],
        "context": ["夫", "妻", "旦那", "彼", "職場", "上司"],
        "risk": ["辛い", "苦しい", "追い詰め", "限界", "もう無理", "疲れた", "鬱"],
    },
    # 浮気不倫の間接的表現パターン
    {
        "name": "浮気示唆",
        "required": [["知り合", "出会い", "会い", "連絡", "メッセージ"]],
        "context": ["海外", "外国", "異性", "男性", "女性", "彼"],
        "risk": ["夫より", "信頼", "嬉しい", "楽しい", "優し", "気持", "秘密", "夢中"],
    },
]


def tokenize_with_fugashi(text: str) -> str:
    """
    使用 fugashi 进行形态素解析，
    将名词、动词、形容词的原形以空格分隔返回。
    作为 BERTopic 的 CountVectorizer 的预处理。

    fugashi を使って形態素解析し、
    名詞・動詞・形容詞の原形をスペース区切りで返す。
    BERTopic の CountVectorizer に渡すための前処理。
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = []
    for word in tagger(text):
        pos = word.feature.pos1
        if pos in PUNCTUATION_POS:
            continue
        if pos in KEEP_POS:
            # 如果有原形(lemma)则使用原形，否则使用表层形 / 原形（lemma）があれば原形を、なければ表層形を使用
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
    与えられたテキストリストに対してトピックモデリングを実行する。

    Args:
        texts: 分析对象的文本列表 / 分析対象のテキストリスト
        dataset_name: 数据集名称（例：'mochiko'） / データセット名（例: 'mochiko'）
        text_type: 文本类型（例：'userInput' 或 'replyText'） / テキスト種別（例: 'userInput' or 'replyText'）
        output_dir: 结果输出目录 / 結果出力ディレクトリ
        n_topics: 指定话题数（None 则自动决定） / トピック数の指定（None なら自動決定）
    """
    print(f"  话题建模: {dataset_name} / {text_type}")
    print(f"  文本数: {len(texts)} / テキスト数: {len(texts)}")

    # 排除空文本 / 空テキストを除外
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(valid_texts) < 3:
        print(f"  ⚠ 有效文本仅有{len(valid_texts)}条，跳过处理 / 有効テキストが{len(valid_texts)}件しかないためスキップ")
        return

    print(f"  有效文本数: {len(valid_texts)} / 有効テキスト数: {len(valid_texts)}")

    # 使用 fugashi 预处理（用于 BERTopic 的 CountVectorizer）/ fugashi で前処理（BERTopic の CountVectorizer 用）
    print("  [1/4] 正在使用 fugashi 进行形态素解析与预处理... / fugashi による形態素解析・前処理中...")
    tokenized_texts = [tokenize_with_fugashi(t) for t in valid_texts]

    # 生成 BERT 嵌入向量 / BERT埋め込みの生成
    print(f"  [2/4] 正在生成 BERT 嵌入向量（{MODEL_NAME}）... / BERT埋め込み生成中...")
    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embeddings = embedding_model.encode(
        valid_texts,
        show_progress_bar=True,
        batch_size=64 if DEVICE == "cuda" else 16,
    )
    print(f"  嵌入维度: {embeddings.shape} / 埋め込み次元: {embeddings.shape}")

    # 使用 BERTopic 执行话题建模 / BERTopic によるトピックモデリング
    print("  正在执行 BERTopic 话题建模... / BERTopic によるトピックモデリング実行中...")

    # UMAP 的可重复性设置 / UMAP の再現性設定
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    # CountVectorizer: 使用 fugashi 预处理后的文本 / fugashi で前処理済みのテキストを使用
    vectorizer = CountVectorizer()

    topic_model = BERTopic(
        embedding_model=None,  # 使用预先计算的嵌入向量 / 事前計算済み埋め込みを使用
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

    # 保存结果 / 結果の保存
    print("  [4/4] 正在保存结果... / 結果保存中...")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset_name}_{text_type}"

    # 保存话题信息 / トピック情報の保存
    topic_info = topic_model.get_topic_info()
    topic_info_path = output_dir / f"{prefix}_topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False, encoding="utf-8-sig")
    print(f"  → トピック情報: {topic_info_path}")

    # 保存各话题的关键词 / 各トピックのキーワード保存
    all_keywords = []
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue  # 跳过离群话题 / アウトライアートピックはスキップ
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

    # 保存文档-话题分配结果 / ドキュメント-トピック割り当ての保存
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

    # 显示话题数摘要 / トピック数のサマリー表示
    n_valid_topics = len([t for t in topic_info["Topic"].values if t != -1])
    n_outliers = len([t for t in topics if t == -1])
    print(f"\n  ■ 结果摘要 / 結果サマリー:")
    print(f"    话题数: {n_valid_topics} / トピック数: {n_valid_topics}")
    print(f"    离群文档数: {n_outliers} / {len(topics)} / アウトライアードキュメント数: {n_outliers} / {len(topics)}")
    print(f"    话题占比: {(len(topics) - n_outliers) / len(topics) * 100:.1f}% / トピック割合: {(len(topics) - n_outliers) / len(topics) * 100:.1f}%")

    # 显示话题一览 / トピック一覧の表示
    print(f"\n  ■ 话题一览 / トピック一覧:")
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue
        count = topic_info.loc[topic_info["Topic"] == topic_id, "Count"].values[0]
        name = topic_info.loc[topic_info["Topic"] == topic_id, "Name"].values[0]
        print(f"    Topic {topic_id} ({count}件): {name}")

    # 2カテゴリ分類CSVを生成
    if source_df is not None:
        print(f"\n  [追加] 2カテゴリ分類を作成中...")
        classify_by_keywords(topic_model, valid_texts, topics, source_df, output_dir, prefix)

    return topic_model


def check_pattern_combos(text: str) -> tuple[bool, str | None]:
    """
    テキスト内でパターンベースの組み合わせを検出する。

    Args:
        text: チェック対象のテキスト

    Returns:
        (is_match, pattern_name): マッチした場合 (True, パターン名)、否则 (False, None)
    """
    if not isinstance(text, str) or not text.strip():
        return False, None

    for pattern in PATTERN_COMBOS:
        # required グループのうち少なくとも1つがマッチするか
        required_match = False
        for req_group in pattern["required"]:
            if any(kw in text for kw in req_group):
                required_match = True
                break
        if not required_match:
            continue

        # context グループのうち少なくとも1つがマッチするか
        context_match = any(kw in text for kw in pattern["context"])
        if not context_match:
            continue

        # required + context でマッチした場合、risk はオプション
        # risk が存在し、かつマッチしない場合はスキップしない（required + context だけで判定）
        # ただし、risk が存在し、マッチする場合は信頼度が上がる

        # 全条件マッチ
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

    Args:
        topic_match_threshold: トピックを負面と判定するシードキーワード一致数の閾値（デフォルト: 2）
    """
    # valid_texts と topics から source_df に topic_id を割り当て
    result_df = source_df.copy()
    topic_id_list = list(topics)

    valid_mask = result_df["userInput"].fillna("").str.strip().str.len() > 0
    valid_indices = result_df.index[valid_mask].tolist()

    result_df["topic_id"] = None
    for idx, tid in zip(valid_indices[:len(topic_id_list)], topic_id_list):
        result_df.at[idx, "topic_id"] = tid

    # BERTopic 半監督モードで2カテゴリ分類用のモデルを訓練
    print("    BERTopic 半監督モードで2カテゴリ分類用モデルを訓練中...")
    tokenized_texts = [tokenize_with_fugashi(t) for t in valid_texts]

    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embeddings = embedding_model.encode(
        valid_texts,
        show_progress_bar=False,
        batch_size=64 if DEVICE == "cuda" else 16,
    )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    vectorizer = CountVectorizer()

    # 2カテゴリ分類用の BERTopic モデル（シードトピック使用）
    category_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        min_topic_size=3,
        seed_topic_list=SEED_TOPICS,
        verbose=False,
        language="japanese",
    )

    category_topics, category_probs = category_model.fit_transform(
        tokenized_texts,
        embeddings=embeddings,
    )

    # シードトピックに対応するトピックIDを特定
    negative_topic_ids = set()
    all_seed_keywords = set()
    for seed_group in SEED_TOPICS:
        all_seed_keywords.update(seed_group)

    topic_info = category_model.get_topic_info()
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            continue
        topic_keywords = category_model.get_topic(topic_id)
        if topic_keywords:
            keyword_list = [kw for kw, _ in topic_keywords]
            for seed_group in SEED_TOPICS:
                matches = sum(1 for kw in seed_group if kw in keyword_list)
                if matches >= topic_match_threshold:
                    negative_topic_ids.add(topic_id)
                    print(f"    負面トピック特定: Topic {topic_id} ({matches}キーワード一致)")
                    break

    # カテゴリ分類（BERTopic半監督 + パターンマッチング + 直接キーワード）
    def assign_category(row):
        # テキストにシードキーワードが直接含まれるか確認（userInput + replyText）
        text_parts = [
            str(row.get("userInput", "")),
            str(row.get("replyText", "")),
        ]
        combined_text = " ".join(text_parts)

        # 直接キーワードマッチング
        if combined_text and any(kw in combined_text for kw in all_seed_keywords):
            return 0

        # パターンベース組み合わせ検出
        is_pattern_match, pattern_name = check_pattern_combos(combined_text)
        if is_pattern_match:
            return 0

        # BERTopic半監督モデルによる分類
        # source_df の valid_indices に対応する category_topics を取得
        row_idx = result_df.index.get_loc(row.name)
        if row_idx < len(category_topics):
            tid = category_topics[row_idx]
            if tid in negative_topic_ids:
                return 0

        # トピックベース分類（元のトピックIDに基づく）
        tid = row["topic_id"]
        if pd.isna(tid) or tid is None:
            return 1
        tid = int(tid)
        if tid == -1:
            return 1
        return 1

    result_df["category"] = result_df.apply(assign_category, axis=1)

    # data ディレクトリに保存
    data_dir = config.DATA_DIR

    # 全件保存
    all_path = data_dir / f"2category_all.csv"
    result_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"\n  → 2カテゴリ（全件）: {all_path}")

    # interrupt / current に分割保存
    for reply_type, label in [("ReplyInterruptPersona", "interrupt"), ("ReplyCurrentPersona", "current")]:
        subset = result_df[result_df["replyType"] == reply_type]
        if subset.empty:
            print(f"  → {label}: データなし、スキップ")
            continue
        path = data_dir / f"2category_{label}.csv"
        subset.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  → 2カテゴリ（{label}）: {path}  ({len(subset)}件)")

    # マッチしたセッション例
    print(f"\n  ■ category 0 のセッション例:")
    matched = result_df[result_df["category"] == 0].head(15)
    for _, row in matched.iterrows():
        text = str(row["userInput"])[:60]
        print(f"    [Topic {row['topic_id']}] {text}...")

    # サマリー
    cat_counts = result_df["category"].value_counts()
    print(f"\n  ■ サマリー:")
    print(f"    category 0（負面）:   {cat_counts.get(0, 0)}件")
    print(f"    category 1（非負面）: {cat_counts.get(1, 0)}件")


# 主处理逻辑 / メイン処理
if __name__ == "__main__":
    output_dir = config.DATA_DIR / "topic_modeling"

    # 加载数据 / データ読み込み
    print("正在加载数据... / データ読み込み中...")
    df_mochiko = pd.read_csv(config.DATA_DIR / "data_mochiko.csv")
    df_pen_sensei = pd.read_csv(config.DATA_DIR / "data_pen_sensei.csv")

    print(f"mochiko: {len(df_mochiko)} 行")
    print(f"pen_sensei: {len(df_pen_sensei)} 行")

    # mochiko 的话题建模 / mochiko のトピックモデリング
    # 用户输入 / ユーザー入力
    mochiko_input_texts = df_mochiko["userInput"].fillna("").tolist()
    run_topic_modeling(
        texts=mochiko_input_texts,
        dataset_name="mochiko",
        text_type="userInput",
        output_dir=output_dir,
    )

    # 机器人回复 / ボット応答
    mochiko_reply_texts = df_mochiko["replyText"].fillna("").tolist()
    run_topic_modeling(
        texts=mochiko_reply_texts,
        dataset_name="mochiko",
        text_type="replyText",
        output_dir=output_dir,
    )

    # pen_sensei 的话题建模 / pen_sensei のトピックモデリング
    # 用户输入 / ユーザー入力
    pen_input_texts = df_pen_sensei["userInput"].fillna("").tolist()
    run_topic_modeling(
        texts=pen_input_texts,
        dataset_name="pen_sensei",
        text_type="userInput",
        output_dir=output_dir,
    )

    # 机器人回复 / ボット応答
    pen_reply_texts = df_pen_sensei["replyText"].fillna("").tolist()
    run_topic_modeling(
        texts=pen_reply_texts,
        dataset_name="pen_sensei",
        text_type="replyText",
        output_dir=output_dir,
    )

    # 合并两者用户输入进行话题建模 / 両者を結合したユーザー入力のトピックモデリング
    print(f"  合并两者的用户输入进行话题建模 / 両者結合ユーザー入力のトピックモデリング")
    all_input_texts = mochiko_input_texts + pen_input_texts
    combined_source_df = pd.concat([df_mochiko, df_pen_sensei], ignore_index=True)
    run_topic_modeling(
        texts=all_input_texts,
        dataset_name="combined",
        text_type="userInput",
        output_dir=output_dir,
        source_df=combined_source_df,
    )

    print(f"  话题建模完成！ / トピックモデリング完了！")
    print(f"  结果输出目录: {output_dir} / 結果出力先: {output_dir}")
