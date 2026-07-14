"""
2カテゴリ分類モジュール（半教師学習方式）
cat0: 離婚、流産、不倫、詐欺、自殺などの負面・危険対話
cat1: その他（一般的な育児相談・生活問題など）
- ruri-v3 プレフィックス: 「トピック: 」を使用（分類・クラスタリング用）
"""
import sys
import warnings
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib

sys.path.insert(0, str(Path(__file__).parent))
import config

from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# 埋め込みモデル
LOCAL_MODEL_PATH = config.MODELS_DIR / "ruri-v3-310m"
MODEL_NAME = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "cl-nagoya/ruri-v3-310m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = config.TOPIC_RANDOM_SEED


def check_pattern_combos(text: str) -> tuple[bool, str | None]:
    """テキスト内でパターンベースの組み合わせを検出"""
    if not isinstance(text, str) or not text.strip():
        return False, None
    for pattern in config.PATTERN_COMBOS:
        required_match = any(any(kw in text for kw in req_group) for req_group in pattern["required"])
        if not required_match:
            continue
        if not pattern["context"]:
            return True, pattern["name"]
        if any(kw in text for kw in pattern["context"]):
            return True, pattern["name"]
    return False, None


def generate_seed_labels(df: pd.DataFrame) -> dict[int, int]:
    """キーワードマッチングでシードラベルを生成"""
    all_negative_keywords = set(config.NEGATIVE_KEYWORDS)
    seed_labels = {}

    for idx, row in df.iterrows():
        user_text = str(row.get("userInput", ""))
        if not user_text.strip():
            continue
        if any(kw in user_text for kw in all_negative_keywords):
            seed_labels[idx] = 0
            continue
        if check_pattern_combos(user_text)[0]:
            seed_labels[idx] = 0
            continue

    positive_keywords = [
        "赤ちゃん", "可愛い", "授乳", "母乳", "おっぱい", "育児",
        "眠れない", "寝る", "食欲", "頭痛", "疲労", "産後",
        "相談", "保健", "センター", "助産", "施設", "地域", "支援", "手伝う",
    ]

    for idx, row in df.iterrows():
        if idx in seed_labels:
            continue
        user_text = str(row.get("userInput", ""))
        if user_text.strip() and any(kw in user_text for kw in positive_keywords):
            seed_labels[idx] = 1

    topic_rules = {
        5: (None, 0),
        2: (None, 1),
        6: (None, 1),
        3: (["流産", "死産", "中絶"], 0),
        0: (["詐欺", "騙", "フィッシング", "登録するだけで", "ベビーモデル", "掲載料"], 0),
        1: (["離婚", "DV", "暴力", "死にたい", "消えたい", "自殺", "終わらせたい"], 0),
        4: (["DV", "暴力", "モラハラ", "パワハラ", "無視", "冷たい"], 0),
    }

    for idx, row in df.iterrows():
        if idx in seed_labels:
            continue
        user_text = str(row.get("userInput", ""))
        tid = row.get("topic_id")
        if not user_text.strip() or pd.isna(tid) or tid not in topic_rules:
            continue
        negative_kws, default_cat = topic_rules[tid]
        if negative_kws is None:
            seed_labels[idx] = default_cat
        elif any(kw in user_text for kw in negative_kws):
            seed_labels[idx] = 0
        else:
            seed_labels[idx] = 1

    return seed_labels


def run_classification():
    """2カテゴリ分類を実行（SVM-RBF 半教師学習）"""
    print("2カテゴリ分類（半教師学習 - SVM-RBF）")

    print("\n[1] データ読み込み...")
    df = pd.read_csv(config.DATA_DIR / "data_with_id.csv")
    doc_topics = pd.read_csv(config.DATA_DIR / "topic_modeling" / "combined_userInput_doc_topics.csv")
    print(f"  data_with_id: {len(df)} 行, doc_topics: {len(doc_topics)} 行")

    df["topic_id"] = doc_topics["topic_id"].values

    print(f"\n[2] 生成嵌入 ({MODEL_NAME})...")
    print(f"  プレフィックス: 「{config.TOPIC_EMBEDDING_PREFIX}」")
    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    valid_texts = df["userInput"].fillna("").tolist()
    prefixed_texts = [f"{config.TOPIC_EMBEDDING_PREFIX}{t}" for t in valid_texts]
    embeddings = embedding_model.encode(
        prefixed_texts,
        show_progress_bar=True,
        batch_size=config.TOPIC_EMBEDDING_BATCH_SIZE_CUDA if DEVICE == "cuda" else config.TOPIC_EMBEDDING_BATCH_SIZE_CPU,
    )
    print(f"  埋め込み次元: {embeddings.shape}")

    print("\n[3] シードラベル生成...")
    seed_labels = generate_seed_labels(df)
    n_cat0 = sum(1 for v in seed_labels.values() if v == 0)
    n_cat1 = sum(1 for v in seed_labels.values() if v == 1)
    print(f"  category 0: {n_cat0} 件, category 1: {n_cat1} 件")

    print("\n[4] SVM-RBF分類器訓練...")
    train_indices = list(seed_labels.keys())
    X_train = embeddings[train_indices]
    y_train = [seed_labels[i] for i in train_indices]

    clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=min(5, len(train_indices)), scoring="f1")
    print(f"  交差検証 F1: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    clf.fit(X_train, y_train)

    # モデル保存
    model_dir = config.MODELS_DIR / "negative_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "classifier.joblib")
    joblib.dump(seed_labels, model_dir / "seed_labels.joblib")
    print(f"  モデル保存: {model_dir / 'classifier.joblib'}")

    print("\n[5] 全文書予測...")
    all_probs = clf.predict_proba(embeddings)

    confidence_threshold = config.CLASSIFY_CONFIDENCE_THRESHOLD
    short_threshold = config.CLASSIFY_SHORT_THRESHOLD
    categories = []
    confidences = []

    for i, (idx, row) in enumerate(df.iterrows()):
        user_text = str(row.get("userInput", ""))
        tokenized = str(doc_topics.iloc[i].get("tokenized_text", ""))

        if not user_text.strip() or len(tokenized.split()) < short_threshold:
            categories.append(1)
            confidences.append(1.0)
            continue

        if idx in seed_labels:
            categories.append(seed_labels[idx])
            confidences.append(1.0)
            continue

        probs = all_probs[i]
        max_prob = max(probs)
        pred_class = probs.argmax()

        if max_prob >= confidence_threshold:
            categories.append(pred_class)
            confidences.append(max_prob)
        else:
            categories.append(1)
            confidences.append(max_prob)

    df["category"] = categories
    df["confidence"] = confidences

    print("\n[6] 結果統計...")
    cat_counts = df["category"].value_counts()
    n_high_conf = sum(1 for c in confidences if c >= 0.85)
    print(f"  category 0 (負面):   {cat_counts.get(0, 0)} 件")
    print(f"  category 1 (非負面): {cat_counts.get(1, 0)} 件")
    print(f"  高信頼度 (>= 0.85):  {n_high_conf} 件")

    print("\n[7] 結果保存...")
    all_path = config.DATA_DIR / "2category_all.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"  → {all_path}  ({len(df)}件)")

    print(f"\ncategory 0 セッション例:")
    for _, row in df[df["category"] == 0].head(10).iterrows():
        print(f"  [Topic {row['topic_id']}] ({row['confidence']:.2f}) {str(row['userInput'])[:50]}...")

    print(f"\ncategory 1 セッション例:")
    for _, row in df[df["category"] == 1].head(10).iterrows():
        print(f"  [Topic {row['topic_id']}] ({row['confidence']:.2f}) {str(row['userInput'])[:50]}...")

    return df


if __name__ == "__main__":
    run_classification()
