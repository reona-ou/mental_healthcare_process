"""
二分类模块：半监督学习方式
cat0: 离婚、流产、出轨、诈骗、自杀等负面或危险对话
cat1: 其它（一般健康咨询和生活问题等）

流程：
1. 读取 topic 建模结果和原始数据
2. 生成 BERT 嵌入
3. 使用关键词匹配生成高置信度种子标签
4. 训练逻辑回归分类器
5. 对所有文档预测，只对高置信度（> 0.85）的文档分类
"""
import sys
import json
import warnings
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import joblib

sys.path.insert(0, str(Path(__file__).parent))
import config

from sentence_transformers import SentenceTransformer
from fugashi import Tagger

warnings.filterwarnings("ignore")

# 加载配置
SEED_TOPICS_PATH = Path(__file__).parent / "seed_topics.json"
with open(SEED_TOPICS_PATH, "r", encoding="utf-8") as f:
    SEED_CONFIG = json.load(f)

NEGATIVE_KEYWORDS = SEED_CONFIG["negative_keywords"]

# 嵌入模型
LOCAL_MODEL_PATH = config.MODELS_DIR / "ruri-v3-310m"
MODEL_NAME = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "cl-nagoya/ruri-v3-310m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = config.TOPIC_RANDOM_SEED

# PATTERN_COMBOS（从 topic.py 移植）
PATTERN_COMBOS = [
    {
        "name": "離婚別離",
        "required": [["離婚", "別居", "親権", "離婚届", "調停", "弁護士", "離婚したい", "離婚する"]],
        "context": ["夫", "妻", "旦那", "主人", "パートナー", "彼氏", "彼女"],
    },
    {
        "name": "浮気不倫",
        "required": [["浮気", "不倫", "愛人", "二股", "不貞", "浮気された", "不貞行為"]],
        "context": ["夫", "妻", "旦那", "主人", "彼", "パートナー", "彼氏", "彼女"],
    },
    {
        "name": "夫以外の幸せ",
        "required": [["夫以外", "他の人", "別の男性", "別の人の"]],
        "context": ["幸せ", "恋", "気持ち", "好き", "愛"],
    },
    {
        "name": "夫隠し",
        "required": [["夫に見つかったら怖い", "見つかったら怖い", "夫に内緒", "夫にバレると怖い", "隠してる", "隠してた"]],
        "context": [],
    },
    {
        "name": "外国人詐欺",
        "required": [["外国", "外国人", "海外の人", "フィリピン", "アメリカ", "イギリス"]],
        "context": ["信頼", "夫より", "優しい", "理解", "相談", "援助"],
    },
    {
        "name": "退役軍人詐欺",
        "required": [["退役軍人", "軍人"]],
        "context": ["貸", "お金", "返す", "会いに来る", "送金"],
    },
    {
        "name": "流産妊娠問題",
        "required": [["流産", "死産", "中絶", "妊娠中絶", "流産した", "流産の心配"]],
        "context": [],
    },
    {
        "name": "ロマンス詐欺",
        "required": [
            ["Facebook", "facebook", "フェイスブック", "SNS", "LINE", "インスタ", "Instagram", "Twitter", "ツイッター", "マッチング", "matching"],
            ["退役軍人", "軍人"],
        ],
        "context": ["海外", "外国", "貸", "送金", "お金", "投資", "返す", "詐欺", "騙された", "会いに来る"],
    },
    {
        "name": "海外金銭",
        "required": [["海外", "外国"]],
        "context": ["貸", "送金", "お金", "投資", "返す", "資金", "融資", "知り合", "出会", "詐欺", "騙された"],
    },
    {
        "name": "詐欺",
        "required": [["詐欺", "フィッシング", "登録するだけで", "月々数万", "掲載料", "ベビーモデル", "詐欺に遭った", "詐欺被害"]],
        "context": [],
    },
    {
        "name": "金銭トラブル",
        "required": [["融資", "資金を持ちかけ", "投資", "借りる", "貸す", "借金"]],
        "context": ["多額", "大金", "ほとんど", "消える", "怪しい", "詐欺", "騙", "不安"],
    },
    {
        "name": "DV暴力",
        "required": [["DV", "暴力", "暴行", "傷害", "殴", "蹴", "脅", "叩", "暴言", "怒鳴", "ドメスティックバイオレンス", "警察沙汰"]],
        "context": ["夫", "妻", "旦那", "彼", "パートナー", "家族", "彼氏", "彼女"],
    },
    {
        "name": "モラハラ",
        "required": [["モラハラ", "モラルハラスメント", "パワハラ", "精神的な虐待"]],
        "context": [],
    },
    {
        "name": "自殺自傷",
        "required": [["死にたい", "消えたい", "自殺", "死のう", "生きる意味", "いらない", "終わらせたい", "死ねない", "死なない", "死にきれ", "自殺したい", "命を絶ちたい"]],
        "context": [],
    },
]


def check_pattern_combos(text: str) -> tuple[bool, str | None]:
    """检查文本是否匹配模式组合"""
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


def generate_seed_labels(df: pd.DataFrame) -> dict[int, int]:
    """
    使用关键词匹配生成高置信度种子标签。
    返回 {row_index: category} 字典。
    """
    all_negative_keywords = set(NEGATIVE_KEYWORDS)
    seed_labels = {}

    for idx, row in df.iterrows():
        user_text = str(row.get("userInput", ""))
        if not user_text.strip():
            continue

        # 检查负面关键词
        if any(kw in user_text for kw in all_negative_keywords):
            seed_labels[idx] = 0
            continue

        # 检查 PATTERN_COMBOS
        is_match, _ = check_pattern_combos(user_text)
        if is_match:
            seed_labels[idx] = 0
            continue

    # 生成 category 1 种子标签
    # 使用明确的正面/中性关键词作为 category 1 的种子
    positive_keywords = [
        "赤ちゃん", "可愛い", "授乳", "母乳", "おっぱい", "育児",
        "眠れない", "寝る", "食欲", "頭痛", "疲労", "産後",
        "相談", "保健", "センター", "助産", "施設",
        "地域", "支援", "手伝う",
    ]

    for idx, row in df.iterrows():
        if idx in seed_labels:
            continue
        user_text = str(row.get("userInput", ""))
        if not user_text.strip():
            continue

        # 如果包含正面/中性关键词，标记为 category 1
        if any(kw in user_text for kw in positive_keywords):
            seed_labels[idx] = 1
            continue

    # 利用 topic 信息生成更多种子标签
    for idx, row in df.iterrows():
        if idx in seed_labels:
            continue
        user_text = str(row.get("userInput", ""))
        tid = row.get("topic_id")
        if not user_text.strip() or pd.isna(tid):
            continue

        # Topic 5 (离婚出轨) → 全部 category 0
        if tid == 5:
            seed_labels[idx] = 0
            continue

        # Topic 2 (产后哺乳)、Topic 6 (产后睡眠) → 全部 category 1
        if tid in (2, 6):
            seed_labels[idx] = 1
            continue

        # Topic 3 (流产妊娠) → 流产相关 category 0，其他 category 1
        if tid == 3:
            if any(kw in user_text for kw in ["流産", "死産", "中絶"]):
                seed_labels[idx] = 0
            else:
                seed_labels[idx] = 1
            continue

        # Topic 0 (育儿支持/诈骗) → 诈骗 category 0，其他 category 1
        if tid == 0:
            if any(kw in user_text for kw in ["詐欺", "騙", "フィッシング", "登録するだけで", "ベビーモデル", "掲載料"]):
                seed_labels[idx] = 0
            else:
                seed_labels[idx] = 1
            continue

        # Topic 1 (育儿离婚) → 离婚/DV/自杀相关 category 0，其他 category 1
        if tid == 1:
            if any(kw in user_text for kw in ["離婚", "DV", "暴力", "死にたい", "消えたい", "自殺", "終わらせたい"]):
                seed_labels[idx] = 0
            else:
                seed_labels[idx] = 1
            continue

        # Topic 4 (人际关系咨询) → DV/モラハラ category 0，其他 category 1
        if tid == 4:
            if any(kw in user_text for kw in ["DV", "暴力", "モラハラ", "パワハラ", "無視", "冷たい"]):
                seed_labels[idx] = 0
            else:
                seed_labels[idx] = 1
            continue

    return seed_labels


def run_classification():
    """执行二分类"""
    print("=" * 60)
    print("二分类（半监督学习方式）")
    print("=" * 60)

    # 1. 读取数据
    print("\n[1] 读取数据...")
    df = pd.read_csv(config.DATA_DIR / "data_with_id.csv")
    doc_topics = pd.read_csv(config.DATA_DIR / "topic_modeling" / "combined_userInput_doc_topics.csv")
    print(f"  data_with_id: {len(df)} 行")
    print(f"  doc_topics: {len(doc_topics)} 行")

    # 合并 topic_id
    df["topic_id"] = doc_topics["topic_id"].values

    # 2. 生成嵌入
    print(f"\n[2] 生成嵌入 ({MODEL_NAME})...")
    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    valid_texts = df["userInput"].fillna("").tolist()
    embeddings = embedding_model.encode(
        valid_texts,
        show_progress_bar=True,
        batch_size=config.TOPIC_EMBEDDING_BATCH_SIZE_CUDA if DEVICE == "cuda" else config.TOPIC_EMBEDDING_BATCH_SIZE_CPU,
    )
    print(f"  嵌入维度: {embeddings.shape}")

    # 3. 生成种子标签
    print("\n[3] 生成种子标签...")
    seed_labels = generate_seed_labels(df)
    n_cat0 = sum(1 for v in seed_labels.values() if v == 0)
    n_cat1 = sum(1 for v in seed_labels.values() if v == 1)
    print(f"  category 0 (种子): {n_cat0} 件")
    print(f"  category 1 (种子): {n_cat1} 件")

    # 4. 训练分类器（尝试多种）
    print("\n[4] 训练分类器...")
    train_indices = list(seed_labels.keys())
    X_train = embeddings[train_indices]
    y_train = [seed_labels[i] for i in train_indices]

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "SVM-RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED),
        "SVM-Linear": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }

    best_clf = None
    best_f1 = 0
    best_name = ""

    for name, clf in classifiers.items():
        cv_scores = cross_val_score(clf, X_train, y_train, cv=min(5, len(train_indices)), scoring="f1")
        f1_mean = cv_scores.mean()
        print(f"  {name}: F1={f1_mean:.3f} (±{cv_scores.std():.3f})")
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_clf = clf
            best_name = name

    print(f"\n  最佳分类器: {best_name} (F1={best_f1:.3f})")
    best_clf.fit(X_train, y_train)

    # 保存模型和种子标签
    model_dir = config.MODELS_DIR / "negative_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, model_dir / "classifier.joblib")
    joblib.dump(seed_labels, model_dir / "seed_labels.joblib")
    print(f"  模型已保存: {model_dir / 'classifier.joblib'}")

    # 5. 预测所有文档
    print("\n[5] 预测所有文档...")
    all_probs = best_clf.predict_proba(embeddings)

    confidence_threshold = 0.85
    categories = []
    confidences = []

    for i, (idx, row) in enumerate(df.iterrows()):
        user_text = str(row.get("userInput", ""))
        if not user_text.strip():
            categories.append(1)
            confidences.append(0.0)
            continue

        # 如果有种子标签，使用种子标签
        if idx in seed_labels:
            categories.append(seed_labels[idx])
            confidences.append(1.0)
            continue

        # 使用分类器预测
        probs = all_probs[i]
        max_prob = max(probs)
        pred_class = probs.argmax()

        if max_prob >= confidence_threshold:
            categories.append(pred_class)
            confidences.append(max_prob)
        else:
            categories.append(-1)
            confidences.append(max_prob)

    df["category"] = categories
    df["confidence"] = confidences

    # 6. 统计结果
    print("\n[6] 结果统计...")
    cat_counts = df["category"].value_counts()
    n_classified = cat_counts.get(0, 0) + cat_counts.get(1, 0)
    n_high_conf = sum(1 for c in confidences if c >= 0.85)
    print(f"  category 0 (负面):   {cat_counts.get(0, 0)} 件")
    print(f"  category 1 (非负面): {cat_counts.get(1, 0)} 件")
    print(f"  未分类 (低置信度):   {cat_counts.get(-1, 0)} 件")
    print(f"  高置信度 (>= 0.85):  {n_high_conf} 件")

    # 7. 保存结果
    print("\n[7] 保存结果...")
    all_path = config.DATA_DIR / "2category_all.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"  → {all_path}  ({len(df)}件)")

    # 打印 category 0 示例
    print(f"\n■ category 0 のセッション例:")
    matched = df[df["category"] == 0].head(15)
    for _, row in matched.iterrows():
        text = str(row["userInput"])[:60]
        conf = row.get("confidence", 0)
        print(f"  [Topic {row['topic_id']}] ({conf:.2f}) {text}...")

    print(f"\n■ category 1 のセッション例:")
    matched = df[df["category"] == 1].head(10)
    for _, row in matched.iterrows():
        text = str(row["userInput"])[:60]
        conf = row.get("confidence", 0)
        print(f"  [Topic {row['topic_id']}] ({conf:.2f}) {text}...")

    print(f"\n■ 未分類のセッション例:")
    unmatched = df[df["category"] == -1].head(10)
    for _, row in unmatched.iterrows():
        text = str(row["userInput"])[:60]
        conf = row.get("confidence", 0)
        print(f"  [Topic {row['topic_id']}] ({conf:.2f}) {text}...")

    return df


if __name__ == "__main__":
    run_classification()
