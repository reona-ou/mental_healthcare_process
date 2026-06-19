# Mental Healthcare Process

---

## 📌 Overview
本研究は、メンタルケアチャットボットの開発を目的とする。
あらかじめ収集した対話ログと専門家の評価を統合したデータセットを用い、自然言語処理（NLP）を用いてメンタルヘルスに悪影響を及ぼす危険性がある要因を特定する。
この要因に基づき、共感性と安全性を備えたモデルを構築する。
またモデルをAWS Bedrockに実装し、LLM-as-a-judgeを使って、共感性、医学的正確性、安全性（医療行為の制限）、有害性（差別・攻撃性）の排除、AI依存の防止、シカファンシーの回避という6つの指標の評価を行う。
これらに基づき、高い信頼性と安全性を保証した心理支援チャットボットを実装し、ユーザーテストを行い、その結果をまとめる。
---

## 📂 Repository Structure

```text
├── data/                          # Data directory
│   ├── sentiment/                 # Sentiment analysis results
│   │   ├── sentiment.csv          # Raw sentiment data
│   │   ├── sentiment_all_diff.csv # All sessions with diff features
│   │   ├── sentiment_diff/        # Per-user sentiment difference analysis
│   │   └── cluster_topic_diff/    # Clustering & topic visualization results
│   │       ├── category0/         # Category 0 (negative) outputs
│   │       └── category1/         # Category 1 (non-negative) outputs
│   ├── topic_modeling/            # BERTopic results
│   └── 2category_all.csv          # 2-category classification results
├── models/                        # Local model checkpoints (GIT IGNORED)
├── src/                           # Core source code
│   ├── config.py                  # Configuration (paths, etc.)
│   ├── sentiment_analysis.py      # Sentiment analysis using transformers
│   ├── sentiment_difference.py    # Sentiment difference computation & visualization
│   ├── sentiment_category_diff.py # Clustering & topic visualization
│   ├── topic.py                   # BERTopic topic modeling & 2-category classification
│   ├── word_process.py            # Text preprocessing
│   ├── Visualization.py           # General visualization utilities
│   ├── create_csv.py              # Data preprocessing
│   └── download_models.py         # Download Hugging Face models locally
├── README.md                      # Project description and guide
├── requirements.txt               # Environment dependencies
└── LICENSE                        # MIT License
```

---

## 🚀 Setup

```bash
# Create conda environment
conda create -n hanai_llm python=3.11
conda activate hanai_llm

# Install dependencies
pip install -r requirements.txt

# Download models locally (optional)
python src/download_models.py
```

---

## 📊 Analysis Scripts

### 1. Sentiment Analysis (`sentiment_analysis.py`)
- Analyzes sentiment of user inputs and bot replies
- Outputs: `data/sentiment/sentiment.csv`

### 2. Sentiment Difference (`sentiment_difference.py`)
- Computes emotion difference matrices (input vs reply)
- Per-user statistics and visualizations
- Outputs: `data/sentiment/sentiment_all_diff.csv` (all sessions)

### 3. Topic Modeling (`topic.py`)
- BERTopic-based topic modeling
- 2-category classification (negative/non-negative)
- Pattern-based detection for specific issues (fraud, DV, divorce, etc.)
- Outputs: `data/2category_all.csv`, `data/topic_modeling/`

### 4. Clustering & Visualization (`sentiment_category_diff.py`)
- UMAP + HDBSCAN clustering on emotion differences
- Category-specific visualizations (UMAP, radar, jitter plots)
- Topic distribution within clusters
- Outputs: `data/sentiment/cluster_topic_diff/category{0,1}/`

---

## 📈 Output Structure

```
data/sentiment/cluster_topic_diff/
├── all_diff_*.html/csv           # All-session results
├── category0/                    # Category 0 (negative)
│   ├── category0_diff_umap_2d.html
│   ├── category0_diff_tsne.html
│   ├── category0_diff_*_radar.html
│   ├── category0_diff_rating_boxplot.html
│   ├── category0_diff_jitter_*.html
│   ├── category0_topic_cluster_distribution.html
│   └── category0_diff_clusters.csv
└── category1/                    # Category 1 (non-negative)
    └── (same structure as category0)
```

---

## 📋 Classification Categories

- **Category 0 (Negative)**: Divorce, affair, DV, fraud, miscarriage, mental health issues
- **Category 1 (Non-negative)**: General conversations, questions, positive interactions

---

## 📝 Notes

- Models are downloaded to `models/` directory (gitignored)
- Large data files are stored in `data/` directory
- Visualization outputs are HTML files (interactive via Plotly)
