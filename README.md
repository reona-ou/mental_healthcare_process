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
│   ├── ruri-v3-310m/              # Embedding model for topic modeling
│   ├── deberta-wrime-emotions/    # Sentiment analysis model
│   └── negative_classifier/       # Trained SVM-RBF classifier
├── src/                           # Core source code
│   ├── config.py                  # Configuration (paths, parameters)
│   ├── topic.py                   # BERTopic topic modeling (KMeans + 统计距离过滤)
│   ├── negative_classify.py       # 2-category classification (SVM-RBF 半监督学习)
│   ├── sentiment_analysis.py      # Sentiment analysis using transformers
│   ├── sentiment_difference.py    # Sentiment difference computation & visualization
│   ├── sentiment_category_diff.py # Clustering & topic visualization
│   ├── word_process.py            # Text preprocessing
│   ├── word_visualization.py      # Word frequency visualization
│   ├── sentiment_summary_charts.py # Sentiment summary charts
│   ├── create_csv.py              # Data preprocessing
│   └── download_models.py         # Download Hugging Face models locally
├── README.md                      # Project description and guide
├── PARAMETER.md                   # Detailed parameter documentation
├── topic.md                       # Topic modeling results
├── requirements.txt               # Environment dependencies
└── LICENSE                        # MIT License
```
---

## 📊 Analysis Scripts

### 1. Sentiment Analysis (`sentiment_analysis.py`)
- Analyzes sentiment of user inputs and bot replies
- Uses DeBERTa model (neuralnaut/deberta-wrime-emotions)
- Outputs: `data/sentiment/sentiment.csv`

### 2. Sentiment Difference (`sentiment_difference.py`)
- Computes emotion difference matrices (input vs reply)
- Per-user statistics and visualizations
- Outputs: `data/sentiment/sentiment_all_diff.csv` (all sessions)

### 3. Topic Modeling (`topic.py`)
- BERTopic-based topic modeling with KMeans clustering
- Short text filtering (tokenized < 2 words → topic -1)
- Statistical distance filtering (mean + 1.5 * std)
- Outputs: `data/topic_modeling/`

### 4. Negative Classification (`negative_classify.py`)
- Semi-supervised learning with SVM-RBF classifier
- Seed labels from keyword matching + topic information
- Confidence threshold: 0.85
- Short texts → category 1 (non-negative)
- Outputs: `data/2category_all.csv`
- Model: `models/negative_classifier/`

### 5. Clustering & Visualization (`sentiment_category_diff.py`)
- UMAP + HDBSCAN clustering on emotion differences
- Category-specific visualizations (UMAP, radar, jitter plots)
- Topic distribution within clusters
- Outputs: `data/sentiment/cluster_topic_diff/category{0,1}/`

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Installation
```bash
pip install -r requirements.txt
```

### Download Models
```bash
python src/download_models.py
```

### Run Analysis
```bash
# 1. Sentiment Analysis
python src/sentiment_analysis.py

# 2. Sentiment Difference
python src/sentiment_difference.py

# 3. Topic Modeling
python src/topic.py

# 4. Negative Classification (requires topic modeling output)
python src/negative_classify.py

# 5. Clustering & Visualization
python src/sentiment_category_diff.py
```

---

## 📝 Notes

- Models are downloaded to `models/` directory (gitignored)
- Data files are stored in `data/` directory
- Visualization outputs are HTML files (interactive via Plotly)
- Topic modeling uses KMeans clustering (7 topics) with statistical distance filtering
- Negative classification uses SVM-RBF with 89.1% F1 score (cross-validation)
