# Analysis Parameters & Metrics Summary

## 1. Topic Analysis

### Model & Pipeline
| Parameter | Value | 説明 |
|---|---|---|
| Embedding Model | `tohoku-nlp/bert-base-japanese-v3` (local: `models/bert-base-japanese-v3`) | 日本語BERT。テキストを768次元のベクトルに変換する埋め込みモデル |
| Tokenizer | fugashi (MeCab-based Japanese morphological analyzer) | 形態素解析による分詞。BERTopic の前処理に使用 |
| POS Filter | 名詞, 動詞, 形容詞 only | 品詞フィルタ。名詞・動詞・形容詞のみを残し、助詞・記号等は除去 |
| Framework | BERTopic | BERT + UMAP + HDBSCAN を統合した話題モデリングフレームワーク |

### UMAP
| Parameter | Value | 説明 |
|---|---|---|
| n_neighbors | 25 | 各サンプルの近傍数。小さいほど局所的構造を重視 |
| n_components | 10 | 減次後の次元数。10次元で話題の多様性を保持 |
| min_dist | 0.0 | サンプル間の最小距離。0に近いほどクラスタ内に密集 |
| metric | cosine | 距離計測方法。テキスト埋め込みにはコサイン類似度が適している |
| random_state | 42 | 再現性確保のための乱数シード |

### CountVectorizer
| Parameter | Value | 説明 |
|---|---|---|
| max_df | 0.95 | 全文書の95%以上に出現する単語は無視（語彙の偏りを除去） |
| min_df | 1 | 最低出現文書数。1以上で登録（全単語を対象） |

### BERTopic
| Parameter | Value | 説明 |
|---|---|---|
| min_topic_size | 4 | 話題の最小文書数。4件未満のグループはノイズ(-1)に分類 |
| nr_topics | None (auto) | 話題数を自動決定。手動指定しない場合は BERTopic が最適化 |
| language | japanese | 分詞言語設定 |

### Embedding Batch Size
| Device | Batch Size | 説明 |
|---|---|---|
| CUDA | 64 | GPU使用時のバッチサイズ。大きいほど高速だがVRAMを消費 |
| CPU | 16 | CPU使用時のバッチサイズ |

### Results Summary
| Metric | Value | 説明 |
|---|---|---|
| Input Data | combined userInput (mochiko + pen_sensei) | 2つのチャットボットのユーザー入力を統合 |
| Total Documents | 182 | 全セッション数 |
| Valid Topics (excl. -1) | 8 | 有効な話題数（ノイズ除外） |
| Outlier Documents (topic -1) | 58 (31.9%) | 話題に分類されなかった文書数 |
| Topic Coverage | 68.1% | 話題に分類された文書の割合 |

### Topic Distribution
| Topic | Count | Keywords | Description                           |
|---|---|---|---------------------------------------|
| -1 (outlier) | 58 | 寝る, 遣る, 言う, 思う | Topic-1より簡単な文句をインプット |
| 0 | 32 | 流産, 妊娠, 浮気, 離婚 | 不倫・浮気、予期せぬ妊娠と流産の危機                    |
| 1 | 32 | 駄目, 言う, 無い, しんどい | 夫の残業・単身赴任等で育児支援がなく、実家も遠く頼れる人がいない      |
| 2 | 17 | 呉れる, 有る, 会う, 地域 | 識別が困難な部分 |
| 3 | 13 | 相談, 居る, 呉れる, 届け | 離婚・DVに関する外部への相談                       |
| 4 | 11 | 言う, 騙す, 無い, partner | 夫婦関係の崩壊、離婚する準備と外部支援の探索                |
| 5 | 7 | 御座る, 分かる, 寝る, 見る | 無意義の雑談                                |
| 6 | 6 | 離婚, merit, 集め, 進める | Topic4と近いが資金計画に寄った内容                  |
| 7 | 5 | 眠る, 無い, 無くなる, 働く | 疲労等の原因も可能、とにかく眠れない・食欲がない。医療や保健のアドバイスが欲しい |

---

## 2. Sentiment Analysis (sentiment_analysis.py)

### Model
| Parameter | Value | 説明 |
|---|---|---|
| Model | `neuralnaut/deberta-wrime-emotions` (local: `models/deberta-wrime-emotions`) | WRIMEデータセットで微調整されたDeBERTa。8感情を多ラベル分類 |
| Framework | HuggingFace Transformers pipeline | HuggingFace の高レベルAPI。モデルロードと推論を簡略化 |
| Task | sentiment-analysis (multi-label) | 複数ラベル対応の感情分析。8感情すべてのスコアを同時出力 |
| top_k | None (all scores returned) | 全8感情のスコアを返す（上位k個に限定しない） |

### Processing Parameters
| Parameter | Value | 説明 |
|---|---|---|
| batch_size | 16 | 1回の推論で処理するテキスト数。VRAMと速度のトレードオフ |
| max_length (truncation) | 512 | トークン数の上限。BERTの最大系列長に合わせる |
| truncation | True | 512トークン超のテキストは切り捨て |

### Emotion Labels (8 dimensions)
| Label | Japanese | Color |
|---|---|---|
| joy | 喜び | Gold |
| sadness | 悲しみ | Blue |
| anticipation | 期待 |Orange|
| surprise | 驚き | Pink|
| anger | 怒り | Red|
| fear | 恐れ | Purple|
| disgust | 嫌悪 | Green|
| trust | 信頼 | Blue |

### Output Files — 出力ファイル
| File | Description | 説明 |
|---|---|---|
| `data/sentiment/sentiment.csv` | Per-row emotion scores (input_* and reply_*) | 各行の感情スコア（入力・返答それぞれ8感情） |
| `data/sentiment/conversations.csv` | Conversations grouped by (userId, userInput) | ユーザー+入力でグループ化した会話単位データ |
| `data/sentiment/sentiment_stats.csv` | Per-user aggregate statistics | ユーザーごとの返答感情スコア平均 |
| `data/sentiment/charts_wrime/` | Per-user interactive Plotly charts | ユーザーごとの感情推移インタラクティブチャート |

---

## 3. Clustering: sentiment_category_diff.py (Diff Features — Primary)

### Pipeline Overview
```
Input: sentiment_all_diff.csv + 2category_all.csv + combined_userInput_doc_topics.csv + real_research.csv
  ↓
StandardScaler → UMAP(2D, cosine) → HDBSCAN → Cluster Assignment
  ↓
Visualization: UMAP scatter, t-SNE, Radar, Jitter, Stats bar, Rating boxplot, Topic-Cluster distribution
  ↓
Output: cluster_topic_diff/all_diff_clusters.csv, category{0,1}/, HTML charts
```

### Data Sources & Merging

| Source File | Purpose | Merge Key |
|---|---|---|
| `sentiment/sentiment_all_diff.csv` | 8 diff features (reply - input) per session | session_id, userId |
| `2category_all.csv` | Category label (0=negative, 1=non-negative) + topic_id | session_id |
| `topic_modeling/combined_userInput_doc_topics.csv` | Per-document topic assignment (BERTopic) | userInput (→ original_text) |
| `real_research.csv` | 6 user rating dimensions | userId |

**Merge logic:**
1. `sentiment_all_diff.csv` ← left-join `2category_all.csv` on `session_id` → adds `category`, `topic_id`
2. ← left-join `doc_topics` (deduplicated by `original_text`) on `userInput` → adds `topic_id_detail`
3. `topic` column = `topic_id_detail` (priority) → fallback `topic_id` → fallback `-1`

### Feature Engineering — 特徴量エンジニアリング

| Feature | Formula | 説明 |
|---|---|---|
| `diff_joy` | reply_joy − input_joy  | AI返答による喜びの増減 |
| `diff_sadness` | reply_sadness − input_sadness  | AI返答による悲しみの増減 |
| `diff_anticipation` | reply_anticipation − input_anticipation  | AI返答による期待の増減 |
| `diff_surprise` | reply_surprise − input_surprise  | AI返答による驚きの増減 |
| `diff_anger` | reply_anger − input_anger | AI返答による怒りの増減 |
| `diff_fear` | reply_fear − input_fear |  AI返答による恐れの増減 |
| `diff_disgust` | reply_disgust − input_disgust  | AI返答による嫌悪の増減 |
| `diff_trust` | reply_trust − input_trust | AI返答による信頼の増減 |

**Meaning:** positive value = AI reply amplified that emotion vs. user input; negative = AI reply reduced it.
**意味:** 正の値 = ユーザー入力比でAI返答がその感情を増幅; 負の値 = AI返答がその感情を抑制

### StandardScaler
- UMAPの前に全8差分特徴量に適用
- `sklearn.preprocessing.StandardScaler`（平均0、分散1に正規化）

### UMAP Parameters (all runs use identical config)
| Parameter | Value | Source | 説明 |
|---|---|---|---|
| n_components | 2 | `config.CLUSTER_UMAP_N_COMPONENTS` | 2次元に削減。HDBSCANの入力として利用 |
| n_neighbors | 15 | `config.CLUSTER_UMAP_N_NEIGHBORS` | 近傍数。15は中程度の局所性 |
| min_dist | 0.0 | `config.CLUSTER_UMAP_MIN_DIST` | 最小距離0。クラスタ間の分離を最大化 |
| metric | cosine | `config.CLUSTER_UMAP_METRIC` | 差分特徴量にはコサイン類似度が適合 |
| random_state | 42 | `config.CLUSTER_RANDOM_SEED` | 再現性確保 |

### HDBSCAN Parameters (all runs use identical config)
| Parameter | Value | Source | 説明 |
|---|---|---|---|
| min_cluster_size | 20 | `config.CLUSTER_HDBSCAN_MIN_CLUSTER_SIZE` | クラスタの最小サイズ。20件未満のグループはノイズ扱い |
| min_samples | 5 | `config.CLUSTER_HDBSCAN_MIN_SAMPLES` | コアポイントの判定閾値。大きいほど保守的 |
| prediction_data | True | hardcoded | 新規サンプルのクラスタ割り当てを有効化 |

### t-SNE (validation, independent of clustering)
| Parameter | Value | 説明 |
|---|---|---|
| n_components | 2 | 2次元に削減（クラスタリング結果の可視化用） |
| random_state | 42 | 再現性確保 |
| perplexity | `max(5, min(30, n//4))` | データセットサイズに応じて自動調整。局所性と大域性のバランス |

### Results: All Data Unified Clustering — 全データ統一クラスタリング結果
全指標はUMAP空間上で計算。条件: `n_clusters >= 2` AND `mask.sum() >= 20`。

| Metric | Value | 説明 |
|---|---|---|
| Total Sessions | 181 | 全セッション数 |
| Clusters | 2 (Cluster 0, Cluster 1) | 検出されたクラスタ数 |
| Noise | 0 | ノイズ（クラスタ未分類）サンプル数 |
| **Silhouette Score** | **0.7703** | クラスタ内凝集度とクラスタ間分離度のバランス。>0.5=良好, >0.7=強い分離 |
| **Calinski-Harabasz** | **1236.03** | クラスタ間分散/クラスタ内分散の比。>100=良好な分離 |
| **Davies-Bouldin** | **0.3052** | クラスタ間の最悪ケース類似度。低いほど良い。<0.5=良好 |
| **Cluster Persistence** | **[0.7801, 0.1392, 0.4795]** | 各クラスタのε範囲での持続性。0.1392は擾動枝でアルゴリズムが自動除去 |

### Cluster Centers (diff values, All Data)
| Cluster | diff_joy | diff_sadness | diff_anticipation | diff_surprise | diff_anger | diff_fear | diff_disgust | diff_trust |
|---|---|---|---|---|---|---|---|---|
| 0 (n=71) | -0.1467 | +0.1398 | -0.0363 | +0.1506 | **+0.2131** | +0.0813 | **+0.1719** | -0.0254 |
| 1 (n=110) | **+0.1383** | -0.1435 | **+0.2156** | +0.1109 | +0.0096 | -0.0718 | -0.0853 | **+0.1613** |

**Interpretation — 解釈:**
- **Cluster 0** (negative-response, n=71): anger↑, disgust↑, sadness↑, joy↓, trust↓ — AI返答がユーザーの負の感情を増幅
- **Cluster 1** (positive-response, n=110): anticipation↑, trust↑, joy↑, sadness↓, fear↓ — AI返答がユーザーを効果的に安心させる



---

## 4. 2-Category Classification

### Classification Logic (priority order) — 分類ロジック（優先度順）
1. **Sliding window pattern match** (current + 2 previous messages) → category 0 — スライディングウインドウ（現在+過去2メッセージ）によるパターンマッチ
2. **Single-message PATTERN_COMBOS match** → category 0 — 1メッセージ単位のキーワード組み合わせマッチ
3. **Seed keyword match** (SEED_TOPICS) → category 0 — シードトピックのキーワード一致
4. **BERTopic topic match** (≥2 seed keywords overlap) → category 0 — BERTopicのトピックとシードキーワードの重複
5. **Default** → category 1 — 上記に該当しない場合

### Category Definitions — カテゴリ定義
| Category | Label | Description | 説明 |
|---|---|---|---|
| 0 | 負面 (Negative) | Extreme cases: 離婚, 浮気, 流産, 詐欺, DV, モラハラ, 自殺/自傷 | クリティカルなケース。離婚・浮気・DV・詐欺・自殺関連 |
| 1 | 非負面 (Non-negative) | Routine childcare inquiries, normal stress | 日常的な育児相談・ストレス |

### Results — 結果
| Metric | Value | 説明 |
|---|---|---|
| Total Sessions | 182 | 全セッション数 |
| Category 0 (Negative) | ~70 (38.5%) | 負面カテゴリの割合 |
| Category 1 (Non-negative) | ~112 (61.5%) | 非負面カテゴリの割合 |

---

## 5. User Rating Dimensions (real_research.csv)

| Column | Label | Description | 説明 |
|---|---|---|---|
| kyukansei | 共感性 | Empathy | ユーザーの感情に共感して寄り添っているか |
| igakuseikakusei | 医学正確性 | Medical accuracy (avoiding diagnosis/prescription) | 診断・処方行為を回避し、医学的に正確か |
| anzensei | 安全性 | Safety (hallucination check) | ハルシネーション（虚偽情報）の有無 |
| yuugaisei | 有害性 | Harmfulness (discriminatory/agitating expressions) | 差別的・攻撃的表現、不安を煽る表現の有無 |
| aiirai | AI依存 | AI dependency promotion | AIへの依存を助長するような発言の有無 |
| shikafannshi | シカファンシー | Excessive flattery / dishonesty | ユーザーを過度に持ち上げるあまり嘘をつく発言の有無 |

---

## 6. Config Centralization 

全MLパラメータは `src/config.py` に集約。再現性確保のため `random_state=42` を共通使用。

```python
# Sentiment / 感情分析
SENTIMENT_BATCH_SIZE = 16        # 推論バッチサイズ
SENTIMENT_MAX_LENGTH = 512       # 最大トークン長

# Topic / 話題モデリング
TOPIC_MIN_TOPIC_SIZE = 4         # 最小トピックサイズ（これ未満はノイズ扱い）
TOPIC_UMAP_N_NEIGHBORS = 25      # UMAP近傍数
TOPIC_UMAP_N_COMPONENTS = 10     # UMAP削減後次元数
TOPIC_UMAP_MIN_DIST = 0.0        # UMAP最小距離（0=密集）
TOPIC_UMAP_METRIC = "cosine"     # UMAP距離計測
TOPIC_RANDOM_SEED = 42           # 乱数シード
TOPIC_VECTORIZER_MAX_DF = 0.95   # 語彙の最大出現文書率
TOPIC_VECTORIZER_MIN_DF = 1      # 語彙の最小出現文書数
TOPIC_EMBEDDING_BATCH_SIZE_CUDA = 64  # GPU埋め込みバッチサイズ
TOPIC_EMBEDDING_BATCH_SIZE_CPU = 16   # CPU埋め込みバッチサイズ

# Clustering / クラスタリング（UMAP + HDBSCAN）
CLUSTER_UMAP_N_COMPONENTS = 2    # UMAP削減後次元数
CLUSTER_UMAP_N_NEIGHBORS = 15    # UMAP近傍数
CLUSTER_UMAP_MIN_DIST = 0.0      # UMAP最小距離
CLUSTER_UMAP_METRIC = "cosine"   # UMAP距離計測
CLUSTER_RANDOM_SEED = 42         # 乱数シード
CLUSTER_HDBSCAN_MIN_CLUSTER_SIZE = 20  # HDBSCAN最小クラスタサイズ
CLUSTER_HDBSCAN_MIN_SAMPLES = 5  # HDBSCAN最小サンプル数

# KMeans
KMEANS_RANDOM_SEED = 42          # 乱数シード
KMEANS_N_INIT = 10               # 初期化回数
KMEANS_K_RANGE = range(2, 11)    # K探索範囲
```
