# パラメータ・指標サマリー

全設定パラメータは `src/config.py` に集約。再現性確保のため `random_state=42` を共通使用。

---

## 1. トピック分析 — topic.py

### パイプライン
```
userInput (182件) → fugashi形態素解析 → 短文フィルタ (<2語→-1)
→ ruri-v3-310m 埋め込み（「トピック: 」プレフィックス付き）
→ UMAP削減 → KMeansクラスタリング → 統計距離フィルタ → トピック割当
```

### 形態素解析パラメータ
| パラメータ | 値 | config定数 | 説明 |
|---|---|---|---|
| POS フィルタ | 名詞, 動詞, 形容詞 | `KEEP_POS` | これらの品詞のみ抽出 |
| 句読点排除 | 補助記号, 記号, 助詞, 助動詞, 接続詞, 感動詞, 接頭詞, 接尾詞 | `PUNCTUATION_POS` | 除外する品詞 |
| ストップワード | 御座る, 分かる, 言う, 無い, 仕舞う, レン 等 | `STOPWORDS` | 一般的すぎる単語・分詞偽影を除去 |

### UMAP
| パラメータ | 値 | config定数 | 説明 |
|---|---|---|---|
| n_neighbors | 15 | `TOPIC_UMAP_N_NEIGHBORS` | 近傍数 |
| n_components | 2 | `TOPIC_UMAP_N_COMPONENTS` | 削減後次元数 |
| min_dist | 0.1 | `TOPIC_UMAP_MIN_DIST` | 最小距離 |
| metric | cosine | `TOPIC_UMAP_METRIC` | 距離計測 |

### KMeans クラスタリング
| パラメータ | 値 | config定数 | 説明 |
|---|---|---|---|
| n_clusters | 7 (自動) | - | min(7, len(texts)//10) |
| n_init | 10 | `KMEANS_N_INIT` | 初期化回数 |
| min_topic_size | 4 | `TOPIC_MIN_TOPIC_SIZE` | 最小トピックサイズ |

### CountVectorizer
| パラメータ | 値 | config定数 | 説明 |
|---|---|---|---|
| max_df | 0.85 | `TOPIC_VECTORIZER_MAX_DF` | 語彙の最大出現文書率 |
| min_df | 2 | `TOPIC_VECTORIZER_MIN_DF` | 語彙の最小出現文書数 |
| ngram_range | (1, 2) | - | ユニグラム+バイグラム |

### BERTopic
| パラメータ | 値 | config定数 | 説明 |
|---|---|---|---|
| nr_topics | None | - | KMeansでクラスタ数を事前決定 |
| language | japanese | - | 分詞言語設定 |
| seed_topic_list | 8カテゴリ | `TOPIC_SEED` | クラスタリング誘導用シード |
| representation_model | MMR (diversity=0.5) | - | キーワード多様性向上 |

### 後処理
1. 短文テキストフィルタ: 分詞後 < 2語 → -1（26件）
2. 統計距離フィルタ: mean + 1.5 * std → -1（11件）

### 結果サマリー
| 指標 | 値 | 説明 |
|---|---|---|
| トピック数 | 7 | 有効なトピック |
| 外れ値 | 37件 (20.3%) | 短文 + 統計異常 |
| カバレッジ | 79.7% | トピックに分類された文書の割合 |

---

## 2. 2カテゴリ分類 — negative_classify.py

### パイプライン
```
data_with_id.csv + doc_topics.csv → ruri-v3-310m 埋め込み
→ シードラベル生成（半教師学習）→ SVM-RBF訓練 → 全文書予測
```

### パラメータ
| パラメータ | 値 | config定数 | 説明 |
|---|---|---|---|
| 信頼度閾値 | 0.85 | `CLASSIFY_CONFIDENCE_THRESHOLD` | この閾値以上の予測のみ採用 |
| 短文テキスト閾値 | 2 | `CLASSIFY_SHORT_THRESHOLD` | 分詞後この語数未満はcategory 1 |
| 負面キーワード | 52語 | `NEGATIVE_KEYWORDS` | 離婚・詐欺・DV・自殺等 |
| パターン検出 | 14パターン | `PATTERN_COMBOS` | ロマンス詐欺・DV等の複合パターン |

### シードラベル生成
| ソース | カテゴリ | 件数 | 説明 |
|---|---|---|---|
| キーワードマッチング | category 0 | 61 | 離婚・詐欺・DV・自殺関連 |
| ポジティブキーワード | category 1 | 91 | 赤ちゃん・授乳・育児関連 |
| トピック情報 | 補助 | 89 | Topic 0→0, Topic 3/4→1等 |

### 分類器
| パラメータ | 値 | 説明 |
|---|---|---|
| アルゴリズム | SVM-RBF | 泛化能力最好（gap=0.051） |
| kernel | rbf | RBFカーネル |
| class_weight | balanced | クラス不平衡を自動補正 |

### 結果
| 指標 | 値 | 説明 |
|---|---|---|
| category 0 (負面) | 59件 (32.4%) | 離婚・浮気・流産・詐欺・DV・自殺 |
| category 1 (非負面) | 123件 (67.6%) | 育児相談・日常ストレス |
| 高信頼度 | 178件 (97.8%) | 信頼度 >= 0.85 |
| 交差検証 F1 | 0.891 | 5-fold CV |

---

## 3. 感情分析 — sentiment_analysis.py

### モデル
| パラメータ | 値 | 説明 |
|---|---|---|
| モデル | neuralnaut/deberta-wrime-emotions | WRIME微調整DeBERTa |
| batch_size | 16 | `SENTIMENT_BATCH_SIZE` |
| max_length | 512 | `SENTIMENT_MAX_LENGTH` |

### 8感情ラベル
| ラベル | 日本語 | 色 |
|---|---|---|
| joy | 喜び | #FFD700 |
| sadness | 悲しみ | #1E90FF |
| anticipation | 期待 | #FF8C00 |
| surprise | 驚き | #FF69B4 |
| anger | 怒り | #DC143C |
| fear | 恐れ | #8B008B |
| disgust | 嫌悪 | #2E8B57 |
| trust | 信頼 | #4169E1 |

---

## 4. 感情クラスタリング — sentiment_category_diff.py

### パイプライン
```
sentiment_all_diff.csv + 2category_all.csv + doc_topics.csv + real_research.csv
→ StandardScaler → UMAP(2D) → HDBSCAN → クラスタリング
```

### UMAPパラメータ
| パラメータ | 値 | config定数 | 説明 |
|---|---|---|---|
| n_components | 2 | `CLUSTER_UMAP_N_COMPONENTS` | 削減後次元数 |
| n_neighbors | 15 | `CLUSTER_UMAP_N_NEIGHBORS` | 近傍数 |
| min_dist | 0.0 | `CLUSTER_UMAP_MIN_DIST` | 最小距離 |
| metric | cosine | `CLUSTER_UMAP_METRIC` | 距離計測 |

### HDBSCANパラメータ
| パラメータ | 値 | config定数 | 説明 |
|---|---|---|---|
| min_cluster_size | 20 | `CLUSTER_HDBSCAN_MIN_CLUSTER_SIZE` | 最小クラスタサイズ |
| min_samples | 5 | `CLUSTER_HDBSCAN_MIN_SAMPLES` | コアポイント閾値 |

### 結果
| 指標 | 値 | 説明 |
|---|---|---|
| クラスタ数 | 2 | Cluster 0, Cluster 1 |
| Silhouette Score | 0.7603 | 良好な分離 |
| Calinski-Harabasz | 1133.03 | 良好な分離 |
| Davies-Bouldin | 0.3160 | 良好な分離 |

---

## 5. config.py 全パラメータ一覧

```python
# ディレクトリ定義
ROOT_DIR / DATA_DIR / LOGS_DIR / MODELS_DIR / SRC_DIR

# 感情分析
SENTIMENT_BATCH_SIZE = 16
SENTIMENT_MAX_LENGTH = 512

# トピックモデリング
TOPIC_MIN_TOPIC_SIZE = 4
TOPIC_UMAP_N_NEIGHBORS = 15
TOPIC_UMAP_N_COMPONENTS = 2
TOPIC_UMAP_MIN_DIST = 0.1
TOPIC_UMAP_METRIC = "cosine"
TOPIC_RANDOM_SEED = 42
TOPIC_VECTORIZER_MAX_DF = 0.85
TOPIC_VECTORIZER_MIN_DF = 2
TOPIC_EMBEDDING_MODEL = "cl-nagoya/ruri-v3-310m"
TOPIC_EMBEDDING_BATCH_SIZE_CUDA = 64
TOPIC_EMBEDDING_BATCH_SIZE_CPU = 16
TOPIC_EMBEDDING_PREFIX = "トピック: "  # ruri-v3 プレフィックス（分類・クラスタリング用）

# クラスタリング
CLUSTER_UMAP_N_COMPONENTS = 2
CLUSTER_UMAP_N_NEIGHBORS = 15
CLUSTER_UMAP_MIN_DIST = 0.0
CLUSTER_UMAP_METRIC = "cosine"
CLUSTER_RANDOM_SEED = 42
CLUSTER_HDBSCAN_MIN_CLUSTER_SIZE = 20
CLUSTER_HDBSCAN_MIN_SAMPLES = 5

# KMeans
KMEANS_RANDOM_SEED = 42
KMEANS_N_INIT = 10
KMEANS_K_RANGE = range(2, 11)

# 形態素解析
KEEP_POS = {"名詞", "動詞", "形容詞"}
PUNCTUATION_POS = {"補助記号", "記号", "助詞", "助動詞", "接続詞", "感動詞", "接頭詞", "接尾詞"}
STOPWORDS = {...}  # 一般的すぎる単語・分詞偽影

# トピックモデリング シード
TOPIC_SEED = [...]  # 8カテゴリ

# 2カテゴリ分類
CLASSIFY_CONFIDENCE_THRESHOLD = 0.85
CLASSIFY_SHORT_THRESHOLD = 2
NEGATIVE_KEYWORDS = [...]  # 52語
PATTERN_COMBOS = [...]     # 14パターン
```

### ruri-v3 プレフィックス仕様
ruri-v3-310m は以下のプレフィックススキームを使用：
```
"" (空文字) → 意味的エンコーディング
"トピック: " → 分類、クラスタリング、トピック情報
"検索クエリ: " → 検索タスクのクエリ
"検索文書: " → 検索タスクの文書
```
`topic.py` と `negative_classify.py` では `トピック: ` プレフィックスを使用。
