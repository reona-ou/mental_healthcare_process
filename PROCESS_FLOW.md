# 処理フロー

## 実行順序

```
1. create_csv.py              → CSV前処理
2. word_process.py            → 単語頻度
3. word_visualization.py      → 単語頻度可視化
4. topic.py                   → トピック分析
5. negative_classify.py       → トピック2カテゴリ分類
6. sentiment_analysis.py      → 感情分析
7. sentiment_difference.py    → 感情差分
8. sentiment_summary_charts.py → 感情可視化
9. sentiment_category_diff.py → 感情クラスタリング
```

全設定パラメータは `src/config.py` に集約。再現性確保のため `random_state=42` を共通使用。

---

## 1. CSV前処理 — create_csv.py

**目的**: チャットボットの対話ログを分析可能なCSV形式に整形する

**入力**:
- `mochiko-line-bot-prod-*.csv` — LINE ボットの対話ログ
- `real_research.csv` — ユーザー評価データ（6指標）

**処理フロー**:

```
対話ログ読み込み (818行)
    ↓
Orchestratorリクエストと返信をFIFOキューでペアリング
    ↓
会話単位に整形（246件にペアリング）
    ↓
無意味入力フィルタ（32件除外）
├── 固定キーワードアクション（「もちこが共感」等7種）
├── ツールコマンド（「子育てまんだら」等9種）
├── 同意確認自動返信
├── 曖昧表現・フィラー句（「そうかも」「かもね」等14種）
└── ペルソナ切替クイックコマンド
    ↓
重複除去（4件除外）
    ↓
データ分離
├── with_id (182件) — 満足度スコア付き
├── cant_find_id (31件) — ID不一致
├── mochiko (168件)
└── pen_sensei (13件)
    ↓
ID照合表作成
```

**出力** (`data/`):
| ファイル | 内容 | 件数 |
|---|---|---|
| `data_with_id.csv` | 全データ（セッションID, ユーザーID, 満足度, ペルソナ, 返信タイプ, ユーザー入力, ボット返信） | 182 |
| `data_mochiko.csv` | もちこペルソナのデータ | 168 |
| `data_pen_sensei.csv` | ペン先生ペルソナのデータ | 13 |
| `data_cant_find_id.csv` | ID不一致データ | 31 |
| `cleaned_data.csv` | フィルタ・重複除去済みデータ | 213 |
| `data.csv` | ペアリング済みデータ | 246 |
| `id_comparison.csv` | チャット vs 評価のID照合表 | - |

---

## 2. 単語頻度 — word_process.py

**目的**: チャットボットの言語使用パターンを分析する

**入力**: `data_mochiko.csv` + `data_pen_sensei.csv` + `data_with_id.csv`

**処理フロー**:

```
データ読み込み
    ↓
fugashi で形態素解析
├── 品詞フィルタ: 名詞・動詞・形容詞のみ
├── 句読点排除: 補助記号, 記号
├── 助詞・助動詞排除（オプション）
└── emoji 抽出・カウント
    ↓
3つの出力カテゴリ
├── チャットボット別（mochiko / pen_sensei）
│   ├── userInput（ユーザー入力）
│   └── replyText（ボット応答）
└── 全ユーザー入力（userId, userInput で重複除去後）
```

**出力** (`data/word_counts/`):
| ディレクトリ | ファイル | 内容 |
|---|---|---|
| `mochiko/` | `{name}_input_words.csv` | もちこ入力の全単語統計 |
| | `{name}_output_words.csv` | もちこ応答の全単語統計 |
| | `{name}_input_n.csv` | もちこ入力の名詞ランキング |
| | `{name}_input_v.csv` | もちこ入力の動詞ランキング |
| | `{name}_input_emojis.csv` | もちこ入力のemojiランキング |
| | `{name}_output_*.csv` | もちこ応答の品詞別ランキング |
| `pen_sensei/` | `{name}_*.csv` | ペン先生の同様データ |
| `input/` | `input_words.csv` | 全ユーザー入力の単語統計 |
| | `input_n.csv` | 全ユーザー入力の名詞ランキング |

---

## 3. 単語頻度可視化 — word_visualization.py

**目的**: 単語頻度データを多角的に可視化する

**入力**: `word_process.py` の出力CSV

**処理フロー**:

```
散布図比較（Mochiko vs Pen Sensei）
├── 全単語 / 名詞 / 動詞 / emoji の4種
├── 共通語は RdYlBu 色スケールで偏向表示
└── 独有語は青/赤で表示 + Top 10排行榜
    ↓
棒グラフ（全ユーザー入力 Top 30）
├── 全単語 / 名詞 / 動詞 / emoji の4種
└── 水平棒グラフ（高頻度順）
    ↓
ワードクラウド（名詞のみ）
├── 全体 output（Mochiko + Pen Sensei 合併）
├── Mochiko output
├── Pen Sensei output
└── 全ユーザー input
    ↓
Treemap（単語頻度ツリーマップ）
├── 全体 output
├── Mochiko output
├── Pen Sensei output
└── 全ユーザー input
```

**出力** (`data/word_counts/`):
| ファイル | 内容 | 形式 |
|---|---|---|
| `output_visualization_*.html` | 散布図（Mochiko vs Pen Sensei） | HTML + SVG |
| `input_visualization_*.html` | 棒グラフ（ユーザー入力 Top 30） | HTML + SVG |
| `wordcloud_*.html` | ワードクラウド（名詞） | HTML + SVG |
| `treemap_*.html` | Treemap（単語頻度ツリーマップ） | HTML + SVG |

---

## 4. トピック分析 — topic.py

**目的**: ユーザー入力をトピックに分類する

**入力**: `data_with_id.csv`

**処理フロー**:

```
userInput 読み込み (182件)
    ↓
fugashi で形態素解析
├── 品詞フィルタ: 名詞・動詞・形容詞のみ
├── ストップワード除去（config.py 定義）
└── 英語サフィックス除去（toilet-toilet → toilet）
    ↓
短文テキスト識別（分詞後 < 2語 → -1）
├── 短文: 26件 → スキップ
└── 対象: 156件 → クラスタリング
    ↓
ruri-v3-310m で 768次元埋め込み生成
    ↓
UMAP 次元削減（768次元 → 2次元）
├── n_neighbors=15, min_dist=0.1
└── metric=cosine
    ↓
KMeans クラスタリング（7クラスタ）
├── min(7, len(texts)//10) で自動決定
└── n_init=10
    ↓
BERTopic トピック表現
├── シードトピック誘導（8カテゴリ）
├── MMR (diversity=0.5) でキーワード多様性向上
└── CountVectorizer (ngram_range=(1,2))
    ↓
統計距離フィルタ（mean + 1.5 * std → -1）
└── 9件が異常値として除外
    ↓
結果保存
```

**出力** (`data/topic_modeling/`):
| ファイル | 内容 |
|---|---|
| `combined_userInput_doc_topics.csv` | 文書ごとのトピック割当（document_index, original_text, tokenized_text, topic_id） |
| `combined_userInput_topic_info.csv` | トピック情報（Topic, Count, Name） |
| `combined_userInput_topic_keywords.csv` | トピック別キーワード（topic_id, keyword, score） |

**結果サマリー**:
| 指標 | 値 |
|---|---|
| トピック数 | 7 |
| 外れ値 | 35件（19.2%） |
| トピックカバレッジ | 80.8% |

**トピック分布**:
| Topic | 件数 | キーワード | 説明 |
|---|---|---|---|
| -1 | 35 | - | 短文 + 統計異常 |
| 0 | 27 | 地域, 保健, 育児, 相談 | 育児支援・地域 |
| 1 | 24 | 赤ちゃん, 離婚, 家事, きつい | 育児・離婚 |
| 2 | 23 | おっぱい, 育児, 寝る | 産後・授乳 |
| 3 | 21 | 流産, 妊娠, 辛い | 流産・妊娠 |
| 4 | 20 | 相談, 苦手, 連絡 | 人間関係・相談 |
| 5 | 18 | 離婚, 浮気, 浮気相手 | 離婚・浮気 |
| 6 | 14 | 寝る, 箇月, 安心 | 産後睡眠 |

---

## 5. トピック2カテゴリ分類 — negative_classify.py

**目的**: トピックを「負面」と「非負面」の2カテゴリに分類する（半教師学習）

**入力**: `data_with_id.csv` + `combined_userInput_doc_topics.csv`

**処理フロー**:

```
データ読み込み（182件）
    ↓
ruri-v3-310m で 768次元埋め込み生成
    ↓
シードラベル生成（半教師学習）
├── キーワードマッチング → category 0（61件）
│   └── 離婚・詐欺・DV・自殺・浮気・流産 関連キーワード
├── ポジティブキーワード → category 1
│   └── 赤ちゃん・授乳・母乳・育児・相談・保健 関連キーワード
└── トピック情報 → 補助ラベル
    ├── Topic 5（離婚・浮気）→ category 0
    ├── Topic 2, 6（産後）→ category 1
    ├── Topic 3（流産）→ 条件付き分類
    ├── Topic 0（育児支援）→ 条件付き分類
    ├── Topic 1（育児離婚）→ 条件付き分類
    └── Topic 4（人間関係）→ 条件付き分類
    ↓
SVM-RBF 分類器訓練
├── kernel: rbf
├── class_weight: balanced
├── 交差検証 F1: 0.891
└── train-CV gap: 0.051（過学習なし）
    ↓
全文書予測
├── 信頼度 >= 0.85 → 予測結果を採用
├── 信頼度 < 0.85 → category 1
├── 短文テキスト（< 2語）→ category 1
└── シードラベル → そのまま採用
    ↓
結果保存
```

**出力**: `data/2category_all.csv`
| カテゴリ | 内容 | 件数 | 割合 |
|---|---|---|---|
| category 0 | 負面（離婚・浮気・流産・詐欺・DV・自殺） | 59 | 32.4% |
| category 1 | 非負面（育児相談・日常ストレス等） | 123 | 67.6% |

**高信頼度**: 178件（97.8%）が信頼度 >= 0.85 で分類

---

## 6. 感情分析 — sentiment_analysis.py

**目的**: 各対話の8感情スコアを分析する

**入力**: `data_with_id.csv`

**処理フロー**:

```
DeBERTa モデル読み込み
├── neuralnaut/deberta-wrime-emotions
└── WRIME データセット微調整済み
    ↓
userInput 感情分析（182件）
└── 8感情スコア（0.0〜1.0）
    ↓
replyText 感情分析（182件）
└── 8感情スコア（0.0〜1.0）
    ↓
(userId, userInput) でグループ化 → 会話構造
├── 同一入力の複数返信を統合
├── Input 感情: 最初の行を使用
└── Reply 感情: 最後の行を使用（ReplyInterruptPersona優先）
    ↓
ユーザー別統計量計算
└── 各感情の平均スコア
    ↓
可視化
└── ユーザー別8感情折れ線グラフ（Reply + Input）
```

**8感情ラベル**:
| 感情 | 日本語 | 色 |
|---|---|---|
| joy | 喜び | #FFD700 (ゴールド) |
| sadness | 悲しみ | #1E90FF (ブルー) |
| anticipation | 期待 | #FF8C00 (オレンジ) |
| surprise | 驚き | #FF69B4 (ピンク) |
| anger | 怒り | #DC143C (レッド) |
| fear | 恐れ | #8B008B (パープル) |
| disgust | 嫌悪 | #2E8B57 (グリーン) |
| trust | 信頼 | #4169E1 (ブルー) |

**出力** (`data/sentiment/`):
| ファイル | 内容 |
|---|---|
| `sentiment.csv` | 全行の感情スコア（input_*, reply_* の16列追加） |
| `conversations.csv` | 会話単位サマリー（conv_id, userId, input情感, reply情感） |
| `sentiment_stats.csv` | ユーザー別統計（各感情の平均） |
| `charts_wrime/` | ユーザー別折れ線グラフ（HTML + SVG） |

---

## 7. 感情差分 — sentiment_difference.py

**目的**: AI返答による感情変化を分析する

**入力**: `sentiment.csv`

**処理フロー**:

```
各ユーザーごとに処理
├── 8×8マトリクスの差分計算
│   └── diff_ij = input_i - reply_j
├── 統計量計算（max, min, mean, median, Q1, Q3）
└── CSV保存（差分 + 統計量）
    ↓
全session統一出力
├── diff_joy = reply_joy - input_joy
├── diff_sadness = reply_sadness - input_sadness
├── ...（8感情分）
└── CSV保存
    ↓
全session統計量計算・保存
    ↓
可視化（2行3列の棒グラフ）
├── max / min / mean / Q1 / median / Q3
└── 各感情の差分値を色分け表示
```

**出力** (`data/sentiment/sentiment_diff/`):
| パス | 内容 |
|---|---|
| `differences/{userId}_differences.csv` | ユーザー別8×8差分行列 |
| `statistics/{userId}_statistics.csv` | ユーザー別8×8統計量 |
| `sentiment_all_diff.csv` | 全session統一差分（diff_joy〜diff_trust） |
| `sentiment_all_diff_statistics.csv` | 全session統計（8感情 × 6統計量） |
| `visualizations/all_sessions_statistics.html` | 統計棒グラフ |

---

## 8. 感情可視化 — sentiment_summary_charts.py

**目的**: 感情分析結果を多角的に可視化する

**入力**: `sentiment.csv`

**処理フロー**:

```
3カテゴリで分析
├── userInput（ユーザー入力）
├── reply（ボット応答）
└── overall（input + reply 混合）
    ↓
各カテゴリにつき4種のチャートを生成
├── 主導感情分布（棒グラフ）
│   └── 各データで最大スコアの感情をカウント
├── 感情レーダー（平均値）
│   └── 8感情の平均スコアをレーダーで表示
├── 統計指標比較（2行3列の棒グラフ）
│   └── mean / median / std / max / min
└── 感情スコア分布（箱ひげ図）
    └── 各感情の分布を箱ひげ図で表示
    ↓
overall はinput + reply を合併して分析
```

**出力** (`data/sentiment/`):
| ディレクトリ | 内容 |
|---|---|
| `input_analysis/` | userInput の4種チャート + 統計CSV |
| `reply_analysis/` | reply の4種チャート + 統計CSV |
| `overall_analysis/` | 全体の4種チャート + 統計CSV |

各ディレクトリのファイル:
- `{prefix}_dominant_emotion.html` — 主導感情分布
- `{prefix}_radar.html` — 感情レーダー
- `{prefix}_stats_comparison.html` — 統計指標比較
- `{prefix}_boxplot.html` — 感情スコア分布
- `{prefix}_statistics.csv` — 統計指標

---

## 9. 感情クラスタリング — sentiment_category_diff.py

**目的**: 感情差分特徴量でクラスタリングし、AI返答パターンを分析する

**入力**:
- `sentiment_all_diff.csv` — 8差分特徴量
- `2category_all.csv` — 2カテゴリラベル
- `combined_userInput_doc_topics.csv` — トピック割当
- `real_research.csv` — ユーザー評価（6指標）

**処理フロー**:

```
データ統合
├── 差分特徴量（8次元: diff_joy〜diff_trust）
├── カテゴリラベル（0=負面, 1=非負面）
├── トピック割当（Topic 0〜6, -1）
└── ユーザー評価（共感性, 医学正確性, 安全性, 有害性, AI依存, シカファンシー）
    ↓
全データ統一クラスタリング
├── StandardScaler（8差分特徴量を正規化）
├── UMAP（2次元削減, cosine距離）
├── HDBSCAN（min_cluster_size=20, min_samples=5）
└── 評価指標
    ├── Silhouette Score: 0.7603（良好）
    ├── Calinski-Harabasz: 1133.03（良好）
    └── Davies-Bouldin: 0.3160（良好）
    ↓
クラスタ解釈
├── Cluster 0 (n=71): anger↑, disgust↑, sadness↑ → AI返答が負の感情を増幅
└── Cluster 1 (n=110): anticipation↑, trust↑, joy↑ → AI返答が効果的に安心
    ↓
カテゴリ別クラスタリング（category 0 / 1）
└── 上記と同様パイプライン
    ↓
可視化（各クラスタにつき8種）
├── UMAP 散布図（2次元）
├── t-SNE（元空間確認）
├── 凝縮樹図（HDBSCAN階層構造）
├── レーダーチャート（差分パターン比較）
├── ユーザー評点箱ひげ図（6指標のクラスタ別分布）
├── 感情差分ジッタープロット（各感情の分布）
├── 統計指標比較（Mean/Max/Min/SD）
└── クラスタ内トピック分布
```

**出力** (`data/sentiment/cluster_topic_diff/`):
| ファイル | 内容 |
|---|---|
| `all_diff_clusters.csv` | 全データクラスタ割当 |
| `all_diff_centers.csv` | クラスタ中心値 |
| `all_diff_umap_2d.html` | UMAP散布図 |
| `all_diff_tsne.html` | t-SNE散布図 |
| `all_diff_radar.html` | レーダーチャート |
| `all_diff_rating_boxplot.html` | 評点箱ひげ図 |
| `all_diff_jitter.html` | ジッタープロット |
| `all_diff_cluster_stats.html` | 統計指標比較 |
| `all_topic_cluster_distribution.html` | トピック分布 |
| `category0/` | category 0（負面）別分析 |
| `category1/` | category 1（非負面）別分析 |
