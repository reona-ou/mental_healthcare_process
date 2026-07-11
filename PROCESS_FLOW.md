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

---

## 1. CSV前処理 — create_csv.py

```
入力: mochiko-line-bot-prod-*.csv + real_research.csv
│
├── チャット履歴読み込み
├── Orchestratorリクエストと返信をペアリング（FIFOキュー）
├── 無意味入力フィルタ（キーワード・パターン・曖昧表現）
├── 重複除去（userId + replyType + userInput + persona）
├── データ分離（with_id / cant_find_id / mochiko / pen_sensei）
└── ID照合表作成
```

**出力** (data/):
- `data_with_id.csv` — 全データ（満足度スコア付き）
- `data_mochiko.csv` — もちこデータ
- `data_pen_sensei.csv` — ペン先生データ
- `data_cant_find_id.csv` — ID不一致データ
- `cleaned_data.csv` — クリーン済みデータ
- `data.csv` — ペアリング済みデータ
- `id_comparison.csv` — ID照合表

---

## 2. 単語頻度 — word_process.py

```
入力: data_mochiko.csv + data_pen_sensei.csv + data_with_id.csv
│
├── fugashi で形態素解析
├── emoji 抽出・カウント
├── 品詞別ランキング（名詞・動詞・emoji）
├── チャットボット別（mochiko / pen_sensei）
│   ├── input 単語統計
│   └── output 単語統計
└── 全ユーザー入力統計（重複除去後）
```

**出力** (data/word_counts/):
- `{name}_input_words.csv` — 入力単語統計
- `{name}_output_words.csv` — 応答単語統計
- `{name}_input_n.csv` — 入力名詞ランキング
- `{name}_input_v.csv` | `{name}_input_emojis.csv` — 同上
- `input/input_words.csv` — 全入力単語統計

---

## 3. 単語頻度可視化 — word_visualization.py

```
入力: word_process.py の出力CSV
│
├── 散布図（Mochiko vs Pen Sensei 単語頻度比較）
│   ├── 全単語
│   ├── 名詞のみ
│   ├── 動詞のみ
│   └── emoji のみ
├── 棒グラフ（全ユーザー入力 Top 30）
│   ├── 全単語
│   ├── 名詞のみ
│   ├── 動詞のみ
│   └── emoji のみ
├── ワードクラウド（名詞）
│   ├── 全体 output（Mochiko + Pen Sensei 合併）
│   ├── Mochiko output
│   ├── Pen Sensei output
│   └── 全ユーザー input
└── Treemap（単語頻度ツリーマップ）
    ├── 全体 output
    ├── Mochiko output
    ├── Pen Sensei output
    └── 全ユーザー input
```

**出力** (data/word_counts/):
- `output_visualization_*.html` — 散布図
- `input_visualization_*.html` — 棒グラフ
- `wordcloud_*.html` — ワードクラウド
- `treemap_*.html` — Treemap

---

## 4. トピック分析 — topic.py

```
入力: data_with_id.csv
│
├── fugashi で形態素解析（名詞・動詞・形容詞のみ）
├── 短文テキストフィルタ（分詞後<2語 → -1）
├── ruri-v3-310m で768次元埋め込み生成
├── KMeans クラスタリング（7クラスタ）
├── 統計距離フィルタ（mean + 1.5 * std → -1）
└── 出力
```

**出力** (data/topic_modeling/):
- `combined_userInput_doc_topics.csv` — 文書ごとのトピック割当
- `combined_userInput_topic_info.csv` — トピック情報
- `combined_userInput_topic_keywords.csv` — トピック別キーワード

**パラメータ**:
- 埋め込み: ruri-v3-310m (768次元)
- UMAP: n_neighbors=15, n_components=2, min_dist=0.1
- クラスタリング: KMeans (7クラスタ)
- MMR: diversity=0.5

---

## 5. トピック2カテゴリ分類 — negative_classify.py

```
入力: data_with_id.csv + doc_topics.csv
│
├── ruri-v3-310m で埋め込み生成
├── シードラベル生成（半教師学習）
│   ├── キーワードマッチング → category 0（離婚・詐欺・DV等）
│   ├── ポジティブキーワード → category 1（育児・授乳等）
│   └── トピック情報 → 補助ラベル
├── SVM-RBF 分類器訓練
├── 全文書予測（信頼度 >= 0.85 で分類）
├── 短文テキスト → category 1
└── 出力
```

**出力**: `data/2category_all.csv`
- category 0: 負面（離婚・浮気・流産・詐欺・DV・自殺）
- category 1: 非負面（育児相談・日常ストレス等）

**パラメータ**:
- 分類器: SVM-RBF (class_weight=balanced)
- 信頼度閾値: 0.85
- 交差検証 F1: 0.891

---

## 6. 感情分析 — sentiment_analysis.py

```
入力: data_with_id.csv
│
├── DeBERTa (neuralnaut/deberta-wrime-emotions) で8感情分析
│   ├── userInput → 8感情スコア
│   └── replyText → 8感情スコア
├── (userId, userInput) でグループ化 → 会話構造
├── ユーザー別統計量計算
└── 出力
```

**8感情**: joy, sadness, anticipation, surprise, anger, fear, disgust, trust

**出力** (data/sentiment/):
- `sentiment.csv` — 全行の感情スコア
- `conversations.csv` — 会話単位サマリー
- `sentiment_stats.csv` — ユーザー別統計
- `charts_wrime/` — ユーザー別折れ線グラフ（HTML+SVG）

---

## 7. 感情差分 — sentiment_difference.py

```
入力: sentiment.csv
│
├── ユーザーごとの差分計算（8×8マトリクス）
│   └── diff = input_感情 - reply_感情
├── 統計量計算（max, min, mean, median, Q1, Q3）
├── 全session統一差分（reply - input）
├── 統計量CSV保存
└── 可視化（6種統計の棒グラフ）
```

**出力** (data/sentiment/sentiment_diff/):
- `{userId}_differences.csv` — ユーザー別差分
- `{userId}_statistics.csv` — ユーザー別統計
- `sentiment_all_diff.csv` — 全session統一差分
- `sentiment_all_diff_statistics.csv` — 全session統計
- `visualizations/` — 統計棒グラフ（HTML+SVG）

---

## 8. 感情可視化 — sentiment_summary_charts.py

```
入力: sentiment.csv
│
├── userInput 分析
│   ├── 主導感情分布（棒グラフ）
│   ├── 感情レーダー（平均値）
│   ├── 統計指標比較（mean/median/std/max/min）
│   └── 感情スコア分布（箱ひげ図）
├── reply 分析（同上）
└── overall 分析（input + reply 混合、同上）
```

**出力** (data/sentiment/):
- `input_analysis/` — userInput 分析
- `reply_analysis/` — reply 分析
- `overall_analysis/` — 全体分析

---

## 9. 感情クラスタリング — sentiment_category_diff.py

```
入力: sentiment_all_diff.csv + 2category_all.csv + doc_topics.csv + real_research.csv
│
├── データ統合（差分特徴量 + カテゴリ + トピック + 評点）
├── 全データ統一クラスタリング
│   ├── StandardScaler
│   ├── UMAP (2D, cosine)
│   ├── HDBSCAN
│   └── 評価指標（Silhouette, CH, DBI）
├── カテゴリ別クラスタリング（category 0 / 1）
│   └── 上記と同様パイプライン
├── 可視化
│   ├── UMAP 散布図
│   ├── t-SNE
│   ├── 凝縮樹図
│   ├── レーダーチャート
│   ├── ユーザー評点箱ひげ図
│   ├── 感情差分ジッタープロット
│   ├── 統計指標比較
│   └── トピック分布
└── 出力
```

**出力** (data/sentiment/cluster_topic_diff/):
- `all_diff_clusters.csv` — 全データクラスタ割当
- `all_diff_centers.csv` — クラスタ中心
- `category{0,1}/` — カテゴリ別分析結果
- 各種HTML+SVG可視化ファイル
