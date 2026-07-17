# 感情差分聚类分析 / Sentiment Difference Clustering Analysis

## 概要 / Overview

基于8种基本情感的差分特征，采用 UMAP 降维 + HDBSCAN 密度聚类方法，对对话的情感变化模式进行无监督聚类，并生成多维度可视化分析。脚本文件：`src/sentiment_category_diff.py`

基于8つの基本感情の差分特徴を用い、UMAP次元削減 + HDBSCAN密度クラスタリングにより、対話の感情変化パターンを教師なしクラスタリングし、多次元可視化分析を生成する。スクリプト: `src/sentiment_category_diff.py`

| 指标 / Indicator | 值 / Value | 说明 / Description |
|---|---|---|
| 输入数据 / Input data | sentiment_all_diff + 2category_all + doc_topics + real_research | 情感差分+二分类标签+主题建模+用户评分 / Sentiment diff + binary labels + topic modeling + user ratings |
| 总样本数 / Total samples | 182 | 全会话数 / Total sessions |
| 聚类特征 / Cluster features | 8维情感差分 (diff_joy ~ diff_trust) | 每种情感的回复前后差值 / Post-reply minus pre-reply diff for each emotion |
| 聚类方法 / Clustering method | UMAP (2D) → HDBSCAN | 密度聚类，无需预设簇数 / Density-based, no predefined k |

---

## 处理流程 / Processing Pipeline

```
数据加载与合并 / Load and merge data
├── sentiment_all_diff.csv — 情感差分 (182条) / Sentiment differences (182 records)
├── 2category_all.csv — 二分类标签 (0=负面, 1=非负面) / Binary labels (0=negative, 1=non-negative)
├── combined_userInput_doc_topics.csv — 主题建模结果 / Topic modeling results
└── real_research.csv — 用户评分 (6项指标) / User ratings (6 indicators)
    ↓
特征提取 / Feature extraction
├── 8维情感差分特征: diff_joy, diff_sadness, ..., diff_trust
└── 填充缺失值为0 / Fill NaN with 0
    ↓
数据标准化 / StandardScaler
└── 零均值单位方差标准化 / Zero-mean unit-variance normalization
    ↓
UMAP 降维 (8D → 2D)
├── n_neighbors=15, min_dist=0.0, metric=cosine
└── random_state=42
    ↓
HDBSCAN 密度聚类
├── min_cluster_size=20, min_samples=5
├── 噪声点标记为 -1 / Noise points labeled as -1
└── prediction_data=True
    ↓
聚类评估 / Clustering evaluation
├── Silhouette Score — 轮廓系数 / Silhouette coefficient
├── Calinski-Harabasz Index — CH指数 / CH index
└── Davies-Bouldin Index — DB指数 / DB index
    ↓
可视化输出 / Visualization output
├── 1) 全数据 UMAP 散布图 / All-data UMAP scatter
├── 2) HDBSCAN 凝缩树图 / Condensed tree plot
├── 3) t-SNE 散布图 / t-SNE scatter
├── 4) 雷达图 — 各簇情感差分轮廓 / Radar chart — emotion diff profile per cluster
├── 5) 用户评分箱线图 / User rating boxplots
├── 6) 情感差分抖动图 / Emotion diff jitter plots
├── 7) 统计指标比较柱状图 / Statistical comparison bar charts
└── 8) 主题分布堆叠图 / Topic distribution stacked bars
    ↓
分类别重复执行 / Repeat per category (cat=0, cat=1)
    ↓
结果整合 / Merge all results
└── 输出: sentiment/cluster_topic_diff/ / Output directory
```

---

## 核心参数 / Core Parameters

### UMAP 降维 / UMAP Dimensionality Reduction

| 参数 / Parameter | 值 / Value | config常量 / Config Constant | 说明 / Description |
|---|---|---|---|
| n_components | 2 | `CLUSTER_UMAP_N_COMPONENTS` | 输出维度 / Output dimensions |
| n_neighbors | 15 | `CLUSTER_UMAP_N_NEIGHBORS` | 近邻数 / Number of neighbors |
| min_dist | 0.0 | `CLUSTER_UMAP_MIN_DIST` | 最小间距 / Minimum distance |
| metric | cosine | `CLUSTER_UMAP_METRIC` | 距离度量 / Distance metric |
| random_state | 42 | `CLUSTER_RANDOM_SEED` | 随机种子 / Random seed |

### HDBSCAN 聚类 / HDBSCAN Clustering

| 参数 / Parameter | 值 / Value | config常量 / Config Constant | 说明 / Description |
|---|---|---|---|
| min_cluster_size | 20 | `CLUSTER_HDBSCAN_MIN_CLUSTER_SIZE` | 最小簇大小 / Minimum cluster size |
| min_samples | 5 | `CLUSTER_HDBSCAN_MIN_SAMPLES` | 核心点阈值 / Core point threshold |
| prediction_data | True | — | 启用软聚类预测 / Enable soft clustering |

### t-SNE 降维 / t-SNE (verification only)

| 参数 / Parameter | 值 / Value | 说明 / Description |
|---|---|---|
| n_components | 2 | 输出维度 / Output dimensions |
| perplexity | max(5, min(30, n//4)) | 自适应 perplexity / Adaptive perplexity |
| random_state | 42 | 随机种子 / Random seed |

---

## 特征量定义 / Feature Definitions

### 8种基本情感差分 / 8 Basic Emotion Differences

| 情感 / Emotion | 特征名 / Feature | 颜色 / Color | 说明 / Description |
|---|---|---|---|
| Joy / 喜悦 | diff_joy | #FFD700 | 快乐、幸福 / Happiness, delight |
| Sadness / 悲伤 | diff_sadness | #1E90FF | 悲伤、失落 / Sadness, loss |
| Anticipation / 期待 | diff_anticipation | #FF8C00 | 期待、盼望 / Expectation, hope |
| Surprise / 惊讶 | diff_surprise | #FF69B4 | 惊讶、意外 / Astonishment, surprise |
| Anger / 愤怒 | diff_anger | #DC143C | 愤怒、不满 / Anger, frustration |
| Fear / 恐惧 | diff_fear | #8B008B | 恐惧、焦虑 / Fear, anxiety |
| Disgust / 厌恶 | diff_disgust | #2E8B57 | 厌恶、反感 / Disgust, aversion |
| Trust / 信任 | diff_trust | #4169E1 | 信任、依赖 / Trust, reliance |

**差分计算**：每个情感维度 = AI回复后的模型情感值 − 用户输入时的模型情感值

**差分の計算**: 各感情次元 = AI返信後のモデル感情値 − ユーザー入力時のモデル感情値

### 用户评分指标 / User Rating Indicators

| 评分项 / Rating | 日文键名 / Key | 说明 / Description |
|---|---|---|
| 共感性 / Empathy | kyokansei | AI回复的共情程度 / AI's empathic response |
| 医学准确性 / Medical Accuracy | igakuseikakusei | 医学信息的正确性 / Correctness of medical info |
| 安全性 / Safety | anzensei | 回复的安全程度 / Safety of response |
| 有害性 / Harmfulness | yuugaisei | 回复的有害程度 / Potential harm of response |
| AI依赖 / AI Dependency | aiirai | 对AI的依赖程度 / Dependency on AI |
| シカファンシー / Sycophancy | shikafanshi | 讨好倾向 / Sycophantic tendency |

---

## 分类处理逻辑 / Per-Category Processing

脚本在全局聚类之后，对两个类别分别执行独立的 UMAP → HDBSCAN 管道：

スクリプトは全体クラスタリングの後、2つのカテゴリに対して個別のUMAP → HDBSCANパイプラインを実行する：

```
全数据聚类 (n=182) → 全局UMAP散布图 + 全局雷达图 + 全局评分箱线图
    ↓
分类别处理 / Per-category processing
├── Category 0 (负面, n≈59) → category0/ 子目录
│   └── run_pipeline(): 独立 UMAP → HDBSCAN → 可视化
└── Category 1 (非负面, n≈123) → category1/ 子目录
    └── run_pipeline(): 独立 UMAP → HDBSCAN → 可视化
    ↓
合并两个分类结果 → 覆盖写入 all_diff_clusters.csv
```

**注意**：分类别管道使用与全局相同的参数 (n_neighbors=15, min_dist=0.0, min_cluster_size=20, min_samples=5)。

**注意**: カテゴリ別パイプラインはグローバルと同じパラメータを使用 (n_neighbors=15, min_dist=0.0, min_cluster_size=20, min_samples=5)。

---

## 可视化类型 / Visualization Types

### 1. UMAP 散布图 / UMAP Scatter Plot

- **文件名格式** / Filename: `{prefix}_diff_umap_2d.html` / `.svg`
- **内容** / Content: 2D UMAP 降维结果，颜色区分簇，灰色为噪声点
- 2D UMAP次元削減結果。色でクラスタを区別、灰色はノイズ

### 2. HDBSCAN 凝缩树 / Condensed Tree Plot

- **文件名格式** / Filename: `{prefix}_diff_condensed_tree.svg`
- **内容** / Content: HDBSCAN 的层次聚类结构，展示簇的分裂与合并过程
- HDBSCANの階層クラスタリング構造を可視化

### 3. t-SNE 散布图 / t-SNE Scatter Plot

- **文件名格式** / Filename: `{prefix}_diff_tsne.html` / `.svg`
- **内容** / Content: t-SNE 降维验证，确认聚类在不同降维方法下的一致性
- t-SNE次元削減による検証。異なる削減手法でのクラスタリング一貫性を確認

### 4. 雷达图 / Radar Chart

- **文件名格式** / Filename: `{prefix}_diff_radar.html` / `.svg`
- **内容** / Content: 每个簇在8种情感差分上的均值轮廓，红色虚线表示差分=0
- 各クラスタの8感情差分の平均プロファイル。赤一点線は差分=0を示す

### 5. 用户评分箱线图 / User Rating Boxplots

- **文件名格式** / Filename: `{prefix}_diff_rating_boxplot.html` / `.svg`
- **内容** / Content: 各簇在6项用户评分上的分布对比
- 各クラスタの6つのユーザー評点の分布比較

### 6. 情感差分抖动图 / Emotion Diff Jitter Plots

- **文件名格式** / Filename: `{prefix}_diff_jitter.html` / `.svg`
- **内容** / Content: 每个簇内各样本在各情感维度的散点分布，附带均值/最大值/最小值连线
- 各クラスタ内各样本の感情差分分布。平均/最大/最小の折れ線付き

### 7. 统计指标比较 / Statistical Comparison

- **文件名格式** / Filename: `{prefix}_diff_cluster_stats.html` / `.svg`
- **内容** / Content: 2×2 子图，分别展示 Mean / Max / Min / Standard Deviation
- 2×2サブプロットで Mean / Max / Min / 標準偏差を比較

### 8. 主题分布堆叠图 / Topic Distribution Stacked Bars

- **文件名格式** / Filename: `{prefix}_topic_cluster_distribution.html` / `.svg`
- **内容** / Content: 各簇内7个主题的比例堆叠柱状图
- 各クラスタ内7トピックの割合積み上げ棒グラフ

---

## 聚类评估指标 / Clustering Evaluation Metrics

| 指标 / Metric | 公式含义 / Meaning | 取值范围 / Range | 越...越好 / Better when |
|---|---|---|---|
| Silhouette Score | 簇内紧密度与簇间分离度的比值 / Ratio of intra-cluster cohesion to inter-cluster separation | [-1, 1] | 越大越好 / Higher is better |
| Calinski-Harabasz Index | 类间方差与类内方差的比值 / Ratio of between-cluster to within-cluster variance | [0, ∞) | 越大越好 / Higher is better |
| Davies-Bouldin Index | 簇间相似度均值 / Average similarity between clusters | [0, ∞) | 越小越好 / Lower is better |

**评估前提条件** / Prerequisites: 簇数 ≥ 2 且非噪声样本数 ≥ 20 时才计算 / Calculated only when clusters ≥ 2 and non-noise samples ≥ 20

---

## 输出文件 / Output Files

### 全局结果 / Global Results

| 文件 / File | 内容 / Content |
|---|---|
| `all_diff_clusters.csv` | 全数据聚类结果 (含 cluster, umap_0, umap_1 列) |
| `all_diff_centers.csv` | 各簇在8维差分上的均值中心 |
| `all_diff_umap_2d.html/.svg` | UMAP 2D 散布图 |
| `all_diff_condensed_tree.svg` | HDBSCAN 凝缩树 |
| `all_diff_tsne.html/.svg` | t-SNE 2D 散布图 |
| `all_diff_radar.html/.svg` | 情感差分雷达图 |
| `all_diff_rating_boxplot.html/.svg` | 用户评分箱线图 |
| `all_diff_jitter.html/.svg` | 情感差分抖动图 |
| `all_diff_cluster_stats.html/.svg` | 统计指标比较图 |
| `all_topic_cluster_distribution.html/.svg` | 主题分布堆叠图 |

### 分类别结果 / Per-Category Results

| 目录 / Directory | 内容 / Content |
|---|---|
| `category0/` | Category 0 (负面) 的独立聚类结果 / Independent clustering for negative category |
| `category1/` | Category 1 (非负面) 的独立聚类结果 / Independent clustering for non-negative category |

每个子目录包含与全局相同格式的文件，文件名前缀为 `category{0|1}_`。

各サブディレクトリにはグローバルと同じフォーマットのファイルが含まれ、ファイル名プレフィックスは `category{0|1}_`。

---

## 聚类结果 / Clustering Results

| 指标 / Indicator | 值 / Value | 说明 / Description |
|---|---|---|
| 全局簇数 / Global clusters | 2 | Cluster 0, Cluster 1 |
| Silhouette Score | 0.7603 | 良好的分离度 / Good separation |
| Calinski-Harabasz | 1133.03 | 良好的分离度 / Good separation |
| Davies-Bouldin | 0.3160 | 良好的分离度 / Good separation |
| 噪声比例 / Noise ratio | 见各分类结果 | label = -1 的样本 / Samples with label = -1 |
