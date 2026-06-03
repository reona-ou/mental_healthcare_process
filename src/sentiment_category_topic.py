"""
感情分析クラスタリングスクリプト
情感分析聚类脚本

Pipeline / 流水线:
  StandardScaler — 標準化 / 标准化
  PCA — 主成分分析による次元削減 / 主成分分析降维
  UMAP — 非線形次元削減 (コサイン類似度) / 非线性降维 (余弦相似度)
  HDBSCAN — 密度ベースクラスタリング / 基于密度的聚类

入力特徴量 / 输入特征: 感情16次元のみ (ユーザー評点は除外)
  入力感情 8種 (joy, sadness, anticipation, surprise, anger, fear, disgust, trust)
  返答感情 8種 (同上)

ユーザー評点はクラスタリングに使用しない。
聚类后通过各クラスタごとの評点分布でクラスタの意味を解釈する。
用户评分不用于聚类，聚类后通过各聚类的评分分布来解释聚类的含义。

注意: persona と replyType はクラスタリングの特徴量には使用しない。
      クラスタ解釈時の分析のみに使用する。
注意: persona 和 replyType 不作为聚类特征，仅用于聚类结果的解释分析。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
import hdbscan
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import config


# データ読み込み / 数据读取
df_sentiment = pd.read_csv(config.DATA_DIR / 'sentiment/sentiment.csv', on_bad_lines='warn')
df_category = pd.read_csv(config.DATA_DIR / '2category_all.csv', on_bad_lines='warn')

for col in ['userId', 'session_id']:
    df_sentiment[col] = df_sentiment[col].astype(str)
    df_category[col] = df_category[col].astype(str)

df = df_sentiment.merge(df_category[['session_id', 'category']], on='session_id', how='left')

# 感情特徴量の定義 / 定义情感特征
input_features = [
    'input_joy', 'input_sadness', 'input_anticipation', 'input_surprise',
    'input_anger', 'input_fear', 'input_disgust', 'input_trust'
]
reply_features = [
    'reply_joy', 'reply_sadness', 'reply_anticipation', 'reply_surprise',
    'reply_anger', 'reply_fear', 'reply_disgust', 'reply_trust'
]
cluster_features = input_features + reply_features

for col in cluster_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 出力設定 / 输出设置
output_dir = config.DATA_DIR / 'sentiment/kmeans_topic'
output_dir.mkdir(parents=True, exist_ok=True)

# レーダーチャート用の設定 / 雷达图设置
emotion_categories = ['joy', 'sadness', 'anticipation', 'surprise', 'anger', 'fear', 'disgust', 'trust']
emotion_labels = [
    'Joy / 喜び', 'Sadness / 悲しみ', 'Anticipation / 期待', 'Surprise / 驚き',
    'Anger / 怒り', 'Fear / 恐れ', 'Disgust / 嫌悪', 'Trust / 信頼'
]
radar_categories = emotion_labels + [emotion_labels[0]]

cluster_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

# ユーザー評点の列名とラベル / 用户评分列名和标签
RATING_COLS = ['kyukansei', 'igakuseikakusei', 'anzensei', 'yuugaisei', 'aiirai', 'shikafannshi']
RATING_LABELS = {
    'kyukansei': '共感性',
    'igakuseikakusei': '医学正確性（診断、処方行為の回避）',
    'anzensei': '安全性（ハルシネーションの有無）',
    'yuugaisei': '有害性（差別的・攻撃的表現、不安を煽る表現）',
    'aiirai': 'AIへの依存を助長するような発言',
    'shikafannshi': 'シカファンシー（ユーザーを過度に持ち上げるあまり嘘をつくような発言）'
}


def make_hover_text(row):
    """ホバー時のテキストを生成 / 生成悬停文本"""
    return (f"session: {row['session_id']}<br>"
            f"persona: {row.get('persona', '?')}<br>"
            f"replyType: {row.get('replyType', '?')}<br>"
            f"input: {str(row.get('userInput', ''))[:40]}...")


def make_radar_layout(title, width=1400, height=650):
    """レーダーチャート共通レイアウトを返す / 返回雷达图通用布局"""
    return dict(
        title=dict(text=title, x=0.5, font=dict(size=20), y=0.98),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor='white'),
        polar2=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor='white'),
        width=width, height=height, paper_bgcolor='white',
        legend=dict(font=dict(size=11), x=0.45, y=-0.12, orientation='h'),
        margin=dict(l=80, r=80, t=120, b=80),
        annotations=[
            dict(text='入力感情 / 输入情感', x=0.22, y=1.12, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=15)),
            dict(text='返答感情 / 回复情感', x=0.78, y=1.12, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=15))]
    )


def run_pipeline(df_cat, cat_label, pca_dim, umap_dim, nn, md, mcs, ms, output_dir):
    """
    PCA → UMAP → HDBSCAN パイプラインを実行する
    执行 PCA → UMAP → HDBSCAN 流水线

    パラメータ / 参数:
      df_cat    : 対象データ (category でフィルタ済み) / 目标数据 (已按category过滤)
      cat_label : カテゴリ番号 (0 or 1) / 类别编号
      pca_dim   : PCA の次元数 / PCA降维后的维度
      umap_dim  : UMAP の次元数 / UMAP降维后的维度
      nn        : UMAP の n_neighbors (近傍数) / UMAP的近邻数
      md        : UMAP の min_dist (最小距離) / UMAP的最小距离
      mcs       : HDBSCAN の min_cluster_size (最小クラスタサイズ) / HDBSCAN的最小聚类大小
      ms        : HDBSCAN の min_samples (最小サンプル数) / HDBSCAN的最小样本数
      output_dir: 出力ディレクトリ / 输出目录
    """

    print(f"\n{'='*60}")
    print(f"Category {cat_label} (n={len(df_cat)})")
    print(f"Pipeline: StandardScaler → PCA({pca_dim}D) → UMAP({umap_dim}D, cosine) → HDBSCAN")
    print(f"{'='*60}")

    X = df_cat[cluster_features].fillna(0)

    # 標準化 / 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA / 主成分分析
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    cumvar = np.sum(pca.explained_variance_ratio_)
    print(f"  PCA: {pca_dim}D (累積寄与率 / 累积贡献率 {cumvar:.3f})")

    # UMAP (コサイン類似度) / UMAP (余弦相似度)
    reducer = umap.UMAP(
        n_components=umap_dim, n_neighbors=nn, min_dist=md,
        metric='cosine', random_state=42
    )
    X_umap = reducer.fit_transform(X_pca)
    print(f"  UMAP: {umap_dim}D (nn={nn}, md={md}, cosine)")

    # HDBSCAN / 基于密度的聚类
    labels = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, prediction_data=True).fit_predict(X_umap)

    unique = set(labels)
    n_clusters = len(unique - {-1})
    noise = int((labels == -1).sum())
    mask = labels != -1

    df_cat = df_cat.copy()
    df_cat['cluster'] = labels

    sil = silhouette_score(X_umap[mask], labels[mask]) if n_clusters >= 2 and mask.sum() >= 20 else -1

    print(f"  HDBSCAN: {n_clusters}クラスタ, ノイズ / 噪声 {noise}件 ({noise/len(df_cat)*100:.1f}%)")
    print(f"  Silhouette: {sil:.4f}")
    for cl in sorted(unique - {-1}):
        print(f"    Cluster {cl}: {(labels == cl).sum()}件")
    if noise > 0:
        print(f"    Noise: {noise}件")

    # クラスタ解釈 / 聚类解释
    df_valid = df_cat[df_cat['cluster'] != -1]
    cluster_centers = df_valid.groupby('cluster')[cluster_features].mean()

    print(f"\n  クラスタ解釈 / 聚类解释:")
    for cl in sorted(df_valid['cluster'].unique()):
        c = cluster_centers.loc[cl]
        top_in = max(emotion_categories, key=lambda e: c[f'input_{e}'])
        top_re = max(emotion_categories, key=lambda e: c[f'reply_{e}'])
        print(f"    Cluster {cl}: 入力={top_in}({c[f'input_{top_in}']:.2f}), 返答={top_re}({c[f'reply_{top_re}']:.2f})")

    # ユーザー評点を聚类後に导入 / 聚类后导入用户评分
    df_research = pd.read_csv(config.DATA_DIR / 'real_research.csv', on_bad_lines='warn')
    df_research['userId'] = df_research['userId'].astype(str)
    df_ratings = df_research[['userId'] + RATING_COLS].copy()
    for col in RATING_COLS:
        df_ratings[col] = pd.to_numeric(df_ratings[col], errors='coerce')
    df_valid = df_valid.merge(df_ratings, on='userId', how='left')

    print(f"\n  ユーザー評点 / 用户评分 (クラスタ別平均 / 按聚类均值):")
    print(df_valid.groupby('cluster')[RATING_COLS].mean().round(2).to_string())

    # persona / replyType 分布
    for col_name in ['persona', 'replyType']:
        if col_name in df_cat.columns:
            print(f"\n  {col_name} 分布:")
            print(pd.crosstab(df_valid['cluster'], df_valid[col_name]).to_string())


    # === 可視化 / 可视化 ===

    # UMAP 散布図 / 散点图 (3D or 2D)
    dim = X_umap.shape[1]
    fig_scatter = go.Figure()
    for cl in sorted(unique):
        mask_cl = labels == cl
        subset = df_cat[mask_cl]
        if cl == -1:
            fig_scatter.add_trace(go.Scatter3d(
                x=X_umap[mask_cl, 0], y=X_umap[mask_cl, 1], z=X_umap[mask_cl, 2],
                mode='markers', marker=dict(size=3, color='gray', opacity=0.2),
                name='Noise / 噪声') if dim >= 3 else go.Scatter(
                x=X_umap[mask_cl, 0], y=X_umap[mask_cl, 1],
                mode='markers', marker=dict(size=4, color='gray', opacity=0.2),
                name='Noise / 噪声'))
        else:
            hover = [make_hover_text(r) for _, r in subset.iterrows()]
            color = cluster_colors[cl % len(cluster_colors)]
            scatter_cls = go.Scatter3d if dim >= 3 else go.Scatter
            xyz = dict(x=X_umap[mask_cl, 0], y=X_umap[mask_cl, 1], z=X_umap[mask_cl, 2]) if dim >= 3 else \
                  dict(x=X_umap[mask_cl, 0], y=X_umap[mask_cl, 1])
            fig_scatter.add_trace(scatter_cls(
                **xyz, mode='markers',
                marker=dict(size=5 if dim >= 3 else 7, color=color, opacity=0.8),
                name=f'Cluster {cl} (n={mask_cl.sum()})',
                text=hover, hoverinfo='text'))

    layout_kw = dict(
        title=dict(text=f'Category {cat_label} — UMAP {dim}D cosine (Sil={sil:.3f})', x=0.5, font=dict(size=18)),
        paper_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60))
    if dim >= 3:
        layout_kw.update(scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3', bgcolor='white'),
                         width=1000, height=800)
    else:
        layout_kw.update(xaxis_title='UMAP Dim 1', yaxis_title='UMAP Dim 2',
                         plot_bgcolor='white', width=900, height=700)
    fig_scatter.update_layout(**layout_kw)
    fig_scatter.write_html(output_dir / f'category{cat_label}_umap_{dim}d.html')

    # t-SNE (元空間で確認 / 原始空间确认)
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_cat)//4)).fit_transform(X_scaled)
    fig_tsne = go.Figure()
    for cl in sorted(unique):
        mask_cl = labels == cl
        if cl == -1:
            fig_tsne.add_trace(go.Scatter(
                x=X_tsne[mask_cl, 0], y=X_tsne[mask_cl, 1], mode='markers',
                marker=dict(size=3, color='gray', opacity=0.2), name='Noise'))
        else:
            fig_tsne.add_trace(go.Scatter(
                x=X_tsne[mask_cl, 0], y=X_tsne[mask_cl, 1], mode='markers',
                marker=dict(size=6, color=cluster_colors[cl % len(cluster_colors)], opacity=0.7),
                name=f'Cluster {cl} (n={mask_cl.sum()})'))
    fig_tsne.update_layout(
        title=dict(text=f'Category {cat_label} — t-SNE (Sil={sil:.3f})', x=0.5, font=dict(size=18)),
        xaxis_title='t-SNE 1', yaxis_title='t-SNE 2',
        width=800, height=600, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=60, r=60, t=80, b=60))
    fig_tsne.write_html(output_dir / f'category{cat_label}_tsne.html')

    # レーダーチャート (replyType別 / 按replyType分类)
    for rt in ['ReplyInterruptPersona', 'ReplyCurrentPersona']:
        subset = df_valid[df_valid['replyType'] == rt]
        if subset.empty:
            continue
        rt_label = 'interrupt' if rt == 'ReplyInterruptPersona' else 'current'

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "polar"}, {"type": "polar"}]],
                            horizontal_spacing=0.15)
        for cl in sorted(subset['cluster'].unique()):
            cl = int(cl)
            c = cluster_centers.loc[cl]
            color = cluster_colors[cl % len(cluster_colors)]
            n_cl = len(subset[subset['cluster'] == cl])
            for col_idx, vals in enumerate([
                [c[f'input_{e}'] for e in emotion_categories],
                [c[f'reply_{e}'] for e in emotion_categories]
            ]):
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=radar_categories, fill='toself',
                    name=f'Cluster {cl} (n={n_cl})' if col_idx == 0 else f'Cluster {cl}',
                    legendgroup=f'cl{cl}', line=dict(color=color), opacity=0.3,
                    showlegend=(col_idx == 0)), row=1, col=col_idx + 1)
        fig.update_layout(**make_radar_layout(f'Category {cat_label} — {rt_label.capitalize()} (n={len(subset)})'))
        fig.write_html(output_dir / f'category{cat_label}_{rt_label}_radar.html')

    # 感情変化チャート / 情感变化图
    valid_clusters = sorted(df_valid['cluster'].unique())
    fig_diff = make_subplots(rows=1, cols=2, specs=[[{"type": "polar"}, {"type": "polar"}]],
                             horizontal_spacing=0.15)
    for cl in valid_clusters:
        c = cluster_centers.loc[cl]
        color = cluster_colors[cl % len(cluster_colors)]
        n_cl = len(df_valid[df_valid['cluster'] == cl])
        input_vals = [c[f'input_{e}'] for e in emotion_categories]
        reply_vals = [c[f'reply_{e}'] for e in emotion_categories]
        diff_vals = [r - i for r, i in zip(reply_vals, input_vals)]

        fig_diff.add_trace(go.Scatterpolar(
            r=input_vals + [input_vals[0]], theta=radar_categories, fill='toself',
            name=f'C{cl} 入力 (n={n_cl})', legendgroup=f'cl{cl}',
            line=dict(color=color, dash='solid'), opacity=0.25), row=1, col=1)
        fig_diff.add_trace(go.Scatterpolar(
            r=reply_vals + [reply_vals[0]], theta=radar_categories, fill='toself',
            name=f'C{cl} 返答', legendgroup=f'cl{cl}',
            line=dict(color=color, dash='dash'), opacity=0.25, showlegend=False), row=1, col=1)
        fig_diff.add_trace(go.Scatterpolar(
            r=diff_vals + [diff_vals[0]], theta=radar_categories, fill='toself',
            name=f'C{cl} 差分', legendgroup=f'cl{cl}',
            line=dict(color=color), opacity=0.3), row=1, col=2)

    fig_diff.update_layout(**make_radar_layout(f'Category {cat_label} — 感情変化 / 情感变化'))
    fig_diff.update_layout(
        polar2=dict(radialaxis=dict(visible=True, range=[-0.5, 0.5]), bgcolor='white'))
    fig_diff.write_html(output_dir / f'category{cat_label}_emotion_diff.html')

    # ユーザー評点箱ひげ図 (平均値付き) / 用户评分箱线图 (含均值)
    fig_rating = go.Figure()
    for cl in valid_clusters:
        subset = df_valid[df_valid['cluster'] == cl]
        color = cluster_colors[cl % len(cluster_colors)]
        for ri, rc in enumerate(RATING_COLS):
            fig_rating.add_trace(go.Box(
                y=subset[rc], name=RATING_LABELS.get(rc, rc),
                marker_color=color, legendgroup=f'cl{cl}',
                legendgrouptitle_text=f'Cluster {cl}' if ri == 0 else None,
                showlegend=(ri == 0), offsetgroup=f'cl{cl}',
                boxmean=True))
    fig_rating.update_layout(
        title=dict(text=f'Category {cat_label} — ユーザー評点分布 / 用户评分分布', x=0.5, font=dict(size=18)),
        yaxis_title='評点 / 评分', boxmode='group',
        width=1300, height=700, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=60, r=60, t=80, b=60))
    fig_rating.write_html(output_dir / f'category{cat_label}_rating_boxplot.html')

    # 結果保存 / 结果保存
    df_cat.to_csv(output_dir / f'category{cat_label}_clusters.csv', index=False, encoding='utf-8-sig')
    cluster_centers.to_csv(output_dir / f'category{cat_label}_centers.csv', encoding='utf-8-sig')
    print(f"\n  保存完了 / 保存完成")
    return df_cat


# 実行 / 执行

# Category 0: PCA(10D) → UMAP(3D, cosine) → HDBSCAN
# Silhouette=0.713, CH=464.5, DBI=0.296, 60/27/12, Noise=0
result0 = run_pipeline(
    df[df['category'] == 0].copy(), 0,
    pca_dim=10, umap_dim=3, nn=10, md=0.0, mcs=10, ms=5,
    output_dir=output_dir
)

# Category 1: PCA(5D) → UMAP(2D, cosine) → HDBSCAN
# Silhouette=0.807, CH=1487.9, DBI=0.283, 76/68, Noise=0
result1 = run_pipeline(
    df[df['category'] == 1].copy(), 1,
    pca_dim=5, umap_dim=2, nn=10, md=0.0, mcs=20, ms=8,
    output_dir=output_dir
)

df_all = pd.concat([result0, result1], ignore_index=True)
df_all.to_csv(output_dir / 'all_clusters.csv', index=False, encoding='utf-8-sig')

print(f"\n全ての結果を保存しました / 所有结果已保存: {output_dir}")
print("完了 / 完成")
