
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
output_dir = config.DATA_DIR / 'sentiment/cluster_topic'
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


def make_radar_layout(title, width=1400, height=700):
    """レーダーチャート共通レイアウトを返す / 返回雷达图通用布局"""
    return dict(
        title=dict(text=title, x=0.5, font=dict(size=18), y=0.98),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor='white'),
        polar2=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor='white'),
        width=width, height=height, paper_bgcolor='white',
        legend=dict(font=dict(size=10), x=0.5, y=-0.1, orientation='h', xanchor='center'),
        margin=dict(l=80, r=80, t=100, b=80),
        annotations=[
            dict(text='入力感情 / 输入情感', x=0.22, y=1.12, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=14)),
            dict(text='返答感情 / 回复情感', x=0.78, y=1.12, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=14))]
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

    print(f"Category {cat_label} (n={len(df_cat)})")
    print(f"Pipeline: StandardScaler → PCA({pca_dim}D) → UMAP({umap_dim}D, cosine) → HDBSCAN")

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

    # ユーザー評点をクラスタリング後に導入 / 聚类后导入用户评分
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
        layout_kw.update(scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3',
                                    bgcolor='white', aspectmode='cube'),
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
        width=900, height=700, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=60, r=60, t=100, b=60))
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

    fig_diff.add_trace(go.Scatterpolar(
        r=[0] * len(radar_categories), theta=radar_categories,
        mode='lines', line=dict(color='gray', width=1.5, dash='dash'),
        name='r=0', showlegend=False), row=1, col=2)
    fig_diff.update_layout(
        title=dict(text=f'Category {cat_label} — 感情変化 / 情感变化', x=0.5, font=dict(size=18), y=0.98),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor='white'),
        polar2=dict(radialaxis=dict(visible=True, range=[-0.5, 0.5]), bgcolor='white'),
        width=1400, height=750, paper_bgcolor='white',
        legend=dict(font=dict(size=10), x=0.5, y=-0.05, orientation='h', xanchor='center'),
        margin=dict(l=80, r=80, t=120, b=80),
        annotations=[
            dict(text='オリジナルデータ', x=0.22, y=1.08, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=14)),
            dict(text='差分', x=0.78, y=1.08, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=14))])
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
        margin=dict(l=60, r=60, t=100, b=60),
        legend=dict(font=dict(size=10), x=0.5, y=-0.08, orientation='h', xanchor='center'))
    fig_rating.write_html(output_dir / f'category{cat_label}_rating_boxplot.html')

    # 感情差分ジッタープロット + 柱状图 (replyType別 / 按replyType分类)
    reply_type_configs = [
        ('all', None, ''),
        ('current', 'ReplyCurrentPersona', ' — Current Persona'),
        ('interrupt', 'ReplyInterruptPersona', ' — Interrupt Persona'),
    ]
    for rt_key, rt_filter, rt_suffix in reply_type_configs:
        df_rt = df_valid if rt_filter is None else df_valid[df_valid['replyType'] == rt_filter]
        if df_rt.empty:
            continue
        n_cl_rt = len(df_rt['cluster'].unique())

        # --- Jitter Plot ---
        fig_jitter = make_subplots(
            rows=1, cols=n_cl_rt,
            subplot_titles=[f'Cluster {cl} (n={len(df_rt[df_rt["cluster"]==cl])})'
                            for cl in sorted(df_rt['cluster'].unique())],
            horizontal_spacing=0.06
        )
        all_diffs = []
        for ci, cl in enumerate(sorted(df_rt['cluster'].unique())):
            subset = df_rt[df_rt['cluster'] == cl]
            color = cluster_colors[cl % len(cluster_colors)]
            means, mins, maxs = [], [], []
            for ei, emo in enumerate(emotion_categories):
                diffs = subset[f'reply_{emo}'].values - subset[f'input_{emo}'].values
                all_diffs.extend(diffs)
                jitter_x = np.full(len(diffs), ei) + np.random.uniform(-0.15, 0.15, len(diffs))
                fig_jitter.add_trace(go.Scatter(
                    x=jitter_x, y=diffs, mode='markers',
                    marker=dict(size=4, color=color, opacity=0.4),
                    name=f'Cluster {cl}' if ei == 0 else None,
                    legendgroup=f'cl{cl}', showlegend=(ei == 0),
                    hovertext=[f'{emo}: {d:.3f}' for d in diffs], hoverinfo='text'
                ), row=1, col=ci + 1)
                means.append(np.mean(diffs))
                mins.append(np.min(diffs))
                maxs.append(np.max(diffs))
            stat_configs = [
                (means, 'solid', 'Mean', True),
                (mins, 'dot', 'Min', False),
                (maxs, 'dash', 'Max', False),
            ]
            for stat_vals, dash, lbl, show_leg in stat_configs:
                fig_jitter.add_trace(go.Scatter(
                    x=list(range(len(emotion_categories))), y=stat_vals,
                    mode='lines+markers', marker=dict(size=5),
                    line=dict(color=color, width=2, dash=dash),
                    name=f'C{cl} {lbl}',
                    legendgroup=f'cl{cl}', showlegend=show_leg
                ), row=1, col=ci + 1)
        jitter_bound = np.ceil(max(abs(np.min(all_diffs)), abs(np.max(all_diffs))) * 10) / 10
        fig_jitter.update_xaxes(
            tickvals=list(range(len(emotion_categories))),
            ticktext=emotion_labels, tickangle=-45
        )
        for ci in range(n_cl_rt):
            fig_jitter.add_hline(y=0, line_dash='dash', line_color='gray',
                                 opacity=0.5, row=1, col=ci + 1)
        fig_jitter.update_layout(
            title=dict(text=f'Category {cat_label}{rt_suffix} — 感情差分ジッタープロット',
                       x=0.5, font=dict(size=18)),
            yaxis_title='差分値 (reply - input)',
            width=max(400 * n_cl_rt, 800), height=600,
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(l=60, r=60, t=100, b=100),
            legend=dict(font=dict(size=10), x=0.5, y=-0.2, orientation='h', xanchor='center')
        )
        for ci in range(n_cl_rt):
            axis_key = f'yaxis{ci + 1}' if ci > 0 else 'yaxis'
            fig_jitter.update_layout(**{axis_key: dict(range=[-jitter_bound, jitter_bound])})
        fig_jitter.write_html(output_dir / f'category{cat_label}_jitter_diff_{rt_key}.html')

        # --- Stats Bar Chart ---
        fig_stats = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mean', 'Max', 'Min', 'Standard Deviation'],
            horizontal_spacing=0.1, vertical_spacing=0.15
        )
        stats_all = []
        for cl in sorted(df_rt['cluster'].unique()):
            subset = df_rt[df_rt['cluster'] == cl]
            color = cluster_colors[cl % len(cluster_colors)]
            stat_means, stat_maxs, stat_mins, stat_sds = [], [], [], []
            for emo in emotion_categories:
                diffs = subset[f'reply_{emo}'].values - subset[f'input_{emo}'].values
                stat_means.append(np.mean(diffs))
                stat_maxs.append(np.max(diffs))
                stat_mins.append(np.min(diffs))
                stat_sds.append(np.std(diffs))
            stats_all.extend(stat_means + stat_maxs + stat_mins + stat_sds)
            for vals, row, col, stat_name in [
                (stat_means, 1, 1, 'mean'), (stat_maxs, 1, 2, 'max'),
                (stat_mins, 2, 1, 'min'), (stat_sds, 2, 2, 'sd'),
            ]:
                fig_stats.add_trace(go.Bar(
                    x=emotion_labels, y=vals, marker_color=color,
                    name=f'Cluster {cl}', legendgroup=f'cl{cl}',
                    showlegend=(stat_name == 'mean')
                ), row=row, col=col)
        stats_bound = np.ceil(max(abs(np.min(stats_all)), abs(np.max(stats_all))) * 10) / 10
        for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            fig_stats.add_hline(y=0, line_dash='dash', line_color='gray',
                                opacity=0.4, row=r, col=c)
        fig_stats.update_layout(
            title=dict(text=f'Category {cat_label}{rt_suffix} — クラスタ別統計比較',
                       x=0.5, font=dict(size=18)),
            yaxis_title='差分値', yaxis3_title='差分値',
            yaxis=dict(range=[-stats_bound, stats_bound]),
            yaxis2=dict(range=[-stats_bound, stats_bound]),
            yaxis3=dict(range=[-stats_bound, stats_bound]),
            yaxis4=dict(range=[-stats_bound, stats_bound]),
            barmode='group',
            width=1200, height=900,
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(l=60, r=60, t=100, b=60),
            legend=dict(font=dict(size=10), x=0.5, y=-0.05, orientation='h', xanchor='center')
        )
        fig_stats.write_html(output_dir / f'category{cat_label}_cluster_stats_{rt_key}.html')

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
