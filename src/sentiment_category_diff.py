
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
df_all = pd.read_csv(config.DATA_DIR / 'sentiment/sentiment_all_diff.csv', on_bad_lines='warn')
df_category = pd.read_csv(config.DATA_DIR / '2category_all.csv', on_bad_lines='warn')

for col in ['userId', 'session_id']:
    df_all[col] = df_all[col].astype(str)
    df_category[col] = df_category[col].astype(str)

# categoryとtopic情報をsession_idでマージ / merge category and topic by session_id
df_all = df_all.merge(df_category[['session_id', 'category', 'topic_id']], on='session_id', how='left')

# doc_topicsからtopic詳細情報を取得（session_idベースで重複排除）/ get topic details from doc_topics (deduplicate by session_id)
df_doc_topics = pd.read_csv(config.DATA_DIR / 'topic_modeling/combined_userInput_doc_topics.csv', on_bad_lines='warn')
df_doc_topics['original_text'] = df_doc_topics['original_text'].fillna('').astype(str)
df_all['userInput'] = df_all['userInput'].fillna('').astype(str)

# userInputで重複排除してからマージ / deduplicate by userInput before merge
df_doc_unique = df_doc_topics.drop_duplicates(subset=['original_text'], keep='first')
df_doc_unique = df_doc_unique.rename(columns={'topic_id': 'topic_id_detail', 'topic_probability': 'topic_prob'})
df_all = df_all.merge(
    df_doc_unique[['original_text', 'topic_id_detail', 'topic_prob']],
    left_on='userInput', right_on='original_text', how='left'
)
df_all.drop(columns=['original_text'], inplace=True, errors='ignore')

print(f"データ数 / 数据量: {len(df_all)}")

# 感情特徴量の定義 / 定义情感特征
emotion_categories = ['joy', 'sadness', 'anticipation', 'surprise', 'anger', 'fear', 'disgust', 'trust']
diff_features = [f'diff_{e}' for e in emotion_categories]
cluster_features = diff_features

# topicカラムの統一 / 统一topic列
if 'topic_id_detail' in df_all.columns:
    df_all['topic'] = df_all['topic_id_detail'].fillna(-1).astype(int)
else:
    df_all['topic'] = df_all.get('topic_id', pd.Series([-1]*len(df_all))).fillna(-1).astype(int)

# 出力設定 / 输出设置
output_dir = config.DATA_DIR / 'sentiment/cluster_topic_diff'
output_dir.mkdir(parents=True, exist_ok=True)

# レーダーチャート用の設定 / 雷达图设置
emotion_labels = [
    'Joy / 喜び', 'Sadness / 悲しみ', 'Anticipation / 期待', 'Surprise / 驚き',
    'Anger / 怒り', 'Fear / 恐れ', 'Disgust / 嫌悪', 'Trust / 信頼'
]
radar_categories = emotion_labels + [emotion_labels[0]]

cluster_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

topic_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
                '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
                '#ffd8b1', '#000075', '#a9a9a9']

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
            f"topic: {row.get('topic', '?')}<br>"
            f"input: {str(row.get('userInput', ''))[:40]}...")


def make_radar_layout(title, width=1400, height=700):
    """レーダーチャート共通レイアウトを返す / 返回雷达图通用布局"""
    return dict(
        title=dict(text=title, x=0.5, font=dict(size=18), y=0.98),
        polar=dict(radialaxis=dict(visible=True, range=[-0.5, 0.5]), bgcolor='white'),
        width=width, height=height, paper_bgcolor='white',
        legend=dict(font=dict(size=10), x=0.5, y=-0.1, orientation='h', xanchor='center'),
        margin=dict(l=80, r=80, t=100, b=80),
    )

# 出力設定 / 输出设置
output_dir = config.DATA_DIR / 'sentiment/cluster_topic_diff'
output_dir.mkdir(parents=True, exist_ok=True)

# レーダーチャート用の設定 / 雷达图设置
emotion_labels = [
    'Joy / 喜び', 'Sadness / 悲しみ', 'Anticipation / 期待', 'Surprise / 驚き',
    'Anger / 怒り', 'Fear / 恐れ', 'Disgust / 嫌悪', 'Trust / 信頼'
]
radar_categories = emotion_labels + [emotion_labels[0]]

cluster_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

topic_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
                '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
                '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
                '#ffd8b1', '#000075', '#a9a9a9']

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
            f"topic: {row.get('topic', '?')}<br>"
            f"input: {str(row.get('userInput', ''))[:40]}...")


def make_radar_layout(title, width=1400, height=700):
    """レーダーチャート共通レイアウトを返す / 返回雷达图通用布局"""
    return dict(
        title=dict(text=title, x=0.5, font=dict(size=18), y=0.98),
        polar=dict(radialaxis=dict(visible=True, range=[-0.5, 0.5]), bgcolor='white'),
        width=width, height=height, paper_bgcolor='white',
        legend=dict(font=dict(size=10), x=0.5, y=-0.1, orientation='h', xanchor='center'),
        margin=dict(l=80, r=80, t=100, b=80),
    )


def run_pipeline(df_cat, cat_label, umap_dim, nn, md, mcs, ms, output_dir):
    """
    UMAP → HDBSCAN パイプラインを実行する (差分特徴量版)
    If 'cluster' column already exists in df_cat, skip clustering and use it directly.
    """

    has_clusters = 'cluster' in df_cat.columns

    if has_clusters:
        print(f"Category {cat_label} (n={len(df_cat)}) — using pre-computed clusters")
    else:
        print(f"Category {cat_label} (n={len(df_cat)})")
        print(f"Features: 8 diff features (reply - input)")
        print(f"Pipeline: StandardScaler → UMAP({umap_dim}D, cosine) → HDBSCAN")

    X = df_cat[cluster_features].fillna(0)

    if has_clusters:
        labels = df_cat['cluster'].values
        X_scaled = StandardScaler().fit_transform(X)
        X_umap = df_cat[['umap_0', 'umap_1']].values
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = umap.UMAP(
            n_components=umap_dim, n_neighbors=nn, min_dist=md,
            metric='cosine', random_state=42
        )
        X_umap = reducer.fit_transform(X_scaled)
        print(f"\n  UMAP: {umap_dim}D (nn={nn}, md={md}, cosine)")

        labels = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, prediction_data=True).fit_predict(X_umap)

        df_cat = df_cat.copy()
        df_cat['umap_0'] = X_umap[:, 0]
        df_cat['umap_1'] = X_umap[:, 1]
        df_cat['cluster'] = labels

    unique = set(labels)
    n_clusters = len(unique - {-1})
    noise = int((labels == -1).sum())
    mask = labels != -1

    if not has_clusters:
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
        top_diff = max(emotion_categories, key=lambda e: abs(c[f'diff_{e}']))
        print(f"    Cluster {cl}: 最大差分={top_diff}(diff={c[f'diff_{top_diff}']:.3f}), "
              f"mean={cluster_centers.loc[cl].mean():.3f}")

    # ユーザー評点をクラスタリング後に導入 / 聚类后导入用户评分
    df_research = pd.read_csv(config.DATA_DIR / 'real_research.csv', on_bad_lines='warn')
    df_research['userId'] = df_research['userId'].astype(str)
    df_ratings = df_research[['userId'] + RATING_COLS].copy()
    for col in RATING_COLS:
        df_ratings[col] = pd.to_numeric(df_ratings[col], errors='coerce')
    df_valid = df_valid.merge(df_ratings, on='userId', how='left')

    print(f"\n  ユーザー評点 / 用户评分 (クラスタ別平均 / 按聚类均值):")
    print(df_valid.groupby('cluster')[RATING_COLS].mean().round(2).to_string())

    for col_name in ['persona', 'replyType']:
        if col_name in df_cat.columns:
            print(f"\n  {col_name} 分布:")
            print(pd.crosstab(df_valid['cluster'], df_valid[col_name]).to_string())


    # === 可視化 / 可视化 ===

    # UMAP 散布図 / 散点图 (2D)
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
        title=dict(text=f'Category {cat_label} — Diff UMAP {dim}D cosine (all data clustering)',
                   x=0.5, font=dict(size=18)),
        paper_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60))
    if dim >= 3:
        layout_kw.update(scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3',
                                    bgcolor='white', aspectmode='cube'),
                         width=1000, height=800)
    else:
        layout_kw.update(xaxis_title='UMAP Dim 1', yaxis_title='UMAP Dim 2',
                         plot_bgcolor='white', width=900, height=700)
    fig_scatter.update_layout(**layout_kw)
    fig_scatter.write_html(output_dir / f'category{cat_label}_diff_umap_{dim}d.html')

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
        title=dict(text=f'Category {cat_label} — Diff t-SNE (all data clustering)',
                   x=0.5, font=dict(size=18)),
        xaxis_title='t-SNE 1', yaxis_title='t-SNE 2',
        width=900, height=700, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=60, r=60, t=100, b=60))
    fig_tsne.write_html(output_dir / f'category{cat_label}_diff_tsne.html')

    # レーダーチャート (replyType別 / 按replyType分类) - 差分バージョン
    for rt in ['ReplyInterruptPersona', 'ReplyCurrentPersona']:
        subset = df_valid[df_valid['replyType'] == rt]
        if subset.empty:
            continue
        rt_label = 'interrupt' if rt == 'ReplyInterruptPersona' else 'current'

        fig = go.Figure()
        for cl in sorted(subset['cluster'].unique()):
            cl = int(cl)
            c = cluster_centers.loc[cl]
            color = cluster_colors[cl % len(cluster_colors)]
            n_cl = len(subset[subset['cluster'] == cl])
            vals = [c[f'diff_{e}'] for e in emotion_categories]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_categories, fill='toself',
                name=f'Cluster {cl} (n={n_cl})',
                line=dict(color=color), opacity=0.3))
        fig.add_trace(go.Scatterpolar(
            r=[0.01] * len(radar_categories), theta=radar_categories,
            mode='lines', line=dict(color='red', width=2, dash='dash'),
            name='差分=0', showlegend=True))
        fig.update_layout(**make_radar_layout(
            f'Category {cat_label} — Diff Radar {rt_label.capitalize()} (n={len(subset)})'))
        fig.write_html(output_dir / f'category{cat_label}_diff_{rt_label}_radar.html')

    # 感情差分チャート / 情感差分图
    valid_clusters = sorted(df_valid['cluster'].unique())
    fig_diff = go.Figure()
    for cl in valid_clusters:
        c = cluster_centers.loc[cl]
        color = cluster_colors[cl % len(cluster_colors)]
        n_cl = len(df_valid[df_valid['cluster'] == cl])
        diff_vals = [c[f'diff_{e}'] for e in emotion_categories]

        fig_diff.add_trace(go.Scatterpolar(
            r=diff_vals + [diff_vals[0]], theta=radar_categories, fill='toself',
            name=f'C{cl} (n={n_cl})',
            line=dict(color=color), opacity=0.3))

    fig_diff.add_trace(go.Scatterpolar(
        r=[0.01] * len(radar_categories), theta=radar_categories,
        mode='lines', line=dict(color='red', width=2, dash='dash'),
        name='差分=0', showlegend=True))
    fig_diff.update_layout(
        title=dict(text=f'Category {cat_label} — 感情差分 / 情感差分', x=0.5, font=dict(size=18), y=0.98),
        polar=dict(radialaxis=dict(visible=True, range=[-0.5, 0.5]), bgcolor='white'),
        width=900, height=700, paper_bgcolor='white',
        legend=dict(font=dict(size=10), x=0.5, y=-0.05, orientation='h', xanchor='center'),
        margin=dict(l=80, r=80, t=100, b=80))
    fig_diff.write_html(output_dir / f'category{cat_label}_diff_radar.html')

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
        title=dict(text=f'Category {cat_label} — Diff ユーザー評点分布 / 用户评分分布', x=0.5, font=dict(size=18)),
        yaxis_title='評点 / 评分', boxmode='group',
        width=1300, height=700, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=60, r=60, t=100, b=60),
        legend=dict(font=dict(size=10), x=0.5, y=-0.08, orientation='h', xanchor='center'))
    fig_rating.write_html(output_dir / f'category{cat_label}_diff_rating_boxplot.html')

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
                diffs = subset[f'diff_{emo}'].values
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
            fig_jitter.add_hline(y=0, line_dash='dash', line_color='red',
                                 line_width=2, opacity=0.7, row=1, col=ci + 1)
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
        fig_jitter.write_html(output_dir / f'category{cat_label}_diff_jitter_{rt_key}.html')

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
                diffs = subset[f'diff_{emo}'].values
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
            fig_stats.add_hline(y=0, line_dash='dash', line_color='red',
                                line_width=2, opacity=0.7, row=r, col=c)
        fig_stats.update_layout(
            title=dict(text=f'Category {cat_label}{rt_suffix} — Diff クラスタ別統計比較',
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
        fig_stats.write_html(output_dir / f'category{cat_label}_diff_cluster_stats_{rt_key}.html')

    # クラスタ内トピック分布 / cluster内topic分布
    if 'topic' in df_cat.columns:
        ct = pd.crosstab(df_cat['cluster'], df_cat['topic'])
        ct_norm = ct.div(ct.sum(axis=1), axis=0)

        fig_ct = go.Figure()
        for tp in sorted(ct_norm.columns):
            fig_ct.add_trace(go.Bar(
                x=[f'Cluster {cl}' for cl in ct_norm.index],
                y=ct_norm[tp].values,
                name=f'Topic {tp}',
                marker_color=topic_colors[tp % len(topic_colors)]))

        fig_ct.update_layout(
            title=dict(text=f'Category {cat_label} — Cluster内Topic分布 / Topic分布within Cluster',
                       x=0.5, font=dict(size=18)),
            barmode='stack', xaxis_title='Cluster', yaxis_title='Proportion',
            paper_bgcolor='white', plot_bgcolor='white',
            width=1000, height=600,
            legend=dict(font=dict(size=10), x=0.5, y=-0.15, orientation='h', xanchor='center'),
            margin=dict(l=60, r=60, t=100, b=80))
        fig_ct.write_html(output_dir / f'category{cat_label}_topic_cluster_distribution.html')

        print(f"\n  Cluster x Topic 交叉集計:")
        print(ct.to_string())

    # 結果保存 / 结果保存
    df_cat.to_csv(output_dir / f'category{cat_label}_diff_clusters.csv', index=False, encoding='utf-8-sig')
    cluster_centers.to_csv(output_dir / f'category{cat_label}_diff_centers.csv', encoding='utf-8-sig')
    print(f"\n  保存完了 / 保存完成")
    return df_cat



# 全データ一括クラスタリング / 全数据统一聚类
print("全データ一括クラスタリング / 全数据统一聚类")

X_all = df_all[cluster_features].fillna(0)

scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X_all)

print(f"特徴量 / 特征: 8 diff features (reply - input)")
print(f"パイプライン / 流水线: StandardScaler → UMAP(2D, cosine) → HDBSCAN")

reducer = umap.UMAP(
    n_components=2, n_neighbors=10, min_dist=0.0,
    metric='cosine', random_state=42
)
X_umap_all = reducer.fit_transform(X_scaled_all)
print(f"\nUMAP: 2D (nn=10, md=0.0, cosine)")

labels_all = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10, prediction_data=True).fit_predict(X_umap_all)

df_all['umap_0'] = X_umap_all[:, 0]
df_all['umap_1'] = X_umap_all[:, 1]
df_all['cluster'] = labels_all

unique_all = set(labels_all)
n_clusters_all = len(unique_all - {-1})
noise_all = int((labels_all == -1).sum())
mask_all = labels_all != -1

sil_all = silhouette_score(X_umap_all[mask_all], labels_all[mask_all]) if n_clusters_all >= 2 and mask_all.sum() >= 20 else -1

print(f"\nHDBSCAN: {n_clusters_all}クラスタ, ノイズ / 噪声 {noise_all}件 ({noise_all/len(df_all)*100:.1f}%)")
print(f"Silhouette: {sil_all:.4f}")
for cl in sorted(unique_all - {-1}):
    print(f"  Cluster {cl}: {(labels_all == cl).sum()}件")
if noise_all > 0:
    print(f"  Noise: {noise_all}件")

# クラスタ解釈 / 聚类解释
df_valid_all = df_all[df_all['cluster'] != -1]
cluster_centers_all = df_valid_all.groupby('cluster')[cluster_features].mean()

print(f"\nクラスタ解釈 / 聚类解释:")
for cl in sorted(df_valid_all['cluster'].unique()):
    c = cluster_centers_all.loc[cl]
    top_diff = max(emotion_categories, key=lambda e: abs(c[f'diff_{e}']))
    print(f"  Cluster {cl}: 最大差分={top_diff}(diff={c[f'diff_{top_diff}']:.3f}), "
          f"mean={cluster_centers_all.loc[cl].mean():.3f}")

# 全体のUMAP可視化 / 全体UMAP可视化
fig_scatter_all = go.Figure()
for cl in sorted(unique_all):
    mask_cl = df_all['cluster'] == cl
    subset = df_all[mask_cl]
    if cl == -1:
        fig_scatter_all.add_trace(go.Scatter(
            x=df_all.loc[mask_cl, 'umap_0'], y=df_all.loc[mask_cl, 'umap_1'],
            mode='markers', marker=dict(size=3, color='gray', opacity=0.2),
            name='Noise / 噪声'))
    else:
        hover = [make_hover_text(r) for _, r in subset.iterrows()]
        color = cluster_colors[cl % len(cluster_colors)]
        fig_scatter_all.add_trace(go.Scatter(
            x=df_all.loc[mask_cl, 'umap_0'], y=df_all.loc[mask_cl, 'umap_1'],
            mode='markers',
            marker=dict(size=5, color=color, opacity=0.7),
            name=f'Cluster {cl} (n={mask_cl.sum()})',
            text=hover, hoverinfo='text'))
fig_scatter_all.update_layout(
    title=dict(text=f'All Data — Diff UMAP 2D cosine (Sil={sil_all:.3f})', x=0.5, font=dict(size=18)),
    xaxis_title='UMAP Dim 1', yaxis_title='UMAP Dim 2',
    paper_bgcolor='white', plot_bgcolor='white',
    width=1000, height=800, margin=dict(l=60, r=60, t=100, b=60))
fig_scatter_all.write_html(output_dir / 'all_diff_umap_2d.html')

# 結果保存 / 全体结果保存
df_all.to_csv(output_dir / 'all_diff_clusters.csv', index=False, encoding='utf-8-sig')
cluster_centers_all.to_csv(output_dir / 'all_diff_centers.csv', encoding='utf-8-sig')

print(f"\n全体の結果を保存しました / 全体结果已保存: {output_dir}")



# カテゴリ別可視化 / 按category分别生成可视化

for cat_label in [0, 1]:
    cat_dir = output_dir / f'category{cat_label}'
    cat_dir.mkdir(parents=True, exist_ok=True)
    df_cat = df_all[df_all['category'] == cat_label].copy()
    run_pipeline(df_cat, cat_label, umap_dim=2, nn=10, md=0.0, mcs=20, ms=10, output_dir=cat_dir)

df_result0 = pd.read_csv(output_dir / 'category0' / 'category0_diff_clusters.csv', on_bad_lines='warn')
df_result1 = pd.read_csv(output_dir / 'category1' / 'category1_diff_clusters.csv', on_bad_lines='warn')
df_concat = pd.concat([df_result0, df_result1], ignore_index=True)
df_concat.to_csv(output_dir / 'all_diff_clusters.csv', index=False, encoding='utf-8-sig')

print(f"\n全ての結果を保存しました / 所有结果已保存: {output_dir}")
print("完了 / 完成")
