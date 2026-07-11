import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
import umap
import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import config

df_all = pd.read_csv(config.DATA_DIR / 'sentiment/sentiment_all_diff.csv', on_bad_lines='warn')
df_category = pd.read_csv(config.DATA_DIR / '2category_all.csv', on_bad_lines='warn')

for col in ['userId', 'session_id']:
    df_all[col] = df_all[col].astype(str)
    df_category[col] = df_category[col].astype(str)

df_doc_topics = pd.read_csv(config.DATA_DIR / 'topic_modeling/combined_userInput_doc_topics.csv', on_bad_lines='warn')
df_doc_topics['original_text'] = df_doc_topics['original_text'].fillna('').astype(str)
df_all['userInput'] = df_all['userInput'].fillna('').astype(str)
df_doc_unique = df_doc_topics.drop_duplicates(subset=['original_text'], keep='first')

df_all = df_all.merge(df_category[['session_id', 'category', 'topic_id']], on='session_id', how='left')
df_all = df_all.merge(df_doc_unique[['original_text', 'topic_id']].rename(columns={'topic_id': 'topic_id_detail'}), left_on='userInput', right_on='original_text', how='left')
df_all.drop(columns=['original_text'], inplace=True, errors='ignore')
df_all['topic'] = df_all['topic_id_detail'].fillna(df_all.get('topic_id', -1)).fillna(-1).astype(int)
df_all.drop(columns=['topic_id', 'topic_id_detail'], errors='ignore', inplace=True)

print(f"Data: {len(df_all)}")

emotion_categories = ['joy', 'sadness', 'anticipation', 'surprise', 'anger', 'fear', 'disgust', 'trust']
diff_features = [f'diff_{e}' for e in emotion_categories]
cluster_features = diff_features

output_dir = config.DATA_DIR / 'sentiment/cluster_topic_diff'
output_dir.mkdir(parents=True, exist_ok=True)

emotion_labels = ['Joy / 喜び', 'Sadness / 悲しみ', 'Anticipation / 期待', 'Surprise / 驚き', 'Anger / 怒り', 'Fear / 恐れ', 'Disgust / 嫌悪', 'Trust / 信頼']
radar_categories = emotion_labels + [emotion_labels[0]]

cluster_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
topic_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9']

RATING_COLS = ['kyukansei', 'igakuseikakusei', 'anzensei', 'yuugaisei', 'aiirai', 'shikafannshi']
RATING_LABELS = {'kyukansei': '共感性', 'igakuseikakusei': '医学正確性', 'anzensei': '安全性', 'yuugaisei': '有害性', 'aiirai': 'AI依存', 'shikafannshi': 'シカファンシー'}

FIG_W, FIG_H = 1200, 700
FIG_WIDE_W, FIG_WIDE_H = 1400, 700
FIG_LARGE_W, FIG_LARGE_H = 1400, 800


def export_fig(fig, base_path):
    fig.write_html(str(base_path) + '.html')
    fig.write_image(str(base_path) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)


def make_hover_text(row):
    return f"session: {row['session_id']}<br>persona: {row.get('persona', '?')}<br>replyType: {row.get('replyType', '?')}<br>topic: {row.get('topic', '?')}<br>input: {str(row.get('userInput', ''))[:40]}..."


def make_radar_layout(title, width=1400, height=700):
    return dict(title=dict(text=title, x=0.5, font=dict(size=18), y=0.98), polar=dict(radialaxis=dict(visible=True, range=[-0.5, 0.5]), bgcolor='white'), width=width, height=height, paper_bgcolor='white', legend=dict(font=dict(size=10), x=0.5, y=-0.1, orientation='h', xanchor='center'), margin=dict(l=80, r=80, t=100, b=80))


def plot_condensed_tree(clusterer, title, output_path):
    fig, ax = plt.subplots(figsize=(16, 8))
    clusterer.condensed_tree_.plot(axis=ax)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    fig.savefig(str(output_path) + '.svg', dpi=300)
    plt.close(fig)


def run_pipeline(df_cat, cat_label, umap_dim, nn, md, mcs, ms, output_dir):
    has_clusters = 'cluster' in df_cat.columns

    if not has_clusters:
        print(f"Category {cat_label} (n={len(df_cat)})")
        print(f"Pipeline: StandardScaler → UMAP({umap_dim}D, cosine) → HDBSCAN")

    X = df_cat[cluster_features].fillna(0)

    if has_clusters:
        labels = df_cat['cluster'].values
        X_scaled = StandardScaler().fit_transform(X)
        X_umap = df_cat[['umap_0', 'umap_1']].values
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        reducer = umap.UMAP(n_components=umap_dim, n_neighbors=nn, min_dist=md, metric='cosine', random_state=config.CLUSTER_RANDOM_SEED)
        X_umap = reducer.fit_transform(X_scaled)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, prediction_data=True)
        labels = clusterer.fit_predict(X_umap)
        df_cat = df_cat.copy()
        df_cat['umap_0'] = X_umap[:, 0]
        df_cat['umap_1'] = X_umap[:, 1]
        df_cat['cluster'] = labels

    unique = set(labels)
    n_clusters = len(unique - {-1})
    noise = int((labels == -1).sum())

    if not has_clusters and n_clusters >= 2 and (labels != -1).sum() >= 20:
        mask = labels != -1
        print(f"  HDBSCAN: {n_clusters} clusters, noise {noise}")
        print(f"  Silhouette: {silhouette_score(X_umap[mask], labels[mask]):.4f}")
        print(f"  Calinski-Harabasz: {calinski_harabasz_score(X_umap[mask], labels[mask]):.2f}")
        print(f"  Davies-Bouldin: {davies_bouldin_score(X_umap[mask], labels[mask]):.4f}")

    df_valid = df_cat[df_cat['cluster'] != -1]
    cluster_centers = df_valid.groupby('cluster')[cluster_features].mean()

    df_research = pd.read_csv(config.DATA_DIR / 'real_research.csv', on_bad_lines='warn')
    df_research['userId'] = df_research['userId'].astype(str)
    df_ratings = df_research[['userId'] + RATING_COLS].copy()
    for col in RATING_COLS:
        df_ratings[col] = pd.to_numeric(df_ratings[col], errors='coerce')
    df_valid = df_valid.merge(df_ratings, on='userId', how='left')

    dim = X_umap.shape[1]
    fig_scatter = go.Figure()
    for cl in sorted(unique):
        mask_cl = labels == cl
        subset = df_cat[mask_cl]
        if cl == -1:
            fig_scatter.add_trace(go.Scatter(x=X_umap[mask_cl, 0], y=X_umap[mask_cl, 1], mode='markers', marker=dict(size=4, color='gray', opacity=0.2), name='Noise'))
        else:
            hover = [make_hover_text(r) for _, r in subset.iterrows()]
            fig_scatter.add_trace(go.Scatter(x=X_umap[mask_cl, 0], y=X_umap[mask_cl, 1], mode='markers', marker=dict(size=7, color=cluster_colors[cl % len(cluster_colors)], opacity=0.8), name=f'Cluster {cl} (n={mask_cl.sum()})', text=hover, hoverinfo='text'))

    fig_scatter.update_layout(title=dict(text=f'Category {cat_label} — Diff UMAP {dim}D', x=0.5, font=dict(size=18)), xaxis_title='UMAP 1', yaxis_title='UMAP 2', paper_bgcolor='white', plot_bgcolor='white', width=FIG_W, height=FIG_H, margin=dict(l=60, r=60, t=100, b=60))
    export_fig(fig_scatter, output_dir / f'category{cat_label}_diff_umap_{dim}d')

    if not has_clusters and hasattr(clusterer, 'condensed_tree_'):
        plot_condensed_tree(clusterer, f'Category {cat_label} — HDBSCAN Condensed Tree', output_dir / f'category{cat_label}_diff_condensed_tree')

    X_tsne = TSNE(n_components=2, random_state=config.CLUSTER_RANDOM_SEED, perplexity=max(5, min(30, len(df_cat)//4))).fit_transform(X_scaled)
    fig_tsne = go.Figure()
    for cl in sorted(unique):
        mask_cl = labels == cl
        if cl == -1:
            fig_tsne.add_trace(go.Scatter(x=X_tsne[mask_cl, 0], y=X_tsne[mask_cl, 1], mode='markers', marker=dict(size=3, color='gray', opacity=0.2), name='Noise'))
        else:
            fig_tsne.add_trace(go.Scatter(x=X_tsne[mask_cl, 0], y=X_tsne[mask_cl, 1], mode='markers', marker=dict(size=6, color=cluster_colors[cl % len(cluster_colors)], opacity=0.7), name=f'Cluster {cl} (n={mask_cl.sum()})'))
    fig_tsne.update_layout(title=dict(text=f'Category {cat_label} — Diff t-SNE', x=0.5, font=dict(size=18)), xaxis_title='t-SNE 1', yaxis_title='t-SNE 2', width=FIG_W, height=FIG_H, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60))
    export_fig(fig_tsne, output_dir / f'category{cat_label}_diff_tsne')

    valid_clusters = sorted(df_valid['cluster'].unique())
    fig_radar = go.Figure()
    for cl in valid_clusters:
        c = cluster_centers.loc[int(cl)]
        vals = [c[f'diff_{e}'] for e in emotion_categories]
        fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=radar_categories, fill='toself', name=f'Cluster {cl} (n={len(df_valid[df_valid["cluster"]==cl])})', line=dict(color=cluster_colors[int(cl) % len(cluster_colors)]), opacity=0.3))
    fig_radar.add_trace(go.Scatterpolar(r=[0.01] * len(radar_categories), theta=radar_categories, mode='lines', line=dict(color='red', width=2, dash='dash'), name='差分=0'))
    fig_radar.update_layout(**make_radar_layout(f'Category {cat_label} — Diff Radar (n={len(df_valid)})'))
    export_fig(fig_radar, output_dir / f'category{cat_label}_diff_radar')

    fig_rating = go.Figure()
    for cl in valid_clusters:
        subset = df_valid[df_valid['cluster'] == cl]
        for ri, rc in enumerate(RATING_COLS):
            fig_rating.add_trace(go.Box(y=subset[rc], name=RATING_LABELS.get(rc, rc), marker_color=cluster_colors[int(cl) % len(cluster_colors)], legendgroup=f'cl{cl}', legendgrouptitle_text=f'Cluster {cl}' if ri == 0 else None, showlegend=(ri == 0), offsetgroup=f'cl{cl}', boxmean=True))
    fig_rating.update_layout(title=dict(text=f'Category {cat_label} — ユーザー評点分布', x=0.5, font=dict(size=18)), yaxis_title='評点', boxmode='group', width=FIG_WIDE_W, height=FIG_WIDE_H, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60), legend=dict(font=dict(size=10), x=0.5, y=-0.08, orientation='h', xanchor='center'))
    export_fig(fig_rating, output_dir / f'category{cat_label}_diff_rating_boxplot')

    n_cl_rt = len(valid_clusters)
    fig_jitter = make_subplots(rows=1, cols=n_cl_rt, subplot_titles=[f'Cluster {cl} (n={len(df_valid[df_valid["cluster"]==cl])})' for cl in valid_clusters], horizontal_spacing=0.06)
    all_diffs = []
    for ci, cl in enumerate(valid_clusters):
        subset = df_valid[df_valid['cluster'] == cl]
        color = cluster_colors[int(cl) % len(cluster_colors)]
        means, mins, maxs = [], [], []
        for ei, emo in enumerate(emotion_categories):
            diffs = subset[f'diff_{emo}'].values
            all_diffs.extend(diffs)
            jitter_x = np.full(len(diffs), ei) + np.random.uniform(-0.15, 0.15, len(diffs))
            fig_jitter.add_trace(go.Scatter(x=jitter_x, y=diffs, mode='markers', marker=dict(size=4, color=color, opacity=0.4), name=f'Cluster {cl}' if ei == 0 else None, legendgroup=f'cl{cl}', showlegend=(ei == 0), hovertext=[f'{emo}: {d:.3f}' for d in diffs], hoverinfo='text'), row=1, col=ci + 1)
            means.append(np.mean(diffs))
            mins.append(np.min(diffs))
            maxs.append(np.max(diffs))
        for stat_vals, dash, lbl, show_leg in [(means, 'solid', 'Mean', True), (mins, 'dot', 'Min', False), (maxs, 'dash', 'Max', False)]:
            fig_jitter.add_trace(go.Scatter(x=list(range(len(emotion_categories))), y=stat_vals, mode='lines+markers', marker=dict(size=5), line=dict(color=color, width=2, dash=dash), name=f'C{cl} {lbl}', legendgroup=f'cl{cl}', showlegend=show_leg), row=1, col=ci + 1)
    jitter_bound = np.ceil(max(abs(np.min(all_diffs)), abs(np.max(all_diffs))) * 10) / 10
    fig_jitter.update_xaxes(tickvals=list(range(len(emotion_categories))), ticktext=emotion_labels, tickangle=-45)
    for ci in range(n_cl_rt):
        fig_jitter.add_hline(y=0, line_dash='dash', line_color='red', line_width=2, opacity=0.7, row=1, col=ci + 1)
    fig_jitter.update_layout(title=dict(text=f'Category {cat_label} — 感情差分ジッタープロット', x=0.5, font=dict(size=18)), yaxis_title='差分値', width=max(400 * n_cl_rt, FIG_W), height=FIG_H, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=60, r=60, t=100, b=100), legend=dict(font=dict(size=10), x=0.5, y=-0.2, orientation='h', xanchor='center'))
    for ci in range(n_cl_rt):
        axis_key = f'yaxis{ci + 1}' if ci > 0 else 'yaxis'
        fig_jitter.update_layout(**{axis_key: dict(range=[-jitter_bound, jitter_bound])})
    export_fig(fig_jitter, output_dir / f'category{cat_label}_diff_jitter')

    fig_stats = make_subplots(rows=2, cols=2, subplot_titles=['Mean', 'Max', 'Min', 'Standard Deviation'], horizontal_spacing=0.1, vertical_spacing=0.15)
    stats_all = []
    for cl in valid_clusters:
        subset = df_valid[df_valid['cluster'] == cl]
        color = cluster_colors[int(cl) % len(cluster_colors)]
        stat_means, stat_maxs, stat_mins, stat_sds = [], [], [], []
        for emo in emotion_categories:
            diffs = subset[f'diff_{emo}'].values
            stat_means.append(np.mean(diffs))
            stat_maxs.append(np.max(diffs))
            stat_mins.append(np.min(diffs))
            stat_sds.append(np.std(diffs))
        stats_all.extend(stat_means + stat_maxs + stat_mins + stat_sds)
        for vals, row, col, stat_name in [(stat_means, 1, 1, 'mean'), (stat_maxs, 1, 2, 'max'), (stat_mins, 2, 1, 'min'), (stat_sds, 2, 2, 'sd')]:
            fig_stats.add_trace(go.Bar(x=emotion_labels, y=vals, marker_color=color, name=f'Cluster {cl}', legendgroup=f'cl{cl}', showlegend=(stat_name == 'mean')), row=row, col=col)
    stats_bound = np.ceil(max(abs(np.min(stats_all)), abs(np.max(stats_all))) * 10) / 10
    for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig_stats.add_hline(y=0, line_dash='dash', line_color='red', line_width=2, opacity=0.7, row=r, col=c)
    fig_stats.update_layout(title=dict(text=f'Category {cat_label} — クラスタ別統計比較', x=0.5, font=dict(size=18)), yaxis_title='差分値', yaxis3_title='差分値', yaxis=dict(range=[-stats_bound, stats_bound]), yaxis2=dict(range=[-stats_bound, stats_bound]), yaxis3=dict(range=[-stats_bound, stats_bound]), yaxis4=dict(range=[-stats_bound, stats_bound]), barmode='group', width=FIG_LARGE_W, height=FIG_LARGE_H, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60), legend=dict(font=dict(size=10), x=0.5, y=-0.05, orientation='h', xanchor='center'))
    export_fig(fig_stats, output_dir / f'category{cat_label}_diff_cluster_stats')

    if 'topic' in df_cat.columns:
        ct = pd.crosstab(df_cat['cluster'], df_cat['topic'])
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        fig_ct = go.Figure()
        for tp in sorted(ct_norm.columns):
            fig_ct.add_trace(go.Bar(x=[f'Cluster {cl}' for cl in ct_norm.index], y=ct_norm[tp].values, name=f'Topic {tp}', marker_color=topic_colors[tp % len(topic_colors)]))
        fig_ct.update_layout(title=dict(text=f'Category {cat_label} — Topic分布 within Cluster', x=0.5, font=dict(size=18)), barmode='stack', xaxis_title='Cluster', yaxis_title='Proportion', paper_bgcolor='white', plot_bgcolor='white', width=FIG_W, height=FIG_H, legend=dict(font=dict(size=10), x=0.5, y=-0.15, orientation='h', xanchor='center'), margin=dict(l=60, r=60, t=100, b=80))
        export_fig(fig_ct, output_dir / f'category{cat_label}_topic_cluster_distribution')

    df_cat.to_csv(output_dir / f'category{cat_label}_diff_clusters.csv', index=False, encoding='utf-8-sig')
    cluster_centers.to_csv(output_dir / f'category{cat_label}_diff_centers.csv', encoding='utf-8-sig')
    return df_cat


print("全データクラスタリング")
X_all = df_all[cluster_features].fillna(0)
scaler = StandardScaler()
X_scaled_all = scaler.fit_transform(X_all)

reducer = umap.UMAP(n_components=config.CLUSTER_UMAP_N_COMPONENTS, n_neighbors=config.CLUSTER_UMAP_N_NEIGHBORS, min_dist=config.CLUSTER_UMAP_MIN_DIST, metric=config.CLUSTER_UMAP_METRIC, random_state=config.CLUSTER_RANDOM_SEED)
X_umap_all = reducer.fit_transform(X_scaled_all)

clusterer_all = hdbscan.HDBSCAN(min_cluster_size=config.CLUSTER_HDBSCAN_MIN_CLUSTER_SIZE, min_samples=config.CLUSTER_HDBSCAN_MIN_SAMPLES, prediction_data=True)
labels_all = clusterer_all.fit_predict(X_umap_all)

df_all['umap_0'] = X_umap_all[:, 0]
df_all['umap_1'] = X_umap_all[:, 1]
df_all['cluster'] = labels_all

unique_all = set(labels_all)
n_clusters_all = len(unique_all - {-1})
noise_all = int((labels_all == -1).sum())
mask_all = labels_all != -1

if n_clusters_all >= 2 and mask_all.sum() >= 20:
    print(f"HDBSCAN: {n_clusters_all} clusters, noise {noise_all}")
    print(f"Silhouette: {silhouette_score(X_umap_all[mask_all], labels_all[mask_all]):.4f}")
    print(f"Calinski-Harabasz: {calinski_harabasz_score(X_umap_all[mask_all], labels_all[mask_all]):.2f}")
    print(f"Davies-Bouldin: {davies_bouldin_score(X_umap_all[mask_all], labels_all[mask_all]):.4f}")

df_valid_all = df_all[df_all['cluster'] != -1]
cluster_centers_all = df_valid_all.groupby('cluster')[cluster_features].mean()

fig_scatter_all = go.Figure()
for cl in sorted(unique_all):
    mask_cl = df_all['cluster'] == cl
    subset = df_all[mask_cl]
    if cl == -1:
        fig_scatter_all.add_trace(go.Scatter(x=df_all.loc[mask_cl, 'umap_0'], y=df_all.loc[mask_cl, 'umap_1'], mode='markers', marker=dict(size=3, color='gray', opacity=0.2), name='Noise'))
    else:
        hover = [make_hover_text(r) for _, r in subset.iterrows()]
        fig_scatter_all.add_trace(go.Scatter(x=df_all.loc[mask_cl, 'umap_0'], y=df_all.loc[mask_cl, 'umap_1'], mode='markers', marker=dict(size=5, color=cluster_colors[cl % len(cluster_colors)], opacity=0.7), name=f'Cluster {cl} (n={mask_cl.sum()})', text=hover, hoverinfo='text'))
fig_scatter_all.update_layout(title=dict(text='All Data — Diff UMAP 2D', x=0.5, font=dict(size=18)), xaxis_title='UMAP 1', yaxis_title='UMAP 2', paper_bgcolor='white', plot_bgcolor='white', width=FIG_W, height=FIG_H, margin=dict(l=60, r=60, t=100, b=60))
export_fig(fig_scatter_all, output_dir / 'all_diff_umap_2d')

plot_condensed_tree(clusterer_all, 'All Data — HDBSCAN Condensed Tree', output_dir / 'all_diff_condensed_tree')

X_tsne_all = TSNE(n_components=2, random_state=config.CLUSTER_RANDOM_SEED, perplexity=max(5, min(30, len(df_all)//4))).fit_transform(X_scaled_all)
fig_tsne_all = go.Figure()
for cl in sorted(unique_all):
    mask_cl = labels_all == cl
    if cl == -1:
        fig_tsne_all.add_trace(go.Scatter(x=X_tsne_all[mask_cl, 0], y=X_tsne_all[mask_cl, 1], mode='markers', marker=dict(size=3, color='gray', opacity=0.2), name='Noise'))
    else:
        fig_tsne_all.add_trace(go.Scatter(x=X_tsne_all[mask_cl, 0], y=X_tsne_all[mask_cl, 1], mode='markers', marker=dict(size=6, color=cluster_colors[cl % len(cluster_colors)], opacity=0.7), name=f'Cluster {cl} (n={mask_cl.sum()})'))
fig_tsne_all.update_layout(title=dict(text='All Data — Diff t-SNE', x=0.5, font=dict(size=18)), xaxis_title='t-SNE 1', yaxis_title='t-SNE 2', width=FIG_W, height=FIG_H, plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60))
export_fig(fig_tsne_all, output_dir / 'all_diff_tsne')

valid_clusters_all = sorted(df_valid_all['cluster'].unique())
fig_radar_all = go.Figure()
for cl in valid_clusters_all:
    c = cluster_centers_all.loc[int(cl)]
    vals = [c[f'diff_{e}'] for e in emotion_categories]
    fig_radar_all.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=radar_categories, fill='toself', name=f'Cluster {cl} (n={len(df_valid_all[df_valid_all["cluster"]==cl])})', line=dict(color=cluster_colors[int(cl) % len(cluster_colors)]), opacity=0.3))
fig_radar_all.add_trace(go.Scatterpolar(r=[0.01] * len(radar_categories), theta=radar_categories, mode='lines', line=dict(color='red', width=2, dash='dash'), name='差分=0'))
fig_radar_all.update_layout(**make_radar_layout(f'All Data — Diff Radar (n={len(df_valid_all)})'))
export_fig(fig_radar_all, output_dir / 'all_diff_radar')

df_research = pd.read_csv(config.DATA_DIR / 'real_research.csv', on_bad_lines='warn')
df_research['userId'] = df_research['userId'].astype(str)
df_ratings = df_research[['userId'] + RATING_COLS].copy()
for col in RATING_COLS:
    df_ratings[col] = pd.to_numeric(df_ratings[col], errors='coerce')
df_valid_all_r = df_valid_all.merge(df_ratings, on='userId', how='left')

fig_rating_all = go.Figure()
for cl in valid_clusters_all:
    subset = df_valid_all_r[df_valid_all_r['cluster'] == cl]
    for ri, rc in enumerate(RATING_COLS):
        fig_rating_all.add_trace(go.Box(y=subset[rc], name=RATING_LABELS.get(rc, rc), marker_color=cluster_colors[int(cl) % len(cluster_colors)], legendgroup=f'cl{cl}', legendgrouptitle_text=f'Cluster {cl}' if ri == 0 else None, showlegend=(ri == 0), offsetgroup=f'cl{cl}', boxmean=True))
fig_rating_all.update_layout(title=dict(text='All Data — ユーザー評点分布', x=0.5, font=dict(size=18)), yaxis_title='評点', boxmode='group', width=FIG_WIDE_W, height=FIG_WIDE_H, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60), legend=dict(font=dict(size=10), x=0.5, y=-0.08, orientation='h', xanchor='center'))
export_fig(fig_rating_all, output_dir / 'all_diff_rating_boxplot')

n_cl_all = len(valid_clusters_all)
fig_jitter_all = make_subplots(rows=1, cols=n_cl_all, subplot_titles=[f'Cluster {cl} (n={len(df_valid_all[df_valid_all["cluster"]==cl])})' for cl in valid_clusters_all], horizontal_spacing=0.06)
all_diffs_all = []
for ci, cl in enumerate(valid_clusters_all):
    subset = df_valid_all[df_valid_all['cluster'] == cl]
    color = cluster_colors[int(cl) % len(cluster_colors)]
    means, mins, maxs = [], [], []
    for ei, emo in enumerate(emotion_categories):
        diffs = subset[f'diff_{emo}'].values
        all_diffs_all.extend(diffs)
        jitter_x = np.full(len(diffs), ei) + np.random.uniform(-0.15, 0.15, len(diffs))
        fig_jitter_all.add_trace(go.Scatter(x=jitter_x, y=diffs, mode='markers', marker=dict(size=4, color=color, opacity=0.4), name=f'Cluster {cl}' if ei == 0 else None, legendgroup=f'cl{cl}', showlegend=(ei == 0), hovertext=[f'{emo}: {d:.3f}' for d in diffs], hoverinfo='text'), row=1, col=ci + 1)
        means.append(np.mean(diffs))
        mins.append(np.min(diffs))
        maxs.append(np.max(diffs))
    for stat_vals, dash, lbl, show_leg in [(means, 'solid', 'Mean', True), (mins, 'dot', 'Min', False), (maxs, 'dash', 'Max', False)]:
        fig_jitter_all.add_trace(go.Scatter(x=list(range(len(emotion_categories))), y=stat_vals, mode='lines+markers', marker=dict(size=5), line=dict(color=color, width=2, dash=dash), name=f'C{cl} {lbl}', legendgroup=f'cl{cl}', showlegend=show_leg), row=1, col=ci + 1)
jitter_bound_all = np.ceil(max(abs(np.min(all_diffs_all)), abs(np.max(all_diffs_all))) * 10) / 10
fig_jitter_all.update_xaxes(tickvals=list(range(len(emotion_categories))), ticktext=emotion_labels, tickangle=-45)
for ci in range(n_cl_all):
    fig_jitter_all.add_hline(y=0, line_dash='dash', line_color='red', line_width=2, opacity=0.7, row=1, col=ci + 1)
fig_jitter_all.update_layout(title=dict(text='All Data — 感情差分ジッタープロット', x=0.5, font=dict(size=18)), yaxis_title='差分値', width=max(400 * n_cl_all, FIG_W), height=FIG_H, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=60, r=60, t=100, b=100), legend=dict(font=dict(size=10), x=0.5, y=-0.2, orientation='h', xanchor='center'))
for ci in range(n_cl_all):
    axis_key = f'yaxis{ci + 1}' if ci > 0 else 'yaxis'
    fig_jitter_all.update_layout(**{axis_key: dict(range=[-jitter_bound_all, jitter_bound_all])})
export_fig(fig_jitter_all, output_dir / 'all_diff_jitter')

fig_stats_all = make_subplots(rows=2, cols=2, subplot_titles=['Mean', 'Max', 'Min', 'Standard Deviation'], horizontal_spacing=0.1, vertical_spacing=0.15)
stats_all_vals = []
for cl in valid_clusters_all:
    subset = df_valid_all[df_valid_all['cluster'] == cl]
    color = cluster_colors[int(cl) % len(cluster_colors)]
    stat_means, stat_maxs, stat_mins, stat_sds = [], [], [], []
    for emo in emotion_categories:
        diffs = subset[f'diff_{emo}'].values
        stat_means.append(np.mean(diffs))
        stat_maxs.append(np.max(diffs))
        stat_mins.append(np.min(diffs))
        stat_sds.append(np.std(diffs))
    stats_all_vals.extend(stat_means + stat_maxs + stat_mins + stat_sds)
    for vals, row, col, stat_name in [(stat_means, 1, 1, 'mean'), (stat_maxs, 1, 2, 'max'), (stat_mins, 2, 1, 'min'), (stat_sds, 2, 2, 'sd')]:
        fig_stats_all.add_trace(go.Bar(x=emotion_labels, y=vals, marker_color=color, name=f'Cluster {cl}', legendgroup=f'cl{cl}', showlegend=(stat_name == 'mean')), row=row, col=col)
stats_bound_all = np.ceil(max(abs(np.min(stats_all_vals)), abs(np.max(stats_all_vals))) * 10) / 10
for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
    fig_stats_all.add_hline(y=0, line_dash='dash', line_color='red', line_width=2, opacity=0.7, row=r, col=c)
fig_stats_all.update_layout(title=dict(text='All Data — クラスタ別統計比較', x=0.5, font=dict(size=18)), yaxis_title='差分値', yaxis3_title='差分値', yaxis=dict(range=[-stats_bound_all, stats_bound_all]), yaxis2=dict(range=[-stats_bound_all, stats_bound_all]), yaxis3=dict(range=[-stats_bound_all, stats_bound_all]), yaxis4=dict(range=[-stats_bound_all, stats_bound_all]), barmode='group', width=FIG_LARGE_W, height=FIG_LARGE_H, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60), legend=dict(font=dict(size=10), x=0.5, y=-0.05, orientation='h', xanchor='center'))
export_fig(fig_stats_all, output_dir / 'all_diff_cluster_stats')

if 'topic' in df_all.columns:
    ct_all = pd.crosstab(df_all['cluster'], df_all['topic'])
    ct_norm_all = ct_all.div(ct_all.sum(axis=1), axis=0)
    fig_ct_all = go.Figure()
    for tp in sorted(ct_norm_all.columns):
        fig_ct_all.add_trace(go.Bar(x=[f'Cluster {cl}' for cl in ct_norm_all.index], y=ct_norm_all[tp].values, name=f'Topic {tp}', marker_color=topic_colors[tp % len(topic_colors)]))
    fig_ct_all.update_layout(title=dict(text='All Data — Topic分布 within Cluster', x=0.5, font=dict(size=18)), barmode='stack', xaxis_title='Cluster', yaxis_title='Proportion', paper_bgcolor='white', plot_bgcolor='white', width=FIG_W, height=FIG_H, legend=dict(font=dict(size=10), x=0.5, y=-0.15, orientation='h', xanchor='center'), margin=dict(l=60, r=60, t=100, b=80))
    export_fig(fig_ct_all, output_dir / 'all_topic_cluster_distribution')

df_all.to_csv(output_dir / 'all_diff_clusters.csv', index=False, encoding='utf-8-sig')
cluster_centers_all.to_csv(output_dir / 'all_diff_centers.csv', encoding='utf-8-sig')

for cat_label in [0, 1]:
    cat_dir = output_dir / f'category{cat_label}'
    cat_dir.mkdir(parents=True, exist_ok=True)
    df_cat = df_all[df_all['category'] == cat_label].copy()
    run_pipeline(df_cat, cat_label, umap_dim=config.CLUSTER_UMAP_N_COMPONENTS, nn=config.CLUSTER_UMAP_N_NEIGHBORS, md=config.CLUSTER_UMAP_MIN_DIST, mcs=config.CLUSTER_HDBSCAN_MIN_CLUSTER_SIZE, ms=config.CLUSTER_HDBSCAN_MIN_SAMPLES, output_dir=cat_dir)

df_result0 = pd.read_csv(output_dir / 'category0' / 'category0_diff_clusters.csv', on_bad_lines='warn')
df_result1 = pd.read_csv(output_dir / 'category1' / 'category1_diff_clusters.csv', on_bad_lines='warn')
pd.concat([df_result0, df_result1], ignore_index=True).to_csv(output_dir / 'all_diff_clusters.csv', index=False, encoding='utf-8-sig')

print("Done")
