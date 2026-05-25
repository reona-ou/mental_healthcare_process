import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

# 加载数据 / データの読み込み
file_path = config.DATA_DIR / 'sentiment' / 'sentiment.csv'
try:
    df = pd.read_csv(file_path, on_bad_lines='warn')
except Exception as e:
    print(f"读取 CSV 出错 / CSV読み取りエラー: {e}")
    exit(1)

print(f"总行数 / 総行数: {len(df)}")
print(f"列名 / 列名: {df.columns.tolist()}")

# 定义特征列 / 特徴列の定義
# 用户输入情绪 / ユーザー入力感情
input_features = [
    'input_joy', 'input_sadness', 'input_anticipation', 'input_surprise',
    'input_anger', 'input_fear', 'input_disgust', 'input_trust'
]

# 回复情绪 / 返答感情
reply_features = [
    'reply_joy', 'reply_sadness', 'reply_anticipation', 'reply_surprise',
    'reply_anger', 'reply_fear', 'reply_disgust', 'reply_trust'
]

# 是否为ReplyInterruptPersona的标记 / ReplyInterruptPersonaかどうかのフラグ
interrupt_persona_feature = ['is_reply_interrupt_persona']

# 检查特征列是否存在 / 特徴列の存在チェック
missing_features = [f for f in (input_features + reply_features + interrupt_persona_feature) if f not in df.columns]
if missing_features:
    # 创建is_reply_interrupt_persona列 / is_reply_interrupt_persona列を作成
    if 'is_reply_interrupt_persona' in missing_features and 'replyType' in df.columns:
        df['is_reply_interrupt_persona'] = (df['replyType'] == 'ReplyInterruptPersona').astype(float)
        print("已从replyType列推导出is_reply_interrupt_persona / replyType列からis_reply_interrupt_personaを導出しました")
    else:
        print(f"缺失特征列 / 欠損特徴列: {missing_features}")
        exit(1)

# Persona one-hot encoding / ペルソナのワンホットエンコーディング
persona_feature_names = []
if 'persona' in df.columns:
    persona_dummies = pd.get_dummies(df['persona'], prefix='persona').astype(float)
    df = pd.concat([df, persona_dummies], axis=1)
    persona_feature_names = persona_dummies.columns.tolist()
    print(f"已对persona列进行one-hot编码，生成特征 / persona列をワンホットエンコーディング、生成特徴: {persona_feature_names}")
else:
    print("警告: 未找到persona列 / 警告: persona列が見つかりません")

all_features = input_features + reply_features + interrupt_persona_feature + persona_feature_names

print(f"\n使用的特征维度 / 使用する特徴次元: {len(all_features)}")
print(f"  - 输入情绪(input) / 入力感情(input): {len(input_features)}维")
print(f"  - 回复情绪(reply) / 返答感情(reply): {len(reply_features)}维")
print(f"  - 是否为ReplyInterruptPersona / ReplyInterruptPersona与否: {len(interrupt_persona_feature)}维")
print(f"  - Persona one-hot / ペルソナ ワンホット: {len(persona_feature_names)}维")

# 数据预处理 / データ前処理
# 转换为数值类型 / 数値型に変換
for col in input_features + reply_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 处理 is_reply_interrupt_persona 列 / is_reply_interrupt_persona列の処理
df['is_reply_interrupt_persona'] = df['is_reply_interrupt_persona'].fillna(0).astype(float)

# 提取特征矩阵 / 特徴行列の抽出
X = df[all_features].fillna(0)

print(f"\n特征矩阵形状 / 特徴行列の形状: {X.shape}")
print(f"缺失值数量 / 欠損値数: {X.isnull().sum().sum()}")

# 查看 is_reply_interrupt_persona 数据情况 / is_reply_interrupt_personaのデータ状況を確認
interrupt_persona_count = (df['is_reply_interrupt_persona'] > 0).sum()
print(f"有ReplyInterruptPersona的数据数 / ReplyInterruptPersonaのデータ数: {interrupt_persona_count} / {len(df)}")

# 标准化特征 / 特徴の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 肘部法则(Elbow Method)确定最优K / エルボー法で最適Kを決定
print("\n肘部法则分析 / エルボー法による分析")
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.4f}")

# 计算肘部点（使用二阶差分） / 肘点を計算（二階差分を使用）
inertia_diff = np.diff(inertias)
inertia_diff2 = np.diff(inertia_diff)
elbow_k_idx = np.argmax(inertia_diff2) + 2  # +2因为二阶差分 / +2は二階差分のため
best_k = list(K_range)[elbow_k_idx]
print(f"\n肘部法则建议的K / エルボー法による推奨K: {best_k}")

# 保存肘部法则图 / エルボー法グラフを保存
output_dir = config.DATA_DIR / 'sentiment/kmeans'
output_dir.mkdir(parents=True, exist_ok=True)

# 使用 Plotly 生成交互式肘部法则图 / Plotlyでインタラクティブなエルボー法グラフを生成
fig_elbow = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Elbow Method — Inertia', 'Silhouette Score')
)

fig_elbow.add_trace(
    go.Scatter(
        x=list(K_range), y=inertias,
        mode='lines+markers',
        name='Inertia',
        marker=dict(color='blue', size=8),
        line=dict(color='blue')
    ),
    row=1, col=1
)
fig_elbow.add_vline(x=best_k, line_dash="dash", line_color="red",
                    annotation_text=f"Best K={best_k}", row=1, col=1)

fig_elbow.add_trace(
    go.Scatter(
        x=list(K_range), y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette',
        marker=dict(color='red', size=8),
        line=dict(color='red')
    ),
    row=1, col=2
)
fig_elbow.add_vline(x=best_k, line_dash="dash", line_color="blue",
                    annotation_text=f"Best K={best_k}", row=1, col=2)

fig_elbow.update_xaxes(title_text="K", row=1, col=1)
fig_elbow.update_xaxes(title_text="K", row=1, col=2)
fig_elbow.update_yaxes(title_text="Inertia", row=1, col=1)
fig_elbow.update_yaxes(title_text="Silhouette Score", row=1, col=2)

fig_elbow.update_layout(
    title=dict(text='KMeans 肘部法则分析 / KMeans エルボー法分析', x=0.5, font=dict(size=22)),
    width=1400, height=500,
    plot_bgcolor='white', paper_bgcolor='white',
    showlegend=False,
    margin=dict(l=80, r=80, t=100, b=60)
)

elbow_path = output_dir / 'kmeans_elbow_plot.html'
fig_elbow.write_html(elbow_path)
print(f"肘部法则图已保存至 / エルボー法グラフを保存しました: {elbow_path}")

# 使用最优K进行最终聚类 / 最適Kで最終クラスタリングを実行
print(f"\n使用 K={best_k} 进行最终聚类 / K={best_k} で最終クラスタリングを実行")
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_scaled)

# 分析聚类中心 / クラスタ中心を分析
# 反标准化回原始量度 / 逆標準化して元の尺度に戻す
cluster_centers_raw = pd.DataFrame(
    scaler.inverse_transform(kmeans_final.cluster_centers_),
    columns=all_features
)
cluster_counts = df['cluster'].value_counts().sort_index()
cluster_centers_raw['count'] = cluster_counts.values

print("\n聚类中心 (原始量度) / クラスタ中心 (元の尺度):")

for i in range(best_k):
    center = cluster_centers_raw.iloc[i]
    print(f"\n--- Cluster {i} (样本数 / サンプル数: {int(center['count'])}) ---")
    print(f"  [输入情绪 / 入力感情] joy={center['input_joy']:.3f} sad={center['input_sadness']:.3f} "
          f"antic={center['input_anticipation']:.3f} surp={center['input_surprise']:.3f} "
          f"anger={center['input_anger']:.3f} fear={center['input_fear']:.3f} "
          f"disg={center['input_disgust']:.3f} trust={center['input_trust']:.3f}")
    print(f"  [回复情绪 / 返答感情] joy={center['reply_joy']:.3f} sad={center['reply_sadness']:.3f} "
          f"antic={center['reply_anticipation']:.3f} surp={center['reply_surprise']:.3f} "
          f"anger={center['reply_anger']:.3f} fear={center['reply_fear']:.3f} "
          f"disg={center['reply_disgust']:.3f} trust={center['reply_trust']:.3f}")
    print(f"  [是否ReplyInterruptPersona / ReplyInterruptPersona与否] {center['is_reply_interrupt_persona']:.3f}")
    if persona_feature_names:
        persona_info = ", ".join([f"{col}={center[col]:.3f}" for col in persona_feature_names])
        print(f"  [Persona / ペルソナ] {persona_info}")

# 每个聚类的特征解读 / 各クラスタの特徴解釈
print("\n\n聚类特征解读 / クラスタ特徴解釈")
for i in range(best_k):
    center = cluster_centers_raw.iloc[i]
    count = int(center['count'])
    
    # 判断主导情绪 / 支配的な感情を判定
    input_emotions = {col.replace('input_', ''): center[col] for col in input_features}
    reply_emotions = {col.replace('reply_', ''): center[col] for col in reply_features}

    top_input = max(input_emotions, key=input_emotions.get)
    top_reply = max(reply_emotions, key=reply_emotions.get)

    is_reply_interrupt = center['is_reply_interrupt_persona'] > 0.5

    print(f"\nCluster {i} ({count}条 / 件):")
    print(f"  用户主导情绪 / ユーザーの支配的感情: {top_input} ({input_emotions[top_input]:.3f})")
    print(f"  回复主导情绪 / 返答の支配的感情: {top_reply} ({reply_emotions[top_reply]:.3f})")
    print(f"  是否为ReplyInterruptPersona / ReplyInterruptPersona与否: {center['is_reply_interrupt_persona']:.3f} ({'是' if is_reply_interrupt else '否'})")
    if persona_feature_names:
        dominant_persona = max(persona_feature_names, key=lambda col: center[col])
        dominant_persona_name = dominant_persona.replace('persona_', '')
        print(f"  主要Persona / 主要ペルソナ: {dominant_persona_name} ({center[dominant_persona]:.3f})")
        persona_details = ", ".join([f"{col.replace('persona_', '')}={center[col]:.3f}" for col in persona_feature_names])
        print(f"  Persona分布 / ペルソナ分布: {persona_details}")

# 保存结果 / 結果を保存
output_path = output_dir / 'conversations_kmeans_clusters.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n聚类结果已保存至 / クラスタリング結果を保存しました: {output_path}")

# 保存聚类中心 / クラスタ中心を保存
centers_path = output_dir / 'kmeans_cluster_centers.csv'
cluster_centers_raw.to_csv(centers_path, index=True, encoding='utf-8-sig')
print(f"聚类中心已保存至 / クラスタ中心を保存しました: {centers_path}")

# 每类样本示例 / 各クラスタのサンプル例
print("\n\n每类样本示例 (前3条) / 各クラスタのサンプル例 (先3件)")
display_cols = ['session_id', 'userId', 'userInput', 'persona', 'replyType']
for i in range(best_k):
    print(f"\n--- Cluster {i} ---")
    subset = df[df['cluster'] == i]
    if not subset.empty:
        available_cols = [c for c in display_cols if c in subset.columns]
        print(subset[available_cols].head(3).to_string(index=False))
    else:
        print("无数据 / データなし")

# 按persona分组统计聚类分布 / ペルソナごとのクラスタ分布を集計
if 'persona' in df.columns:
    print("\n\n各角色在各聚类中的分布 / 各ペルソナのクラスタ分布")
    cross_tab = pd.crosstab(df['persona'], df['cluster'], margins=True)
    print(cross_tab)
    
    cross_path = output_dir / 'kmeans_cluster_by_persona.csv'
    cross_tab.to_csv(cross_path, encoding='utf-8-sig')
    print(f"角色-聚类交叉表已保存至 / ペルソナ-クラスタクロス集計表を保存しました: {cross_path}")

# Plotly 交互式可视化 / Plotlyインタラクティブ可視化

# 雷达图：各聚类的输入情绪轮廓 / レーダーチャート：各クラスタの入力感情プロファイル
emotion_categories = ['joy', 'sadness', 'anticipation', 'surprise', 'anger', 'fear', 'disgust', 'trust']
# 英日双语标签 / 英日バイリンガルラベル
emotion_labels = [
    'Joy / 喜び',
    'Sadness / 悲しみ',
    'Anticipation / 期待',
    'Surprise / 驚き',
    'Anger / 怒り',
    'Fear / 恐れ',
    'Disgust / 嫌悪',
    'Trust / 信頼'
]
radar_categories = emotion_labels + [emotion_labels[0]]  # 闭合 / 閉じる

# 调色板 / カラーパレット
cluster_colors = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
]

fig_radar = go.Figure()
for i in range(best_k):
    center = cluster_centers_raw.iloc[i]
    # 输入情绪 / 入力感情
    input_vals = [center[f'input_{e}'] for e in emotion_categories]
    input_vals_closed = input_vals + [input_vals[0]]

    fig_radar.add_trace(
        go.Scatterpolar(
            r=input_vals_closed,
            theta=radar_categories,
            fill='toself',
            name=f'Cluster {i}',
            line=dict(color=cluster_colors[i % len(cluster_colors)]),
            opacity=0.3
        )
    )

fig_radar.update_layout(
    title=dict(text='各聚类 — 输入情绪轮廓 / 各クラスタ — 入力感情プロファイル', x=0.5, font=dict(size=22)),
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1]),
        bgcolor='white'
    ),
    width=1200, height=700,
    paper_bgcolor='white',
    legend=dict(font=dict(size=11)),
    margin=dict(l=80, r=80, t=100, b=60)
)

radar_path = output_dir / 'cluster_radar_input.html'
fig_radar.write_html(radar_path)
print(f"输入情绪雷达图已保存至 / 入力感情レーダーチャートを保存しました: {radar_path}")

# 雷达图：各聚类的回复情绪轮廓 / レーダーチャート：各クラスタの返答感情プロファイル
fig_radar_reply = go.Figure()
for i in range(best_k):
    center = cluster_centers_raw.iloc[i]
    reply_vals = [center[f'reply_{e}'] for e in emotion_categories]
    reply_vals_closed = reply_vals + [reply_vals[0]]

    fig_radar_reply.add_trace(
        go.Scatterpolar(
            r=reply_vals_closed,
            theta=radar_categories,
            fill='toself',
            name=f'Cluster {i}',
            line=dict(color=cluster_colors[i % len(cluster_colors)]),
            opacity=0.3
        )
    )

fig_radar_reply.update_layout(
    title=dict(text='各聚类 — 回复情绪轮廓 / 各クラスタ — 返答感情プロファイル', x=0.5, font=dict(size=22)),
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1]),
        bgcolor='white'
    ),
    width=1200, height=700,
    paper_bgcolor='white',
    legend=dict(font=dict(size=11)),
    margin=dict(l=80, r=80, t=100, b=60)
)

radar_reply_path = output_dir / 'cluster_radar_reply.html'
fig_radar_reply.write_html(radar_reply_path)
print(f"回复情绪雷达图已保存至 / 返答感情レーダーチャートを保存しました: {radar_reply_path}")

print("\n所有可视化已保存至 / 全ての可視化を保存しました:", output_dir)
print("\n完成 / 完了")
