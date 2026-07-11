"""
感情分析サマリーチャート生成
- userInput / reply / overall の3カテゴリ
- 主導感情分布、レーダー、統計指標、箱ひげ図
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

# 出力ディレクトリ
SENTIMENT_DIR = config.DATA_DIR / "sentiment"

# 8感情定義
EMOTIONS = ['joy', 'sadness', 'anticipation', 'surprise', 'anger', 'fear', 'disgust', 'trust']
EMOTION_LABELS = {'joy': '喜び', 'sadness': '悲しみ', 'anticipation': '期待', 'surprise': '驚き', 'anger': '怒り', 'fear': '恐れ', 'disgust': '嫌悪', 'trust': '信頼'}
EMOTION_COLORS = {'joy': '#FFD700', 'sadness': '#1E90FF', 'anticipation': '#FF8C00', 'surprise': '#FF69B4', 'anger': '#DC143C', 'fear': '#8B008B', 'disgust': '#2E8B57', 'trust': '#4169E1'}
RADAR_LABELS = [EMOTION_LABELS[e] for e in EMOTIONS] + [EMOTION_LABELS[EMOTIONS[0]]]

# 統一サイズ定義
FIG_W, FIG_H = 1200, 700
FIG_WIDE_W, FIG_WIDE_H = 1400, 700
FIG_LARGE_W, FIG_LARGE_H = 1400, 800


def export_fig(fig, base_path):
    """plotly の fig を HTML + SVG の2形式で出力"""
    fig.write_html(str(base_path) + '.html')
    fig.write_image(str(base_path) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)


def compute_stats(scores_df):
    """各感情の統計指標を計算"""
    stats = {}
    for emo in EMOTIONS:
        vals = scores_df[emo].dropna()
        stats[emo] = {'mean': vals.mean(), 'median': vals.median(), 'std': vals.std(), 'max': vals.max(), 'min': vals.min()}
    return pd.DataFrame(stats, index=['mean', 'median', 'std', 'max', 'min']).T


def get_dominant_emotions(scores_df):
    """各データの主導感情を取得"""
    return scores_df[EMOTIONS].idxmax(axis=1)


def create_dominant_emotion_chart(dominant_counts, title, output_path):
    """主導感情の棒グラフ"""
    labels = [EMOTION_LABELS[e] for e in dominant_counts.index]
    colors = [EMOTION_COLORS[e] for e in dominant_counts.index]
    fig = go.Figure(go.Bar(x=labels, y=dominant_counts.values, marker_color=colors, text=dominant_counts.values, textposition='outside'))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=18)), xaxis_title='感情', yaxis_title='件数', width=FIG_W, height=FIG_H, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60))
    export_fig(fig, output_path.with_suffix(''))


def create_radar_chart(stats_df, title, output_path):
    """レーダーチャート（平均値）"""
    vals = stats_df['mean'].tolist() + [stats_df['mean'].tolist()[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=RADAR_LABELS, fill='toself', line=dict(color='#636EFA', width=2), opacity=0.4, name='Mean'))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=18)), polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor='white'), width=FIG_WIDE_W, height=FIG_WIDE_H, paper_bgcolor='white', margin=dict(l=80, r=80, t=100, b=80))
    export_fig(fig, output_path.with_suffix(''))


def create_stats_bar_chart(stats_df, title, output_path):
    """統計指標の比較棒グラフ（mean/median/std/max/min）"""
    labels = [EMOTION_LABELS[e] for e in stats_df.index]
    colors = [EMOTION_COLORS[e] for e in stats_df.index]
    stat_names = ['mean', 'median', 'std', 'max', 'min']
    fig = make_subplots(rows=2, cols=3, subplot_titles=['平均値', '中央値', '標準偏差', '最大値', '最小値'], horizontal_spacing=0.08, vertical_spacing=0.15)
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

    all_vals = [v for stat in stat_names for v in stats_df[stat].tolist()]
    bound = max(abs(min(all_vals)), abs(max(all_vals))) * 1.1

    for (row, col), stat in zip(positions, stat_names):
        fig.add_trace(go.Bar(x=labels, y=stats_df[stat].tolist(), marker_color=colors, name=stat, showlegend=False), row=row, col=col)
        if stat in ('mean', 'median'):
            fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.5, row=row, col=col)

    for r, c in positions:
        fig.update_layout(**{f'yaxis{r if c == 1 else ""}{"" if (r, c) == (1, 1) else (r - 1) * 3 + c}': dict(range=[-bound, bound])})

    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=18)), width=FIG_LARGE_W, height=FIG_LARGE_H, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=60, r=60, t=100, b=100), legend=dict(font=dict(size=10)))
    export_fig(fig, output_path.with_suffix(''))


def create_emotion_boxplot(scores_df, title, output_path):
    """各感情の箱ひげ図"""
    fig = go.Figure()
    for emo in EMOTIONS:
        fig.add_trace(go.Box(y=scores_df[emo].dropna(), name=EMOTION_LABELS[emo], marker_color=EMOTION_COLORS[emo], boxmean=True))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=18)), yaxis_title='スコア', width=FIG_WIDE_W, height=FIG_WIDE_H, paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=60, r=60, t=100, b=60))
    export_fig(fig, output_path.with_suffix(''))


def analyze_group(scores_df, prefix, output_dir):
    """グループごとに全チャートを生成"""
    output_dir.mkdir(parents=True, exist_ok=True)
    col_names = [f'{prefix}_{e}' for e in EMOTIONS]
    renamed = scores_df[col_names].copy()
    renamed.columns = EMOTIONS

    stats_df = compute_stats(renamed)
    dominant = get_dominant_emotions(renamed)
    dominant_counts = dominant.value_counts().reindex(EMOTIONS, fill_value=0)

    # CSV保存
    stats_df.to_csv(output_dir / f'{prefix}_statistics.csv', encoding='utf-8-sig')

    # チャート生成
    create_dominant_emotion_chart(dominant_counts, f'{prefix.upper()} — 主導感情分布', output_dir / f'{prefix}_dominant_emotion.html')
    create_radar_chart(stats_df, f'{prefix.upper()} — 感情レーダー', output_dir / f'{prefix}_radar.html')
    create_stats_bar_chart(stats_df, f'{prefix.upper()} — 統計指標比較', output_dir / f'{prefix}_stats_comparison.html')
    create_emotion_boxplot(renamed, f'{prefix.upper()} — 感情スコア分布', output_dir / f'{prefix}_boxplot.html')
    return stats_df


if __name__ == "__main__":
    print("感情分析サマリーチャート生成\n")

    df = pd.read_csv(SENTIMENT_DIR / 'sentiment.csv')
    print(f"データ数: {len(df)}, ユーザー数: {df['userId'].nunique()}\n")

    # userInput 分析
    print("=== userInput 分析 ===")
    stats_input = analyze_group(df, 'input', SENTIMENT_DIR / 'input_analysis')

    # reply 分析
    print("\n=== reply 分析 ===")
    stats_reply = analyze_group(df, 'reply', SENTIMENT_DIR / 'reply_analysis')

    # overall 分析（input + reply 混合）
    print("\n=== overall 分析 ===")
    overall_dir = SENTIMENT_DIR / 'overall_analysis'
    overall_dir.mkdir(parents=True, exist_ok=True)

    overall_df = pd.DataFrame({emo: pd.concat([df[f'input_{emo}'], df[f'reply_{emo}']], ignore_index=True) for emo in EMOTIONS})
    overall_stats = compute_stats(overall_df)
    overall_dominant = get_dominant_emotions(overall_df)
    overall_dominant_counts = overall_dominant.value_counts().reindex(EMOTIONS, fill_value=0)

    overall_stats.to_csv(overall_dir / 'overall_statistics.csv', encoding='utf-8-sig')
    create_dominant_emotion_chart(overall_dominant_counts, '全体 — 主導感情分布', overall_dir / 'overall_dominant_emotion.html')
    create_radar_chart(overall_stats, '全体 — 感情レーダー', overall_dir / 'overall_radar.html')
    create_stats_bar_chart(overall_stats, '全体 — 統計指標比較', overall_dir / 'overall_stats_comparison.html')
    create_emotion_boxplot(overall_df, '全体 — 感情スコア分布', overall_dir / 'overall_boxplot.html')

    print(f"\n完了")
    print(f"出力先: {SENTIMENT_DIR}")
    print(f"  input_analysis/   — userInput 感情分析")
    print(f"  reply_analysis/   — chatbotReply 感情分析")
    print(f"  overall_analysis/ — 全体感情分析")
