"""
トピックモデリング可視化スクリプト
topic.py の出力 CSV からキーワード棒グラフを生成する。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

FIG_WIDE_W, FIG_WIDE_H = 1400, 700

TOPIC_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
]


def export_fig(fig, base_path):
    """HTML + SVG の2形式で出力（SVGは標題を除去）"""
    fig.write_html(str(base_path) + ".html")
    fig_copy = go.Figure(fig)
    fig_copy.update_layout(title=None)
    fig_copy.write_image(str(base_path) + ".svg", width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)


def plot_keyword_bars(df_kw, topic_ids, output_path, prefix):
    """キーワード棒グラフ（各トピック Top-10、サブプロット）"""
    n_cols = min(3, len(topic_ids))
    n_rows = (len(topic_ids) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Topic {t}" for t in topic_ids],
        vertical_spacing=0.08, horizontal_spacing=0.12,
    )
    for idx, tid in enumerate(topic_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        kws = df_kw[df_kw["topic_id"] == tid].nlargest(10, "score").sort_values("score")
        fig.add_trace(go.Bar(
            x=kws["score"], y=kws["keyword"], orientation="h",
            marker_color=TOPIC_COLORS[tid % len(TOPIC_COLORS)],
            showlegend=False,
        ), row=row, col=col)

    fig.update_layout(
        title=dict(text=f"{prefix} — Topic Keywords (Top 10)", x=0.5, font=dict(size=16)),
        paper_bgcolor="white", width=FIG_WIDE_W,
        height=max(400, n_rows * 300),
        margin=dict(l=120, r=40, t=80, b=40),
    )
    export_fig(fig, output_path)
    print(f"  ✓ キーワード棒グラフ")


def main():
    topic_dir = config.DATA_DIR / "topic_modeling"
    prefix = "combined_userInput"

    kw_path = topic_dir / f"{prefix}_topic_keywords.csv"
    if not kw_path.exists():
        print("CSV ファイルが見つかりません。先に topic.py を実行してください。")
        return

    df_kw = pd.read_csv(kw_path)
    topic_ids = sorted(df_kw["topic_id"].unique())
    print(f"トピック数: {len(topic_ids)}")

    vis_dir = topic_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("可視化生成中...")
    plot_keyword_bars(df_kw, topic_ids, vis_dir / f"{prefix}_topic_keywords_bars", prefix)
    print(f"\n可視化完了: {vis_dir}")


if __name__ == "__main__":
    main()
