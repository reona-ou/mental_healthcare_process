import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

FIG_WIDE_W, FIG_WIDE_H = 1400, 800


def export_fig(fig, base_path):
    fig.write_html(str(base_path) + '.html')
    svg_fig = go.Figure(fig)
    svg_fig.layout.title.text = ""
    svg_fig.update_layout(margin=dict(l=100, r=80, t=20, b=80))
    svg_fig.write_image(str(base_path) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)


df = pd.read_csv(config.DATA_DIR / "sentiment/sentiment.csv")
emotions = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]

stat_funcs = {
    "max": lambda x: np.max(x), "min": lambda x: np.min(x),
    "Q1(25%)": lambda x: np.percentile(x, 25), "mean": lambda x: np.mean(x),
    "median": lambda x: np.median(x), "Q3(75%)": lambda x: np.percentile(x, 75),
}
vis_stat_order = ["max", "min", "mean", "Q1(25%)", "median", "Q3(75%)"]

output_dir = config.DATA_DIR / "sentiment/sentiment_diff"
output_dir.mkdir(parents=True, exist_ok=True)

for user_id, user_df in df.groupby("userId"):
    user_rows = []
    stats_results = []

    for _, row in user_df.iterrows():
        record = {k: row[k] for k in ["session_id", "userId", "replyType", "persona", "userInput", "replyText"]}
        for i_e in emotions:
            for r_e in emotions:
                record[f"diff_{i_e}_{r_e}"] = row[f"input_{i_e}"] - row[f"reply_{r_e}"]
        user_rows.append(record)

    for stat_name, stat_func in stat_funcs.items():
        for i_e in emotions:
            for r_e in emotions:
                diff_values = user_df[f"input_{i_e}"].values - user_df[f"reply_{r_e}"].values
                stats_results.append({"stat": stat_name, "input_emotion": i_e, "reply_emotion": r_e, "value": stat_func(diff_values)})

    if user_rows:
        diff_path = output_dir / "differences" / f"{user_id}_differences.csv"
        pd.DataFrame(user_rows).to_csv(diff_path, index=False)

    if stats_results:
        stats_path = output_dir / "statistics" / f"{user_id}_statistics.csv"
        pd.DataFrame(stats_results).to_csv(stats_path, index=False)

print(f"Per-user CSVs saved to {output_dir}")

print("Generating all-session output...")
all_rows = []
for _, row in df.iterrows():
    record = {k: row[k] for k in ["session_id", "userId", "replyType", "persona", "userInput", "replyText"]}
    for emo in emotions:
        record[f"diff_{emo}"] = row[f"reply_{emo}"] - row[f"input_{emo}"]
    all_rows.append(record)

all_df = pd.DataFrame(all_rows)
all_path = config.DATA_DIR / "sentiment" / "sentiment_all_diff.csv"
all_df.to_csv(all_path, index=False, encoding="utf-8-sig")

all_stats_results = []
for stat_name, stat_func in stat_funcs.items():
    for emo in emotions:
        diff_col = f"diff_{emo}"
        if diff_col in all_df.columns:
            all_stats_results.append({"stat": stat_name, "emotion": emo, "value": stat_func(all_df[diff_col].values)})

all_stats_df = pd.DataFrame(all_stats_results)
all_stats_path = config.DATA_DIR / "sentiment" / "sentiment_all_diff_statistics.csv"
all_stats_df.to_csv(all_stats_path, index=False, encoding="utf-8-sig")

if not all_stats_df.empty:
    all_values = all_stats_df["value"].values
    abs_max = max(abs(all_values.min()), abs(all_values.max()))

    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{stat}" for stat in vis_stat_order], vertical_spacing=0.15, horizontal_spacing=0.12)

    for idx, stat_name in enumerate(vis_stat_order):
        row_idx, col_idx = idx // 3 + 1, idx % 3 + 1
        stat_data = all_stats_df[all_stats_df["stat"] == stat_name]
        values = [stat_data[stat_data["emotion"] == emo]["value"].values[0] if len(stat_data[stat_data["emotion"] == emo]["value"].values) > 0 else 0 for emo in emotions]

        fig.add_trace(go.Bar(
            x=emotions, y=values, text=[f"{v:.2f}" for v in values], textposition="auto", textfont=dict(size=12),
            marker=dict(colorscale="RdBu_r", cmin=-abs_max, cmax=abs_max, color=values, showscale=(idx == 0), colorbar=dict(title="Value", thickness=15, len=0.3, y=0.5 if idx == 0 else None)),
        ), row=row_idx, col=col_idx)

    fig.update_layout(title=dict(text="All Sessions Sentiment Difference Statistics (Reply - Input)", font=dict(size=20), x=0.5, xanchor="center"), height=800, width=1400, template="plotly_white", showlegend=False, margin=dict(l=80, r=80, t=120, b=80))

    for i in range(1, 7):
        row_idx, col_idx = (i - 1) // 3 + 1, (i - 1) % 3 + 1
        fig.update_xaxes(title_text="Emotion" if row_idx == 2 else "", tickangle=45, tickfont=dict(size=12), row=row_idx, col=col_idx)
        fig.update_yaxes(title_text="Difference Value" if col_idx == 1 else "", tickfont=dict(size=12), zeroline=True, zerolinecolor="gray", zerolinewidth=1, matches="y" if col_idx != 1 else None, row=row_idx, col=col_idx)

    for annotation in fig.layout.annotations:
        annotation.font.size = 16

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    export_fig(fig, vis_dir / "all_sessions_statistics")

print("Done")
