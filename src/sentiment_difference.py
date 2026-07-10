"""
sentiment.csv の感情差分分析（8×8マトリクス）/ sentiment.csv 的情感差分分析（8×8矩阵）
- 按用户输出CSV / ユーザーごとにCSVを出力
- 保存每次对话的差分 / 各対話の差分を保存
- 按用户计算并保存统计量（max, min, median, mean, Q1, Q3）/ ユーザーごとに統計量（max, min, median, mean, Q1, Q3）を計算・保存
- 全session汇总热力图+条形图（HTML+SVG输出）/ 全sessionまとめヒートマップ+棒グラフ（HTML+SVG出力）
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

# === 統一サイズ定義 / Unified figure sizes ===
FIG_WIDE_W, FIG_WIDE_H = 1400, 800


def export_fig(fig, base_path):
    """plotly の fig を HTML + SVG の2形式で出力。SVGにはタイトルなし"""
    fig.write_html(str(base_path) + '.html')
    # SVG用：タイトルを除去したコピーを作成
    svg_fig = go.Figure(fig)
    svg_fig.layout.title.text = ""
    svg_fig.update_layout(margin=dict(l=100, r=80, t=20, b=80))
    svg_fig.write_image(str(base_path) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)

# 数据加载 / データ読み込み
df = pd.read_csv(config.DATA_DIR / "sentiment/sentiment.csv")

# 8个情感维度 / 8つの感情次元
emotions = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]

# 统计量类型（包含mean）/ 統計量の種類（meanを追加）
stat_funcs = {
    "max": lambda x: np.max(x),
    "min": lambda x: np.min(x),
    "Q1(25%)": lambda x: np.percentile(x, 25),
    "mean": lambda x: np.mean(x),
    "median": lambda x: np.median(x),
    "Q3(75%)": lambda x: np.percentile(x, 75),
}

# 可视化用的统计量顺序 / 可視化用の統計量順序
vis_stat_order = ["max", "min", "mean", "Q1(25%)", "median", "Q3(75%)"]

# 输出目录 / 出力ディレクトリ
output_dir = config.DATA_DIR / "sentiment/sentiment_diff"
output_dir.mkdir(parents=True, exist_ok=True)

# 按用户处理 / ユーザーごとに処理
for user_id, user_df in df.groupby("userId"):
    user_rows = []        # 该用户的全部对话差分 / このユーザーの全対話差分
    stats_results = []    # 该用户的统计量 / このユーザーの統計量

    # 计算每次对话的差分 / 各対話ごとの差分を計算
    for _, row in user_df.iterrows():
        record = {
            "session_id": row["session_id"],
            "userId": row["userId"],
            "replyType": row["replyType"],
            "persona": row["persona"],
            "userInput": row["userInput"],
            "replyText": row["replyText"],
        }
        for i_e in emotions:
            for r_e in emotions:
                col_name = f"diff_{i_e}_{r_e}"
                record[col_name] = row[f"input_{i_e}"] - row[f"reply_{r_e}"]
        user_rows.append(record)

    # 计算统计量（8×8矩阵）/ 統計量を計算（8×8マトリクス）
    for stat_name, stat_func in stat_funcs.items():
        for i_e in emotions:
            for r_e in emotions:
                diff_values = user_df[f"input_{i_e}"].values - user_df[f"reply_{r_e}"].values
                val = stat_func(diff_values)
                stats_results.append({
                    "stat": stat_name,
                    "input_emotion": i_e,
                    "reply_emotion": r_e,
                    "value": val,
                })

    # 保存对话差分CSV / 対話差分CSVを保存
    if user_rows:
        diff_df = pd.DataFrame(user_rows)
        diff_path = output_dir / "differences" / f"{user_id}_differences.csv"
        diff_df.to_csv(diff_path, index=False)
        print(f"差分CSV已保存 / 差分CSV保存: {diff_path} ({len(diff_df)}行)")

    # 保存统计量CSV / 統計量CSVを保存
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        stats_path = output_dir / "statistics" / f"{user_id}_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"统计量CSV已保存 / 統計量CSV保存: {stats_path} ({len(stats_df)}行)")

print(f"\n全部用户CSV处理完成，输出目录: {output_dir} / 全ユーザーのCSV処理が完了しました。出力先: {output_dir}")

# 全session一括出力 / 全session统一输出
print(f"\n全session一括出力を生成中... / 正在生成全session统一输出...")

all_rows = []
for _, row in df.iterrows():
    record = {
        "session_id": row["session_id"],
        "userId": row["userId"],
        "replyType": row["replyType"],
        "persona": row["persona"],
        "userInput": row["userInput"],
        "replyText": row["replyText"],
    }
    for emo in emotions:
        record[f"diff_{emo}"] = row[f"reply_{emo}"] - row[f"input_{emo}"]
    all_rows.append(record)

all_df = pd.DataFrame(all_rows)
all_path = config.DATA_DIR / "sentiment" / "sentiment_all_diff.csv"
all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
print(f"全session差分CSV已保存 / 全session差分CSV保存: {all_path} ({len(all_df)}行)")

# === 全session差分可视化 / 全session差分の可視化 ===
print("\n全session差分可视化を生成中... / 正在生成全session差分可视化...")

# 计算8个情感维度的统计量 / 8つの感情次元の統計量を計算
all_stats_results = []
for stat_name, stat_func in stat_funcs.items():
    for emo in emotions:
        diff_col = f"diff_{emo}"
        if diff_col in all_df.columns:
            diff_values = all_df[diff_col].values
            val = stat_func(diff_values)
            all_stats_results.append({
                "stat": stat_name,
                "emotion": emo,
                "value": val,
            })

# 转换为DataFrame / DataFrameに変換
all_stats_df = pd.DataFrame(all_stats_results)

# 保存统计量CSV / 統計量CSVを保存
all_stats_path = config.DATA_DIR / "sentiment" / "sentiment_all_diff_statistics.csv"
all_stats_df.to_csv(all_stats_path, index=False, encoding="utf-8-sig")
print(f"全session统计量CSV已保存 / 全session統計量CSV保存: {all_stats_path} ({len(all_stats_df)}行)")

# 可视化 / 可視化
if not all_stats_df.empty:
    # 获取值范围 / 値の範囲を取得
    all_values = all_stats_df["value"].values
    vmin = all_values.min()
    vmax = all_values.max()
    abs_max = max(abs(vmin), abs(vmax))

    # 创建2行3列的子图 / 2行3列のサブプロットを作成
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{stat}" for stat in vis_stat_order],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    for idx, stat_name in enumerate(vis_stat_order):
        row_idx = idx // 3 + 1
        col_idx = idx % 3 + 1

        stat_data = all_stats_df[all_stats_df["stat"] == stat_name]

        # 构建1×8的矩阵 / 1×8のマトリクスを構築
        values = []
        for emo in emotions:
            val = stat_data[stat_data["emotion"] == emo]["value"].values
            if len(val) > 0:
                values.append(val[0])
            else:
                values.append(0)

        # 创建条形图 / 棒グラフを作成
        fig.add_trace(
            go.Bar(
                x=emotions,
                y=values,
                text=[f"{v:.2f}" for v in values],
                textposition="auto",
                textfont=dict(size=12),
                marker=dict(
                    colorscale="RdBu_r",
                    cmin=-abs_max,
                    cmax=abs_max,
                    color=values,
                    showscale=(idx == 0),  # 仅第一个子图显示颜色条 / 最初のサブプロットのみカラーバー表示
                    colorbar=dict(
                        title="Value",
                        thickness=15,
                        len=0.3,
                        y=0.5 if idx == 0 else None,
                    ),
                ),
            ),
            row=row_idx,
            col=col_idx
        )

    # 布局设置 / レイアウト設定
    fig.update_layout(
        title=dict(
            text="All Sessions Sentiment Difference Statistics (Reply - Input)",
            font=dict(size=20),
            x=0.5,
            xanchor="center"
        ),
        height=800,
        width=1400,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=80, r=80, t=120, b=80)
    )

    # 设置各子图的轴标签（仅边缘显示标题）/ 軸ラベル設定（端のみ表示）
    for i in range(1, 7):
        row_idx = (i - 1) // 3 + 1
        col_idx = (i - 1) % 3 + 1
        is_bottom = (row_idx == 2)
        is_left = (col_idx == 1)
        fig.update_xaxes(
            title_text="Emotion" if is_bottom else "",
            tickangle=45,
            tickfont=dict(size=12),
            title_font=dict(size=14),
            row=row_idx, col=col_idx
        )
        fig.update_yaxes(
            title_text="Difference Value" if is_left else "",
            tickfont=dict(size=12),
            title_font=dict(size=14),
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
            matches="y" if not is_left else None,
            row=row_idx, col=col_idx
        )

    # 调整子图标题的字体大小 / サブプロットタイトルのフォントサイズ調整
    for annotation in fig.layout.annotations:
        annotation.font.size = 16

    # 保存为HTML+SVG / HTML+SVGとして保存
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    base = vis_dir / "all_sessions_statistics"
    export_fig(fig, base)
    print(f"全session可视化已保存 / 全session可視化保存: {base}.html")
