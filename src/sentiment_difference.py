"""
sentiment.csv の感情差分分析（8×8マトリクス）/ sentiment.csv 的情感差分分析（8×8矩阵）
- 按用户输出CSV / ユーザーごとにCSVを出力
- 保存每次对话的差分 / 各対話の差分を保存
- 按用户计算并保存统计量（max, min, median, mean, Q1, Q3）/ ユーザーごとに統計量（max, min, median, mean, Q1, Q3）を計算・保存
- 用Plotly可视化各统计量的8×8矩阵（带颜色热力图，HTML输出）/ 各統計量の8×8マトリクスをPlotlyで可視化（色付きヒートマップ、HTML出力）
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

# 数据加载 / データ読み込み
df = pd.read_csv(config.DATA_DIR / "sentiment/sentiment.csv")

# 8个情感维度 / 8つの感情次元
emotions = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]

input_emos = [f"input_{e}" for e in emotions]
reply_emos = [f"reply_{e}" for e in emotions]

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
vis_stat_order = ["max", "min", "Q1(25%)", "mean", "median", "Q3(75%)"]

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

    # Plotly可视化 / Plotly可視化
    if stats_results:
        stats_df = pd.DataFrame(stats_results)

        # 获取全部统计量的值范围（用于统一颜色映射）/ 全統計量の値の範囲を取得（カラーマップの統一用）
        all_values = stats_df["value"].values
        vmin = all_values.min()
        vmax = all_values.max()
        abs_max = max(abs(vmin), abs(vmax))

        # 3行2列的子图（留有余地的布局）/ 3行2列のサブプロット（余裕のあるレイアウト）
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[f"{stat}" for stat in vis_stat_order],
            vertical_spacing=0.10,
            horizontal_spacing=0.12
        )

        for idx, stat_name in enumerate(vis_stat_order):
            row_idx = idx // 2 + 1
            col_idx = idx % 2 + 1

            stat_data = stats_df[stats_df["stat"] == stat_name]

            # 构建8×8矩阵 / 8×8マトリクスを構築
            matrix = np.zeros((len(emotions), len(emotions)))
            for i, i_e in enumerate(emotions):
                for j, r_e in enumerate(emotions):
                    val = stat_data[
                        (stat_data["input_emotion"] == i_e) &
                        (stat_data["reply_emotion"] == r_e)
                        ]["value"].values
                    if len(val) > 0:
                        matrix[i, j] = val[0]

            # 用于文本标签的数组 / テキストラベル用の配列
            text_labels = [[f"{matrix[i, j]:.2f}" for j in range(len(emotions))] for i in range(len(emotions))]

            # 添加热力图 / ヒートマップ追加
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=emotions,
                    y=emotions,
                    text=text_labels,
                    texttemplate="%{text}",
                    textfont={"size": 11},
                    colorscale="RdBu_r",
                    zmid=0,
                    zmin=-abs_max,
                    zmax=abs_max,
                    colorbar=dict(
                        title="Value",
                        thickness=15,
                        len=0.3,
                        y=0.5 if idx == 0 else None,
                    ),
                    showscale=(idx == 0),  # 仅第一个子图显示颜色条 / 最初のサブプロットのみカラーバー表示
                ),
                row=row_idx,
                col=col_idx
            )

        # 布局设置 / レイアウト設定
        fig.update_layout(
            title=dict(
                text=f"User: {user_id[:30]}...<br>Sentiment Difference Statistics (Input - Reply)",
                font=dict(size=20),
                x=0.5,
                xanchor="center"
            ),
            height=1800,
            width=1200,
            template="plotly_white",
            margin=dict(l=100, r=80, t=120, b=80)
        )

        # 设置各子图的轴标签 / 各サブプロットの軸ラベルを設定
        for i in range(1, 7):
            row_idx = (i - 1) // 2 + 1
            col_idx = (i - 1) % 2 + 1
            fig.update_xaxes(
                title_text="Reply Emotion",
                tickangle=45,
                tickfont=dict(size=11),
                title_font=dict(size=13),
                row=row_idx, col=col_idx
            )
            fig.update_yaxes(
                title_text="Input Emotion",
                tickfont=dict(size=11),
                title_font=dict(size=13),
                row=row_idx, col=col_idx
            )

        # 调整子图标题的字体大小 / サブプロットタイトルのフォントサイズ調整
        for annotation in fig.layout.annotations:
            annotation.font.size = 16

        # 保存为HTML文件 / HTMLファイルとして保存
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        html_path = vis_dir / f"{user_id}_statistics.html"
        fig.write_html(str(html_path))
        print(f"可视化已保存 / 可視化保存: {html_path}")

print(f"\n全部用户处理完成，输出目录: {output_dir} / 全ユーザーの処理が完了しました。出力先: {output_dir}")

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
