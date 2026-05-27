"""
sentiment.csv の感情差分分析（8×8マトリクス）
- ユーザーごとにCSVを出力
- interrupt / current の返答を分離
- 各対話の差分を保存
- ユーザーごとにinterrupt/currentの統計量（max, min, median, mean, Q1, Q3）を計算・保存
- 各統計量の8×8マトリクスをPlotlyで可視化（色付きヒートマップ、HTML出力）
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

# データ読み込み
df = pd.read_csv(config.DATA_DIR / "sentiment/sentiment.csv")

# 8つの感情次元
emotions = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]

input_emos = [f"input_{e}" for e in emotions]
reply_emos = [f"reply_{e}" for e in emotions]

# 統計量の種類（meanを追加）
stat_funcs = {
    "max": lambda x: np.max(x),
    "min": lambda x: np.min(x),
    "Q1(25%)": lambda x: np.percentile(x, 25),
    "mean": lambda x: np.mean(x),
    "median": lambda x: np.median(x),
    "Q3(75%)": lambda x: np.percentile(x, 75),
}

# 可視化用の統計量順序
vis_stat_order = ["max", "min", "Q1(25%)", "mean", "median", "Q3(75%)"]

# replyType のマッピング
reply_type_map = {
    "ReplyCurrentPersona": "current",
    "ReplyInterruptPersona": "interrupt",
}

# 出力ディレクトリ
output_dir = config.DATA_DIR / "sentiment/sentiment_diff"
output_dir.mkdir(parents=True, exist_ok=True)

# ユーザーごとに処理
for user_id, user_df in df.groupby("userId"):
    user_rows = []        # このユーザーの全対話差分
    stats_results = []    # このユーザーの統計量

    for reply_type_label, reply_type_short in reply_type_map.items():
        subset = user_df[user_df["replyType"] == reply_type_label]

        if subset.empty:
            continue

        # 各対話ごとの差分を計算
        for _, row in subset.iterrows():
            record = {
                "session_id": row["session_id"],
                "userId": row["userId"],
                "replyType": reply_type_short,
                "persona": row["persona"],
                "userInput": row["userInput"],
                "replyText": row["replyText"],
            }
            for i_e in emotions:
                for r_e in emotions:
                    col_name = f"diff_{i_e}_{r_e}"
                    record[col_name] = row[f"input_{i_e}"] - row[f"reply_{r_e}"]
            user_rows.append(record)

        # 統計量を計算（8×8マトリクス）
        for stat_name, stat_func in stat_funcs.items():
            for i_e in emotions:
                for r_e in emotions:
                    diff_values = subset[f"input_{i_e}"].values - subset[f"reply_{r_e}"].values
                    val = stat_func(diff_values)
                    stats_results.append({
                        "replyType": reply_type_short,
                        "stat": stat_name,
                        "input_emotion": i_e,
                        "reply_emotion": r_e,
                        "value": val,
                    })

    # 対話差分CSVを保存
    if user_rows:
        diff_df = pd.DataFrame(user_rows)
        diff_path = output_dir / "differences" / f"{user_id}_differences.csv"
        diff_df.to_csv(diff_path, index=False)
        print(f"差分CSV保存: {diff_path} ({len(diff_df)}行)")

    # 統計量CSVを保存
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        stats_path = output_dir / "statistics" / f"{user_id}_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"統計量CSV保存: {stats_path} ({len(stats_df)}行)")

    # ========== Plotly可視化 ==========
    if stats_results:
        # replyTypeごとに可視化（currentとinterruptを別々にHTML出力）
        for reply_type_short in stats_df["replyType"].unique():
            rt_df = stats_df[stats_df["replyType"] == reply_type_short]

            # 全統計量の値の範囲を取得（カラーマップの統一用）
            all_values = rt_df["value"].values
            vmin = all_values.min()
            vmax = all_values.max()
            abs_max = max(abs(vmin), abs(vmax))

            # 3行2列のサブプロット（余裕のあるレイアウト）
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[f"{stat}" for stat in vis_stat_order],
                vertical_spacing=0.10,
                horizontal_spacing=0.12
            )

            for idx, stat_name in enumerate(vis_stat_order):
                row_idx = idx // 2 + 1
                col_idx = idx % 2 + 1

                stat_data = rt_df[rt_df["stat"] == stat_name]

                # 8×8マトリクスを構築
                matrix = np.zeros((len(emotions), len(emotions)))
                for i, i_e in enumerate(emotions):
                    for j, r_e in enumerate(emotions):
                        val = stat_data[
                            (stat_data["input_emotion"] == i_e) &
                            (stat_data["reply_emotion"] == r_e)
                            ]["value"].values
                        if len(val) > 0:
                            matrix[i, j] = val[0]

                # テキストラベル用の配列
                text_labels = [[f"{matrix[i, j]:.2f}" for j in range(len(emotions))] for i in range(len(emotions))]

                # ヒートマップ追加
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
                        showscale=(idx == 0),  # 最初のサブプロットのみカラーバー表示
                    ),
                    row=row_idx,
                    col=col_idx
                )

            # レイアウト設定
            fig.update_layout(
                title=dict(
                    text=f"User: {user_id[:30]}...<br>Reply Type: {reply_type_short} | Sentiment Difference Statistics (Input - Reply)",
                    font=dict(size=20),
                    x=0.5,
                    xanchor="center"
                ),
                height=1800,
                width=1200,
                template="plotly_white",
                margin=dict(l=100, r=80, t=120, b=80)
            )

            # 各サブプロットの軸ラベルを設定
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

            # サブプロットタイトルのフォントサイズ調整
            for annotation in fig.layout.annotations:
                annotation.font.size = 16

            # HTMLファイルとして保存（current/interrupt別フォルダ）
            vis_dir = output_dir / "visualizations" / reply_type_short
            vis_dir.mkdir(parents=True, exist_ok=True)
            html_path = vis_dir / f"{user_id}_statistics.html"
            fig.write_html(str(html_path))
            print(f"可視化保存: {html_path}")

print(f"\n全ユーザーの処理が完了しました。出力先: {output_dir}")