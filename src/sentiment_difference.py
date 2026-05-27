"""
sentiment.csv の感情差分分析（8×8マトリクス）
- ユーザーごとにCSVを出力
- interrupt / current の返答を分離
- 各対話の差分を保存
- ユーザーごとにinterrupt/currentの統計量（max, min, median, Q1, Q3）を計算・保存
"""
import pandas as pd
import numpy as np
import config

# データ読み込み
df = pd.read_csv(config.DATA_DIR / "sentiment/sentiment.csv")

# 8つの感情次元
emotions = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]

input_emos = [f"input_{e}" for e in emotions]
reply_emos = [f"reply_{e}" for e in emotions]

# 統計量の種類
stat_funcs = {
    "max": lambda x: np.max(x),
    "min": lambda x: np.min(x),
    "median": lambda x: np.median(x),
    "Q1(25%)": lambda x: np.percentile(x, 25),
    "Q3(75%)": lambda x: np.percentile(x, 75),
}

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
        diff_path = output_dir / "differences"/f"{user_id}_differences.csv"
        diff_df.to_csv(diff_path, index=False)
        print(f"差分CSV保存: {diff_path} ({len(diff_df)}行)")

    # 統計量CSVを保存
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        stats_path = output_dir /"statistics"/ f"{user_id}_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"統計量CSV保存: {stats_path} ({len(stats_df)}行)")

print(f"\n全ユーザーの処理が完了しました。出力先: {output_dir}")