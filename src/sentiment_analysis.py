"""
Sentiment Analysis using neuralnaut/deberta-wrime-emotions (WRIME微調DeBERTa)
- 8种情感 / 8つの感情: joy, sadness, anticipation, surprise, anger, fear, disgust, trust
- 使用 data_with_id.csv / data_with_id.csv を使用
- 按 (userId, userInput) 分组为对话 / (userId, userInput) でグループ化して会話に分ける
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import config
import warnings
warnings.filterwarnings("ignore")

# 输出目录 / 出力ディレクトリ
SENTIMENT_DIR = config.DATA_DIR / "sentiment"
SENTIMENT_DIR.mkdir(exist_ok=True)

# 8种情感标签 / 8つの感情ラベル
EMOTION_LABELS = {
    'joy': '喜び',
    'sadness': '悲しみ',
    'anticipation': '期待',
    'surprise': '驚き',
    'anger': '怒り',
    'fear': '恐れ',
    'disgust': '嫌悪',
    'trust': '信頼'
}

# 每种情感的颜色 / 各感情の色
EMOTION_COLORS = {
    'joy': '#FFD700',          # 金色 / ゴールド
    'sadness': '#1E90FF',      # 蓝色 / ブルー
    'anticipation': '#FF8C00', # 橙色 / オレンジ
    'surprise': '#FF69B4',     # 粉红 / ピンク
    'anger': '#DC143C',        # 红色 / レッド
    'fear': '#8B008B',         # 紫色 / パープル
    'disgust': '#2E8B57',      # 绿色 / グリーン
    'trust': '#4169E1'         # 蓝色 / ブルー
}


# 设备选择 & 加载情感分析模型 / デバイス選択 & 感情分析モデルのロード
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    print(f"使用CUDA设备: {device_name} / CUDAデバイス: {device_name}")
else:
    device = torch.device("cpu")
    device_name = "CPU"
    print(f"CUDA不可用，使用CPU / CUDA利用不可、CPUを使用")

LOCAL_MODEL_PATH = config.MODELS_DIR / 'deberta-wrime-emotions'
MODEL_NAME = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else 'neuralnaut/deberta-wrime-emotions'
print(f"正在加载模型 {MODEL_NAME} ... / モデル {MODEL_NAME} を読み込み中...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)  # 明确将模型移动到GPU / モデルを明示的にGPUに移動
model.eval()  # 设置为评估模式 / 評価モードに設定
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tok,
    top_k=None,
    device=device
)
print("模型加载完成。 / モデルの読み込み完了。")
print(f"情感标签: {model.config.id2label} / 感情ラベル: {model.config.id2label}")


# 批量情感分析（返回8种情感分数）/ バッチ感情分析（8つの感情スコアを返す）
def analyze_emotions_batch(texts, batch_size=32):
    """
    批量进行8种情感分析
    8つの感情をバッチで分析する
    Returns: list of dict, each dict maps emotion_name -> score
    """
    results = [{}] * len(texts)
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if pd.isna(text) or str(text).strip() == "":
            continue
        valid_indices.append(i)
        valid_texts.append(str(text)[:512])

    if not valid_texts:
        return results

    try:
        for start in range(0, len(valid_texts), batch_size):
            end = min(start + batch_size, len(valid_texts))
            batch_texts = valid_texts[start:end]
            batch_indices = valid_indices[start:end]

            batch_results = sentiment_pipeline(batch_texts, batch_size=batch_size)

            for idx, res in zip(batch_indices, batch_results):
                scores = {item['label']: item['score'] for item in res}
                results[idx] = scores
    except Exception as e:
        print(f"  批量分析错误: {e}")

    return results


# 加载数据并进行情感分析 / データをロードして感情分析を実行
def process_dataset(filepath):
    print(f"处理数据集: {filepath} / データセット処理中: {filepath}")

    df = pd.read_csv(filepath)
    print(f"数据行数: {len(df)} / データ行数: {len(df)}")
    print(f"用户数: {df['userId'].nunique()} / ユーザー数: {df['userId'].nunique()}")
    print(f"replyType 分布: / replyType 分布:")
    print(df['replyType'].value_counts())
    print(f"persona 分布: / persona 分布:")
    print(df['persona'].value_counts())

    # 分析 userInput 情感 / userInput の感情を分析
    print("\n正在分析 userInput 的情感... / userInput の感情を分析中...")
    input_emotions = analyze_emotions_batch(df['userInput'].tolist())
    for emotion in EMOTION_LABELS.keys():
        df[f'input_{emotion}'] = [d.get(emotion, np.nan) for d in input_emotions]

    # 分析 replyText 情感 / replyText の感情を分析
    print("正在分析 replyText の感情を分析中... / replyText の感情を分析中...")
    reply_emotions = analyze_emotions_batch(df['replyText'].tolist())
    for emotion in EMOTION_LABELS.keys():
        df[f'reply_{emotion}'] = [d.get(emotion, np.nan) for d in reply_emotions]

    print(f"\n情感分析完成。数据列: {[c for c in df.columns if c.startswith('input_') or c.startswith('reply_')]} / 感情分析完了。データ列: {[c for c in df.columns if c.startswith('input_') or c.startswith('reply_')]}")
    return df


# 构建对话数据结构 / 会話データ構造を構築
def build_conversations(df):
    # 按 (userId, userInput) 分组 / (userId, userInput) でグループ化
    conversations = []
    for (uid, user_input), group in df.groupby(['userId', 'userInput']):
        group = group.sort_values('session_id')

        conv = {
            'userId': uid,
            'userInput': user_input,
            'session_ids': group['session_id'].tolist(),
            'input': user_input,
        }

        # Input 情感分数（取第一条，同一输入相同）/ Input 感情スコア（最初の行を取得、同一入力は同じ）
        first = group.iloc[0]
        for emotion in EMOTION_LABELS.keys():
            conv[f'input_{emotion}'] = first.get(f'input_{emotion}', np.nan)

        # ReplyCurrentPersona（当前角色回复）
        # ReplyCurrentPersona（現在のペルソナへの返信）
        rcp = group[group['replyType'] == 'ReplyCurrentPersona']
        for emotion in EMOTION_LABELS.keys():
            if len(rcp) > 0:
                conv[f'rcp_{emotion}'] = rcp.iloc[0].get(f'reply_{emotion}', np.nan)
            else:
                conv[f'rcp_{emotion}'] = np.nan
        conv['rcp_persona'] = rcp.iloc[0]['persona'] if len(rcp) > 0 else None
        conv['current'] = rcp.iloc[0]['replyText'] if len(rcp) > 0 else None

        # ReplyInterruptPersona（打断角色回复）
        # ReplyInterruptPersona（割り込みペルソナへの返信）
        rip = group[group['replyType'] == 'ReplyInterruptPersona']
        for emotion in EMOTION_LABELS.keys():
            if len(rip) > 0:
                conv[f'rip_{emotion}'] = rip.iloc[0].get(f'reply_{emotion}', np.nan)
            else:
                conv[f'rip_{emotion}'] = np.nan
        conv['rip_persona'] = rip.iloc[0]['persona'] if len(rip) > 0 else None
        conv['interrupt'] = rip.iloc[0]['replyText'] if len(rip) > 0 else None

        conversations.append(conv)

    conv_df = pd.DataFrame(conversations)

    # 为每组对话添加序号 Cxxx / 各会話に番号 Cxxx を付与
    conv_df.insert(0, 'conv_id', [f'C{i+1:03d}' for i in range(len(conv_df))])

    print(f"\n对话数: {len(conv_df)}（按 userId+userInput 分组）/ 会話数: {len(conv_df)}（userId+userInput でグループ化）")
    return conv_df



# 用户统计 / ユーザー統計
def compute_user_statistics(conv_df):
    print(f"\n用户统计 / ユーザー統計")
    user_stats = []
    for uid, group in conv_df.groupby('userId'):
        stats = {
            'userId': uid,
            'conversations': len(group),
        }
        # 各情感的平均分数 / 各感情の平均スコア
        for emotion in EMOTION_LABELS.keys():
            rcp_scores = group[f'rcp_{emotion}'].dropna()
            rip_scores = group[f'rip_{emotion}'].dropna()
            stats[f'rcp_{emotion}_mean'] = rcp_scores.mean() if len(rcp_scores) > 0 else np.nan
            stats[f'rip_{emotion}_mean'] = rip_scores.mean() if len(rip_scores) > 0 else np.nan
        user_stats.append(stats)

    stats_df = pd.DataFrame(user_stats)
    print(stats_df.to_string(index=False))
    return stats_df


# 为每个用户生成图表 / 各ユーザーのグラフを生成
def create_charts(conv_df, output_dirname='charts_wrime'):
    """
    每个用户两张子图：
    上图: ReplyCurrentPersona 8种情感折线 + ReplyInterruptPersona 8种情感柱状（标明persona）
    下图: Input 8种情感折線

    各ユーザーに2つのサブプロット：
    上: ReplyCurrentPersona 8感情の折れ線 + ReplyInterruptPersona 8感情の棒グラフ（ペルソナ表示）
    下: Input 8感情の折れ線
    """
    print(f"\n正在生成图表... / グラフを生成中...")

    charts_dir = SENTIMENT_DIR / output_dirname
    charts_dir.mkdir(exist_ok=True)

    # persona 颜色用于柱状 / personaの色は棒グラフに使用
    users = conv_df['userId'].unique()
    generated_files = []

    for uid in users:
        user_conv = conv_df[conv_df['userId'] == uid].sort_values(
            ['session_ids'],
            key=lambda x: x.apply(lambda s: s[0] if isinstance(s, list) else s)
        ).reset_index(drop=True)

        short_id = uid  # 使用完整原始userId / 完元のuserIdを使用
        n = len(user_conv)
        x_positions = list(range(1, n + 1))

        persona_symbols = {
            'mochiko': 'circle',
            'muchiko': 'square',
            'pen_sensei': 'diamond',
        }

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                '<br>ReplyCurrentPersona',
                '<br>User Input:',
                '<br>ReplyInterruptPersona'
            ),
            vertical_spacing=0.10,
            row_heights=[0.35, 0.35, 0.30]
        )



        # RCP 8条情感折线，marker与折线关联（lines+markers模式）
        # 通过为每个点设置不同的marker symbol来区分persona
        # RCP 8感情の折れ線、マーカーと折れ線を関連付け（lines+markersモード）
        # 各ポイントに異なるマーカーシンボルを設定してペルソナを区別
        persona_list_for_markers = user_conv['rcp_persona'].tolist()
        for emotion in EMOTION_LABELS.keys():
            rcp_scores = user_conv[f'rcp_{emotion}'].tolist()
            # 根据persona为每个点设置对应的marker symbol
            # ペルソナに応じて各ポイントのマーカーシンボルを設定
            symbols = []
            for p in persona_list_for_markers:
                if p in persona_symbols:
                    symbols.append(persona_symbols[p])
                else:
                    symbols.append('circle')  # 默认圆形 / デフォルトは円形
            fig.add_trace(
                go.Scatter(
                    x=x_positions, y=rcp_scores,
                    mode='lines+markers',
                    name=f'{EMOTION_LABELS[emotion]}',
                    line=dict(color=EMOTION_COLORS[emotion], width=2),
                    marker=dict(
                        size=10,
                        symbol=symbols,
                        color=EMOTION_COLORS[emotion],
                        line=dict(width=1, color='white')
                    ),
                    legendgroup=f'rcp_{emotion}',
                    showlegend=True,
                    hovertemplate=[
                        f'{persona_list_for_markers[i]} | {EMOTION_LABELS[emotion]}: %{{y:.4f}}<extra></extra>'
                        for i in range(len(rcp_scores))
                    ]
                ),
                row=1, col=1
            )


        annotations = []

        # RIP 柱状（独立第二子图，按emotion颜色上色）
        # RIP 棒グラフ（独立したサブプロット、感情の色で着色）
        rip_positions = []
        for j, (_, row) in enumerate(user_conv.iterrows()):
            if pd.notna(row.get(f'rip_joy', np.nan)):
                rip_positions.append(j)

        if rip_positions:
            n_emotions = len(EMOTION_LABELS)
            bar_width = 0.6 / n_emotions
            emotion_list = list(EMOTION_LABELS.keys())

            for ei, emotion in enumerate(emotion_list):
                rip_x_list = []
                rip_y_list = []
                rip_persona_list = []

                for j in rip_positions:
                    row = user_conv.iloc[j]
                    score = row.get(f'rip_{emotion}', np.nan)
                    if pd.notna(score):
                        offset = (ei - n_emotions / 2 + 0.5) * bar_width
                        rip_x_list.append(x_positions[j] + offset)
                        rip_y_list.append(score)
                        rip_persona_list.append(row.get('rip_persona', '?'))

                if rip_x_list:
                    fig.add_trace(
                        go.Bar(
                            x=rip_x_list,
                            y=rip_y_list,
                            marker_color=EMOTION_COLORS[emotion],
                            opacity=0.75,
                            name=f'{EMOTION_LABELS[emotion]} (Interrupt)',
                            legendgroup=f'rip_{emotion}',
                            showlegend=True,
                            textposition='inside',
                            textfont=dict(size=8, color='white'),
                            width=bar_width * 0.9,
                            hovertemplate=[
                                f'{EMOTION_LABELS[emotion]} ({p})<br>Score: %{{y:.4f}}<extra></extra>'
                                for p in rip_persona_list
                            ]
                        ),
                        row=3, col=1
                    )


        # 中间图：Input 折线 / 中央: Input 折れ線

        for emotion in EMOTION_LABELS.keys():
            input_scores = user_conv[f'input_{emotion}'].tolist()
            fig.add_trace(
                go.Scatter(
                    x=x_positions, y=input_scores,
                    mode='lines+markers',
                    name=f'{EMOTION_LABELS[emotion]} (Input)',
                    line=dict(color=EMOTION_COLORS[emotion], width=2),
                    marker=dict(size=6),
                    legendgroup=f'input_{emotion}',
                    hovertemplate=f'Input {EMOTION_LABELS[emotion]}: %{{y:.4f}}<extra></extra>'
                ),
                row=2, col=1
            )

        # userInput 底部标注 / userInput の底部アノテーション
        for j, (_, row) in enumerate(user_conv.iterrows()):
            input_text = str(row['userInput'])[:25]
            if len(str(row['userInput'])) > 25:
                input_text += '...'
            annotations.append(dict(
                x=x_positions[j], y=-0.08,
                xref='x', yref='y2',
                text=input_text,
                showarrow=False,
                font=dict(size=7, color='#888'),
                textangle=-45,
                xanchor='right'
            ))

        # 布局 / レイアウト
        y_range = [-0.05, 1.05]

        fig.update_layout(
            title=dict(
                text=f'User {short_id}<br>',
                x=0.5, font=dict(size=15)
            ),
            height=1400,
            width=max(1200, n * 100 + 300),
            template='plotly_white',
            barmode='group',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.12,
                xanchor='center',
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(l=80, r=80, t=200, b=250),
            annotations=annotations,
        )

        # Row 1 (RCP折线): x轴刻度旁显示rcp_persona
        # Row 1（RCP折れ線）: x軸の目盛り横にrcp_personaを表示
        rcp_ticktext = []
        for j in range(n):
            persona = user_conv.iloc[j].get('rcp_persona', '')
            if persona:
                rcp_ticktext.append(f"{j+1}<br><span style='font-size:8px;color:#666'>{persona}</span>")
            else:
                rcp_ticktext.append(str(j+1))
        fig.update_xaxes(title_text='对话序号/やり取り番号', row=1, col=1, tickvals=x_positions, ticktext=rcp_ticktext, range=[0.3, n + 0.7])
        fig.update_yaxes(title_text='情感分数/点数', row=1, col=1, range=y_range, showgrid=True, gridcolor='#f0f0f0')

        # Row 2 (Input折线): x轴普通刻度
        # Row 2（Input折れ線）: x軸は通常の目盛り
        fig.update_xaxes(title_text='对话序号/やり取り番号', row=2, col=1, tickvals=x_positions, range=[0.3, n + 0.7])
        fig.update_yaxes(title_text='情感分数/点数', row=2, col=1, range=y_range, showgrid=True, gridcolor='#f0f0f0')

        # Row 3 (RIP柱状): x轴刻度旁显示rip_persona
        # Row 3（RIP棒グラフ）: x軸の目盛り横にrip_personaを表示
        rip_ticktext = []
        for j in range(n):
            persona = user_conv.iloc[j].get('rip_persona', '')
            if pd.notna(persona) and persona:
                rip_ticktext.append(f"{j+1}<br><span style='font-size:8px;color:#666'>{persona}</span>")
            else:
                rip_ticktext.append(str(j+1))
        fig.update_xaxes(title_text='对话序号/やり取り番号', row=3, col=1, tickvals=x_positions, ticktext=rip_ticktext, range=[0.3, n + 0.7])
        fig.update_yaxes(title_text='情感分数/点数', row=3, col=1, range=y_range, showgrid=True, gridcolor='#f0f0f0')

        output_path = charts_dir / f'user_{short_id}.html'
        fig.write_html(output_path)
        generated_files.append(output_path)
        print(f"  {output_path.name}")

    print(f"\n共生成 {len(generated_files)} 个图表，保存至: {charts_dir} / {len(generated_files)} 個のグラフを生成、保存先: {charts_dir}")
    return generated_files




if __name__ == "__main__":
    import time
    start_time = time.time()

    # 加载并分析 / ロードして分析
    df = process_dataset(config.DATA_DIR / 'data_with_id.csv')

    # 构建对话结构 / 会話構造を構築
    conv_df = build_conversations(df)

    # 用户统计 / ユーザー統計
    stats = compute_user_statistics(conv_df)

    # 保存 / 保存
    output_csv = SENTIMENT_DIR / 'sentiment.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n情感明细已保存至: {output_csv} / 感情詳細を保存しました: {output_csv}")

    conv_csv = SENTIMENT_DIR / 'conversations.csv'
    conv_df.to_csv(conv_csv, index=False, encoding='utf-8-sig')
    print(f"对话汇总已保存至: {conv_csv} / 会話サマリーを保存しました: {conv_csv}")

    stats_csv = SENTIMENT_DIR / 'sentiment_stats.csv'
    stats.to_csv(stats_csv, index=False, encoding='utf-8-sig')
    print(f"用户统计已保存至: {stats_csv} / ユーザー統計を保存しました: {stats_csv}")

    # 生成图表 / グラフを生成
    create_charts(conv_df, 'charts_wrime')

    elapsed = time.time() - start_time
    print(f"全部处理完成！总耗时: {elapsed:.1f}秒 / 全ての処理完了！総経過時間: {elapsed:.1f}秒")
    print(f"输出文件 ({SENTIMENT_DIR}): / 出力ファイル ({SENTIMENT_DIR}):")
    print(f"  - sentiment_wrime.csv        (情感分析明细 / 感情分析詳細)")
    print(f"  - conversations_wrime.csv    (对话汇总 / 会話サマリー)")
    print(f"  - sentiment_stats_wrime.csv  (用户统计 / ユーザー統計)")
    print(f"  - charts_wrime/              (每个用户独立图表 / 各ユーザーの個別グラフ)")
