"""
感情分析モジュール（neuralnaut/deberta-wrime-emotions）
- 8感情: joy, sadness, anticipation, surprise, anger, fear, disgust, trust
- data_with_id.csv を使用
- (userId, userInput) でグループ化して会話に分ける
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

# 統一サイズ定義
FIG_W, FIG_H = 1200, 700
FIG_WIDE_W, FIG_WIDE_H = 1400, 700


def export_fig(fig, base_path):
    """plotly の fig を HTML + SVG の2形式で出力"""
    fig.write_html(str(base_path) + '.html')
    fig.write_image(str(base_path) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)


# 出力ディレクトリ
SENTIMENT_DIR = config.DATA_DIR / "sentiment"
SENTIMENT_DIR.mkdir(exist_ok=True)

# 8感情ラベル
EMOTION_LABELS = {'joy': '喜び', 'sadness': '悲しみ', 'anticipation': '期待', 'surprise': '驚き', 'anger': '怒り', 'fear': '恐れ', 'disgust': '嫌悪', 'trust': '信頼'}

# 各感情の色
EMOTION_COLORS = {'joy': '#FFD700', 'sadness': '#1E90FF', 'anticipation': '#FF8C00', 'surprise': '#FF69B4', 'anger': '#DC143C', 'fear': '#8B008B', 'disgust': '#2E8B57', 'trust': '#4169E1'}

# デバイス選択
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"CUDA使用: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CPUモード")

# 感情分析モデルのロード
LOCAL_MODEL_PATH = config.MODELS_DIR / 'deberta-wrime-emotions'
MODEL_NAME = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else 'neuralnaut/deberta-wrime-emotions'
print(f"モデル読込: {MODEL_NAME}...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tok, top_k=None, device=device)
print("モデル読込完了")


def analyze_emotions_batch(texts, batch_size=config.SENTIMENT_BATCH_SIZE):
    """バッチで8感情を分析"""
    results = [{}] * len(texts)
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if pd.isna(text) or str(text).strip() == "":
            continue
        valid_indices.append(i)
        valid_texts.append(str(text))

    if not valid_texts:
        return results

    try:
        for start in range(0, len(valid_texts), batch_size):
            end = min(start + batch_size, len(valid_texts))
            batch_results = sentiment_pipeline(valid_texts[start:end], batch_size=batch_size, truncation=True, max_length=config.SENTIMENT_MAX_LENGTH)
            for idx, res in zip(valid_indices[start:end], batch_results):
                results[idx] = {item['label']: item['score'] for item in res}
    except Exception as e:
        print(f"  バッチ分析エラー: {e}")

    return results


def process_dataset(filepath):
    """データセットを読み込み感情分析を実行"""
    print(f"データセット処理: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  行数: {len(df)}, ユーザー数: {df['userId'].nunique()}")

    print("userInput の感情分析中...")
    input_emotions = analyze_emotions_batch(df['userInput'].tolist())
    for emotion in EMOTION_LABELS.keys():
        df[f'input_{emotion}'] = [d.get(emotion, np.nan) for d in input_emotions]

    print("replyText の感情分析中...")
    reply_emotions = analyze_emotions_batch(df['replyText'].tolist())
    for emotion in EMOTION_LABELS.keys():
        df[f'reply_{emotion}'] = [d.get(emotion, np.nan) for d in reply_emotions]

    return df


def build_conversations(df):
    """(userId, userInput) でグループ化して会話構造を構築"""
    conversations = []
    for (uid, user_input), group in df.groupby(['userId', 'userInput']):
        group = group.sort_values('session_id')
        conv = {'userId': uid, 'userInput': user_input, 'session_ids': group['session_id'].tolist()}

        # Input感情スコア（最初の行を取得）
        first = group.iloc[0]
        for emotion in EMOTION_LABELS.keys():
            conv[f'input_{emotion}'] = first.get(f'input_{emotion}', np.nan)

        # Reply感情スコア（最後の行を使用）
        reply_data = group.iloc[-1]
        for emotion in EMOTION_LABELS.keys():
            conv[f'reply_{emotion}'] = reply_data.get(f'reply_{emotion}', np.nan)
        conv['reply_persona'] = reply_data['persona']
        conv['replyText'] = reply_data['replyText']
        conv['replyType'] = reply_data['replyType']

        conversations.append(conv)

    conv_df = pd.DataFrame(conversations)
    conv_df.insert(0, 'conv_id', [f'C{i+1:03d}' for i in range(len(conv_df))])
    print(f"\n会話数: {len(conv_df)}")
    return conv_df


def compute_user_statistics(conv_df):
    """ユーザーごとの統計量を計算"""
    user_stats = []
    for uid, group in conv_df.groupby('userId'):
        stats = {'userId': uid, 'conversations': len(group)}
        for emotion in EMOTION_LABELS.keys():
            reply_scores = group[f'reply_{emotion}'].dropna()
            stats[f'reply_{emotion}_mean'] = reply_scores.mean() if len(reply_scores) > 0 else np.nan
        user_stats.append(stats)
    return pd.DataFrame(user_stats)


def create_charts(conv_df, output_dirname='charts_wrime'):
    """ユーザーごとに8感情の折れ線グラフを生成"""
    charts_dir = SENTIMENT_DIR / output_dirname
    charts_dir.mkdir(exist_ok=True)

    # ペルソナごとのマーカーシンボル
    persona_symbols = {'mochiko': 'circle', 'muchiko': 'square', 'pen_sensei': 'diamond'}
    users = conv_df['userId'].unique()
    generated_files = []

    for uid in users:
        user_conv = conv_df[conv_df['userId'] == uid].sort_values(
            ['session_ids'], key=lambda x: x.apply(lambda s: s[0] if isinstance(s, list) else s)
        ).reset_index(drop=True)

        n = len(user_conv)
        x_positions = list(range(1, n + 1))
        persona_list = user_conv['reply_persona'].tolist()

        # ペルソナ名をチャットボット名にマッピング
        persona_display_map = {'mochiko': 'chatbot_mo', 'muchiko': 'chatbot_mu', 'pen_sensei': 'chatbot_p'}

        # 2行1列のサブプロット（上: Reply, 下: Input）
        fig = make_subplots(rows=2, cols=1, subplot_titles=('<br>Reply', '<br>User Input:'), vertical_spacing=0.12, row_heights=[0.5, 0.5])

        # Reply 8感情の折れ線
        for emotion in EMOTION_LABELS.keys():
            reply_scores = user_conv[f'reply_{emotion}'].tolist()
            symbols = [persona_symbols.get(p, 'circle') for p in persona_list]
            fig.add_trace(go.Scatter(
                x=x_positions, y=reply_scores, mode='lines+markers',
                name=EMOTION_LABELS[emotion], line=dict(color=EMOTION_COLORS[emotion], width=2),
                marker=dict(size=10, symbol=symbols, color=EMOTION_COLORS[emotion], line=dict(width=1, color='white')),
                legendgroup=f'reply_{emotion}', showlegend=True,
                hovertemplate=[f'{persona_display_map.get(persona_list[i], persona_list[i])} | {EMOTION_LABELS[emotion]}: %{{y:.4f}}<extra></extra>' for i in range(len(reply_scores))]
            ), row=1, col=1)

        # Input 8感情の折れ線
        for emotion in EMOTION_LABELS.keys():
            input_scores = user_conv[f'input_{emotion}'].tolist()
            fig.add_trace(go.Scatter(
                x=x_positions, y=input_scores, mode='lines+markers',
                name=f'{EMOTION_LABELS[emotion]} (Input)', line=dict(color=EMOTION_COLORS[emotion], width=2),
                marker=dict(size=6), legendgroup=f'input_{emotion}',
                hovertemplate=f'Input {EMOTION_LABELS[emotion]}: %{{y:.4f}}<extra></extra>'
            ), row=2, col=1)

        # userInput のアノテーション
        annotations = []
        for j, (_, row) in enumerate(user_conv.iterrows()):
            input_text = str(row['userInput'])[:25]
            if len(str(row['userInput'])) > 25:
                input_text += '...'
            annotations.append(dict(x=x_positions[j], y=-0.08, xref='x', yref='y2', text=input_text, showarrow=False, font=dict(size=7, color='#888'), textangle=-45, xanchor='right'))

        y_range = [-0.05, 1.05]
        fig.update_layout(title=dict(text=f'User {uid}<br>', x=0.5, font=dict(size=15)), height=1000, width=max(1200, n * 100 + 300), template='plotly_white', legend=dict(orientation='h', yanchor='top', y=-0.12, xanchor='center', x=0.5, font=dict(size=10)), margin=dict(l=80, r=80, t=150, b=200), annotations=annotations)

        # x軸にペルソナ名を表示
        reply_ticktext = [f"{j+1}<br><span style='font-size:8px;color:#666'>{persona_display_map.get(user_conv.iloc[j].get('reply_persona', ''), user_conv.iloc[j].get('reply_persona', ''))}</span>" if user_conv.iloc[j].get('reply_persona', '') else str(j+1) for j in range(n)]
        fig.update_xaxes(title_text='やり取り番号', row=1, col=1, tickvals=x_positions, ticktext=reply_ticktext, range=[0.3, n + 0.7])
        fig.update_yaxes(title_text='感情スコア', row=1, col=1, range=y_range, showgrid=True, gridcolor='#f0f0f0')
        fig.update_xaxes(title_text='やり取り番号', row=2, col=1, tickvals=x_positions, range=[0.3, n + 0.7])
        fig.update_yaxes(title_text='感情スコア', row=2, col=1, range=y_range, showgrid=True, gridcolor='#f0f0f0')

        output_path = charts_dir / f'user_{uid}'
        export_fig(fig, output_path)
        generated_files.append(output_path)

    print(f"\n{len(generated_files)}件のグラフを生成: {charts_dir}")
    return generated_files


if __name__ == "__main__":
    import time
    start_time = time.time()

    df = process_dataset(config.DATA_DIR / 'data_with_id.csv')
    conv_df = build_conversations(df)
    stats = compute_user_statistics(conv_df)

    df.to_csv(SENTIMENT_DIR / 'sentiment.csv', index=False, encoding='utf-8-sig')
    conv_df.to_csv(SENTIMENT_DIR / 'conversations.csv', index=False, encoding='utf-8-sig')
    stats.to_csv(SENTIMENT_DIR / 'sentiment_stats.csv', index=False, encoding='utf-8-sig')

    create_charts(conv_df, 'charts_wrime')

    print(f"\n処理完了: {time.time() - start_time:.1f}秒")
    print(f"出力先: {SENTIMENT_DIR}")
