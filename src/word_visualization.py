import pandas as pd
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image
import io
import base64
import config

# === 統一サイズ定義 / Unified figure sizes ===
FIG_W, FIG_H = 1200, 700
FIG_WIDE_W, FIG_WIDE_H = 1400, 700


def export_fig(fig, base_path):
    """plotly の fig を HTML + SVG の2形式で出力"""
    fig.write_html(str(base_path) + '.html')
    fig.write_image(str(base_path) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)



# 分类函数：判断单词属于哪一类
# 分類関数：単語がどのカテゴリに属するかを判定する

def get_category(row):
    if pd.isna(row['rank_m']): return 'Pen Only'
    if pd.isna(row['rank_p']): return 'Mochiko Only'
    return 'Common'


# 对两个 chatbot 的单词频率数据生成散点对比图
# 2つのchatbotの単語頻度データに対して散布図比較プロットを生成する
# 共通词（Common）用 RdYlBu 色阶表示偏向，独有词分别用蓝/红表示
# 共通語（Common）はRdYlBu色スケールで偏向を表示、独自語は青/赤で表示
def build_scatter_plot(df_mochiko, df_pen_sensei, title_suffix, output_filename):
    """
    Args:
        df_mochiko:     Mochiko の単語カウント DataFrame（word, count 列必須）
        df_pen_sensei:  Pen Sensei の単語カウント DataFrame（word, count 列必須）
        title_suffix:   图表标题后缀（如 'Nouns', 'Verbs', 'Emoji'）/ グラフタイトルのサフィックス
        output_filename: 输出 HTML 文件名 / 出力HTMLファイル名
    """

    # 计算排名百分比 (0=最高频, 1=最低频)
    # ランク百分率を計算（0=最高頻度, 1=最低頻度）
    df_m = df_mochiko.copy()
    df_p = df_pen_sensei.copy()
    df_m['rank_m'] = df_m['count'].rank(ascending=False, method='first', pct=True)
    df_p['rank_p'] = df_p['count'].rank(ascending=False, method='first', pct=True)

    # 合并数据集 / データセットを結合
    df_merged = pd.merge(
        df_m[['word', 'rank_m', 'count']],
        df_p[['word', 'rank_p', 'count']],
        on='word', how='outer', suffixes=('_m', '_p')
    )

    # 处理缺失值：将独有词映射到 Infrequent 之外的区域 (1.15)
    # 欠損値の処理：独自語をInfrequent外の領域(1.15)にマッピング
    df_merged['rank_m_filled'] = df_merged['rank_m'].fillna(1.15)
    df_merged['rank_p_filled'] = df_merged['rank_p'].fillna(1.15)

    df_merged['category'] = df_merged.apply(get_category, axis=1)

    # 核心逻辑：计算倾向性颜色 (蓝-黄-红 渐变)
    # コアロジック：傾向色を計算（青-黄-赤のグラデーション）
    df_merged['diff'] = df_merged['rank_p'] - df_merged['rank_m']
    df_merged.loc[df_merged['category'] == 'Mochiko Only', 'diff'] = 1.0
    df_merged.loc[df_merged['category'] == 'Pen Only', 'diff'] = -1.0

    # 非线性缩放（平方根）让高频区域更分散
    # 非線形スケーリング（平方根）で高頻度領域をより分散させる
    df_merged['rank_m_scaled'] = np.sqrt(df_merged['rank_m_filled'])
    df_merged['rank_p_scaled'] = np.sqrt(df_merged['rank_p_filled'])

    # 获取前 10 名 / トップ10を取得
    top10_m = df_m.sort_values('count', ascending=False).head(10)['word'].tolist()
    top10_p = df_p.sort_values('count', ascending=False).head(10)['word'].tolist()

    # 构造排行榜显示文字 / ランキング表示テキストを構築
    text_top10_m = "<b>Mochiko</b><br>" + "<br>".join([f"{i+1}. {w}" for i, w in enumerate(top10_m)])
    text_top10_p = "<b>Pen Sensei</b><br>" + "<br>".join([f"{i+1}. {w}" for i, w in enumerate(top10_p)])

    # 绘图 / プロット
    fig = go.Figure()

    # Pen Only 的词用 annotations 纵向显示
    pen_only = df_merged[df_merged['category'] == 'Pen Only']

    for cat in ['Common', 'Mochiko Only', 'Pen Only']:
        mask = df_merged['category'] == cat
        df_sub = df_merged[mask]

        # 针对独有词，调整文字标签位置
        # 独自語のラベル位置を調整
        pos = 'top center'
        if cat == 'Mochiko Only': pos = 'middle left'

        fig.add_trace(
            go.Scatter(
                x=df_sub['rank_p_scaled'],
                y=df_sub['rank_m_scaled'],
                mode='markers+text' if cat != 'Pen Only' else 'markers',
                name=cat,
                text=df_sub['word'] if cat != 'Pen Only' else None,
                customdata=np.stack((df_sub['count_m'], df_sub['count_p']), axis=-1) if len(df_sub) > 0 else None,
                textposition=pos,
                textfont=dict(
                    family="Arial Black",
                    size=np.where(df_sub['category'] == 'Common', 9, 7) if cat != 'Pen Only' else 7,
                    color='rgba(40, 40, 40, 0.8)'
                ),
                marker=dict(
                    size=np.clip(12 - df_sub['rank_m_filled'].fillna(1) * 8, 4, 12) if cat == 'Common' else 6,
                    color=df_sub['diff'],
                    colorscale='RdYlBu',
                    cmin=-1,
                    cmax=1,
                    showscale=(cat == 'Common'),
                    opacity=0.7 if cat == 'Common' else 0.5,
                    line=dict(width=0.5, color='white') if cat == 'Common' else None
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Type: " + cat + "<br>"
                    "Mochiko Count: %{customdata[0]}<br>"
                    "Pen Sensei Count: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )

    # Pen Only 的词用 annotations 纵向显示，向上偏移避免遮挡
    pen_annotations = []
    for _, row in pen_only.iterrows():
        pen_annotations.append(dict(
            x=row['rank_p_scaled'],
            y=row['rank_m_scaled'],
            text=row['word'],
            showarrow=False,
            font=dict(size=7, family="Arial Black", color='rgba(40, 40, 40, 0.8)'),
            textangle=-90,
            xanchor='center',
            yanchor='top',
            yshift=-8
        ))

    # 布局调整 / レイアウト調整
    fig.update_layout(
        title=dict(
            text=f'Word Frequency Correlation — {title_suffix}',
            x=0.5, y=0.97,
            font=dict(size=20, family="Arial", color="#333")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=0
        ),
        annotations=pen_annotations + [
            # Mochiko 排行榜（右侧上方）/ Mochikoランキング（右側上部）
            dict(
                x=1.02, y=0.98,
                xref="paper", yref="paper",
                text=text_top10_m,
                showarrow=False, align="left",
                font=dict(size=11, family="Arial", color="#333"),
                bgcolor="rgba(248,248,248,0.9)",
                bordercolor="#ddd", borderwidth=1, borderpad=8
            ),
            # Pen Sensei 排行榜（右侧下方）/ Pen Senseiランキング（右側下部）
            dict(
                x=1.02, y=0.48,
                xref="paper", yref="paper",
                text=text_top10_p,
                showarrow=False, align="left",
                font=dict(size=11, family="Arial", color="#333"),
                bgcolor="rgba(248,248,248,0.9)",
                bordercolor="#ddd", borderwidth=1, borderpad=8
            )
        ],
        margin=dict(l=70, r=180, t=70, b=60),
        width=1400,
        height=800,

        xaxis=dict(
            title=dict(text='Pen Sensei Frequency', font=dict(size=13, color="#555")),
            range=[1.25, -0.05],
            tickvals=[1.15, 1, 0.5, 0],
            ticktext=['Only Mochiko', 'Infrequent', 'Average', 'Frequent'],
            tickfont=dict(size=11, color="#666"),
            showgrid=True, gridcolor='#eee', gridwidth=1,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text='Mochiko Frequency', font=dict(size=13, color="#555")),
            range=[1.25, -0.05],
            tickvals=[1.15, 1, 0.5, 0],
            ticktext=['Only Pen Sensei', 'Infrequent', 'Average', 'Frequent'],
            tickfont=dict(size=11, color="#666"),
            showgrid=True, gridcolor='#eee', gridwidth=1,
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    # 添加背景对角虚线 / 背景の対角点線を追加
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="rgba(180,180,180,0.3)", width=1, dash="dot")
    )

    # 保存 / 保存
    output_file = config.DATA_DIR / 'word_counts' / output_filename
    base = str(output_file).replace('.html', '')
    export_fig(fig, base)
    print(f"已保存至: {base}.html")




# word_process.py 出力ファイルのパス定義
# word_process.py 输出文件的路径定义

BASE_M = config.DATA_DIR / 'word_counts/mochiko/mochiko'
BASE_P = config.DATA_DIR / 'word_counts/pen_sensei/pen_sensei'


# output（bot応答）の比較図 / output（bot回复）的对比图


# 全品詞（output）/ 全品词（output）
build_scatter_plot(
    pd.read_csv(f'{BASE_M}_output_words.csv'),
    pd.read_csv(f'{BASE_P}_output_words.csv'),
    title_suffix='Output — All Words',
    output_filename='output_visualization_words.html'
)

# 名詞のみ（output）/ 仅名词（output）
build_scatter_plot(
    pd.read_csv(f'{BASE_M}_output_n.csv'),
    pd.read_csv(f'{BASE_P}_output_n.csv'),
    title_suffix='Output — Nouns',
    output_filename='output_visualization_nouns.html'
)

# 動詞のみ（output）/ 仅动词（output）
build_scatter_plot(
    pd.read_csv(f'{BASE_M}_output_v.csv'),
    pd.read_csv(f'{BASE_P}_output_v.csv'),
    title_suffix='Output — Verbs',
    output_filename='output_visualization_verbs.html'
)

# emoji のみ（output）/ 仅emoji（output）
build_scatter_plot(
    pd.read_csv(f'{BASE_M}_output_emojis.csv'),
    pd.read_csv(f'{BASE_P}_output_emojis.csv'),
    title_suffix='Output — Emoji',
    output_filename='output_visualization_emojis.html'
)

# input（全ユーザー入力）の単語頻度ランキング図
# input（全量用户输入）的词频排行榜图
#
# data_with_id.csv から (userId, userInput) で重複除去した
# 全ユーザー入力に対して、品詞別 Top 30 を棒グラフで表示する
# 从 data_with_id.csv 按 (userId, userInput) 去重后，
# 对全量用户输入按品词展示 Top 30 柱状图


# 名詞のワードクラウドを生成
# 生成名词词云
def build_word_cloud(csv_path, title, output_filename, top_n=80, min_count=1):
    """
    对 CSV 数据生成词云
    CSVデータに対してワードクラウドを生成する

    Args:
        csv_path:       CSV 文件路径（word, count 列必須）/ CSVファイルパス
        title:          图表标题 / グラフタイトル
        output_filename: 输出文件名（不含扩展名）/ 出力ファイル名（拡張子なし）
        top_n:          取前 N 个高频词 / 上位 N 件の高頻度語を取得
        min_count:      最小出现次数 / 最小出現回数
    """
    df = pd.read_csv(csv_path)
    df = df[df['count'] >= min_count].head(top_n)

    freq_dict = dict(zip(df['word'], df['count']))
    n_words = len(freq_dict)

    font_path = r"C:\Users\Reona\AppData\Local\Microsoft\Windows\Fonts\SarasaMonoJ-Regular.ttf"

    # 词少时缩小画布、减少间距，使词云更紧凑
    if n_words <= 20:
        w, h, margin, mh = 900, 600, 2, 0.9
    elif n_words <= 50:
        w, h, margin, mh = 1200, 700, 4, 0.8
    else:
        w, h, margin, mh = 1600, 900, 10, 0.65

    wc = WordCloud(
        font_path=font_path,
        width=w,
        height=h,
        background_color='white',
        max_words=top_n,
        colormap='plasma',
        prefer_horizontal=mh,
        min_font_size=14 if n_words <= 20 else 12,
        max_font_size=150,
        relative_scaling=0.5,
        margin=margin,
    )
    wc.generate_from_frequencies(freq_dict)

    # 保存 PNG
    img = wc.to_image()
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_data = "data:image/png;base64," + img_str

    # 保存 SVG
    svg_str = wc.to_svg()
    base = output_filename.replace('.html', '')

    output_dir = config.DATA_DIR / 'word_counts'

    # SVG 文件
    svg_file = output_dir / f'{base}.svg'
    with open(svg_file, 'w', encoding='utf-8') as f:
        f.write(svg_str)

    # HTML 文件
    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #f5f5f5; font-family: "Hiragino Sans", "Yu Gothic", sans-serif; padding: 32px; }}
h1 {{ text-align: center; color: #222; font-size: 24px; font-weight: 600; margin-bottom: 24px; letter-spacing: 0.5px; }}
.container {{ max-width: {w + 48}px; margin: 0 auto; text-align: center; background: #fff; border-radius: 8px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
.stats {{ margin-top: 16px; color: #888; font-size: 13px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="container">
<img src="{img_data}" alt="{title}">
<div class="stats">{n_words} words | min count: {min_count}</div>
</div>
</body>
</html>"""

    html_file = output_dir / output_filename
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"已保存至: {html_file}")


def build_bar_chart(csv_path, title_suffix, output_filename, top_n=30):
    """
    对单一数据集生成水平柱状图（Top N 高频词）
    単一データセットに対して水平棒グラフ（Top N 高頻度語）を生成する

    Args:
        csv_path:       CSV 文件路径（word, count 列必須）/ CSVファイルパス
        title_suffix:   图表标题后缀 / グラフタイトルのサフィックス
        output_filename: 输出 HTML 文件名 / 出力HTMLファイル名
        top_n:          显示前 N 名 / 上位 N 件を表示
    """
    df = pd.read_csv(csv_path)
    df_top = df.head(top_n).iloc[::-1]  # 反转使最高频在上方 / 最高頻度を上部に表示するため反転

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_top['count'],
            y=df_top['word'],
            orientation='h',
            marker=dict(
                color=df_top['count'],
                colorscale='Blues',
                showscale=False
            ),
            text=df_top['count'],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(
            text=f'User Input Word Frequency — {title_suffix}',
            x=0.5, font=dict(size=22)
        ),
        margin=dict(l=120, r=80, t=100, b=60),
        width=1000,
        height=800,
        xaxis=dict(
            title='Count',
            showgrid=True, gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            title='',
            tickfont=dict(size=11)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    output_file = config.DATA_DIR / 'word_counts' / output_filename
    base = str(output_file).replace('.html', '')
    export_fig(fig, base)
    print(f"已保存至: {base}.html")


# input 用データのパス定義 / input 数据路径定义
BASE_INPUT = config.DATA_DIR / 'word_counts/input/input'

# 全品詞（input）/ 全品词（input）
build_bar_chart(
    f'{BASE_INPUT}_words.csv',
    title_suffix='All Words',
    output_filename='input_visualization_words.html'
)

# 名詞のみ（input）/ 仅名词（input）
build_bar_chart(
    f'{BASE_INPUT}_n.csv',
    title_suffix='Nouns',
    output_filename='input_visualization_nouns.html'
)

# 動詞のみ（input）/ 仅动词（input）
build_bar_chart(
    f'{BASE_INPUT}_v.csv',
    title_suffix='Verbs',
    output_filename='input_visualization_verbs.html'
)

# emoji のみ（input）/ 仅emoji（input）
build_bar_chart(
    f'{BASE_INPUT}_emojis.csv',
    title_suffix='Emoji',
    output_filename='input_visualization_emojis.html'
)

# 名詞のワードクラウド / 名词词云

# 全体 output（Mochiko + Pen Sensei 合并，仅名词）
df_m_all = pd.read_csv(f'{BASE_M}_output_n.csv')
df_p_all = pd.read_csv(f'{BASE_P}_output_n.csv')
df_combined = pd.concat([df_m_all[['word', 'count']], df_p_all[['word', 'count']]])
df_combined = df_combined.groupby('word', as_index=False)['count'].sum().sort_values('count', ascending=False)

combined_path = config.DATA_DIR / 'word_counts' / '_combined_output_n.csv'
df_combined.to_csv(combined_path, index=False)

build_word_cloud(
    str(combined_path),
    title='Output — Nouns Word Cloud',
    output_filename='wordcloud_output_nouns.html',
    min_count=10
)

# Mochiko output 名詞
build_word_cloud(
    f'{BASE_M}_output_n.csv',
    title='Mochiko — Nouns Word Cloud',
    output_filename='wordcloud_mochiko_nouns.html',
    min_count=5
)

# Pen Sensei output 名詞
build_word_cloud(
    f'{BASE_P}_output_n.csv',
    title='Pen Sensei — Nouns Word Cloud',
    output_filename='wordcloud_pen_sensei_nouns.html',
    min_count=5
)

# input 名詞（全ユーザー入力）
build_word_cloud(
    f'{BASE_INPUT}_n.csv',
    title='User Input — Nouns Word Cloud',
    output_filename='wordcloud_input_nouns.html',
    min_count=10
)
