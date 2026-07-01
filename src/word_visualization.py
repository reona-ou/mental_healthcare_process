import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

    for cat in ['Common', 'Mochiko Only', 'Pen Only']:
        mask = df_merged['category'] == cat
        df_sub = df_merged[mask]

        # 针对独有词，调整文字标签位置
        # 独自語のラベル位置を調整
        pos = 'top center'
        if cat == 'Mochiko Only': pos = 'middle left'
        if cat == 'Pen Only': pos = 'bottom center'

        fig.add_trace(
            go.Scatter(
                x=df_sub['rank_p_scaled'],
                y=df_sub['rank_m_scaled'],
                mode='markers+text',
                name=cat,
                text=df_sub['word'],
                customdata=np.stack((df_sub['count_m'], df_sub['count_p']), axis=-1) if len(df_sub) > 0 else None,
                textposition=pos,
                textfont=dict(
                    family="Arial Black",
                    # 独有词字号设小一点，避免边缘太拥挤
                    # 独自語のフォントサイズを小さくし、端の混雑を回避
                    size=np.where(df_sub['category'] == 'Common', 9, 7),
                    color='rgba(40, 40, 40, 0.8)'
                ),
                marker=dict(
                    size=6,
                    color=df_sub['diff'],
                    colorscale='RdYlBu',
                    cmin=-1,   # 强制设定颜色轴最小值（红色端）/ 色軸の最小値を強制設定（赤端）
                    cmax=1,    # 强制设定颜色轴最大值（蓝色端）/ 色軸の最大値を強制設定（青端）
                    showscale=(cat == 'Common'),  # 仅共同词显示颜色条 / 共通語のみカラーバーを表示
                    opacity=0.6 if cat == 'Common' else 0.4
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

    # 布局调整 / レイアウト調整
    fig.update_layout(
        title=dict(
            text=f'Word Frequency Correlation — {title_suffix}',
            x=0.5, font=dict(size=22)),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=1.07,
            font=dict(size=12)
        ),
        annotations=[
            # Mochiko 排行榜（右侧上方）/ Mochikoランキング（右側上部）
            dict(
                x=1.14, y=0.9,
                xref="paper", yref="paper",
                text=text_top10_m,
                showarrow=False, align="left",
                font=dict(size=13, family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderwidth=1, borderpad=10
            ),
            # Pen Sensei 排行榜（右侧下方）/ Pen Senseiランキング（右側下部）
            dict(
                x=1.16, y=0.45,
                xref="paper", yref="paper",
                text=text_top10_p,
                showarrow=False, align="left",
                font=dict(size=13, family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderwidth=1, borderpad=10
            )
        ],
        # 增加右边距，给排行榜留位置 / ランキング用の右マージンを確保
        margin=dict(l=80, r=200, t=100, b=80),

        # 尺寸设定为 16:9 / サイズを16:9に設定
        width=1600,
        height=900,

        xaxis=dict(
            title='Pen Sensei Frequency',
            range=[1.25, -0.05],
            tickvals=[1.15, 1, 0.5, 0],
            ticktext=['Only Mochiko', 'Infrequent', 'Average', 'Frequent'],
            showgrid=True, gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            title='Mochiko Frequency',
            range=[1.25, -0.05],
            tickvals=[1.15, 1, 0.5, 0],
            ticktext=['Only Pen Sensei', 'Infrequent', 'Average', 'Frequent'],
            showgrid=True, gridcolor='#f0f0f0'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    # 添加背景对角虚线 / 背景の対角点線を追加
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color="rgba(150,150,150,0.2)", width=1, dash="dot")
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
