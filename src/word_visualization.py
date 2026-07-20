"""
単語頻度可視化スクリプト
散布図、棒グラフ、Treemap を生成する。
"""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import config

# 統一サイズ定義
FIG_W, FIG_H = 1200, 700
FIG_WIDE_W, FIG_WIDE_H = 1400, 700


def export_fig(fig, base_path):
    """plotly の fig を HTML + SVG の2形式で出力"""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.write_html(str(base_path) + '.html')
    fig.write_image(str(base_path) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)


def get_category(row):
    """単語がどのカテゴリに属するかを判定"""
    if pd.isna(row['rank_m']): return 'Pen Only'
    if pd.isna(row['rank_p']): return 'Mochiko Only'
    return 'Common'


def build_scatter_plot(df_chatbot_mo, df_chatbot_p, title_suffix, output_filename):
    """2つのチャットボットの単語頻度データに対して散布図比較プロットを生成"""
    df_m = df_chatbot_mo.copy()
    df_p = df_chatbot_p.copy()
    df_m['rank_m'] = df_m['count'].rank(ascending=False, method='first', pct=True)
    df_p['rank_p'] = df_p['count'].rank(ascending=False, method='first', pct=True)

    df_merged = pd.merge(
        df_m[['word', 'rank_m', 'count']],
        df_p[['word', 'rank_p', 'count']],
        on='word', how='outer', suffixes=('_m', '_p')
    )

    df_merged['rank_m_filled'] = df_merged['rank_m'].fillna(1.15)
    df_merged['rank_p_filled'] = df_merged['rank_p'].fillna(1.15)
    df_merged['category'] = df_merged.apply(get_category, axis=1)

    df_merged['diff'] = df_merged['rank_p'] - df_merged['rank_m']
    df_merged.loc[df_merged['category'] == 'Mochiko Only', 'diff'] = 1.0
    df_merged.loc[df_merged['category'] == 'Pen Only', 'diff'] = -1.0

    df_merged['rank_m_scaled'] = np.sqrt(df_merged['rank_m_filled'])
    df_merged['rank_p_scaled'] = np.sqrt(df_merged['rank_p_filled'])

    top10_m = df_m.sort_values('count', ascending=False).head(10)['word'].tolist()
    top10_p = df_p.sort_values('count', ascending=False).head(10)['word'].tolist()
    text_top10_m = "<b>Chatbot MO</b><br>" + "<br>".join([f"{i+1}. {w}" for i, w in enumerate(top10_m)])
    text_top10_p = "<b>Chatbot P</b><br>" + "<br>".join([f"{i+1}. {w}" for i, w in enumerate(top10_p)])

    fig = go.Figure()
    pen_only = df_merged[df_merged['category'] == 'Pen Only']
    pen_annotations = []

    for cat in ['Common', 'Mochiko Only', 'Pen Only']:
        mask = df_merged['category'] == cat
        df_sub = df_merged[mask]
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
                    cmin=-1, cmax=1,
                    showscale=(cat == 'Common'),
                    opacity=0.7 if cat == 'Common' else 0.5,
                    line=dict(width=0.5, color='white') if cat == 'Common' else None
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>Type: " + cat + "<br>"
                    "Chatbot MO Count: %{customdata[0]}<br>Chatbot P Count: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    for _, row in pen_only.iterrows():
        pen_annotations.append(dict(
            x=row['rank_p_scaled'], y=row['rank_m_scaled'],
            text=row['word'], showarrow=False,
            font=dict(size=7, family="Arial Black", color='rgba(40, 40, 40, 0.8)'),
            textangle=-90, xanchor='center', yanchor='top', yshift=-8
        ))

    fig.update_layout(
        title=dict(text=f'Word Frequency Correlation — {title_suffix}', x=0.5, y=0.97, font=dict(size=20, family="Arial", color="#333")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=11), bgcolor="rgba(255,255,255,0.8)", borderwidth=0),
        annotations=pen_annotations + [
            dict(x=1.02, y=0.98, xref="paper", yref="paper", text=text_top10_m, showarrow=False, align="left", font=dict(size=11, family="Arial", color="#333"), bgcolor="rgba(248,248,248,0.9)", bordercolor="#ddd", borderwidth=1, borderpad=8),
            dict(x=1.02, y=0.48, xref="paper", yref="paper", text=text_top10_p, showarrow=False, align="left", font=dict(size=11, family="Arial", color="#333"), bgcolor="rgba(248,248,248,0.9)", bordercolor="#ddd", borderwidth=1, borderpad=8),
        ],
        margin=dict(l=70, r=180, t=70, b=60), width=1400, height=800,
        xaxis=dict(title=dict(text='Chatbot P Frequency', font=dict(size=13, color="#555")), range=[1.25, -0.05], tickvals=[1.15, 1, 0.5, 0], ticktext=['Only Chatbot MO', 'Infrequent', 'Average', 'Frequent'], tickfont=dict(size=11, color="#666"), showgrid=True, gridcolor='#eee', gridwidth=1, zeroline=False),
        yaxis=dict(title=dict(text='Chatbot MO Frequency', font=dict(size=13, color="#555")), range=[1.25, -0.05], tickvals=[1.15, 1, 0.5, 0], ticktext=['Only Chatbot P', 'Infrequent', 'Average', 'Frequent'], tickfont=dict(size=11, color="#666"), showgrid=True, gridcolor='#eee', gridwidth=1, zeroline=False),
        plot_bgcolor='white', paper_bgcolor='white',
    )

    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="rgba(180,180,180,0.3)", width=1, dash="dot"))

    output_file = config.DATA_DIR / 'word_counts' / output_filename
    base = str(output_file).replace('.html', '')
    export_fig(fig, base)
    print(f"已保存至: {base}.html")


def build_treemap(csv_path, title, output_filename, top_n=50, min_count=1):
    """CSVデータに対してTreemap（単語頻度ツリーマップ）を生成"""
    df = pd.read_csv(csv_path)
    df = df[df['count'] >= min_count].head(top_n)

    treemap_kwargs = dict(
        labels=df['word'], parents=[''] * len(df), values=df['count'],
        texttemplate='<b>%{label}</b><br><i>f=%{value}</i>',
        textposition='bottom right',
        textfont=dict(size=24),
        marker=dict(colors=df['count'], colorscale='YlGn', showscale=True, colorbar=dict(title='count')),
        hovertemplate='<b>%{customdata}</b><br>Count: %{value}<extra></extra>', customdata=df['word'],
    )

    fig = go.Figure(go.Treemap(**treemap_kwargs))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=22)), margin=dict(l=10, r=10, t=80, b=10), width=1200, height=800)

    output_file = config.DATA_DIR / 'word_counts' / output_filename
    base = str(output_file).replace('.html', '')
    fig.write_html(str(base) + '.html')

    fig_no_title = go.Figure(go.Treemap(**treemap_kwargs))
    fig_no_title.update_layout(margin=dict(l=10, r=10, t=10, b=10), width=1200, height=800)
    fig_no_title.write_image(str(base) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)
    print(f"已保存至: {base}.html / .svg")


def build_bar_chart(csv_path, title_suffix, output_filename, top_n=30):
    """単一データセットに対して水平棒グラフ（Top N 高頻度語）を生成"""
    df = pd.read_csv(csv_path)
    df_top = df.head(top_n).iloc[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_top['count'], y=df_top['word'], orientation='h',
        marker=dict(color=df_top['count'], colorscale='Blues', showscale=False),
        text=df_top['count'], textposition='outside', textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=f'User Input Word Frequency — {title_suffix}', x=0.5, font=dict(size=22)),
        margin=dict(l=120, r=80, t=100, b=60), width=1000, height=800,
        xaxis=dict(title='Count', showgrid=True, gridcolor='#f0f0f0'),
        yaxis=dict(title='', tickfont=dict(size=11)),
        plot_bgcolor='white', paper_bgcolor='white',
    )

    output_file = config.DATA_DIR / 'word_counts' / output_filename
    base = str(output_file).replace('.html', '')
    export_fig(fig, base)
    print(f"已保存至: {base}.html")


if __name__ == "__main__":
    BASE_M = config.DATA_DIR / 'word_counts/chatbot_mo/chatbot_mo'
    BASE_P = config.DATA_DIR / 'word_counts/chatbot_p/chatbot_p'
    BASE_INPUT = config.DATA_DIR / 'word_counts/input/input'
    WC_DIR = config.DATA_DIR / 'word_counts'

    # 散布図（比較）→ 根目录
    build_scatter_plot(pd.read_csv(f'{BASE_M}_output_words.csv'), pd.read_csv(f'{BASE_P}_output_words.csv'), title_suffix='Output — All Words', output_filename='output_visualization_words.html')
    build_scatter_plot(pd.read_csv(f'{BASE_M}_output_n.csv'), pd.read_csv(f'{BASE_P}_output_n.csv'), title_suffix='Output — Nouns', output_filename='output_visualization_nouns.html')
    build_scatter_plot(pd.read_csv(f'{BASE_M}_output_v.csv'), pd.read_csv(f'{BASE_P}_output_v.csv'), title_suffix='Output — Verbs', output_filename='output_visualization_verbs.html')
    build_scatter_plot(pd.read_csv(f'{BASE_M}_output_emojis.csv'), pd.read_csv(f'{BASE_P}_output_emojis.csv'), title_suffix='Output — Emoji', output_filename='output_visualization_emojis.html')

    # 棒グラフ（input）→ input/
    for suffix, fname in [('All Words', 'input_visualization_words.html'), ('Nouns', 'input_visualization_nouns.html'), ('Verbs', 'input_visualization_verbs.html'), ('Emoji', 'input_visualization_emojis.html')]:
        csv_type = 'words' if 'All' in suffix else ('n' if 'Nouns' in suffix else ('v' if 'Verbs' in suffix else 'emojis'))
        build_bar_chart(f'{BASE_INPUT}_{csv_type}.csv', title_suffix=suffix, output_filename=f'input/{fname}')

    # 合并名词数据
    df_m_all = pd.read_csv(f'{BASE_M}_output_n.csv')
    df_p_all = pd.read_csv(f'{BASE_P}_output_n.csv')
    df_combined = pd.concat([df_m_all[['word', 'count']], df_p_all[['word', 'count']]]).groupby('word', as_index=False)['count'].sum().sort_values('count', ascending=False)
    combined_path = WC_DIR / '_combined_output_n.csv'
    df_combined.to_csv(combined_path, index=False)

    # Treemap → 各chatbot目录
    build_treemap(str(combined_path), title='Output — Nouns Treemap', output_filename='treemap_output_nouns.html', top_n=50, min_count=20)
    build_treemap(f'{BASE_M}_output_n.csv', title='Chatbot MO — Nouns Treemap', output_filename='chatbot_mo/treemap_chatbot_mo_nouns.html', top_n=50, min_count=5)
    build_treemap(f'{BASE_P}_output_n.csv', title='Chatbot P — Nouns Treemap', output_filename='chatbot_p/treemap_chatbot_p_nouns.html', top_n=50, min_count=5)
    build_treemap(f'{BASE_INPUT}_n.csv', title='User Input — Nouns Treemap', output_filename='input/treemap_input_nouns.html', top_n=50, min_count=5)
