import pandas as pd
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud
import io
import base64
import config

FIG_W, FIG_H = 1200, 700
FIG_WIDE_W, FIG_WIDE_H = 1400, 700


def export_fig(fig, base_path):
    fig.write_html(str(base_path) + '.html')
    fig.write_image(str(base_path) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)


def get_category(row):
    if pd.isna(row['rank_m']): return 'Pen Only'
    if pd.isna(row['rank_p']): return 'Mochiko Only'
    return 'Common'


def build_scatter_plot(df_mochiko, df_pen_sensei, title_suffix, output_filename):
    df_m = df_mochiko.copy()
    df_p = df_pen_sensei.copy()
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
    text_top10_m = "<b>Mochiko</b><br>" + "<br>".join([f"{i+1}. {w}" for i, w in enumerate(top10_m)])
    text_top10_p = "<b>Pen Sensei</b><br>" + "<br>".join([f"{i+1}. {w}" for i, w in enumerate(top10_p)])

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
                    "Mochiko Count: %{customdata[0]}<br>Pen Sensei Count: %{customdata[1]}<extra></extra>"
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
        xaxis=dict(title=dict(text='Pen Sensei Frequency', font=dict(size=13, color="#555")), range=[1.25, -0.05], tickvals=[1.15, 1, 0.5, 0], ticktext=['Only Mochiko', 'Infrequent', 'Average', 'Frequent'], tickfont=dict(size=11, color="#666"), showgrid=True, gridcolor='#eee', gridwidth=1, zeroline=False),
        yaxis=dict(title=dict(text='Mochiko Frequency', font=dict(size=13, color="#555")), range=[1.25, -0.05], tickvals=[1.15, 1, 0.5, 0], ticktext=['Only Pen Sensei', 'Infrequent', 'Average', 'Frequent'], tickfont=dict(size=11, color="#666"), showgrid=True, gridcolor='#eee', gridwidth=1, zeroline=False),
        plot_bgcolor='white', paper_bgcolor='white',
    )

    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="rgba(180,180,180,0.3)", width=1, dash="dot"))

    output_file = config.DATA_DIR / 'word_counts' / output_filename
    base = str(output_file).replace('.html', '')
    export_fig(fig, base)
    print(f"已保存至: {base}.html")


def build_word_cloud(csv_path, title, output_filename, top_n=80, min_count=1):
    df = pd.read_csv(csv_path)
    df = df[df['count'] >= min_count].head(top_n)
    freq_dict = dict(zip(df['word'], df['count']))
    n_words = len(freq_dict)

    if n_words <= 20:
        w, h, margin, mh = 900, 600, 2, 0.9
    elif n_words <= 50:
        w, h, margin, mh = 1200, 700, 4, 0.8
    else:
        w, h, margin, mh = 1600, 900, 10, 0.65

    wc = WordCloud(
        width=w, height=h, background_color='white', max_words=top_n,
        colormap='plasma', prefer_horizontal=mh,
        min_font_size=14 if n_words <= 20 else 12, max_font_size=150,
        relative_scaling=0.5, margin=margin,
    )
    wc.generate_from_frequencies(freq_dict)

    img = wc.to_image()
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_data = "data:image/png;base64," + img_str

    svg_str = wc.to_svg()
    base = output_filename.replace('.html', '')
    output_dir = config.DATA_DIR / 'word_counts'

    with open(output_dir / f'{base}.svg', 'w', encoding='utf-8') as f:
        f.write(svg_str)

    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #f5f5f5; font-family: "Hiragino Sans", "Yu Gothic", sans-serif; padding: 32px; }}
h1 {{ text-align: center; color: #222; font-size: 24px; font-weight: 600; margin-bottom: 24px; letter-spacing: 0.5px; }}
.container {{ max-width: {w + 48}px; margin: 0 auto; text-align: center; background: #fff; border-radius: 8px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
.stats {{ margin-top: 16px; color: #888; font-size: 13px; }}
</style></head><body>
<h1>{title}</h1>
<div class="container"><img src="{img_data}" alt="{title}">
<div class="stats">{n_words} words | min count: {min_count}</div></div>
</body></html>"""

    with open(output_dir / output_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"已保存至: {output_dir / output_filename}")


def build_treemap(csv_path, title, output_filename, top_n=50, min_count=1):
    df = pd.read_csv(csv_path)
    df = df[df['count'] >= min_count].head(top_n)
    df['label'] = df['word'] + '\n' + df['count'].astype(str)

    fig = go.Figure(go.Treemap(
        labels=df['label'], parents=[''] * len(df), values=df['count'],
        textinfo='label', textfont=dict(size=24),
        marker=dict(colors=df['count'], colorscale='YlGn', showscale=True, colorbar=dict(title='count')),
        hovertemplate='<b>%{customdata}</b><br>Count: %{value}<extra></extra>', customdata=df['word'],
    ))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=22)), margin=dict(l=10, r=10, t=80, b=10), width=1200, height=800)

    output_file = config.DATA_DIR / 'word_counts' / output_filename
    base = str(output_file).replace('.html', '')
    fig.write_html(str(base) + '.html')

    fig_no_title = go.Figure(go.Treemap(
        labels=df['label'], parents=[''] * len(df), values=df['count'],
        textinfo='label', textfont=dict(size=24),
        marker=dict(colors=df['count'], colorscale='YlGn', showscale=True, colorbar=dict(title='count')),
        hovertemplate='<b>%{customdata}</b><br>Count: %{value}<extra></extra>', customdata=df['word'],
    ))
    fig_no_title.update_layout(margin=dict(l=10, r=10, t=10, b=10), width=1200, height=800)
    fig_no_title.write_image(str(base) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)
    print(f"已保存至: {base}.html / .svg")


def build_bar_chart(csv_path, title_suffix, output_filename, top_n=30):
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
    BASE_M = config.DATA_DIR / 'word_counts/mochiko/mochiko'
    BASE_P = config.DATA_DIR / 'word_counts/pen_sensei/pen_sensei'
    BASE_INPUT = config.DATA_DIR / 'word_counts/input/input'

    build_scatter_plot(pd.read_csv(f'{BASE_M}_output_words.csv'), pd.read_csv(f'{BASE_P}_output_words.csv'), title_suffix='Output — All Words', output_filename='output_visualization_words.html')
    build_scatter_plot(pd.read_csv(f'{BASE_M}_output_n.csv'), pd.read_csv(f'{BASE_P}_output_n.csv'), title_suffix='Output — Nouns', output_filename='output_visualization_nouns.html')
    build_scatter_plot(pd.read_csv(f'{BASE_M}_output_v.csv'), pd.read_csv(f'{BASE_P}_output_v.csv'), title_suffix='Output — Verbs', output_filename='output_visualization_verbs.html')
    build_scatter_plot(pd.read_csv(f'{BASE_M}_output_emojis.csv'), pd.read_csv(f'{BASE_P}_output_emojis.csv'), title_suffix='Output — Emoji', output_filename='output_visualization_emojis.html')

    for suffix, fname in [('All Words', 'input_visualization_words.html'), ('Nouns', 'input_visualization_nouns.html'), ('Verbs', 'input_visualization_verbs.html'), ('Emoji', 'input_visualization_emojis.html')]:
        csv_type = 'words' if 'All' in suffix else ('n' if 'Nouns' in suffix else ('v' if 'Verbs' in suffix else 'emojis'))
        build_bar_chart(f'{BASE_INPUT}_{csv_type}.csv', title_suffix=suffix, output_filename=fname)

    df_m_all = pd.read_csv(f'{BASE_M}_output_n.csv')
    df_p_all = pd.read_csv(f'{BASE_P}_output_n.csv')
    df_combined = pd.concat([df_m_all[['word', 'count']], df_p_all[['word', 'count']]]).groupby('word', as_index=False)['count'].sum().sort_values('count', ascending=False)
    combined_path = config.DATA_DIR / 'word_counts' / '_combined_output_n.csv'
    df_combined.to_csv(combined_path, index=False)

    build_word_cloud(str(combined_path), title='Output — Nouns Word Cloud', output_filename='wordcloud_output_nouns.html', min_count=10)
    build_word_cloud(f'{BASE_M}_output_n.csv', title='Mochiko — Nouns Word Cloud', output_filename='wordcloud_mochiko_nouns.html', min_count=5)
    build_word_cloud(f'{BASE_P}_output_n.csv', title='Pen Sensei — Nouns Word Cloud', output_filename='wordcloud_pen_sensei_nouns.html', min_count=5)
    build_word_cloud(f'{BASE_INPUT}_n.csv', title='User Input — Nouns Word Cloud', output_filename='wordcloud_input_nouns.html', min_count=10)

    build_treemap(str(combined_path), title='Output — Nouns Treemap', output_filename='treemap_output_nouns.html', top_n=50, min_count=10)
    build_treemap(f'{BASE_M}_output_n.csv', title='Mochiko — Nouns Treemap', output_filename='treemap_mochiko_nouns.html', top_n=50, min_count=5)
    build_treemap(f'{BASE_P}_output_n.csv', title='Pen Sensei — Nouns Treemap', output_filename='treemap_pen_sensei_nouns.html', top_n=50, min_count=5)
    build_treemap(f'{BASE_INPUT}_n.csv', title='User Input — Nouns Treemap', output_filename='treemap_input_nouns.html', top_n=50, min_count=10)
