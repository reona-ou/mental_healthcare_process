"""
Treemap SVG 日本語→英語 翻訳ツール

word_visualization.py の build_treemap と同じ方法で CSV → Treemap SVG を生成するが、
ラベルを英語に翻訳して出力する。
翻訳後に重複ラベルがある場合は「英語 (元の日本語)」で区別する。

用法:
  python translate_svg.py              # 处理所有 treemap CSV
  python translate_svg.py path/to/csv  # 处理单个 CSV
"""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
from deep_translator import GoogleTranslator
import config

# word_visualization.py と同じサイズ定義
FIG_WIDE_W, FIG_WIDE_H = 1400, 700

WC_DIR = config.DATA_DIR / 'word_counts'

# (csv_path相對於WC_DIR, title, output_html路徑相對於WC_DIR, top_n, min_count)
TREEMAP_TASKS: list[tuple[str, str, str, int, int]] = [
    ('_combined_output_n.csv',          'Output — Nouns Treemap',       'treemap_output_nouns_word_translate.html',              50, 20),
    ('input/input_n.csv',               'User Input — Nouns Treemap',   'input/treemap_input_nouns_word_translate.html',         50, 5),
]


def _translate_words(words: list[str]) -> dict[str, str]:
    unique = list(dict.fromkeys(words))
    translator = GoogleTranslator(source='ja', target='en')
    try:
        results = translator.translate_batch(unique)
        return dict(zip(unique, results))
    except Exception as e:
        print(f"  [WARN] 批量翻译失败，回退到逐个翻译: {e}", file=sys.stderr)
        mapping = {}
        for w in unique:
            try:
                mapping[w] = translator.translate(w) or w
            except Exception:
                mapping[w] = w
        return mapping


def _deduplicate_labels(translated: list[str], originals: list[str]) -> list[str]:
    """翻译后对重复标签追加原文区分，如 'baby (赤ちゃん)' / 'baby (ベビー)'。"""
    counts = Counter(translated)
    result = list(translated)
    seen: dict[str, int] = {}
    for i, en in enumerate(result):
        if counts[en] > 1:
            seen[en] = seen.get(en, 0) + 1
            result[i] = f"{en} ({originals[i]})"
    return result


def translate_treemap(csv_path: Path, title: str, output_filename: str,
                      top_n: int = 50, min_count: int = 1):
    """build_treemap と同じ Treemap を生成。ラベルのみ英語に翻訳。HTML + SVG 出力。"""
    df = pd.read_csv(csv_path)
    df = df[df['count'] >= min_count].head(top_n)

    if df.empty:
        print(f"  跳过（过滤后无数据）: {csv_path.name}")
        return

    # 翻译
    word_map = _translate_words(df['word'].tolist())
    raw_translated = [word_map.get(w, w) for w in df['word']]
    translated = _deduplicate_labels(raw_translated, df['word'].tolist())
    n_translated = sum(1 for o, t in zip(df['word'], translated) if o != t)
    print(f"[{csv_path.name}] {len(df)} 词, 翻译 {n_translated} 个")

    # build_treemap と同じ kwargs（labels と customdata のみ翻訳済み）
    treemap_kwargs = dict(
        labels=translated,
        parents=[''] * len(df),
        values=df['count'],
        texttemplate='<b>%{label}</b><br><i>f=%{value}</i>',
        textposition='bottom right',
        textfont=dict(family="Arial", size=18),
        marker=dict(
            colors=df['count'],
            colorscale='YlGn',
            showscale=True,
            colorbar=dict(title='count'),
        ),
        hovertemplate='<b>%{customdata}</b><br>Count: %{value}<extra></extra>',
        customdata=translated,
    )

    fig = go.Figure(go.Treemap(**treemap_kwargs))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=22)),
        margin=dict(l=10, r=10, t=80, b=10),
        width=1200, height=800
    )

    output_file = WC_DIR / output_filename
    output_file.parent.mkdir(parents=True, exist_ok=True)
    base = str(output_file).replace('.html', '')
    fig.write_html(str(base) + '.html')

    fig_no_title = go.Figure(go.Treemap(**treemap_kwargs))
    fig_no_title.update_layout(margin=dict(l=10, r=10, t=10, b=10), width=1200, height=800)
    fig_no_title.write_image(str(base) + '.svg', width=FIG_WIDE_W, height=FIG_WIDE_H, scale=1)
    print(f"已保存至: {base}.html / .svg")


def main():
    if len(sys.argv) > 1:
        csv_arg = Path(sys.argv[1])
        if csv_arg.is_file():
            output = csv_arg.stem + '_word_translate.html'
            translate_treemap(csv_arg, title=csv_arg.stem, output_filename=output)
            return
        elif csv_arg.is_dir():
            pass
        else:
            print(f"エラー: {csv_arg} が見つかりません", file=sys.stderr)
            sys.exit(1)

    for csv_rel, title, html_rel, top_n, min_count in TREEMAP_TASKS:
        csv_path = WC_DIR / csv_rel
        if not csv_path.exists():
            print(f"  跳过（CSV 不存在）: {csv_path}")
            continue
        translate_treemap(csv_path, title, html_rel, top_n, min_count)


if __name__ == '__main__':
    main()