"""
単語頻度分析スクリプト
2つのチャットボット（chatbot_mo / chatbot_p）の会話データに対して、
ユーザー入力とボット応答の単語統計を行う。
名詞・動詞・emojiの出現頻度ランキングを出力する。
"""
import config
import pandas as pd
from fugashi import Tagger
import csv
import emoji
from collections import Counter

# データ読み込み
df_chatbot_mo = pd.read_csv(config.DATA_DIR / 'data_chatbot_mo.csv')
df_chatbot_p = pd.read_csv(config.DATA_DIR / 'data_chatbot_p.csv')

# 形態素解析器
tagger = Tagger()
punctuation_pos = {'補助記号', '記号'}
particle_pos = {'助詞', '助動詞'}

# ストップワードリスト
STOPWORD_LIST = [
    "あそこ", "あたり", "あちら", "あっち", "あと", "あな", "あなた", "あれ", "いくつ", "いつ", "いま", "いや", "いろいろ", "うち", "おおまか", "おまえ", "おれ", "がい", "かく", "かたち", "かやの", "から", "がら", "きた", "くせ", "ここ", "こっち", "こと", "ごと", "こちら", "ごっちゃ", "これ", "これら", "ごろ", "さまざま", "さらい", "さん", "しかた", "しよう", "すか", "ずつ", "すね", "すべて", "ぜんぶ", "そう", "そこ", "そちら", "そっち", "そで", "それ", "それぞれ", "それなり", "たくさん", "たち", "たび", "ため", "だめ", "ちゃ", "ちゃん", "てん", "とおり", "とき", "どこ", "どこか", "ところ", "どちら", "どっか", "どっち", "どれ", "なか", "なかば", "なに", "など", "なん", "はじめ", "はず", "はるか", "ひと", "ひとつ", "ふく", "ぶり", "べつ", "へん", "ぺん", "ほう", "ほか", "まさ", "まし", "まとも", "まま", "みたい", "みつ", "みなさん", "みんな", "もと", "もの", "もん", "やつ", "よう", "よそ", "わけ", "わたし", "ハイ", "上", "中", "下", "字", "年", "月", "日", "時", "分", "秒", "週", "火", "水", "木", "金", "土", "国", "都", "道", "府", "県", "市", "区", "町", "村", "各", "第", "方", "何", "的", "度", "文", "者", "性", "体", "人", "他", "今", "部", "課", "係", "外", "類", "達", "気", "室", "口", "誰", "用", "界", "会", "首", "男", "女", "別", "話", "私", "屋", "店", "家", "場", "等", "見", "際", "観", "段", "略", "例", "系", "論", "形", "間", "地", "員", "線", "点", "書", "品", "力", "法", "感", "作", "元", "手", "数", "彼", "彼女", "子", "内", "楽", "喜", "怒", "哀", "輪", "頃", "化", "境", "俺", "奴", "高", "校", "婦", "伸", "紀", "誌", "レ", "行", "列", "事", "士", "台", "集", "様", "所", "歴", "器", "名", "情", "連", "毎", "式", "簿", "回", "匹", "個", "席", "束", "歳", "目", "通", "面", "円", "玉", "枚", "前", "後", "左", "右", "次", "先", "春", "夏", "秋", "冬", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "億", "兆", "下記", "上記", "時間", "今回", "前回", "場合", "一つ", "年生", "自分", "ヶ所", "ヵ所", "カ所", "箇所", "ヶ月", "ヵ月", "カ月", "箇月", "名前", "本当", "確か", "時点", "全部", "関係", "近く", "方法", "我々", "違い", "多く", "扱い", "新た", "その後", "半ば", "結局", "様々", "以前", "以後", "以降", "未満", "以上", "以下", "幾つ", "毎日", "自体", "向こう", "何人", "手段", "同じ", "感じ", "てる", "いる", "なる", "れる", "する", "ある", "こと", "これ", "さん", "して", "くれる", "やる", "くださる", "そう", "せる", "した", "思う", "できる", "くる", "みる", "しまう", "それ", "ここ", "ちゃん", "くん", "て", "に", "を", "は", "の", "が", "と", "た", "し", "で", "ない", "も", "な", "い", "か", "ので", "よう",
]
STOPWORD_SET = set(STOPWORD_LIST)


def get_word_counts(texts, exclude_particles=False):
    """テキストリストに対して形態素解析を行い、単語の出現回数をカウント"""
    word_data = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        for emoji_item in emoji.emoji_list(text):
            word_data.append((emoji_item['emoji'], 'emoji'))
        text_without_emoji = emoji.replace_emoji(text, replace='')
        for word in tagger(text_without_emoji):
            pos = word.feature.pos1
            if pos in punctuation_pos:
                continue
            if exclude_particles and pos in particle_pos:
                continue
            if word.surface in STOPWORD_SET:
                continue
            word_data.append((word.surface, pos))
    return Counter(word_data)


def save_word_counts(filename, word_counts):
    """単語カウント結果をCSVに保存"""
    output_path = config.DATA_DIR / 'word_counts' / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'pos', 'count'])
        for (word, pos), count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, pos, count])


def save_rankings_by_pos(name, word_counts):
    """品詞別にランキングを生成してCSVに保存"""
    nouns = Counter({k: v for k, v in word_counts.items() if k[1] == '名詞'})
    save_word_counts(f'{name}_n.csv', nouns)
    verbs = Counter({k: v for k, v in word_counts.items() if k[1] == '動詞'})
    save_word_counts(f'{name}_v.csv', verbs)
    emojis = Counter({k: v for k, v in word_counts.items() if k[1] == 'emoji'})
    save_word_counts(f'{name}_emojis.csv', emojis)


def process_chatbot(name, df):
    """単一チャットボットのデータを処理し、input/outputの統計を保存"""
    texts_input = df['userInput'].fillna('')
    word_counts_input = get_word_counts(texts_input, exclude_particles=True)
    texts_output = df['replyText'].fillna('')
    word_counts_output = get_word_counts(texts_output, exclude_particles=True)
    save_word_counts(f'{name}/{name}_input_words.csv', word_counts_input)
    save_word_counts(f'{name}/{name}_output_words.csv', word_counts_output)
    save_rankings_by_pos(f'{name}/{name}_input', word_counts_input)
    save_rankings_by_pos(f'{name}/{name}_output', word_counts_output)


def process_all_inputs():
    """data_with_id.csv から全ユーザー入力を読み込み、(userId, userInput) で重複除去後に統計"""
    df_all = pd.read_csv(config.DATA_DIR / 'data_with_id.csv')
    df_deduped = df_all.drop_duplicates(subset=['userId', 'userInput'])
    texts_input = df_deduped['userInput'].fillna('')
    word_counts = get_word_counts(texts_input, exclude_particles=True)
    save_word_counts('input/input_words.csv', word_counts)
    save_rankings_by_pos('input/input', word_counts)


if __name__ == "__main__":
    process_chatbot('chatbot_mo', df_chatbot_mo)
    process_chatbot('chatbot_p', df_chatbot_p)
    process_all_inputs()
