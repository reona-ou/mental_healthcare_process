import config
import pandas as pd
from fugashi import Tagger
import csv
import emoji
from collections import Counter

# 数据读取 / データ読み込み
df_mochiko = pd.read_csv(config.DATA_DIR / 'data_mochiko.csv')
df_pen_sensei = pd.read_csv(config.DATA_DIR / 'data_pen_sensei.csv')

# 初始化分词器 / 形態素解析器の初期化
tagger = Tagger()

# 定义要排除的词性（介词/连词类）/ 除外する品詞（助詞・接続詞）
exclude_pos = ['助詞', '接続詞']


# ============================================================
# 形态素解析による单词计数处理
# 形態素解析による単語カウント処理
#
# 对2个chatbot（mochiko / pen_sensei）的会话数据进行分析，
# 2つのchatbot（mochiko / pen_sensei）の会話データに対して、
# 分别对用户输入(input)和机器人回复(output)进行单词统计，
# ユーザー入力(input)とボット応答(output)それぞれに対して単語統計を行い、
# 输出名词・动词・emoji的出现频率排行榜。
# 名詞・動詞・emojiの出現頻度ランキングを出力する。
# ============================================================


def get_word_counts(texts, exclude_particles=False):
    """
    对文本列表进行形态素解析，统计单词出现次数
    テキストリストに対して形態素解析を行い、単語の出現回数をカウントする

    Args:
        texts: 文本列表 / テキストのリスト
        exclude_particles: 是否排除助词/助动词 / 助詞・助動詞を除外するかどうか
    """
    # 根据词典版本调整标点集合
    # 辞書のバージョンに応じて句読点の集合を調整
    # UniDic 常用: '補助記号', '助詞', '助動詞'
    # IPADIC 常用: '記号', '助詞', '助動詞'
    punctuation_pos = {'補助記号', '記号'}
    # 助词/助动词集合（类似英文的stop words）
    # 助詞・助動詞の集合（英語のstop wordsに相当）
    particle_pos = {'助詞', '助動詞'}
    word_data = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue

        # 先提取并计数emoji / まずテキストから絵文字を抽出してカウントする
        emoji_list = emoji.emoji_list(text)
        for emoji_item in emoji_list:
            emoji_char = emoji_item['emoji']
            word_data.append((emoji_char, 'emoji'))

        # 移除emoji后进行形态素解析 / emojiを除去したテキストで形態素解析を行う
        text_without_emoji = emoji.replace_emoji(text, replace='')

        for word in tagger(text_without_emoji):
            pos = word.feature.pos1

            # 排除标点符号 / 句読点を除外する
            if pos in punctuation_pos:
                continue
            # 根据开关排除助词/助动词 / スイッチに応じて助詞・助動詞を除外する
            if exclude_particles and pos in particle_pos:
                continue
            # 直接使用原形（不做词形还原）/ 原形をそのまま使用（レマタイズしない）
            word_data.append((word.surface, pos))

    word_counts = Counter(word_data)
    return word_counts


# 保存结果到CSV / 結果をCSVに保存する
def save_word_counts(filename, word_counts):
    """
    将单词计数结果保存为CSV文件 / 単語カウント結果をCSVファイルに保存する
    输出文件位于 data/word_counts/ 目录下
    出力ファイルは data/word_counts/ ディレクトリ配下
    """
    with open(config.DATA_DIR / 'word_counts' / filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'pos', 'count'])
        for (word, pos), count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, pos, count])


# 名词/动词/emoji排行榜生成函数 / 名詞・動詞・emojiランキング生成関数
def save_rankings_by_pos(name, word_counts):
    """
    按品词分类生成排行榜并保存为CSV
    品詞別にランキングを生成してCSVに保存する
    """
    # 名词排行榜 / 名詞ランキング
    nouns = Counter({k: v for k, v in word_counts.items() if k[1] == '名詞'})
    save_word_counts(f'{name}_n.csv', nouns)

    # 动词排行榜 / 動詞ランキング
    verbs = Counter({k: v for k, v in word_counts.items() if k[1] == '動詞'})
    save_word_counts(f'{name}_v.csv', verbs)

    # Emoji排行榜 / emojiランキング
    emojis = Counter({k: v for k, v in word_counts.items() if k[1] == 'emoji'})
    save_word_counts(f'{name}_emojis.csv', emojis)


#   input_words.csv     - 用户输入的单词统计 / ユーザー入力の単語統計
#   output_words.csv    - 回复的单词统计 / 応答の単語統計
#   input_n.csv         - 用户输入的名词排行 / ユーザー入力の名詞ランキング
#   output_n.csv        - 回复的名词排行 / 応答の名詞ランキング
#   input_v.csv         - 用户输入的动词排行 / ユーザー入力の動詞ランキング
#   output_v.csv        - 回复的动词排行 / 応答の動詞ランキング
#   input_emojis.csv    - 用户输入的emoji排行 / ユーザー入力のemojiランキング
#   output_emojis.csv   - 回复的emoji排行 / 応答のemojiランキング


def process_chatbot(name, df):
    """
    对单个chatbot的数据进行处理，分别统计input和output
    単一chatbotのデータを処理し、inputとoutputをそれぞれ統計する

    Args:
        name: chatbot名称（'mochiko' 或 'pen_sensei'）/ chatbot名
        df: 对应的数据DataFrame / 対応するデータのDataFrame
    """
    # 用户输入（userInput）/ ユーザー入力（userInput）
    texts_input = df['userInput'].fillna('')
    word_counts_input = get_word_counts(texts_input, exclude_particles=True)

    # 机器人回复（replyText）/ ボット応答（replyText）
    texts_output = df['replyText'].fillna('')
    word_counts_output = get_word_counts(texts_output, exclude_particles=True)

    # 保存用户输入的单词统计 / ユーザー入力の単語統計を保存
    save_word_counts(f'{name}/{name}_input_words.csv', word_counts_input)
    # 保存机器人回复的单词统计 / ボット応答の単語統計を保存
    save_word_counts(f'{name}/{name}_output_words.csv', word_counts_output)

    # 保存用户输入的品词排行榜 / ユーザー入力の品詞別ランキングを保存
    save_rankings_by_pos(f'{name}/{name}_input', word_counts_input)
    # 保存机器人回复的品词排行榜 / ボット応答の品詞別ランキングを保存
    save_rankings_by_pos(f'{name}/{name}_output', word_counts_output)


# 处理mochiko数据 / mochikoデータの処理
process_chatbot('mochiko', df_mochiko)

# 处理pen_sensei数据 / pen_senseiデータの処理
process_chatbot('pen_sensei', df_pen_sensei)


# ============================================================
# data_with_id.csv から全ユーザー入力を抽出・解析
# data_with_id.csv 中提取全量用户输入并解析
#
# 同一 userId の同一 userInput は重複カウントしない
# 相同 userId 的相同 userInput 不会重复计数
# ============================================================

def process_all_inputs():
    """
    从 data_with_id.csv 读取所有用户输入，按 (userId, userInput) 去重后统计
    data_with_id.csv から全ユーザー入力を読み込み、(userId, userInput) で重複除去後に統計する
    """
    df_all = pd.read_csv(config.DATA_DIR / 'data_with_id.csv')

    # 按 (userId, userInput) 去重 / (userId, userInput) で重複除去
    df_deduped = df_all.drop_duplicates(subset=['userId', 'userInput'])
    texts_input = df_deduped['userInput'].fillna('')
    word_counts = get_word_counts(texts_input, exclude_particles=True)

    # 保存全量单词统计 / 全体の単語統計を保存
    save_word_counts('input/input_words.csv', word_counts)

    # 直接复用 save_rankings_by_pos 函数，无需手动过滤
    save_rankings_by_pos('input/input', word_counts)


# 处理全量用户输入 / 全ユーザー入力の処理
process_all_inputs()
