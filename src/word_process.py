import config
import pandas as pd
from fugashi import Tagger
import csv
import emoji
from collections import Counter

df_mochiko = pd.read_csv(config.DATA_DIR / 'data_mochiko.csv')
df_pen_sensei = pd.read_csv(config.DATA_DIR / 'data_pen_sensei.csv')

tagger = Tagger()
punctuation_pos = {'補助記号', '記号'}
particle_pos = {'助詞', '助動詞'}


def get_word_counts(texts, exclude_particles=False):
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
            word_data.append((word.surface, pos))
    return Counter(word_data)


def save_word_counts(filename, word_counts):
    with open(config.DATA_DIR / 'word_counts' / filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'pos', 'count'])
        for (word, pos), count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([word, pos, count])


def save_rankings_by_pos(name, word_counts):
    nouns = Counter({k: v for k, v in word_counts.items() if k[1] == '名詞'})
    save_word_counts(f'{name}_n.csv', nouns)
    verbs = Counter({k: v for k, v in word_counts.items() if k[1] == '動詞'})
    save_word_counts(f'{name}_v.csv', verbs)
    emojis = Counter({k: v for k, v in word_counts.items() if k[1] == 'emoji'})
    save_word_counts(f'{name}_emojis.csv', emojis)


def process_chatbot(name, df):
    texts_input = df['userInput'].fillna('')
    word_counts_input = get_word_counts(texts_input, exclude_particles=True)
    texts_output = df['replyText'].fillna('')
    word_counts_output = get_word_counts(texts_output, exclude_particles=True)
    save_word_counts(f'{name}/{name}_input_words.csv', word_counts_input)
    save_word_counts(f'{name}/{name}_output_words.csv', word_counts_output)
    save_rankings_by_pos(f'{name}/{name}_input', word_counts_input)
    save_rankings_by_pos(f'{name}/{name}_output', word_counts_output)


def process_all_inputs():
    df_all = pd.read_csv(config.DATA_DIR / 'data_with_id.csv')
    df_deduped = df_all.drop_duplicates(subset=['userId', 'userInput'])
    texts_input = df_deduped['userInput'].fillna('')
    word_counts = get_word_counts(texts_input, exclude_particles=True)
    save_word_counts('input/input_words.csv', word_counts)
    save_rankings_by_pos('input/input', word_counts)


if __name__ == "__main__":
    process_chatbot('mochiko', df_mochiko)
    process_chatbot('pen_sensei', df_pen_sensei)
    process_all_inputs()
