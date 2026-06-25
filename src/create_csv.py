import re
from collections import deque
import pandas as pd
import config

# 读取原始CSV / 生のCSVデータを読み込む
df_chat = pd.read_csv(config.DATA_DIR / 'mochiko-line-bot-prod-20260318 (1).csv')
df_point = pd.read_csv(config.DATA_DIR / 'real_research.csv')

# 用于存储提取后的数据 / 抽出後のデータを格納する変数
orchestrator_queue = deque()  # Orchestrator请求的FIFO队列 / OrchestratorリクエストのFIFOキュー
session_buffer = {}  # 存放流程状态 / フローの状態を保存するバッファ（キー: createdAt）
processed_rows = []  # 整形后的行数据列表 / 整形済みの行データリスト
session_count = 1    # 会话序号计数器 / セッション（会話ターン）の通番カウンター

# 暂存区 / 一時保存用変数
last_input = None
current_reply_queue = []
interrupt_reply_queue = []

# 定义权重（各项合计1.0） / 重み付け定義（各項目の合計1.0）
weights = {
    'kyukansei': 0.1,        # 共感性 / 共感性
    'igakuseikakusei': 0.25,  # 医学正确性 / 医学的正確性
    'anzensei': 0.35,         # 安全性 / 安全性
    'yuugaisei': 0.2,         # 有害性 / 有害性
    'aiirai': 0.05,           # AI依赖 / AI依頼
    'shikafannshi': 0.05      # 资格表示 / 資格表示
}
# 风险扣分项的列 / リスク（マイナス要因）となる評価項目の列名リスト
negative_cols = ['igakuseikakusei', 'anzensei', 'yuugaisei', 'aiirai', 'shikafannshi']
# 计算负面风险加权和 / ダメな要素の加重和（リスクスコア）を計算
risk_score = df_point[negative_cols].mul([weights[col] for col in negative_cols]).sum(axis=1)
# 计算共感奖励得分 / 正面な要素（共感性）の加重得点（ボーナススコア）を計算
bonus_score = df_point['kyukansei'] * weights['kyukansei']
# 综合计算：基础100 - 风险*10 + 奖励*10，数值越大越好 / 総合スコア計算：基本100 - リスク*10 + ボーナス*10、数値が大きいほど満足度が高い
df_point['satisfaction'] = (100 - (risk_score * 10) + (bonus_score * 10))
# 提取 userId 和 satisfaction / userIdと満足度スコアのみ抽出
df_point_result = df_point[['userId', 'satisfaction']].copy()
# 构建 userId -> satisfaction 的字典，用于快速查找 / userId→satisfactionの辞書を作成（高速ルックアップ用）
point_result_dict = dict(zip(df_point_result['userId'], df_point_result['satisfaction']))


# 辅助函数：生成一条记录的字典 / ヘルパー関数：1レコード分の辞書を作成する
# 人格判定逻辑 / ペルソナの判定ロジック：
#   - ReplyInterruptPersona（中断回复）→ 使用 suggested / 提案されたペルソナを使用
#   - ReplyCurrentPersona（当前人格回复）→ 使用 current / 現在のペルソナを使用
#   - 其他 → 'Unknown' / それ以外 → 'Unknown'
def create_record(s, reply_row, count):
    if reply_row['action'] == 'ReplyInterruptPersona':
        effective_parent = s['suggested']
    elif reply_row['action'] == 'ReplyCurrentPersona':
        effective_parent = s['current']
    else:
        effective_parent = 'Unknown'
    return {
        'session_id': f'S{count:03d}',          # 会话ID / セッションID
        'userId': s['user_id'],                   # 用户ID / ユーザーID
        'point': point_result_dict.get(s['user_id']),  # 满足度得分，不存在则为None / 満足度スコア（存在しない場合はNone）
        'persona': effective_parent,              # 担当人格名 / 担当ペルソナ名
        'replyType': reply_row['action'],         # 回复类型 / 返信タイプ
        'userInput': s['input'],                  # 用户输入文本 / ユーザーの入力テキスト
        'replyText': reply_row['replyText']       # 机器人回复文本 / ボットの返信テキスト
    }


# 主处理：遍历聊天历史，将 Orchestrator 请求与回复进行配对 / メイン処理：チャット履歴を走査し、Orchestratorリクエストと返信をペアリング
for i in range(len(df_chat)):
    row = df_chat.iloc[i]

    # 捕获 OrchestratorResult（用户输入解析结果），加入 FIFO 队列 / OrchestratorResult（ユーザー入力の解析結果）をキャプチャしFIFOキューに追加
    if row['action'] == 'OrchestratorResult':
        p_id = row['createdAt']
        session_buffer[p_id] = {
            'input': row['userInput'],           # 用户输入文本 / ユーザーの入力テキスト
            'suggested': row['suggestedParent'],  # 建议人格 / 提案されたペルソナ
            'current': row['currentParent'],      # 当前人格 / 現在のペルソナ
            'current_reply': None,                # 当前人格的回复，稍后设置 / 現在ペルソナからの返信（後で設定）
            'interrupt_reply': None,              # 中断人格的回复，稍后设置 / 割り込みペルソナからの返信（後で設定）
            'user_id': row['userId']              # 用户ID / ユーザーID
        }
        orchestrator_queue.append(p_id)

    # 捕获回复，与队列头部（最早的请求）进行匹配 / 返信をキャプチャし、キューの先頭（最も古いリクエスト）とマッチング
    elif row['action'] in ['ReplyCurrentPersona', 'ReplyInterruptPersona']:
        if orchestrator_queue:
            target_pid = orchestrator_queue[0]  # 取队列头部（最早请求） / キューの先頭を取得

            if row['action'] == 'ReplyCurrentPersona':
                session_buffer[target_pid]['current_reply'] = row
                # 建议人格与当前人格相同 → 无需中断，session完成，出队 / 提案と現在が同じ→割り込み不要、セッション完了、デキュー
                if session_buffer[target_pid]['suggested'] == session_buffer[target_pid]['current']:
                    orchestrator_queue.popleft()
            else:
                # 中断回复到达 → session完成，出队 / 割り込み返信到着→セッション完了、デキュー
                session_buffer[target_pid]['interrupt_reply'] = row
                orchestrator_queue.popleft()

    # 归档：将已完成的 session 追加到 processed_rows / アーカイブ：完了したセッションをprocessed_rowsに追加
    keys_to_delete = []
    for p_id, s in session_buffer.items():
        # 判断是否需要中断（建议人格与当前人格不同时需要中断） / 割り込み要否判定（提案と現在のペルソナが異なる場合に必要）
        has_interrupt_needed = (s['suggested'] != s['current'])
        # current_reply 已到，且（无需中断 或 中断回复也已到） / current_reply到着済み かつ（割り込み不要 or 割り込み返信も到着済み）
        if (s['current_reply'] is not None) and \
                ((not has_interrupt_needed) or (s['interrupt_reply'] is not None)):
            # 如果同时存在interrupt_reply，只保留interrupt_reply（优先使用中断回复）
            # 割り込み返信が存在する場合、interrupt_replyのみを保持（割り込み返信を優先）
            if s['interrupt_reply'] is not None:
                processed_rows.append(create_record(s, s['interrupt_reply'], session_count))
            else:
                processed_rows.append(create_record(s, s['current_reply'], session_count))
            session_count += 1
            keys_to_delete.append(p_id)

    # 从缓冲区删除已完成的 session / 完了済みセッションをバッファから削除
    for p_id in keys_to_delete:
        del session_buffer[p_id]

# 将处理结果转换为 DataFrame / 処理結果をDataFrameに変換
df_final = pd.DataFrame(processed_rows)



# 无意义对话筛选
# 固定关键词动作（FIXED_KEYWORD_ACTIONS）的触发文本 / 固定キーワードアクションのトリガーテキスト
FIXED_KEYWORD_TEXTS = [
    "もちこが共感",
    "ほっこりおはなし",
    "もちことモヤモヤまとめ",
    "励ましもちこ",
    "一緒に調べる",
    "ぺん先生にきいてみる",
    "むちことおしゃべり",
]

# 工具命令（handle_tool_command）文本 / ツールコマンドのテキスト
TOOL_COMMAND_TEXTS = [
    "うちのこいくつ？",
    "子育てまんだら",
    "AIたちの楽しい使い方",
    "研究同意の確認",
    "ストレッチを始める",
    "スクワットを始める",
    "筋トレ動画をみる",
    "ストレスBOMB",
    "ID",
]

# 同意确认自动回复中的特征文本 / 同意確認の自動返信に含まれる特徴テキスト
CONSENT_KEYWORDS = "その機能を使うには、まず研究利用への同意が必要"

# 合并所有需要排除的固定文本 / 除外すべき固定テキストを統合
ALL_EXCLUDED_TEXTS = FIXED_KEYWORD_TEXTS + TOOL_COMMAND_TEXTS

# 除外する曖昧表現・フィラー句 / 除外する曖昧表現・フィラー句
MEANINGLESS_PHRASES = [
    "そうかも", "そうかもね",
    "かもしれない", "かも", "かもね",
    "なんか", "なんだか",
    "ちょっとどうしよう", "どうしよう",
    "たぶん", "おそらく",
    "きっと大丈夫", "大丈夫",
    "ありがとう","分かりました"
]


def is_meaningless_input(text):
    """
    判断是否为无意义输入：
    - 非字符串 / 非文字列
    - 包含人格切换快捷指令（如「もちことお話しする」）
    - 包含固定关键词动作文本
    - 包含工具命令文本
    - 包含同意确认自动回复特征
    - 仅由曖昧表現・フィラー構成
    - 输入过短（仅空白/符号等）

    意味のない入力を判定：
    - 非文字列
    - ペルソナ切替クイックコマンド（例：「もちことお話しする」）
    - 固定キーワードアクションのテキスト
    - ツールコマンドのテキスト
    - 同意確認自動返信の特徴テキストを含む
    - 曖昧表現・フィラーのみで構成されている
    - 入力が短すぎる（空白・記号のみなど）
    """
    if not isinstance(text, str):
        return True

    stripped = text.strip()
    if len(stripped) == 0:
        return True

    # 人格切换快捷指令模式 / ペルソナ切替クイックコマンドパターン
    persona_switch_pattern = re.compile(
        r'(ペン先生|もちこ|むちこ).*?(お話しする|きいてみる|相談する|に交代する|相談してみる|相談します)',
        re.IGNORECASE
    )
    if persona_switch_pattern.search(stripped):
        return True

    # 固定关键词动作 & 工具命令 / 固定キーワードアクション＆ツールコマンド
    if stripped in ALL_EXCLUDED_TEXTS:
        return True

    # 同意确认自动回复 / 同意確認の自動返信
    if CONSENT_KEYWORDS in stripped:
        return True

    # 曖昧表現・フィラーのみで構成されているか / 曖昧表現・フィラーのみで構成されているか
    if stripped in MEANINGLESS_PHRASES:
        return True

    return False


# 注：df_final 中只包含 ReplyCurrentPersona 和 ReplyInterruptPersona 两种 replyType，
# FixedKeywordSwitch、PostbackSwitch、ConsentRequired、ErrorOrchestrator 等无意义动作
# 在主循环的配对逻辑中已被自然排除，无需额外过滤 action 列。
# 注：df_final には ReplyCurrentPersona と ReplyInterruptPersona の2種類の replyType のみ含まれ、
# FixedKeywordSwitch、PostbackSwitch、ConsentRequired、ErrorOrchestrator などの意味のないアクションは
# メインループのペアリングロジックで自然に除外されているため、action列の追加フィルタは不要。

# 仅保留有效数据：
# 1. userInput长度>=4
# 2. userInput不匹配无意义模式

# 有効データのみ抽出：
# 1. userInputの長さ>=4
# 2. userInputが意味のないパターンにマッチしない
mask_valid = (df_final['userInput'].str.len() >= 4) & (~df_final['userInput'].apply(is_meaningless_input))
df_clean = df_final[mask_valid].copy()
# 去重：按 userId, replyType, userInput, persona 组合，保留首次出现 / 重複除去：userId, replyType, userInput, personaの組み合わせで最初の出現を残す
df_clean = df_clean.drop_duplicates(
    subset=['userId', 'replyType', 'userInput', 'persona'],
    keep='first'
)


# 清理特殊符号：将换行符、制表符、回车符替换为空格 / 特殊文字クリーニング：改行・タブ・復帰をスペースに置換
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    return text


# 替换用户输入中的特殊字符 / ユーザー入力内の特殊文字列をスペースに置換
df_clean['userInput'] = df_clean['userInput'].str.replace('\\n', ' ').str.replace('\\t', ' ').str.replace('\\r', ' ')
# 替换回复文本中的特殊字符 / 返信テキスト内の特殊文字列をスペースに置換
df_clean['replyText'] = df_clean['replyText'].str.replace('\\n', ' ').str.replace('\\t', ' ').str.replace('\\r', ' ')

# 分离数据 / データの分離
# 有满足度得分的数据（问卷回答存在） / 満足度スコアが存在するデータ（アンケート回答あり）
df_with_id = df_clean[df_clean['point'].notna()].copy()
# 无满足度得分的数据（问卷回答不存在、ID不匹配） / 満足度スコアが無いデータ（アンケート回答なし、ID不一致）
df_cant_find_id = df_clean[df_clean['point'].isna()].copy()
# 仅提取人格「mochiko」的数据 / ペルソナ「もちこ」のデータのみ抽出
df_mochiko = df_with_id[df_with_id['persona'] == 'mochiko'].copy()
# 仅提取人格「pen_sensei」的数据 / ペルソナ「ペン先生」のデータのみ抽出
df_pen_sensei = df_with_id[df_with_id['persona'] == 'pen_sensei'].copy()

# ID对照表 / ID照合表
# 聊天数据中出现的唯一userId列表 / チャットデータのユニークuserId一覧
ids_chat = pd.DataFrame(df_chat['userId'].unique(), columns=['chat_userId'])
# 问卷数据中出现的唯一userId列表 / アンケートデータのユニークuserId一覧
ids_point = pd.DataFrame(df_point['userId'].unique(), columns=['point_userId'])
# 使用outer join（全连接）对照，缺失方显示为NaN / outer join（外部結合）で照合、一方にしか無いIDはNaNで確認可能
df_id_comparison = pd.merge(
    ids_chat,
    ids_point,
    left_on='chat_userId',
    right_on='point_userId',
    how='outer'
)

# 导出CSV / CSVファイルへのエクスポート
df_id_comparison.to_csv(config.DATA_DIR / 'id_comparison.csv', index=False, encoding='utf-8-sig')

df_final.to_csv(config.DATA_DIR / 'data.csv', index=False, encoding='utf-8-sig')
df_clean.to_csv(config.DATA_DIR / 'cleaned_data.csv', index=False, encoding='utf-8-sig')
df_with_id.to_csv(config.DATA_DIR / 'data_with_id.csv', index=False, encoding='utf-8-sig')
df_cant_find_id.to_csv(config.DATA_DIR / 'data_cant_find_id.csv', index=False, encoding='utf-8-sig')
df_mochiko.to_csv(config.DATA_DIR / 'data_mochiko.csv', index=False, encoding='utf-8-sig')
df_pen_sensei.to_csv(config.DATA_DIR / 'data_pen_sensei.csv', index=False, encoding='utf-8-sig')

# 输出处理结果的行数 / 処理結果の件数を出力
print(f"num of rows(mochiko): {len(df_mochiko)}")
print(f"num of rows(pen_sensei): {len(df_pen_sensei)}")