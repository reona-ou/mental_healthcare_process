# トピックモデリング結果 / Topic Modeling Results

## 概要 / Overview

ruri-v3-310m 埋め込みモデル + KMeans クラスタリング + 半教師シード誘導 + MMR + 統計距離フィルタにより、7つの有効トピックを識別。35件の文書が外れ値として分類された。

| 指標 | 値 | 説明 |
|---|---|---|
| 入力データ | combined userInput | 2チャットボットのユーザー入力統合 / Combined user inputs from 2 chatbots |
| 総文書数 | 182件 | 全セッション数 / Total sessions |
| 有効トピック数 | 7 | 有効な話題数 / Valid topics |
| 外れ値 | 35件 (19.2%) | 短文 + 統計異常 / Short texts + statistical outliers |
| トピックカバレッジ | 80.8% | トピックに分類された文書の割合 / Topic coverage rate |

---

## 処理パイプライン / Pipeline

```
userInput (182件)
    ↓
fugashi で形態素解析（名詞・動詞・形容詞のみ）
    ↓
短文テキストフィルタ（分詞後 < 2語 → -1）: 26件除外
    ↓
ruri-v3-310m で 768次元埋め込み生成
    ↓
UMAP 次元削減（768 → 2次元）
├── n_neighbors=15, min_dist=0.1, metric=cosine
    ↓
KMeans クラスタリング（7クラスタ）
├── n_init=10, random_state=42
    ↓
BERTopic トピック表現
├── シードトピック誘導（8カテゴリ）
├── MMR (diversity=0.5) でキーワード多様性向上
└── CountVectorizer (ngram_range=(1,2))
    ↓
統計距離フィルタ（mean + 1.5 * std → -1）: 9件除外
    ↓
結果保存
```

---

## 外れ値（Topic -1）: 35件 (19.2%) / Outliers

### 内訳 / Breakdown
- **短文テキスト**: 26件 — 分詞後2語未満の文書
- **統計距離異常**: 9件 — トピック中心から mean + 1.5 * std を超える文書

### 例 / Examples
| テキスト | 種類 |
|---|---|
| ありがとう | 短文 |
| これ一択 | 短文 |
| ほっこりおなはし | 短文 |
| 男のトイレの時間長くない？ | 短文 |
| しちゃってるかも | 短文 |

---

## トピック一覧 / Topic List

### Topic 0: 育児支援・地域 / Parenting Support & Community (27件)

**キーワード**: 地域, 保健, 心配, 育児, 相談, 赤ちゃん, 友達, 予定

**内容**:
地域の支援機関（保健センター、助産師会、ファミリーサポート等）を通じて育児支援を求める会話。具体的には：
- 保健センターへの相談
- 助産師との面談希望
- 子育て支援センターの利用
- 地域の育児支援イベント参加
- 赤ちゃんの成長发育に関する相談

**具体例**:
> 「育児や子どもの発達について保健センターの他に気軽に相談できるところってありますか？」
> 「助産師さんとゆっくり話がしたい時はどの施設がいいですか？」
> 「こないだ電話かかってきて、地域の方みたいで…手助けしたげるよと言ってもらいました！」

---

### Topic 1: 育児・離婚 / Parenting & Divorce (24件)

**キーワード**: 赤ちゃん, 離婚, 家事, 育児, きつい, 話し掛ける, 嫌い

**内容**:
育児ストレスによる夫婦間の矛盾と離婚の検討。具体的には：
- ワンオペ育児での夫の非協力
- 家事・育児分担の不満
- 夫とのコミュニケーション不足
- 離婚の検討
- 産後の体型変化と自信喪失

**具体例**:
> 「夫は赤ちゃんにたくさん話しかけて優しくするのに、私にはあまり話しかけてくれなくなった」
> 「産後2か月経って体重が10キロ減ったけど、お腹ボヨボヨで腰回りが太くて、女として終わった気がする。」
> 「夫が全く手伝ってくれないので、イライラします。」

---

### Topic 2: 産後・授乳 / Postpartum & Breastfeeding (23件)

**キーワード**: おっぱい, 育児, 寝る, しんどい, ワンオペ, 安心, 母乳

**内容**:
産後の授乳、睡眠問題、母子の健康に関する不安と安心。具体的には：
- 母乳の量や出方への不安
- おっぱいサプリメントへの関心
- 産後の体型変化
- ワンオペ育児の孤独感
- 赤ちゃんの成長への安堵

**具体例**:
> 「入院中は母乳もたくさん出てたのに、最近出なくなりました。おっぱいも張っていないようです。」
> 「最近、おっぱいが出なくなってきたので、ママ友に相談したら、おっぱいが出るいいサプリがあると聞いたんだけど…」
> 「おっぱい飲んですやすや寝てるのを見ると、安心する。」

---

### Topic 3: 流産・妊娠 / Miscarriage & Pregnancy (21件)

**キーワード**: 流産, 妊娠, 自分, 辛い, 落ち込む, 嬉しい, 経験

**内容**:
流産・妊娠に関する感情の揺れと個人の経験。具体的には：
- 過去の流産体験の追憶
- 妊娠への期待と不安
- 流産後の自分への責め
- 医師への相談
- 今後の妊娠への不安

**具体例**:
> 「過去に流産したことがまだ受け止めきれなくて、この子を見ていてもたまに涙が出てくる」
> 「流産の原因はなんですか？私の場合は仕事のしすぎかと思います。」
> 「あの子のことも抱っこしてあげたかったな、どんな子だったのかなと考えてしまう」

---

### Topic 4: 人間関係・相談 / Interpersonal & Consultation (20件)

**キーワード**: 相談, 苦手, 連絡, しんどい, 無視, 友達, 嫌い

**内容**:
人間関係の悩み、相談へのハードル、コミュニケーションの問題。具体的には：
- 相談することへの抵抗感
- 友人・家族への相談の困難さ
- 夫からの無視やモラハラ
- 行政への相談方法
- 信頼できる人の不在

**具体例**:
> 「人に話したことはないかな。話したら、相手を困らせてしまうし…」
> 「相談するのが苦手なのです。」
> 「友達もいないし、実家の両親は年を取っているから相談できません。」

---

### Topic 5: 離婚・浮気 / Divorce & Infidelity (18件)

**キーワード**: 離婚, 浮気, 浮気相手, 考える, 冷たい, 視野

**内容**:
離婚の意思決定、不倫問題、夫婦関係の崩壊。具体的には：
- 離婚の検討と具体的な行動
- 浮気相手との関係
- 夫の冷たい態度
- 離婚届の送付
- 弁護士への相談

**具体例**:
> 「結婚を終わらせたい」
> 「浮気されるくらいなら離婚したいし、離婚する前に浮気されたら死にたい」
> 「パートナーは昨日言った通り、冷たいの。離婚を視野にいれて弁護士さんに話をしてるよ。」

---

### Topic 6: 産後睡眠 / Postpartum Sleep (14件)

**キーワード**: 寝る, 箇月, 安心, しんどい, 授乳, 夜中

**内容**:
産後の睡眠障害、授乳期の健康問題。具体的には：
- 夜中の授乳による睡眠不足
- 赤ちゃんの睡眠リズムへの対応
- 産後の体調不良
- 安心感の共有

**具体例**:
> 「たまひよアプリで子どもの睡眠時間を測ってるけど、生後2カ月だけど、夜中3時間で起きてしまいます！」
> 「夜眠れなくて少し辛い」
> 「寝かしつけ中。でも乳幼児突然死症候群になったらどうしようとか…」

---

## トピック分布 / Topic Distribution

| トピック | 件数 | 割合 | 内容 |
|---|---|---|---|
| -1 (外れ値) | 35 | 19.2% | 短文 + 統計異常 / Short texts + statistical outliers |
| 0 | 27 | 14.8% | 育児支援・地域 / Parenting Support & Community |
| 1 | 24 | 13.2% | 育児・離婚 / Parenting & Divorce |
| 2 | 23 | 12.6% | 産後・授乳 / Postpartum & Breastfeeding |
| 3 | 21 | 11.5% | 流産・妊娠 / Miscarriage & Pregnancy |
| 4 | 20 | 11.0% | 人間関係・相談 / Interpersonal & Consultation |
| 5 | 18 | 9.9% | 離婚・浮気 / Divorce & Infidelity |
| 6 | 14 | 7.7% | 産後睡眠 / Postpartum Sleep |

---

## 詐欺関連の分類について / About Fraud Classification

### 現状 / Current Status
詐欺関連文書（14件）は、独立したトピックとして分類されていない。理由は以下の通り：

1. **データ量不足**: 詐欺文書は全体の7.7%（14件/182件）と少なく、独立したクラスタを形成するのに不十分 / Insufficient data: fraud documents are only 7.7% (14/182), too few to form an independent cluster
2. **語彙の重複**: 詐欺文書には「相談」「心配」「夫」など、他のトピックと共通する語彙が含まれる / Vocabulary overlap: fraud documents share common words with other topics
3. **KMeansの制約**: KMeansは全文書をクラスタに強制割り当てるため、outlierとして分類されない / KMeans forces all documents into clusters, preventing outlier classification

### 詐欺文書の現在の分類 / Current Fraud Distribution
| トピック | 件数 | 内容 |
|---|---|---|
| Topic 0 (育児支援) | 10件 | 育児支援 + 詐欺混入 / Parenting + fraud mixed |
| Topic 5 (離婚・浮気) | 2件 | 出転関連 + 詐欺 / Infidelity + fraud |
| Topic -1 (外れ値) | 2件 | 短文 / Short texts |

### 2カテゴリ分類での対応 / 2-Category Classification
topic分析とは別に、`negative_classify.py` による2カテゴリ分類では：
- 詐欺文書は **category 0（負面）** として正しく分類されている
- キーワードマッチング + SVM-RBF分類器により、全14件の詐欺文書が正しく識別

Separately, `negative_classify.py` correctly classifies all fraud documents:
- Fraud documents are classified as **category 0 (negative)**
- All 14 fraud documents are properly identified via keyword matching + SVM-RBF

### 改善の選択肢 / Options for Improvement
1. **現状維持**: topic分析では詐欺を独立トピックとせず、2カテゴリ分類で対応 / Keep fraud in topic 0, rely on 2-category classification
2. **クラスタ数増加**: cluster数を9-10に増やして詐欺分離を試みる（他のトピックが小さくなるリスク） / Increase clusters to 9-10 to separate fraud (risk of smaller other topics)
3. **半教師学習**: 詐欺のシードトピックを強化して誘導（データ量不足で効果が限定的） / Strengthen fraud seed topics (limited effect due to insufficient data)
