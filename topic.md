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

## 外れ値（Topic -1）: 35件 (19.2%) / Outliers: 35 docs (19.2%)

### 内訳 / Breakdown
- **短文テキスト / Short texts**: 26件 — 分詞後2語未満の文書 / 26 docs — Documents with fewer than 2 tokens
- **統計距離異常 / Statistical outliers**: 9件 — トピック中心から mean + 1.5 * std を超える文書 / 9 docs — Documents exceeding mean + 1.5 * std from topic centroid

### 例 / Examples
| テキスト / Text | 種類 / Type |
|---|---|
| ありがとう / Thank you | 短文 / Short text |
| これ一択 / This is the only choice | 短文 / Short text |
| ほっこりおなはし / Heartwarming story | 短文 / Short text |
| 男のトイレの時間長くない？ / Don't men take too long in the restroom? | 短文 / Short text |
| しちゃってるかも / Might be doing it | 短文 / Short text |

---

## トピック一覧 / Topic List

### Topic 0: 育児支援・地域 / Parenting Support & Community (27件 / 27 docs)

**キーワード / Keywords**: 地域 (community), 保健 (health center), 心配 (worry), 育児 (parenting), 相談 (consultation), 赤ちゃん (baby), 友達 (friend), 予定 (schedule)

**内容 / Content**:
地域の支援機関を通じて育児支援を求める会話。保健センターへの相談、助産師との面談、子育て支援センターの利用など。
Conversations seeking parenting support through local community organizations. Includes consultations at health centers, interviews with midwives, and the use of childcare support centers.

**具体例 / Examples**:
> 「育児や子どもの発達について保健センターの他に気軽に相談できるところってありますか？」
> "Is there anywhere besides the health center where I can easily seek advice about parenting and child development?"
>
> 「助産師さんとゆっくり話がしたい時はどの施設がいいですか？」
> "Which facility is best if I want to have a relaxed conversation with a midwife?"
>
> 「こないだ電話かかってきて、地域の方みたいで…手助けしたげるよと言ってもらいました！」
> "I got a call the other day, seemingly from someone in the local community... They offered to give me a helping hand!"

---

### Topic 1: 育児・離婚 / Parenting & Divorce (24件 / 24 docs)

**キーワード / Keywords**: 赤ちゃん (baby), 離婚 (divorce), 家事 (housework), 育児 (parenting), きつい (harsh), 話し掛ける (talk to), 嫌い (hate)

**内容 / Content**:
育児ストレスによる夫婦間の矛盾と離婚の検討。ワンオペ育児、家事分担の不満、夫とのコミュニケーション不足など。
Marital conflicts and considerations of divorce due to parenting stress. Includes solo parenting, dissatisfaction with housework distribution, and lack of communication with husbands.

**具体例 / Examples**:
> 「夫は赤ちゃんにたくさん話しかけて優しくするのに、私にはあまり話しかけてくれなくなった」
> "My husband talks a lot to the baby and is very gentle, but he barely speaks to me anymore."
>
> 「産後2か月経って体重が10キロ減ったけど、お腹ボヨボヨで腰回りが太くて、女として終わった気がする。」
> "Two months postpartum, I've lost 10 kg, but my stomach is flabby and my waist is thick. I feel like I'm finished as a woman."
>
> 「夫が全く手伝ってくれないので、イライラします。」
> "My husband doesn't help out at all, which makes me feel so frustrated."

---

### Topic 2: 産後・授乳 / Postpartum & Breastfeeding (23件 / 23 docs)

**キーワード / Keywords**: おっぱい (breast), 育児 (parenting), 寝る (sleep), しんどい (tough), ワンオペ (solo parenting), 安心 (relief), 母乳 (breast milk)

**内容 / Content**:
産後の授乳、睡眠問題、母子の健康に関する不安と安心。母乳の量への不安、おっぱいサプリへの関心など。
Anxieties and reliefs regarding postpartum breastfeeding, sleep issues, and maternal/infant health. Includes concerns about breast milk supply and interest in breastfeeding supplements.

**具体例 / Examples**:
> 「入院中は母乳もたくさん出てたのに、最近出なくなりました。おっぱいも張っていないようです。」
> "While I was hospitalized, I had plenty of breast milk, but recently it has stopped coming out. My breasts don't even feel engorged anymore."
>
> 「最近、おっぱいが出なくなってきたので、ママ友に相談したら、おっぱいが出るいいサプリがあると聞いたんだけど…」
> "Since my breast milk has been drying up lately, I talked to a fellow mom friend, and she mentioned there's a good supplement that helps with milk supply..."
>
> 「おっぱい飲んですやすや寝てるのを見ると、安心する。」
> "Seeing my baby drinking milk and sleeping peacefully brings me so much relief."

---

### Topic 3: 流産・妊娠 / Miscarriage & Pregnancy (21件 / 21 docs)

**キーワード / Keywords**: 流産 (miscarriage), 妊娠 (pregnancy), 自分 (myself), 辛い (painful), 落ち込む (depressed), 嬉しい (happy), 経験 (experience)

**内容 / Content**:
流産・妊娠に関する感情の揺れと個人の経験。過去の流産体験の追憶、妊娠への期待と不安など。
Emotional fluctuations and personal experiences regarding miscarriage and pregnancy. Includes memories of past miscarriage experiences, as well as expectations and anxieties about pregnancy.

**具体例 / Examples**:
> 「過去に流産したことがまだ受け止めきれなくて、この子を見ていてもたまに涙が出てくる」
> "I still haven't fully come to terms with my past miscarriage, and sometimes tears just fall even while I'm looking at this child."
>
> 「流産の原因はなんですか？私の場合は仕事のしすぎかと思います。」
> "What causes a miscarriage? In my case, I think it might have been due to overworking."
>
> 「あの子のことも抱っこしてあげたかったな、どんな子だったのかなと考えてしまう」
> "I often find myself thinking that I wanted to hold that baby too, wondering what kind of child they would have been."

---

### Topic 4: 人間関係・相談 / Interpersonal & Consultation (20件 / 20 docs)

**キーワード / Keywords**: 相談 (consultation), 苦手 (not good at), 連絡 (contact), しんどい (exhausting), 無視 (ignore), 友達 (friend), 嫌い (dislike)

**内容 / Content**:
人間関係の悩み、相談へのハードル、コミュニケーションの問題。相談することへの抵抗感、友人・家族への相談の困難さなど。
Interpersonal relationship worries, barriers to seeking advice, and communication issues. Includes reluctance to consult others and difficulties in talking to friends or family.

**具体例 / Examples**:
> 「人に話したことはないかな。話したら、相手を困らせてしまうし…」
> "I don't think I've ever spoken to anyone about it. If I do, I feel like I'd just trouble them..."
>
> 「相談するのが苦手なのです。」
> "I am not good at consulting or opening up to others."
>
> 「友達もいないし、実家の両親は年を取っているから相談できません。」
> "I don't have any friends, and my parents back home are elderly, so I can't consult them."

---

### Topic 5: 離婚・浮気 / Divorce & Infidelity (18件 / 18 docs)

**キーワード / Keywords**: 離婚 (divorce), 浮気 (affair), 浮気相手 (affair partner), 考える (consider), 冷たい (cold), 視野 (scope)

**内容 / Content**:
離婚の意思決定、不倫問題、夫婦関係の崩壊。離婚の検討、浮気相手との関係、夫の冷たい態度など。
Decision-making on divorce, infidelity issues, and the breakdown of marital relations. Includes considering divorce, relationships with affair partners, and partner's cold attitudes.

**具体例 / Examples**:
> 「結婚を終わらせたい」
> "I want to end my marriage."
>
> 「浮気されるくらいなら離婚したいし、離婚する前に浮気されたら死にたい」
> "I'd rather divorce than be cheated on, and if I get cheated on before a divorce, I feel like I'd want to die."
>
> 「パートナーは昨日言った通り、冷たいの。離婚を視野にいれて弁護士さんに話をしてるよ。」
> "As I said yesterday, my partner is cold. I am considering divorce and am currently speaking with a lawyer."

---

### Topic 6: 産後睡眠 / Postpartum Sleep (14件 / 14 docs)

**キーワード / Keywords**: 寝る (sleep), 箇月 (months), 安心 (relief), しんどい (tough), 授乳 (breastfeeding), 夜中 (nighttime)

**内容 / Content**:
産後の睡眠障害、授乳期の健康問題。夜中の授乳による睡眠不足、赤ちゃんの睡眠リズムへの対応など。
Postpartum sleep disorders and health issues during the breastfeeding period. Includes sleep deprivation from nighttime feeding and dealing with the baby's sleep rhythm.

**具体例 / Examples**:
> 「たまひよアプリで子どもの睡眠時間を測ってるけど、生後2カ月だけど、夜中3時間で起きてしまいます！」
> "I'm tracking my child's sleep time with the Tamahiyo app. Even though they are 2 months old, they still wake up every 3 hours in the middle of the night!"
>
> 「夜眠れなくて少し辛い」
> "It's a bit tough because I can't sleep at night."
>
> 「寝かしつけ中。でも乳幼児突然死症候群になったらどうしようとか…」
> "I'm putting my baby to sleep right now. But I can't stop worrying about what if Sudden Infant Death Syndrome (SIDS) happens..."

---

## トピック分布 / Topic Distribution

| トピック / Topic | 件数 / Count | 割合 / Percentage | 内容 / Content |
|---|---|---|---|
| -1 (外れ値 / Outliers) | 35 | 19.2% | 短文 + 統計異常 / Short texts + statistical outliers |
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
Fraud-related documents (14 cases) are not classified as an independent topic. The reasons are as follows:

1. **データ量不足 / Insufficient Data**: 詐欺文書は全体の7.7%（14件/182件）と少なく、独立したクラスタを形成するのに不十分 / Fraud documents are only 7.7% (14/182) of the total, which is too few to form an independent cluster.
2. **語彙の重複 / Vocabulary Overlap**: 詐欺文書には「相談」「心配」「夫」など、他のトピックと共通する語彙が含まれる / Fraud documents share common words like "consultation", "worry", and "husband" with other topics.
3. **KMeansの制約 / KMeans Constraints**: KMeansは全文書をクラスタに強制割り当てるため、outlierとして分類されない / KMeans forces all documents into specific clusters, preventing them from being classified as outliers.

### 詐欺文書の現在の分類 / Current Fraud Distribution
| トピック / Topic | 件数 / Count | 内容 / Content |
|---|---|---|
| Topic 0 (育児支援 / Parenting Support) | 10件 / 10 docs | 育児支援 + 詐欺混入 / Parenting support + fraud mixed |
| Topic 5 (離婚・浮気 / Divorce & Infidelity) | 2件 / 2 docs | 不倫関連 + 詐欺 / Infidelity-related + fraud |
| Topic -1 (外れ値 / Outliers) | 2件 / 2 docs | 短文 / Short texts |

### 2カテゴリ分類での対応 / 2-Category Classification
topic分析とは別に、`negative_classify.py` による2カテゴリ分類では：
Separately from the topic analysis, the 2-category classification by `negative_classify.py` shows:
- 詐欺文書は **category 0（負面）** として正しく分類されている
  Fraud documents are correctly classified as **category 0 (Negative)**.
- キーワードマッチング + SVM-RBF分類器により、全14件の詐欺文書が正しく識別
  All 14 fraud documents are properly identified via keyword matching + SVM-RBF classifier.
