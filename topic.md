# トピックモデリング結果 / Topic Modeling Results

## 概要 / Overview

ruri-v3-310m 埋め込みモデル（`トピック: ` プレフィックス付き）+ KMeans クラスタリング + 半教師シード誘導 + MMR + 統計距離フィルタにより、7つの有効トピックを識別。37件の文書が外れ値として分類された。

| 指標 | 値 | 説明 |
|---|---|---|
| 入力データ | combined userInput | 2チャットボットのユーザー入力統合 / Combined user inputs from 2 chatbots |
| 総文書数 | 182件 | 全セッション数 / Total sessions |
| 有効トピック数 | 7 | 有効な話題数 / Valid topics |
| 外れ値 | 37件 (20.3%) | 短文 + 統計異常 / Short texts + statistical outliers |
| トピックカバレッジ | 79.7% | トピックに分類された文書の割合 / Topic coverage rate |

---

## 処理パイプライン / Pipeline

```
userInput (182件)
    ↓
fugashi で形態素解析（名詞・動詞・形容詞のみ）
    ↓
短文テキストフィルタ（分詞後 < 2語 → -1）: 26件除外
    ↓
ruri-v3-310m で 768次元埋め込み生成（元テキスト +「トピック: 」プレフィックス）
    ↓
UMAP 次元削減（768 → 2次元）
├── n_neighbors=15, min_dist=0.1, metric=cosine
    ↓
KMeans クラスタリング（7クラスタ）
├── n_init=10, random_state=42
    ↓
BERTopic トピック表現（分詞済みテキストを使用）
├── シードトピック誘導（8カテゴリ）
├── MMR (diversity=0.5) でキーワード多様性向上
└── CountVectorizer (ngram_range=(1,2))
    ↓
統計距離フィルタ（mean + 1.5 * std → -1）: 11件除外
    ↓
結果保存
```

---

## 外れ値（Topic -1）: 37件 (20.3%) / Outliers: 37 docs (20.3%)

### 内訳 / Breakdown
- **短文テキスト / Short texts**: 26件 — 分詞後2語未満の文書 / 26 docs — Documents with fewer than 2 tokens
- **統計距離異常 / Statistical outliers**: 11件 — トピック中心から mean + 1.5 * std を超える文書 / 11 docs — Documents exceeding mean + 1.5 * std from topic centroid

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

### Topic 0: 離婚・浮気 / Divorce & Infidelity (26件 / 26 docs)

**キーワード / Keywords**: 離婚 (divorce), 浮気 (affair), 考える (consider), 変わる (change), 産後 (postpartum), 箇月 (months), きつい (harsh), 浮気相手 (affair partner), 家事 (housework)

**内容 / Content**:
離婚の意思決定、不倫問題、夫婦関係の崩壊。浮気相手との関係、夫の冷たい態度、育児ストレスによる離婚の検討など。
Decision-making on divorce, infidelity issues, and the breakdown of marital relations. Includes relationships with affair partners, partner's cold attitudes, and considering divorce due to parenting stress.

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

### Topic 1: 人間関係・支援 / Interpersonal & Support (26件 / 26 docs)

**キーワード / Keywords**: 貰う (receive), 両親 (parents), 一緒 (together), 連絡 (contact), 心配 (worry), 忙しい (busy), 時間 (time), 友達 (friend), 不安 (anxiety)

**内容 / Content**:
人間関係の悩み、相談へのハードル、コミュニケーションの問題。友人・家族への相談の困難さ、夫との関係問題など。
Interpersonal relationship worries, barriers to seeking advice, and communication issues. Includes difficulties in talking to friends or family, and relationship problems with husbands.

**具体例 / Examples**:
> 「友達もいないし、実家の両親は年を取っているから相談できません。」
> "I don't have any friends, and my parents back home are elderly, so I can't consult them."
>
> 「人に話したことはないかな。話したら、相手を困らせてしまうし…」
> "I don't think I've ever spoken to anyone about it. If I do, I feel like I'd just trouble them..."
>
> 「相談するのが苦手なのです。」
> "I am not good at consulting or opening up to others."

---

### Topic 2: 相談・苦手 / Consultation & Difficulty (20件 / 20 docs)

**キーワード / Keywords**: 苦手 (not good at), 欲しい (want), 相談出来る (can consult), 伝える (convey), 手伝う (help), 友達 (friend), 無視 (ignore)

**内容 / Content**:
相談することへの抵抗感、コミュニケーションの問題。夫への不満、相談へのハードルなど。
Reluctance to consult others, communication issues. Includes dissatisfaction with husbands and barriers to seeking advice.

**具体例 / Examples**:
> 「人に話したことはないかな。話したら、相手を困らせてしまうし、自分もうまく反応できなくてその人のことを嫌いになりたくない」
> "I don't think I've ever spoken to anyone about it. If I do, I feel like I'd just trouble them..."
>
> 「相談するのが苦手なのです。」
> "I am not good at consulting or opening up to others."
>
> 「行政にどのようにそうだんしたらよいでしょう」
> "How should I consult with the municipal government?"

---

### Topic 3: 産後・授乳 / Postpartum & Breastfeeding (20件 / 20 docs)

**キーワード / Keywords**: おっぱい (breast), 眠る (sleep), 寝る (sleep), 安心 (relief), 母乳 (breast milk), 箇月 (months), 最近 (recently), 時間 (time), しんどい (tough)

**内容 / Content**:
産後の授乳、睡眠問題、母子の健康に関する不安と安心。母乳の量への不安、赤ちゃんの睡眠リズムへの対応など。
Anxieties and reliefs regarding postpartum breastfeeding, sleep issues, and maternal/infant health. Includes concerns about breast milk supply and dealing with the baby's sleep rhythm.

**具体例 / Examples**:
> 「入院中は母乳もたくさん出てたのに、最近出なくなりました。おっぱいも張っていないようです。」
> "While I was hospitalized, I had plenty of breast milk, but recently it has stopped coming out. My breasts don't even feel engorged anymore."
>
> 「おっぱい飲んですやすや寝てるのを見ると、安心する。」
> "Seeing my baby drinking milk and sleeping peacefully brings me so much relief."
>
> 「たまひよアプリで子どもの睡眠時間を測ってるけど、生後2カ月だけど、夜中3時間で起きてしまいます！」
> "I'm tracking my child's sleep time with the Tamahiyo app. Even though they are 2 months old, they still wake up every 3 hours in the middle of the night!"

---

### Topic 4: 育児支援・地域 / Parenting Support & Community (19件 / 19 docs)

**キーワード / Keywords**: 育児 (parenting), 地域 (community), 保健 (health center), 最近 (recently), 保健センター (health center), センター (center), 感じ (feeling), 名前 (name), 予定 (schedule)

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

### Topic 5: 流産・妊娠 / Miscarriage & Pregnancy (18件 / 18 docs)

**キーワード / Keywords**: 流産 (miscarriage), 妊娠 (pregnancy), 辛い (painful), 嬉しい (happy), 知れる (know), 自分 (myself), 昨日 (yesterday), 仕事 (work), 考える (consider)

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

### Topic 6: 育児ストレス / Parenting Stress (16件 / 16 docs)

**キーワード / Keywords**: しんどい (tough), 駄目 (no good), 可愛い (cute), 頑張る (do one's best), 育児 (parenting), 母乳 (breast milk), ベビー (baby), 死ぬ (die), 頼る (rely)

**内容 / Content**:
育児ストレス、孤独感、追い詰められた気持ち。一人で抱える育児の苦しみ、死にたいという気持ちなど。
Parenting stress, loneliness, and feeling overwhelmed. Includes the suffering of raising children alone and thoughts of wanting to die.

**具体例 / Examples**:
> 「しんどい こどもいるし死ねないし死なないけど だれにも言えない 夫もできる範囲でやってくれてるし 近くに頼れる親とか親戚いないし 友達もいないし」
> "It's tough. I have kids so I can't die, but I can't tell anyone. My husband does what he can, but I have no parents or relatives nearby, and no friends."
>
> 「や、完璧にはできてないよ ダメダメだよ ダメダメなことにもしんどくなるよ、、 夫は育児もしながら仕事も頑張ってるのに 家にいる私、全然できてない」
> "No, I'm not doing it perfectly. I'm a mess, and I get tired of being a mess... My husband is working hard at both parenting and his job, but I'm at home and can't do anything."
>
> 「ひとりでやってるのがつらいから、むしろ本当に1人になったほうが諦めつくかなって 夜考えるんだよねー 誰かにそばにいて欲しい」
> "Doing it alone is so hard that I sometimes think maybe it would be easier to truly be alone. I think about it at night... I just want someone by my side."

---

## トピック分布 / Topic Distribution

| トピック / Topic | 件数 / Count | 割合 / Percentage | 内容 / Content |
|---|---|---|---|
| -1 (外れ値 / Outliers) | 37 | 20.3% | 短文 + 統計異常 / Short texts + statistical outliers |
| 0 | 26 | 14.3% | 離婚・浮気 / Divorce & Infidelity |
| 1 | 26 | 14.3% | 人間関係・支援 / Interpersonal & Support |
| 2 | 20 | 11.0% | 相談・苦手 / Consultation & Difficulty |
| 3 | 20 | 11.0% | 産後・授乳 / Postpartum & Breastfeeding |
| 4 | 19 | 10.4% | 育児支援・地域 / Parenting Support & Community |
| 5 | 18 | 9.9% | 流産・妊娠 / Miscarriage & Pregnancy |
| 6 | 16 | 8.8% | 育児ストレス / Parenting Stress |

---

## トピック分析の改善点 / Improvements

### プレフィックス追加の効果 / Effect of Adding Prefix
ruri-v3モデルに「トピック: 」プレフィックスを追加したことで、以下の改善が見られた：
Adding the "トピック: " prefix to the ruri-v3 model showed the following improvements:

1. **トピック分離の向上 / Improved Topic Separation**: 離婚・浮気トピックが明確に分離された / Divorce & Infidelity topic is now clearly separated
2. **育児ストレスの抽出 / Parenting Stress Extraction**: しんどい・死にたい等の感情が独立したトピックとして抽出された / Emotions like "tough" and "want to die" are now extracted as a separate topic
3. **詐欺文書の分類 / Fraud Classification**: 詐欺関連文書は主にTopic 1（人間関係・支援）に分類 / Fraud-related documents are mainly classified in Topic 1 (Interpersonal & Support)

### 詐欺関連の分類について / About Fraud Classification

#### 現状 / Current Status
詐欺関連文書（14件）は、独立したトピックとして分類されていない。理由は以下の通り：
Fraud-related documents (14 cases) are not classified as an independent topic. The reasons are as follows:

1. **データ量不足 / Insufficient Data**: 詐欺文書は全体の7.7%（14件/182件）と少なく、独立したクラスタを形成するのに不十分 / Fraud documents are only 7.7% (14/182) of the total, which is too few to form an independent cluster.
2. **語彙の重複 / Vocabulary Overlap**: 詐欺文書には「相談」「心配」「夫」など、他のトピックと共通する語彙が含まれる / Fraud documents share common words like "consultation", "worry", and "husband" with other topics.
3. **KMeansの制約 / KMeans Constraints**: KMeansは全文書をクラスタに強制割り当てるため、outlierとして分類されない / KMeans forces all documents into specific clusters, preventing them from being classified as outliers.

#### 詐欺文書の現在の分類 / Current Fraud Distribution
| トピック / Topic | 件数 / Count | 内容 / Content |
|---|---|---|
| Topic 1 (人間関係・支援 / Interpersonal & Support) | 8件 / 8 docs | 人間関係 + 詐欺混入 / Interpersonal + fraud mixed |
| Topic 4 (育児支援・地域 / Parenting Support & Community) | 2件 / 2 docs | ロマンス詐欺 / Romance fraud |
| Topic -1 (外れ値 / Outliers) | 2件 / 2 docs | フィッシング詐欺 + 詐欺被害の疑い / Phishing + suspected fraud |

#### 詐欺関連文書一覧 / Fraud-Related Documents
| Session ID | Topic | テキスト / Text |
|---|---|---|
| 0 | 1 | フィッシング詐欺にあってしまって、請求が来るのではないか不安でどうしよう |
| 16 | 1 | 最近インスタで友達になった友人から、ベビーモデル応募しない？って誘われたんだ |
| 17 | 1 | 雑誌のモデルみたいなんだけど、掲載料が少しかかるみたい |
| 18 | 1 | そうなの？ 友だちはそんな事言わなかったよ。私、騙されてる？ |
| 27 | 1 | このままじゃダメだと思う。登録するだけで数万円手に入るっていうサイトを紹介してもらった |
| 33 | 1 | お金の心配も、月々数万円もらえるっていう案件のバイトでどうにかしようともう |
| 57 | -1 | フィッシング詐欺にもあってしまって、もうだめなことばかり |
| 97 | 4 | 男性なの。退役軍人で今度会いに来てくれるのだけど、色々とありお金をお貸ししてます |
| 100 | 4 | ペン先生　Facebookで知り合った男性が、退役軍人で今度会いに来てくれるのだけど |
| 145 | 1 | 夫と話してみようかな...。でも仕事が忙しそう。友人から、ベンチャー企業立ち上げに誘われているみたいで |
| 146 | -1 | 詐欺じゃないよね？ |
| 147 | 1 | 夫が上手い話に騙されてないかなーって。そうだよね、やっぱり詐欺かもしれないし | 

#### 2カテゴリ分類での対応 / 2-Category Classification
topic分析とは別に、`negative_classify.py` による2カテゴリ分類では：
Separately from the topic analysis, the 2-category classification by `negative_classify.py` shows:
- 詐欺文書は **category 0（負面）** として正しく分類されている
  Fraud documents are correctly classified as **category 0 (Negative)**.
- キーワードマッチング + SVM-RBF分類器により、全14件の詐欺文書が正しく識別
  All 14 fraud documents are properly identified via keyword matching + SVM-RBF classifier.

### 分類ミス / Misclassifications

以下の文書は分類が不適切です：
The following documents are misclassified:

| Session ID | 現在のTopic / Current Topic | 本来のTopic / Correct Topic | 理由 / Reason |
|---|---|---|---|
| 56 | Topic 1 (人間関係) | Topic 6 (育児ストレス) | 「子供を放り出してしまうんじゃないか」は育児ストレス / Parenting stress |
| 174 | Topic 1 (人間関係) | Topic -1 (外れ値) | 「昨日外出先から帰ったらいなくって」は短文 / Short text |
| 176 | Topic 1 (人間関係) | Topic 0 (離婚・浮気) | 「警察にはもう少し待って連絡」は夫の行方不明 / Husband missing |
| 177 | Topic 1 (人間関係) | Topic 0 (離婚・浮気) | 「警察沙汰にするのは怖い」は夫の行方不明 / Husband missing |

