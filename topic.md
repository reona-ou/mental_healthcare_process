# Topic Modeling Results

基于 BERTopic 话题建模的结果（ruri-v3-310m 嵌入模型 + KMeans聚类 + 半监督模式 + MMR），共识别出 7 个有效话题（Topic 0-6），26 篇短文本标记为离群（Topic -1）。

---

### Topic -1: Short Texts / 短文本 / 短いテキスト (26件)

* **主题说明**:
  * **中**: tokenized 后少于 2 个词的短文本，无法进行有效的话题分类。
  * **日**: 分詞後に2語未満の短いテキスト。有効なトピック分類が不可能。
  * **EN**: Short texts with fewer than 2 tokens after preprocessing, insufficient for topic classification.

---

### Topic 0: 育児支援・地域 / 育儿支援与地域 / Parenting Support & Community (30件)

* **核心关键词**: `地域`, `保健`, `心配`, `育児`, `相談`, `赤ちゃん`
* **主题说明**:
  * **中**: 通过地域支援机构（保健中心等）寻求育儿帮助。
  * **日**: 地域の支援機関（保健センター等）を通じて育児支援を求める。
  * **EN**: Seeking parenting help through community support organizations such as health centers.

---

### Topic 1: 育児・離婚 / 育儿与离婚 / Parenting & Divorce (25件)

* **核心关键词**: `赤ちゃん`, `離婚`, `家事`, `育児`, `きつい`, `話し掛ける`
* **主题说明**:
  * **中**: 育儿压力导致的夫妻矛盾与离婚考虑。
  * **日**: 育児ストレスによる夫婦間の矛盾と離婚の検討。
  * **EN**: Marital conflicts and divorce considerations stemming from parenting stress.

---

### Topic 2: 産後・授乳 / 产后哺乳 / Postpartum & Breastfeeding (24件)

* **核心关键词**: `おっぱい`, `育児`, `寝る`, `しんどい`, `ワンオペ`, `安心`
* **主题说明**:
  * **中**: 产后哺乳、睡眠问题、母婴健康相关的焦虑与安心。
  * **日**: 産後・授乳・睡眠に関する不安と安心。
  * **EN**: Postpartum breastfeeding, sleep issues, and maternal anxiety/relief.

---

### Topic 3: 流産・妊娠 / 流产妊娠 / Miscarriage & Pregnancy (22件)

* **核心关键词**: `流産`, `妊娠`, `自分`, `辛い`, `落ち込む`, `嬉しい`
* **主题说明**:
  * **中**: 流产、妊娠相关的情感波动与个人经历。
  * **日**: 流産・妊娠に関する感情の揺れと個人の経験。
  * **EN**: Emotional fluctuations and personal experiences related to miscarriage and pregnancy.

---

### Topic 4: 人間関係・相談 / 人际关系咨询 / Interpersonal & Consultation (21件)

* **核心关键词**: `相談`, `苦手`, `連絡`, `しんどい`, `無視`, `友達`
* **主题说明**:
  * **中**: 人际关系困扰、寻求咨询帮助、沟通问题。
  * **日**: 人間関係の悩み、相談の求め、コミュニケーション問題。
  * **EN**: Interpersonal difficulties, seeking consultation, and communication issues.

---

### Topic 5: 離婚・浮気 / 离婚出轨 / Divorce & Infidelity (19件)

* **核心关键词**: `離婚`, `浮気`, `浮気相手`, `考える`, `冷たい`, `視野`
* **主题说明**:
  * **中**: 离婚决策、出轨问题、夫妻关系破裂。
  * **日**: 離婚の意思決定、不倫問題、夫婦関係の崩壊。
  * **EN**: Divorce decisions, infidelity issues, and marital breakdown.

---

### Topic 6: 産後睡眠 / 产后睡眠 / Postpartum Sleep (15件)

* **核心关键词**: `寝る`, `箇月`, `安心`, `しんどい`, `授乳`, `夜中`
* **主题说明**:
  * **中**: 产后睡眠障碍、哺乳期健康问题。
  * **日**: 産後の睡眠障害、授乳期の健康問題。
  * **EN**: Postpartum sleep disorders and breastfeeding health issues.

---

## 话题分布统计 / Topic Distribution / トピック分布

| 话题 / Topic | 数量 | 占比 | 主题 / Theme |
|------|------|------|------|
| -1 (短文本) | 26 | 14.3% | 短文本 / 短いテキスト / Short Texts |
| 0 | 30 | 16.5% | 育儿支援地域 / 育児支援・地域 / Parenting Support & Community |
| 1 | 25 | 13.7% | 育儿离婚 / 育児・離婚 / Parenting & Divorce |
| 2 | 24 | 13.2% | 产后哺乳 / 産後・授乳 / Postpartum & Breastfeeding |
| 3 | 22 | 12.1% | 流产妊娠 / 流産・妊娠 / Miscarriage & Pregnancy |
| 4 | 21 | 11.5% | 人际关系咨询 / 人間関係・相談 / Interpersonal & Consultation |
| 5 | 19 | 10.4% | 离婚出轨 / 離婚・浮気 / Divorce & Infidelity |
| 6 | 15 | 8.2% | 产后睡眠 / 産後睡眠 / Postpartum Sleep |

---

## 改进说明

相比之前的结果，本次优化解决了以下问题：

1. **-1 outlier 问题**：从 49 篇（26.9%）降至 26 篇（14.3%），且全部是合理的短文本
2. **topic 内容混乱**：消除了伪影词（レン、仕舞う等），topic 关键词更加连贯
3. **topic 分布均匀**：从不均匀分布（43/9/12/16/15/12/9）改善为均匀分布（30/25/24/22/21/19/15）
4. **短文本过滤**：tokenized 后少于 2 个词的短文本不参与聚类，避免污染 topic
