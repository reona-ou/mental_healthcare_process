# Mental Healthcare Process

---

## 📌 Overview
本研究は、メンタルケアチャットボットの開発を目的とする。
あらかじめ収集した対話ログと専門家の評価を統合したデータセットを用い、自然言語処理（NLP）を用いてメンタルヘルスに悪影響を及ぼす危険性がある要因を特定する。
この要因に基づき、共感性と安全性を備えたモデルを構築する。
またモデルをAWS Bedrockに実装し、LLM-as-a-judgeを使って、共感性、医学的正確性、安全性（医療行為の制限）、有害性（差別・攻撃性）の排除、AI依存の防止、シカファンシーの回避という6つの指標の評価を行う。
これらに基づき、高い信頼性と安全性を保証した心理支援チャットボットを実装し、ユーザーテストを行い、その結果をまとめる。
---

## 📂 Repository Structure

```text
├── data/                  # Data directory (contains sample or processed data)
├── models/                # Local model checkpoints directory (GIT IGNORED)
├── src/                   # Core source code for training and evaluation
│   ├── download_models.py # Script to download and cache Hugging Face models locally├── README.md              # Project description and guide
├── requirements.txt       # Environment dependencies
└── LICENSE                # MIT License