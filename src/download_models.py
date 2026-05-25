"""
模型下载脚本 / モデルダウンロードスクリプト

将项目中所有需要的预训练模型下载到 models/ 目录。
プロジェクトで必要な事前学習済みモデルをすべて models/ ディレクトリにダウンロードする。

使用方法 / 使い方:
    python src/download_models.py

下载的模型 / ダウンロードされるモデル:
    1. neuralnaut/deberta-wrime-emotions   (情感分析 / 感情分析)  - sentiment_analysis.py
    2. tohoku-nlp/bert-base-japanese-v3    (日语BERT / 日本語BERT) - topic.py
"""

import sys
from pathlib import Path
import config

# 项目路径 / プロジェクトパス
SRC_DIR = config.SRC_DIR
MODELS_DIR = config.MODELS_DIR

# 模型列表 / モデル一覧
MODELS = [
    {
        "repo_id": "neuralnaut/deberta-wrime-emotions",
        "description": "WRIME微調DeBERTa (8種感情分析) / WRIME微調整DeBERTa (8種感情分析)",
        "used_by": f"{SRC_DIR / 'sentiment_analysis.py'}",
    },
    {
        "repo_id": "tohoku-nlp/bert-base-japanese-v3",
        "description": "日语BERT Base V3 (トピックモデリング埋め込み) / 日本語BERT Base V3 (トピックモデリング埋め込み)",
        "used_by": f"{SRC_DIR / 'topic.py'}",
    },
]


def download_model(repo_id: str, local_dir: Path) -> None:
    """
    使用 huggingface_hub 下载模型到指定目录。
    huggingface_hub を使用してモデルを指定ディレクトリにダウンロードする。
    """
    from huggingface_hub import snapshot_download

    model_name = repo_id.split("/")[-1]
    save_dir = local_dir / model_name

    print(f"  保存先: {save_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(save_dir),
        local_dir_use_symlinks=False,
    )

    print(f"下载完成 / ダウンロード完了: {repo_id}\n")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("模型下载开始 / モデルダウンロード開始")
    print(f"保存目录 / 保存先: {MODELS_DIR}")

    for i, model_info in enumerate(MODELS, 1):
        repo_id = model_info["repo_id"]
        desc = model_info["description"]
        used_by = model_info["used_by"]

        print(f"\n[{i}/{len(MODELS)}] {desc}")
        print(f"仓库 / リポジトリ: {repo_id}")
        print(f"使用文件 / 使用ファイル: {used_by}")

        try:
            download_model(repo_id, MODELS_DIR)
        except Exception as e:
            print(f"下载失败 / ダウンロード失敗: {e}")
            print(f"请检查网络连接或HuggingFace访问权限。")
            print(f"ネットワーク接続またはHuggingFaceのアクセス権限を確認してください。")
            sys.exit(1)

    print("所有模型下载完成！ / 全モデルのダウンロード完了！")
    print(f"模型保存位置 / モデル保存先: {MODELS_DIR}")

    # 显示已下载的模型目录 / ダウンロード済みモデルディレクトリを表示
    print("\n已下载的模型 / ダウンロード済みモデル:")
    for item in sorted(MODELS_DIR.iterdir()):
        if item.is_dir():
            size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 * 1024)
            print(f"  📁 {item.name}/ ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()