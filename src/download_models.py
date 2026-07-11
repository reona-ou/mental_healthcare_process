"""
モデルダウンロードスクリプト
プロジェクトで必要な事前学習済みモデルを models/ ディレクトリにダウンロードする。

使い方:
    python src/download_models.py

ダウンロードされるモデル:
    1. neuralnaut/deberta-wrime-emotions   (感情分析)  - sentiment_analysis.py
    2. cl-nagoya/ruri-v3-310m              (テキスト埋め込み) - topic.py
"""
import sys
from pathlib import Path
import config

SRC_DIR = config.SRC_DIR
MODELS_DIR = config.MODELS_DIR

# ダウンロード対象モデル一覧
MODELS = [
    {
        "repo_id": "neuralnaut/deberta-wrime-emotions",
        "description": "WRIME微調整DeBERTa (8種感情分析)",
        "used_by": f"{SRC_DIR / 'sentiment_analysis.py'}",
    },
    {
        "repo_id": "cl-nagoya/ruri-v3-310m",
        "description": "ruri-v3-310m (日本語テキスト埋め込み)",
        "used_by": f"{SRC_DIR / 'topic.py'}",
    },
]


def download_model(repo_id: str, local_dir: Path) -> None:
    """huggingface_hub を使用してモデルをダウンロード"""
    from huggingface_hub import snapshot_download
    model_name = repo_id.split("/")[-1]
    save_dir = local_dir / model_name
    print(f"  保存先: {save_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(save_dir), local_dir_use_symlinks=False)
    print(f"ダウンロード完了: {repo_id}\n")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("モデルダウンロード開始")
    print(f"保存先: {MODELS_DIR}")

    for i, model_info in enumerate(MODELS, 1):
        repo_id = model_info["repo_id"]
        desc = model_info["description"]
        used_by = model_info["used_by"]
        print(f"\n[{i}/{len(MODELS)}] {desc}")
        print(f"リポジトリ: {repo_id}")
        print(f"使用ファイル: {used_by}")
        try:
            download_model(repo_id, MODELS_DIR)
        except Exception as e:
            print(f"ダウンロード失敗: {e}")
            print(f"ネットワーク接続またはHuggingFaceのアクセス権限を確認してください。")
            sys.exit(1)

    print("全モデルのダウンロード完了！")
    print(f"モデル保存先: {MODELS_DIR}")

    print("\nダウンロード済みモデル:")
    for item in sorted(MODELS_DIR.iterdir()):
        if item.is_dir():
            size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 * 1024)
            print(f"  {item.name}/ ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
