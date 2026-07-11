"""
共通設定ファイル
全MLパラメータをここに集約。再現性確保のため random_state=42 を共通使用。
"""
from pathlib import Path

# ディレクトリ定義
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR = ROOT_DIR / "src"

# 感情分析パラメータ
SENTIMENT_BATCH_SIZE = 16        # 推論バッチサイズ
SENTIMENT_MAX_LENGTH = 512       # 最大トークン長

# トピックモデリングパラメータ
TOPIC_MIN_TOPIC_SIZE = 4         # 最小トピックサイズ
TOPIC_UMAP_N_NEIGHBORS = 15      # UMAP近傍数
TOPIC_UMAP_N_COMPONENTS = 2      # UMAP削減後次元数
TOPIC_UMAP_MIN_DIST = 0.1        # UMAP最小距離
TOPIC_UMAP_METRIC = "cosine"     # UMAP距離計測
TOPIC_RANDOM_SEED = 42           # 乱数シード
TOPIC_VECTORIZER_MAX_DF = 0.85   # 語彙の最大出現文書率
TOPIC_VECTORIZER_MIN_DF = 2      # 語彙の最小出現文書数
TOPIC_EMBEDDING_MODEL = "cl-nagoya/ruri-v3-310m"
TOPIC_EMBEDDING_BATCH_SIZE_CUDA = 64  # GPUバッチサイズ
TOPIC_EMBEDDING_BATCH_SIZE_CPU = 16   # CPUバッチサイズ

# クラスタリングパラメータ（UMAP + HDBSCAN）
CLUSTER_UMAP_N_COMPONENTS = 2
CLUSTER_UMAP_N_NEIGHBORS = 15
CLUSTER_UMAP_MIN_DIST = 0.0
CLUSTER_UMAP_METRIC = "cosine"
CLUSTER_RANDOM_SEED = 42
CLUSTER_HDBSCAN_MIN_CLUSTER_SIZE = 20
CLUSTER_HDBSCAN_MIN_SAMPLES = 5

# KMeansパラメータ
KMEANS_RANDOM_SEED = 42
KMEANS_N_INIT = 10
KMEANS_K_RANGE = range(2, 11)
