import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 依赖库运行测试 / 依存ライブラリ動作テスト
# 用于验证requirements.txt中所有库是否能正常运行
# requirements.txtの全ライブラリが正常に動作するかを確認するためのテスト


import sys

def test_library(name, import_name=None):
    """测试单个库是否能正常导入 / 個々のライブラリが正常にインポートできるかをテスト"""
    try:
        mod = __import__(import_name or name)
        version = getattr(mod, '__version__', 'バージョン情報なし')
        print(f"  [OK] {name}: {version}")
        return True
    except Exception as e:
        print(f"  [NG] {name}: 导入失败 / インポート失敗 - {e}")
        return False

print("依赖库测试开始 / 依存ライブラリテスト開始")

results = {}

# PyTorch & CUDA
print("\n[PyTorch & CUDA]")
import torch
results['torch'] = test_library('torch')
print(torch.__version__)
print(torch.version.cuda)
print(f"  CUDA可用性 / CUDA利用可能: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Transformers
print("\n[Transformers]")
results['transformers'] = test_library('transformers')

# Sentence Transformers
print("\n[Sentence Transformers]")
results['sentence_transformers'] = test_library('sentence_transformers')

# Data Science
print("\n[データサイエンス / 数据科学]")
results['numpy'] = test_library('numpy')
results['pandas'] = test_library('pandas')
results['scikit_learn'] = test_library('scikit_learn', 'sklearn')

# Visualization
print("\n[可視化 / 可视化]")
results['plotly'] = test_library('plotly')

# Topic Modeling
print("\n[トピックモデリング / 主题建模]")
results['bertopic'] = test_library('bertopic')
results['umap_learn'] = test_library('umap_learn', 'umap')

# Japanese Processing
print("\n[日本語処理 / 日语处理]")
results['fugashi'] = test_library('fugashi')
results['emoji'] = test_library('emoji')

# Summary
passed = sum(1 for v in results.values() if v)
total = len(results)
print(f"测试结果 / テスト結果: {passed}/{total} 通过 / 成功")

if passed == total:
    print("所有依赖库运行正常 / 全てのライブラリが正常に動作しています")
else:
    failed = [k for k, v in results.items() if not v]
    print(f"失败的库 / 失敗したライブラリ: {', '.join(failed)}")
    sys.exit(1)
