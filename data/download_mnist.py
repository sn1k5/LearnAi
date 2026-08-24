"""下载经典手写数字识别数据集 MNIST 到本地。

MNIST 由 Yann LeCun 等人整理，包含 6 万张训练图 + 1 万张测试图，
每张为 28x28 的灰度手写数字（0~9）。官方提供 4 个 gzip 文件：
  train-images-idx3-ubyte.gz  /  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz   /  t10k-labels-idx1-ubyte.gz

本脚本使用标准库 urllib 直接从官方镜像下载，无需安装第三方下载库。
下载完成后解压为 .idx3-ubyte / .idx1-ubyte 原始字节文件，供 test_mnist.py 解析。
"""

import gzip
import os
import urllib.request

# 基于脚本所在目录解析默认数据目录，避免依赖当前工作目录 (cwd)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")

# 官方下载源（LeCun 维护的镜像）
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILE_NAMES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_mnist(data_dir=_DEFAULT_DATA_DIR, base_url=BASE_URL):
    """下载并解压 MNIST 四个文件到 data_dir。

    返回: 解压后的原始文件路径列表（.idx3-ubyte / .idx1-ubyte）。
    """
    os.makedirs(data_dir, exist_ok=True)
    raw_paths = []
    for fname in FILE_NAMES:
        gz_path = os.path.join(data_dir, fname)
        raw_name = fname.replace(".gz", "")
        raw_path = os.path.join(data_dir, raw_name)
        raw_paths.append(raw_path)

        # 若已存在解压后的原始文件，则跳过下载
        if os.path.exists(raw_path):
            print(f"[跳过] 已存在: {raw_path}")
            continue

        # 下载 .gz（若不存在）
        if not os.path.exists(gz_path):
            url = base_url + fname
            print(f"[下载] {url}")
            urllib.request.urlretrieve(url, gz_path)

        # 解压
        print(f"[解压] {gz_path} -> {raw_path}")
        with gzip.open(gz_path, "rb") as f_in, open(raw_path, "wb") as f_out:
            f_out.write(f_in.read())
        # 删除 .gz，只保留原始字节文件
        os.remove(gz_path)

    return raw_paths


if __name__ == "__main__":
    paths = download_mnist()
    print("\n下载完成，文件列表:")
    for p in paths:
        print("  ", p)
