#此代码为转化pts文件为npy文件，方便模型训练，务必安装以下的算法包#
#pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
#pip install numpy tqdm open3d

# =========================================================
# 描述: 自动读取所有 .pts 文件，按 8:2 划分训练/测试集，并生成成对的 .npy 文件。
# =========================================================

import numpy as np
import os
import glob
import random
from sklearn.utils import shuffle

# --- 配置参数 ---
# **请根据您的实际环境修改这些路径**
INPUT_DIR = '/kaggle/input/teatest/test01'  # 你的原始数据根目录 (包含 .pts 文件)
OUTPUT_DIR = '/kaggle/working/data/plant'  # 你的输出目录 (将生成 train/test 文件夹)
SPLIT_RATIO = 0.8  # 训练集比例 (80%)
PARTIAL_RATIO = 0.3  # 生成 partial 点云时保留的点数比例

# 设置随机种子以确保每次运行的拆分结果一致
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 目录准备 ---
TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'train')
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'test')
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# 1. 读取所有完整点云文件
all_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.pts')))
total_files = len(all_files)

if total_files == 0:
    print(f"❌ 错误: 在目录 {INPUT_DIR} 中未找到任何 *.pts 文件。请检查路径。")
    exit()

print(f"✅ 总文件数: {total_files}")

# 2. 自动拆分 8:2
num_train = int(total_files * SPLIT_RATIO)

# 打乱文件列表 (确保随机性)
random.shuffle(all_files)

train_files = all_files[:num_train]
test_files = all_files[num_train:]

print(f"分割完成: 训练集 ({len(train_files)} 文件), 测试集 ({len(test_files)} 文件)")


# 3. 核心生成函数
def generate_partial_cloud(complete_pts, complete_file, partial_ratio, output_dir, split):
    """
    加载完整点云，生成部分点云，并保存成对的 .npy 文件。
    """
    n_points = len(complete_pts)
    partial_n = int(n_points * partial_ratio)

    # 使用 sklearn.utils.shuffle 打乱点云并取前 partial_n 个点作为不完整点云
    # 注意: complete_pts 应该是 N x 3 数组
    partial_pts = shuffle(complete_pts, random_state=RANDOM_SEED)[:partial_n]

    base_name = os.path.basename(complete_file).replace('.pts', '')

    # 构造输出文件名: partial_train_filename.npy
    partial_filename = f'partial_{base_name}.npy'
    complete_filename = f'complete_{base_name}.npy'

    np.save(os.path.join(output_dir, partial_filename), partial_pts)
    np.save(os.path.join(output_dir, complete_filename), complete_pts)

    # print(f"  -> 生成 {split} {base_name}: partial ({partial_n}点), complete ({n_points}点)")


# 4. 执行生成
print("-" * 30)
print(f"开始生成训练集 pairs (保存在 {TRAIN_OUTPUT_DIR})...")
for i, complete_file in enumerate(train_files):
    try:
        # 加载文件，只取前三列 (X, Y, Z)
        complete_pts = np.loadtxt(complete_file, dtype=np.float32)[:, :3]
        generate_partial_cloud(complete_pts, complete_file, PARTIAL_RATIO, TRAIN_OUTPUT_DIR, 'train')
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个训练样本...")
    except Exception as e:
        print(f"❌ 警告: 处理训练文件 {os.path.basename(complete_file)} 时出错: {e}")

print("-" * 30)
print(f"开始生成测试集 pairs (保存在 {TEST_OUTPUT_DIR})...")
for i, complete_file in enumerate(test_files):
    try:
        complete_pts = np.loadtxt(complete_file, dtype=np.float32)[:, :3]
        generate_partial_cloud(complete_pts, complete_file, PARTIAL_RATIO, TEST_OUTPUT_DIR, 'test')
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个测试样本...")
    except Exception as e:
        print(f"❌ 警告: 处理测试文件 {os.path.basename(complete_file)} 时出错: {e}")

print("-" * 30)
print("✅ 所有数据生成和拆分完成！")
