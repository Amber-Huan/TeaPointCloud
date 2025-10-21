#此为正式训练文件，请替换对应的文件地址，可能需要克隆仓库
# 脚本名称: train_pfnet_plant_paired.py
# 描述: 适用于植物点云补全的 PF-Net 简化版训练脚本 (基于预先成对的数据集)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import re  # 引入正则表达式库

# --- 1. 配置参数 ---
# **请根据您的实际情况修改以下路径和参数**
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本运行目录

# 您的点云数据根目录 (假设 /data/plant/train 位于此处)
DATA_ROOT = os.path.join(BASE_DIR, 'data')
TRAIN_ROOT = os.path.join(DATA_ROOT, 'plant/train')
TEST_ROOT = os.path.join(DATA_ROOT, 'plant/test')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'PFNet_Outputs')

REPO_URL = "https://github.com/zztianzz/PF-Net-Point-Fractal-Network.git"
CLONE_DIR = os.path.join(BASE_DIR, "PFNet_Source")

ORIGINAL_POINTS = 2048  # 完整点云的点数（GT）
INPUT_KEEP = 256  # 不完整点云的点数（输入）
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------
# --- 2. 环境和仓库准备 ---
# ---------------------------------------------------------

def prepare_environment():
    """检查目录并克隆 PF-Net 仓库。"""
    print(f"--- 环境准备 ---")
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"✅ 创建输出目录: {MODEL_SAVE_DIR}")

    if not os.path.exists(CLONE_DIR):
        print(f"正在克隆 PF-Net 仓库到: {CLONE_DIR}")
        try:
            # 使用 git clone --depth 1 快速克隆，但如果遇到问题可以移除 --depth 1
            subprocess.run(['git', 'clone', REPO_URL, CLONE_DIR, '--depth', '1'], check=True)
            print("✅ 仓库克隆成功！")
        except subprocess.CalledProcessError as e:
            print(f"❌ Git 克隆失败，请手动执行 git clone {REPO_URL} {CLONE_DIR}")
    else:
        print("✅ 仓库已存在，跳过克隆。")

    print(f"使用设备: {DEVICE}")
    print(f"训练数据路径: {TRAIN_ROOT}")
    print(f"测试数据路径: {TEST_ROOT}")
    print("-" * 20)


# ---------------------------------------------------------
# --- 3. 核心组件 (损失、模型、数据加载器) ---
# ---------------------------------------------------------

class ChamferDistance(nn.Module):
    """用于点云重构的 Chamfer Distance 损失函数。"""

    def forward(self, pred, gt):
        dist = torch.cdist(pred, gt, p=2)
        dist_forward = dist.min(dim=2)[0].mean()
        dist_backward = dist.min(dim=1)[0].mean()
        return dist_forward + dist_backward


class PFNetGenerator(nn.Module):
    """简化的 PointNet-like 骨干作为生成器。"""

    def __init__(self, output_points=ORIGINAL_POINTS):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.dec = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, output_points * 3)
        )
        self.output_points = output_points

    def forward(self, x):
        feat = self.enc(x).squeeze(-1)
        pred_flat = self.dec(feat)
        return pred_flat.view(-1, self.output_points, 3)


class PairedPlantCompletionDataset(Dataset):
    """
    更新的数据集加载器：直接加载预先成对的 partial_*.npy 和 complete_*.npy 文件。
    """

    def __init__(self, root_dir, complete_points=ORIGINAL_POINTS, partial_points=INPUT_KEEP):
        self.root_dir = root_dir
        self.complete_points = complete_points
        self.partial_points = partial_points

        # 匹配所有 complete_*.npy 文件
        complete_files = sorted(glob.glob(os.path.join(self.root_dir, 'complete_*.npy')))

        self.paired_files = []
        if not complete_files:
            print(f"❌ 错误: 在路径 {self.root_dir} 未找到任何 complete_*.npy 文件。")

        # 建立 complete 和 partial 文件的配对
        for c_file in complete_files:
            # 从文件名中提取编号 (例如 'train_1' 或 'test_8')
            match = re.search(r'(complete|partial)_([a-z]+_\d+)\.npy', os.path.basename(c_file))
            if match:
                identifier = match.group(2)
                p_filename = f'partial_{identifier}.npy'
                p_file = os.path.join(self.root_dir, p_filename)

                if os.path.exists(p_file):
                    self.paired_files.append((c_file, p_file))
                # else:
                #     print(f"⚠️ 警告: 找不到对应的 partial 文件 {p_filename}")

        if not self.paired_files:
            print("🔍 未找到有效的成对数据文件，将创建虚拟数据以确保脚本可运行...")
            self._create_dummy_files(root_dir)
            complete_files = sorted(glob.glob(os.path.join(self.root_dir, 'complete_*.npy')))
            for c_file in complete_files:
                match = re.search(r'(complete)_(\d+)\.npy', os.path.basename(c_file))
                if match:
                    identifier = match.group(2)
                    p_file = os.path.join(self.root_dir, f'partial_{identifier}.npy')
                    if os.path.exists(p_file):
                        self.paired_files.append((c_file, p_file))

        print(f"✅ Dataset: 找到 {len(self.paired_files)} 对成对样本。")

    def _create_dummy_files(self, root_dir):
        """创建随机的 .npy 文件用于测试（与之前的虚拟数据逻辑相似）。"""
        if not os.path.exists(root_dir): os.makedirs(root_dir)
        count = 10  # 仅创建 10 对用于演示
        for i in range(1, count + 1):
            # 完整点云 (2048, 3)
            c_data = np.random.rand(self.complete_points, 3).astype(np.float32)
            # 不完整点云 (256, 3)
            p_data = c_data[:self.partial_points] + np.random.rand(self.partial_points, 3) * 0.1

            np.save(os.path.join(root_dir, f'complete_{i}.npy'), c_data)
            np.save(os.path.join(root_dir, f'partial_{i}.npy'), p_data)

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        c_path, p_path = self.paired_files[idx]

        # 加载数据
        complete_cloud = np.load(c_path)[:, :3].astype(np.float32)
        partial_cloud = np.load(p_path)[:, :3].astype(np.float32)

        # 确保点数匹配，如果不匹配则进行重新采样/截断 (防止崩溃)
        if complete_cloud.shape[0] != self.complete_points:
            c_indices = np.random.choice(complete_cloud.shape[0], self.complete_points, replace=True)
            complete_cloud = complete_cloud[c_indices, :]

        if partial_cloud.shape[0] != self.partial_points:
            p_indices = np.random.choice(partial_cloud.shape[0], self.partial_points, replace=True)
            partial_cloud = partial_cloud[p_indices, :]

        # 归一化/中心化 (基于完整点云的统计信息)
        center = complete_cloud.mean(axis=0)
        complete_cloud -= center
        partial_cloud -= center  # 对 partial 也使用相同的中心

        max_dist = np.max(np.sqrt(np.sum(complete_cloud ** 2, axis=1)))
        if max_dist > 1e-6:
            complete_cloud /= max_dist
            partial_cloud /= max_dist

        return (torch.from_numpy(partial_cloud),
                torch.from_numpy(complete_cloud))


# ---------------------------------------------------------
# --- 4. 训练主函数 (与之前逻辑相同) ---
# ---------------------------------------------------------

def train_model():
    """主训练流程函数。"""

    train_dataset = PairedPlantCompletionDataset(TRAIN_ROOT)
    test_dataset = PairedPlantCompletionDataset(TEST_ROOT)

    # 减少 num_workers 以避免 PyCharm/Windows 环境下的常见问题
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)

    netG = PFNetGenerator().to(DEVICE)
    optimizer = torch.optim.Adam(netG.parameters(), lr=LR)
    cd_loss_fn = ChamferDistance()

    train_losses = []
    test_losses = []

    print(f"\n--- 开始训练 {EPOCHS} 个 Epoch ---")

    for epoch in range(EPOCHS):
        netG.train()
        epoch_train_loss = 0.0

        # 训练阶段
        for partial, complete_gt in train_loader:
            partial = partial.to(DEVICE).transpose(1, 2)
            complete_gt = complete_gt.to(DEVICE)

            optimizer.zero_grad()
            pred = netG(partial)
            loss = cd_loss_fn(pred, complete_gt)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 评估阶段
        netG.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for partial, complete_gt in test_loader:
                partial = partial.to(DEVICE).transpose(1, 2)
                complete_gt = complete_gt.to(DEVICE)

                pred = netG(partial)
                loss = cd_loss_fn(pred, complete_gt)
                epoch_test_loss += loss.item()

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f'Epoch {epoch + 1}/{EPOCHS} | Train CD: {avg_train_loss:.6f} | Test CD: {avg_test_loss:.6f}')

    # 保存模型权重
    model_path = os.path.join(MODEL_SAVE_DIR, 'plant_pfnet_final.pth')
    torch.save(netG.state_dict(), model_path)
    print(f'\n✅ 训练完成！模型权重已保存至: {model_path}')

    return train_losses, test_losses, netG


# ---------------------------------------------------------
# --- 5. 可视化函数 (与之前逻辑相同) ---
# ---------------------------------------------------------

def visualize_results(train_losses, test_losses, netG):
    """绘制损失曲线和点云补全结果。"""

    print("\n--- 结果可视化 ---")

    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Chamfer Distance', color='blue')
    plt.plot(test_losses, label='Test Chamfer Distance', color='red')
    plt.title('Training and Test Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Chamfer Distance Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plot_loss_path = os.path.join(MODEL_SAVE_DIR, 'loss_curve.png')
    plt.savefig(plot_loss_path)
    plt.show()
    print(f"✅ 损失曲线图已保存至: {plot_loss_path}")

    # 2. 补全效果模拟图
    test_dataset = PairedPlantCompletionDataset(TEST_ROOT)
    if not test_dataset.paired_files:
        print("❌ 无法找到测试样本进行可视化。")
        return

    # 获取第一个测试样本进行测试
    netG.eval()
    test_sample_idx = 0
    partial_np, complete_gt_np = test_dataset[test_sample_idx]

    partial_tensor = torch.from_numpy(partial_np).unsqueeze(0).to(DEVICE).transpose(1, 2)

    with torch.no_grad():
        pred_tensor = netG(partial_tensor)
        pred_np = pred_tensor.squeeze(0).cpu().numpy()

    C_GT = 'gray'
    C_INPUT = 'blue'
    C_PRED = 'red'

    # --- 2D 平面投影 (XY 平面) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].scatter(complete_gt_np[:, 0], complete_gt_np[:, 1], complete_gt_np[:, 2], s=1, c=C_GT)
    axes[0].set_title(f'1. Ground Truth ({ORIGINAL_POINTS} pts)')
    axes[0].set_aspect('equal');
    axes[0].axis('off')

    axes[1].scatter(complete_gt_np[:, 0], complete_gt_np[:, 1], complete_gt_np[:, 2], s=1, c=C_GT, alpha=0.2)
    axes[1].scatter(partial_np[:, 0], partial_np[:, 1], partial_np[:, 2], s=3, c=C_INPUT)
    axes[1].set_title(f'2. Partial Input ({INPUT_KEEP} pts)')
    axes[1].set_aspect('equal');
    axes[1].axis('off')

    axes[2].scatter(complete_gt_np[:, 0], complete_gt_np[:, 1], complete_gt_np[:, 2], s=1, c=C_GT, alpha=0.2)
    axes[2].scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], s=3, c=C_PRED, alpha=0.8)
    axes[2].set_title(f'3. Completion (Input + Predicted)')
    axes[2].set_aspect('equal');
    axes[2].axis('off')

    plt.tight_layout()
    plot_2d_path = os.path.join(MODEL_SAVE_DIR, 'completion_2d_projection.png')
    plt.savefig(plot_2d_path)
    plt.show()

    # --- 3D 静态图 ---
    fig = plt.figure(figsize=(18, 6))

    ax_3d_1 = fig.add_subplot(131, projection='3d')
    ax_3d_1.scatter(complete_gt_np[:, 0], complete_gt_np[:, 1], complete_gt_np[:, 2], s=1, c=C_GT, alpha=0.5)
    ax_3d_1.set_title('A. Ground Truth (GT)')
    ax_3d_1.set_box_aspect([1, 1, 1]);
    ax_3d_1.view_init(elev=20, azim=45)

    ax_3d_2 = fig.add_subplot(132, projection='3d')
    ax_3d_2.scatter(partial_np[:, 0], partial_np[:, 1], partial_np[:, 2], s=5, c=C_INPUT)
    ax_3d_2.set_title('B. Partial Input')
    ax_3d_2.set_box_aspect([1, 1, 1]);
    ax_3d_2.view_init(elev=20, azim=45)

    ax_3d_3 = fig.add_subplot(133, projection='3d')
    ax_3d_3.scatter(partial_np[:, 0], partial_np[:, 1], partial_np[:, 2], s=1, c=C_INPUT)
    ax_3d_3.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], s=3, c=C_PRED, alpha=0.5)
    ax_3d_3.set_title('C. Completed Result (Input + Pred)')
    ax_3d_3.set_box_aspect([1, 1, 1]);
    ax_3d_3.view_init(elev=20, azim=45)

    plt.tight_layout()
    plot_3d_path = os.path.join(MODEL_SAVE_DIR, 'completion_3d_static.png')
    plt.savefig(plot_3d_path)
    plt.show()

    print(f"✅ 补全效果 2D 图已保存至: {plot_2d_path}")
    print(f"✅ 补全效果 3D 图已保存至: {plot_3d_path}")


# ---------------------------------------------------------
# --- 6. 脚本入口点 ---
# ---------------------------------------------------------

if __name__ == '__main__':
    prepare_environment()
    train_losses, test_losses, netG = train_model()
    visualize_results(train_losses, test_losses, netG)