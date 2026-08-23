"""micrograd 演示：用标量 autograd 引擎训练 toy 二分类（Karpathy micrograd demo 风格）。

功能概览：
  1. 构造 8 个二维 toy 样本（四象限对角分布，标签 y ∈ {+1, -1}）。
  2. mlp(2, [4, 4, 1]) 前向打分，hinge 损失直接用引擎算子构造：
     loss = sum(relu(1 - y_i * score_i))——展示 autograd 的意义：
     损失公式怎么写，梯度就怎么来，全程无需手写任何求导代码。
  3. 纯 SGD 训练循环：m.zero_grad() -> 前向累加 loss -> loss.backward()
     -> 遍历 parameters() 手工更新 p.data -= lr * p.grad。
  4. 训练结束逐样本打印坐标 / 真实标签 / score / 预测类别，汇报正确率。

本文件为新增演示脚本，只依赖 engine.py / nn.py，不修改任何现有实现。

运行方式：
  cd LearnAi && python micrograd_impl/demo.py   // 从项目根运行
"""

import os
import random
import sys

# 将项目根加入 sys.path，使 `from micrograd_impl.nn import ...` 可解析
#（直接运行本文件时，sys.path[0] 是 micrograd_impl/ 目录而非项目根）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from micrograd_impl.engine import value
from micrograd_impl.nn import mlp


# ------------------------- 1. toy 二分类数据集 -------------------------
# 四象限对角分布的二维样本，标签 y ∈ {+1, -1}：
#   +1 类：落在主对角线 y = x 上（一、三象限）：(1,1) (-1,-1) (2,2) (-2,-2)
#   -1 类：偏离主对角线（四、二象限及偏离点）：(1,-1) (-1,1) (-2,-1) (2,-2)
# 几何含义：贴近主对角线 -> +1，远离主对角线 -> -1；
# -1 类分布在主对角线两侧，线性不可分，正好需要隐藏层 tanh 的非线性。
XS = [
    (1.0, 1.0), (-1.0, -1.0), (2.0, 2.0), (-2.0, -2.0),    # 标签 +1
    (1.0, -1.0), (-1.0, 1.0), (-2.0, -1.0), (2.0, -2.0),   # 标签 -1
]
YS = [+1, +1, +1, +1, -1, -1, -1, -1]

# ------------------------- 2. 超参数 -------------------------
SEED = 42       # 固定随机种子（nn.py 参数初始化用 random），保证结果可复现
STEPS = 100     # 训练步数（纯 SGD，每步遍历全部样本一遍）
LR = 0.1        # 学习率


# ------------------------- 3. hinge 损失（直接用引擎算子构造） -------------------------
def hinge_loss(model, xs, ys):
    """对全部样本累加 hinge 损失：loss = sum(relu(1 - y_i * score_i))。

    把数学式子原样『翻译』成 value 计算图（乘法 / 减法 / relu），
    梯度完全由 loss.backward() 自动回传——这就是 autograd 的意义。
    """
    loss = value(0.0)
    for (x0, x1), yi in zip(xs, ys):
        score = model([x0, x1])[0]           # mlp 返回输出列表，取唯一的 score
        loss = loss + (1.0 - yi * score).relu()
    return loss


# ------------------------- 4. 训练与评估 -------------------------
def main():
    random.seed(SEED)

    # 2 维输入 -> 两个 4 神经元隐藏层（tanh，nn.py 默认保证）-> 1 维线性输出
    m = mlp(2, [4, 4, 1])
    print(f">> 网络: mlp(2, [4, 4, 1])，参数量 = {len(m.parameters())}")
    print(f">> 样本数 = {len(XS)}，训练 {STEPS} 步，lr = {LR}")

    # 训练前初始损失
    print(f"step   0 | loss = {hinge_loss(m, XS, YS).data:.4f}")

    # 纯 SGD 训练循环：清梯度 -> 前向累加 loss -> 反向传播 -> 手工更新参数
    for step in range(1, STEPS + 1):
        m.zero_grad()                        # 1) 清空全部参数梯度
        loss = hinge_loss(m, XS, YS)         # 2) 前向：全部样本累加 hinge 损失
        loss.backward()                      # 3) 反向：autograd 一次回传整张图
        for p in m.parameters():             # 4) 手工 SGD 更新参数
            p.data -= LR * p.grad
        if step % 10 == 0:
            print(f"step {step:3d} | loss = {loss.data:.4f}")

    # 训练结束：逐样本打印坐标 / 真实标签 / score / 预测类别（score > 0 为 +1）
    print(">> 训练结束，逐样本结果：")
    print(f"最终 loss = {hinge_loss(m, XS, YS).data:.4f}")
    print("坐标            真实    score       预测")
    errors = 0
    for (x0, x1), yi in zip(XS, YS):
        score = m([x0, x1])[0].data
        pred = 1 if score > 0 else -1
        if pred != yi:
            errors += 1
        print(f"({x0:+.1f}, {x1:+.1f})    {yi:+d}      {score:+.4f}     {pred:+d}")

    print("-" * 40)
    if errors == 0:
        print("全部样本分类正确")
    else:
        print(f"存在分类错误：错误数 = {errors}（正确 {len(XS) - errors}/{len(XS)}）")


if __name__ == "__main__":
    main()
