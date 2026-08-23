"""micrograd_impl 数值梯度校验脚本：中心差分 vs 自动微分逐点对拍。

功能概览：
  1. 校验原理：对输入 x_i 施加微小扰动 h，用中心差分
     (f(x+h) - f(x-h)) / (2h) 近似数值梯度，与 value.backward()
     回传得到的解析梯度逐点对比，最大误差在容差内即视为一致。
  2. 逐算子校验：add / sub / rsub / mul / div / rdiv / pow /
     tanh / exp / log / relu，含 a+2、2+a、a-2、2-a、3*a、2/a
     等混合类型运算（relu 导数在 0 处不连续，只在远离 0 的点测试）。
  3. 复合表达式校验：多输入混合运算的联合梯度校验。
  4. MLP 校验：mlp(3, [4, 2, 1]) 前向建图，loss = sum((pred-target)**2)，
     backward 后检查全部 parameters() 的 grad 非 None 且有限
     （math.isfinite），并校验 loss 对各输入（也做成 value 传入）的
     解析梯度与数值梯度一致。
  5. 汇总 PASS / FAIL 计数：全部通过打印「全部 N 项校验通过」，
     存在失败则打印失败项详情并以退出码 1 结束。

本文件为新增，不修改 engine.py / nn.py。

运行方式：
  cd LearnAi && python micrograd_impl/test.py    // 从项目根运行
"""

import math
import os
import random
import sys

# 将项目根加入 sys.path，使 `from micrograd_impl.engine import value` 可解析
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from micrograd_impl.engine import value
from micrograd_impl.nn import mlp

# ------------------------- 全局配置 -------------------------
H = 1e-6        # 中心差分扰动步长
TOL = 1e-5      # 解析梯度与数值梯度的最大误差容差


# ------------------------- 1. 中心差分数值梯度 -------------------------
def numerical_grad(f, xs, h=H):
    """中心差分求各输入的数值梯度。

    f: 表达式构造函数，输入 value 列表 -> loss value；
    xs: 输入点（float 列表）；h: 扰动步长。
    对每个分量 i：grad_i ≈ (f(x+h·e_i) - f(x-h·e_i)) / (2h)。
    """
    grads = []
    for i in range(len(xs)):
        plus = list(xs)
        minus = list(xs)
        plus[i] = xs[i] + h
        minus[i] = xs[i] - h
        fp = f([value(x) for x in plus]).data
        fm = f([value(x) for x in minus]).data
        grads.append((fp - fm) / (2.0 * h))
    return grads


# ------------------------- 2. 单用例校验（解析 vs 数值） -------------------------
def check_grad(results, label, expr, xs, names=None):
    """校验一个梯度用例：expr(输入 value 列表) -> loss value。

    解析梯度：inputs=value(xs) -> expr -> loss.backward() -> 各输入 .grad；
    数值梯度：numerical_grad 中心差分。逐输入比较，最大误差 <= TOL 判 PASS。
    """
    if names is None:
        names = ("a", "b", "c", "d")[: len(xs)]
    # 解析梯度（自动微分）
    inputs = [value(x) for x in xs]
    loss = expr(inputs)
    loss.backward()
    analytic = [v.grad for v in inputs]
    # 数值梯度（中心差分）
    numeric = numerical_grad(expr, xs)
    # 逐输入误差（nan/inf 参与比较自然判 FAIL）
    errs = [abs(a - n) for a, n in zip(analytic, numeric)]
    max_err = max(errs)
    ok = math.isfinite(max_err) and max_err <= TOL
    in_str = " ".join(f"{n}={x:.1f}" for n, x in zip(names, xs))
    an_str = ",".join(f"{a:.2f}" for a in analytic)
    nu_str = ",".join(f"{n:.2f}" for n in numeric)
    print(f"{'PASS' if ok else 'FAIL'} {label}: {in_str} | "
          f"解析={an_str} 数值={nu_str} 误差={max_err:.1e}")
    if not ok:
        for n, a, m, e in zip(names, analytic, numeric, errs):
            print(f"    详情 输入 {n}: 解析={a:.6e} 数值={m:.6e} "
                  f"误差={e:.3e}（容差 {TOL:.1e}）")
    results.append((label, ok))


# ------------------------- 3. 逐算子校验用例 -------------------------
# 每个小节: (算子名说明, [(label, 表达式构造函数, 输入点列表), ...])
OP_SECTIONS = [
    ("add 加法（含 a+2、2+a 混合类型）", [
        ("add(a+b)", lambda v: v[0] + v[1], [2.0, -3.0]),
        ("add(a+2)", lambda v: v[0] + 2, [2.5]),
        ("add(2+a)", lambda v: 2 + v[0], [-1.2]),
    ]),
    ("sub/rsub 减法（含 a-2、2-a）", [
        ("sub(a-b)", lambda v: v[0] - v[1], [2.0, -3.0]),
        ("sub(a-2)", lambda v: v[0] - 2, [3.5]),
        ("rsub(2-a)", lambda v: 2 - v[0], [1.5]),
    ]),
    ("mul 乘法（含 3*a）", [
        ("mul(a*b)", lambda v: v[0] * v[1], [2.0, -3.0]),
        ("mul(3*a)", lambda v: 3 * v[0], [-1.5]),
    ]),
    ("div/rdiv 除法（正值输入避免除零）", [
        ("div(a/b)", lambda v: v[0] / v[1], [3.0, 2.0]),
        ("rdiv(2/a)", lambda v: 2 / v[0], [4.0]),
    ]),
    ("pow 幂（正值输入）", [
        ("pow(a**2)", lambda v: v[0] ** 2, [1.5]),
        ("pow(a**3)", lambda v: v[0] ** 3, [2.0]),
    ]),
    ("tanh 双曲正切", [
        ("tanh(a)", lambda v: v[0].tanh(), [0.7]),
        ("tanh(a)", lambda v: v[0].tanh(), [-1.3]),
    ]),
    ("exp 指数（小值避免溢出）", [
        ("exp(a)", lambda v: v[0].exp(), [0.5]),
        ("exp(a)", lambda v: v[0].exp(), [-1.3]),
    ]),
    ("log 自然对数（正值输入）", [
        ("log(a)", lambda v: v[0].log(), [2.0]),
        ("log(a)", lambda v: v[0].log(), [0.5]),
    ]),
    ("relu（导数在 0 处不连续，取远离 0 的点）", [
        ("relu(a>0)", lambda v: v[0].relu(), [2.0]),
        ("relu(a<0)", lambda v: v[0].relu(), [-1.5]),
    ]),
]


# ------------------------- 4. 复合表达式用例 -------------------------
COMPOSED_CASES = [
    ("复合(a*b + a.tanh() - b.exp()/a + (a*b)**2)",
     lambda v: v[0] * v[1] + v[0].tanh() - v[1].exp() / v[0]
               + (v[0] * v[1]) ** 2,
     [1.5, 0.8]),
    ("复合(a.relu()*b.log() + (a-b)**3)",
     lambda v: v[0].relu() * v[1].log() + (v[0] - v[1]) ** 3,
     [1.2, 0.6]),
]


# ------------------------- 5. MLP 校验 -------------------------
def check_mlp(results):
    """MLP 梯度校验：参数梯度性质检查 + 输入梯度解析/数值对拍。"""
    random.seed(42)                       # 固定随机初始化，保证可复现
    model = mlp(3, [4, 2, 1])
    xs = [0.5, -0.3, 0.8]
    target = [0.7]

    def build_loss(inputs):
        """前向建图并构造 loss = sum((pred - target)**2)。"""
        pred = model(inputs)
        return sum((p - t) ** 2 for p, t in zip(pred, target))

    # 1) 参数梯度：backward 后所有 parameters() 的 grad 非 None 且有限
    loss = build_loss([value(x) for x in xs])
    loss.backward()
    params = model.parameters()
    bad = [i for i, p in enumerate(params)
           if p.grad is None or not math.isfinite(p.grad)]
    ok = not bad
    if ok:
        print(f"PASS mlp参数梯度: 参数 {len(params)} 个 | "
              f"grad 全部非 None 且有限")
    else:
        print(f"FAIL mlp参数梯度: 参数 {len(params)} 个 | "
              f"梯度异常下标 {bad}")
        for i in bad:
            print(f"    详情 参数[{i}].grad = {params[i].grad!r}")
    results.append(("mlp参数梯度", ok))

    # 2) 输入梯度：输入也做成 value 传入网络，解析梯度 vs 数值梯度
    check_grad(results, "mlp输入梯度", build_loss, xs,
               names=("x0", "x1", "x2"))


# ------------------------- 6. 汇总与入口 -------------------------
def main():
    results = []                          # [(label, ok), ...]

    print(">> 逐算子校验：中心差分 (f(x+h)-f(x-h))/2h vs "
          "value.backward()（h=1e-6，容差 1e-5）")
    for section, cases in OP_SECTIONS:
        print(f"-- {section} --")
        for label, expr, xs in cases:
            check_grad(results, label, expr, xs)

    print(">> 复合表达式校验（多输入联合）")
    for label, expr, xs in COMPOSED_CASES:
        check_grad(results, label, expr, xs)

    print(">> MLP 校验：mlp(3, [4, 2, 1])，loss = sum((pred - target)**2)")
    check_mlp(results)

    # 汇总：统计 PASS/FAIL，全部通过打印总数，否则非零退出码结束
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = len(results) - n_pass
    print("-" * 60)
    print(f"汇总: PASS {n_pass} 项 / FAIL {n_fail} 项")
    if n_fail == 0:
        print(f"全部 {len(results)} 项校验通过")
    else:
        print("失败项：")
        for label, ok in results:
            if not ok:
                print(f"  FAIL {label}")
        sys.exit(1)


if __name__ == "__main__":
    main()
