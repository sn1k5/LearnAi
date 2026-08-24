"""可保存/早停的通用 Mixin（numpy_impl）：为 network / network_batch / my_network / my_network_batch 提供

统一的模型保存、加载、best 模型快照与 early stop 逻辑。

设计要点：
  - 模型以 .npz 存储（原生 numpy 格式，无额外依赖），内含每层权重 W/B 与结构元数据
    （input_size / output_size / 各层 shape / is_output），加载时据此直接重建网络。
  - 训练循环通过本 mixin 提供的 _begin_monitor / _maybe_checkpoint / _should_early_stop
    三个钩子接入：
      * save_best  : 监控某验证指标（默认 val_acc），指标改善则自动存为 best 模型。
      * early_stop : 监控指标连续 patience 轮无改善则提前停止。
"""

import os

import numpy as np


# 监控指标配置：key -> (取值函数, 越大越好?)
# 训练侧算出 (val_loss, val_acc) 后，根据 monitor 选择其一作为早停/保存依据。
_MONITORS = {
    "val_acc":  lambda val_loss, val_acc: val_acc,
    "val_loss": lambda val_loss, val_acc: val_loss,
}


class _SaveableMixin:
    """为各 network 类提供保存/加载/早停的通用实现。

    子类需提供：
      - self.layers : layer / layer_batch 实例列表，每个含 .W / .B / .input_size /
        .output_size / .is_output。
      - self.input_size / self.output_size （所有子类均已具备）。
    子类需实现 _build_from_shapes(struct) 以在 load 时重建层结构。
    """

    # ------------------------- 结构描述（子类可重写以暴露更多字段） -------------------------
    def _arch_meta(self):
        """返回结构元数据 dict（不含权重）。"""
        return {
            "input_size": int(self.input_size),
            "output_size": int(self.output_size),
            "layer_shapes": [
                (int(l.input_size), int(l.output_size), bool(l.is_output))
                for l in self.layers
            ],
            "class": type(self).__name__,
        }

    def _build_from_shapes(self, layer_shapes):
        """根据 layer_shapes 重建层（默认用 layer_batch / layer 的通用约定）。

        子类若构造签名特殊（如 hidden_shape 列表、hidden_dim + num_hidden_layers），
        应重写本方法以正确重建，否则权重张量形状与层数量不匹配。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 需要重写 _build_from_shapes 以正确重建层结构"
        )

    # ------------------------- 保存 -------------------------
    def save(self, path):
        """把当前模型（结构元数据 + 每层 W/B）保存为 .npz 文件。

        path 不含扩展名时自动补 .npz；若指向目录则存为 model.npz。
        """
        if os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, "model.npz")
        if not path.endswith(".npz"):
            path = path + ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        meta = self._arch_meta()
        # 把结构元数据序列化进 npz（用 0 维数组承载，加载时再还原）
        save_dict = {
            "meta_input_size": np.array(meta["input_size"]),
            "meta_output_size": np.array(meta["output_size"]),
            "meta_class": np.array(meta["class"]),
            "meta_layer_shapes": np.array(
                meta["layer_shapes"], dtype=object
            ),
        }
        for i, l in enumerate(self.layers):
            save_dict[f"W_{i}"] = np.asarray(l.W)
            save_dict[f"B_{i}"] = np.asarray(l.B)
        np.savez(path, **save_dict)
        return path

    # ------------------------- 加载 -------------------------
    @classmethod
    def load(cls, path):
        """从 .npz 文件加载模型，返回新建的 cls 实例（权重已恢复）。

        path 不含扩展名时自动补 .npz。
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path, allow_pickle=True)
        meta_class = str(data["meta_class"])
        if meta_class != cls.__name__:
            raise ValueError(
                f"模型文件类别为 {meta_class}，与当前类 {cls.__name__} 不一致"
            )
        layer_shapes = list(data["meta_layer_shapes"])
        net = cls.__new__(cls)  # 跳过 __init__，避免随机初始化覆盖
        net.input_size = int(data["meta_input_size"])
        net.output_size = int(data["meta_output_size"])
        net._build_from_shapes(layer_shapes)
        for i, l in enumerate(net.layers):
            l.W = np.asarray(data[f"W_{i}"], dtype=float).reshape(
                l.input_size, l.output_size
            )
            l.B = np.asarray(data[f"B_{i}"], dtype=float).reshape(l.output_size)
        return net

    # ------------------------- 早停 / best 监控辅助 -------------------------
    def _begin_monitor(self, monitor):
        """初始化监控状态，返回内部状态 dict（由 train 持有）。"""
        if monitor not in _MONITORS:
            raise ValueError(f"monitor 仅支持 {list(_MONITORS)}，收到 {monitor!r}")
        return {
            "monitor": monitor,
            "best_value": None,      # 历史最优监控值
            "best_epoch": -1,
            "no_improve": 0,         # 连续无改善轮数
            "best_path": None,       # 若启用 save_best 则记录路径
        }

    def _metric_value(self, state, val_loss, val_acc):
        """根据 monitor 配置从 (val_loss, val_acc) 取出当前监控值。"""
        return _MONITORS[state["monitor"]](val_loss, val_acc)

    def _is_better(self, state, cur):
        """判断 cur 是否优于历史最优（loss 越小越好，acc 越大越好）。"""
        if state["monitor"] == "val_loss":
            return state["best_value"] is None or cur < state["best_value"]
        return state["best_value"] is None or cur > state["best_value"]

    def _maybe_checkpoint(self, state, epoch, val_loss, val_acc, save_best, best_path):
        """每轮验证后调用：更新最优指标，必要时存 best 模型。

        返回是否刷新了 best（供日志使用）。
        """
        if save_best is None and best_path is None:
            return False
        cur = self._metric_value(state, val_loss, val_acc)
        improved = self._is_better(state, cur)
        if improved:
            state["best_value"] = cur
            state["best_epoch"] = epoch
            state["no_improve"] = 0
            if best_path is not None:
                saved = self.save(best_path)
                state["best_path"] = saved
            return True
        state["no_improve"] += 1
        return False

    def _should_early_stop(self, state, epoch, patience):
        """判断是否应早停（基于连续无改善轮数）。patience<=0 表示不早停。"""
        if patience is not None and patience > 0:
            if state["no_improve"] >= patience:
                return True
        return False

    def _monitor_summary(self, state):
        """返回 best 指标摘要字符串（供日志使用）。"""
        if state["best_value"] is None:
            return ""
        return (
            f"  best_{state['monitor']} = {state['best_value']:.4f}"
            f"(epoch {state['best_epoch'] + 1})"
        )
