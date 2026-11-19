# audio-mixer

## 题面

接入 5.1 片源后，混音总线一开就失真：dB 换算误用 10*log10，-6 dB 实际只剩 -3 dB；声像电位器把线性律写死，`-3db` 等功率律下中间声道偏低 3 dB；混音按最短声道截断，长声道尾部被整段丢掉；求和后不做限幅，叠加超过 1.0 后回放直接削波；延迟补偿形同虚设，总线路由后整体偏移；抖动用全局 `random`，同一素材两次导出结果不同；5.1 下混立体声把中置与 LFE 都直接加进左右，整体电平凭空 +6 dB；量化不做钳制，-1.2 回绕成正向噪声；批量混音遇到 NaN 只跳过该条，输出里继续留下 NaN；声道布局转换按索引硬映射，mono→5.1 把中置放到了 LFE。mixer/bus.py 是核心，tests/test_bus.py 15个红。要修的是：① dB 换算统一改用 20*log10，增益 0 返回 -inf、负增益抛 `MixError`；② `pan_to_gains` 真正实现 `linear` 与 `-3db` 两套声像律；③ 混音以最长声道为基准补零，不得丢掉长总线尾部；④ 求和后经 `apply_limiter` 软限幅，输出绝对值不超过 ceiling；⑤ `compensate_latency` 按帧对齐，负延迟抛 `MixError`；⑥ `dither` 使用固定 seed 的局部 TPDF，保证确定性；⑦ 5.1→stereo 时中置按 -3 dB 下混、LFE 不直接叠加；⑧ 量化结果钳制在 [-1, 1]，负向过载不得回绕；⑨ `mix_batch` 遇到任一非有限值整批回滚；⑩ 布局转换按语义映射，mono→5.1 进中置而不是 LFE。

## 复现

```console
$ python -m pytest tests -q
15 failed
```

初始状态必须恰好是 `15 failed`：没有 collection error、import error 或 syntax error，15 条测试全部因 `NotImplementedError` 失败。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只使用 Python 标准库（`math`、`struct`、`array`）与 pytest；离线运行、不使用全局随机（`dither` 的 TPDF 完全由显式 seed 决定）、不 `sleep`。
- 采样率与帧数由调用方显式传入；模块内不读时钟、不联网、不写文件。
- 音频统一表示为 `list[list[float]]`（声道 × 样本），浮点样本范围 -1.0..1.0；断言容差 1e-6（dB 往返为 1e-9）。
- `tests/test_bus.py` 是验收规格，不可修改；修复只发生在 `mixer/bus.py`（`mixer/__init__.py` 仅做导出）。
