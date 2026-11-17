# 长 K 分段累加：原机受阻，qjh004 回归通过

最初在 qjh000 完成编译后，八张卡均有外部作业，数值验证受阻。随后转到用户指定的 qjh004，全部 3024 项长 K 候选组合通过，并完成新增路径的完整 memcheck、racecheck 和 synccheck；详细范围、参考的剩余失败和证据见 [qjh004 报告](../qjh004-20261116/REPORT.md)。以下保留实现机制和原始受阻记录；实验仍默认关闭，性能尚未验收。

原实现让同一组 WGMMA accumulator 沿完整 K 持续累加。新路径将它限制在 256 个 K 元素：完成四个 K=64 的 tile 并等待 WGMMA 后，把 64 个部分和分别用普通 FP32 加法并入另一组总和寄存器，清零部分和，并执行 wgmma.fence 后进入下一段。最后一个不足 256 的分段也合并，最终总和只转换一次 BF16。不增加 workspace 或 kernel launch，原来的 TMA full/empty barrier 与 buffer 复用规则继续执行。

该方案缩短了 Tensor Core 持续累加链，但总和合并本身仍存在 FP32 舍入误差，不能保证任意长 K、任意幅度与强消去输入都满足固定容差。256 是待实测的初始分段长度，后续应以原失败输入和 FP64 基准检验效果。当前没有证据把原失败精确归因于某种硬件舍入机制。

代码覆盖向量加载、TMA 1–4 级和独立分工 TMA 1–4 级，使用独立模板实例。只有设置 BARE_GEMM_STABLE_LONG_K=1 且 K≥16384 才选择新实例；默认仍使用原实例，原五个 K≤4096 的正式用例不走分段路径。其代价是多一组 64 个 FP32 寄存器，以及段间加法、清零与 fence，可能降低驻留或吞吐。没有开展性能测量，也没有改动锁频和监控服务。

目标 CUDA 12.8 编译的全部实例寄存器用量为 90–168，无 spill stores/loads。高寄存器版本仍需实际检查驻留与吞吐，不能由“无 spill”推断没有性能代价。本地 ruff、pyright 与原有 15 项 pytest 通过；这些不代替 GPU 数值或 sanitizer 验证。

复现命令：

```bash
cmake --build build -j 4
BARE_GEMM_STABLE_LONG_K=1 python3 scripts/run_coverage.py --gpu 0 --suite long-k
```

long-k 套件包含 14 个形状，检查 K=16376/16384/16392 的路径切换、16632/16640/16648 的分段尾块、32760/32768/32776 的长归约尾块、65536 的原失败长度，以及 M/N 的小矩阵和尾块。全部 9 条路径各使用 8 种分布、3 个种子、3 次刷新，共 3024 项组合、9072 次候选 launch。此处是待执行的测试规模，不是已完成统计。结果保持原容差与 FP64 基准，分别记录 candidate_status、reference_status；CUTLASS 参考的原有失败不会被隐藏。

目标服务器保留 results/segmented-longk-20261116 中的不可变二进制、源码快照、构建日志和繁忙检查回执。本地证据见同名 evidence 目录。后续需要先找到一张空闲卡完成上述数值检查，再补长 K 的 memcheck、racecheck 和 synccheck；既有只到 K=520 的 sanitizer 不能证明新增模板路径通过。
