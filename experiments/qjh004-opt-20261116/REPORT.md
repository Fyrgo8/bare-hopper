# H20 GEMM 优化：小输出长 K 使用 Split-K，输出改成 BF16 成对写回

本轮继续使用 CUDA C++ 与内联 PTX，候选没有引入 CUTLASS、CuTe 或 cuBLAS 计算模块。固定 CUTLASS 仅作为性能参考，独立 FP64 DGEMM 用于数值检查。最终二进制 SHA256 为 019c9455781dc1a88d1b4ee2635ae2571164b7190786b0a0da72646be83a2176。

原来的小输出长 K 矩阵只有少量输出 CTA，却需要每个 CTA 串行完成全部 K。现在把 K 按 256 拆开，每个 CTA 将自己的 FP32 部分和写到互不重叠的临时空间，再由第二个 kernel 按 K 顺序进行 FP32 归并，最后转换一次 BF16。归并不使用原子加法，两次 launch 都在同一个调用方 stream 上，均计入候选耗时。临时空间在 prepare 阶段分配，每次 launch 完整覆盖所有有效部分和与输出，不复用上次输入的计算结果。

默认仅在 K≥16384、输出 tile 数≤32、临时空间≤64 MiB、拆分数≤65535 时使用 Split-K；否则回到串行分段路径。BF16 输入输出和 FP32 累加保持不变。较大输出的 Split-K 实验在 1024×1024×16384 上变慢，已撤回，不作为默认路径。

TMA 缓冲并非越多越好。两级版本在若干正式大矩阵上回退，三级则改善了长 K 串行路径。因此默认 K<16384 使用单级，长 K 使用三级；设置 BARE_GEMM_STAGES 可覆盖选择。BARE_GEMM_STABLE_LONG_K 和 BARE_GEMM_SPLIT_K 默认均启用，显式设置 0 可保留原实现进行对照。独立 producer/consumer 仍需显式启用。每个线程将相邻两个 FP32 accumulator 转为一个 BF16 对并一次写回；N 必须为 8 的倍数，因此有效输出末端不会出现只有一个 BF16 可写的半对。

## 最终二进制的正确性与性能

2026-11-17 在 qjh004 H20 上完成全部 520 个回归任务。full 套件为 5352 项组合，long 为 3024 项，dispatch 为 1512 项，合计 9888 项，每项刷新 3 次，共 29664 次候选 launch，全部满足原容差。这里的 launch 指完整候选调用，Split-K 调用内部包含计算与归并两个 kernel。最大归一化误差分别为 0.3882802895、0.6149215353、0.3883731255，阈值为 1。固定参考分别失败 21、126、27 项，共 174 项；没有改变容差或隐藏参考失败。另将全部配置环境变量清空，对 8 个计时形状的 dyadic、uniform、logscale 输入各刷新 3 次，24 项候选检查也全部通过。

最终二进制完成常规 7 个边界形状、9 条路径、3 种 sanitizer 的 27 个任务，以及新增 Split-K 的 4 个边界形状、8 条 TMA 路径、3 种 sanitizer 的 24 个任务。两组全部退出码为 0，memcheck/synccheck 无错误，racecheck 无错误或警告。本地 ruff、指定虚拟环境解释器的 pyright、15 项 pytest 和 diff 空白检查通过。

其他 GPU 检查全部退出后，对默认自动策略进行 30 轮 AB/BA 配对计时，rotating 输入轮转超过 4 倍 L2，预热 32 次并自动校准迭代数。计时输入为原 dyadic 分布，前后校验均通过。耗时为各自样本中位数；加速比为配对 candidate/reference 比中位数的倒数，可能与表内两个中位数相除略有差异。

| M×N×K | 参考 ms | 候选 ms | 参考/候选 | 加速比 95% bootstrap 区间 |
| --- | ---: | ---: | ---: | --- |
| 4096×4096×4096 | 1.053851 | 1.008092 | 1.045394× | [1.045338, 1.045427] |
| 1024×4096×4096 | 0.304958 | 0.274145 | 1.112557× | [1.112361, 1.112809] |
| 4096×4096×1024 | 0.280097 | 0.255678 | 1.095491× | [1.095178, 1.095732] |
| 3968×4096×4096 | 0.981156 | 0.985188 | 0.995858× | [0.994976, 0.996239] |
| 4104×4088×4096 | 1.057502 | 1.008877 | 1.048244× | [1.048196, 1.048308] |
| 65×136×32768 | 0.591403 | 0.057185 | 10.341977× | [10.335793, 10.347067] |
| 65×136×65536 | 1.177385 | 0.102717 | 11.462412× | [11.460151, 11.466460] |
| 1024×1024×16384 | 0.300857 | 0.304560 | 0.987835× | [0.987797, 0.987851] |

八项的 candidate/reference 区间上界均小于 1.03，诊断 timing_status 为 pass。M 尾块仍慢约 0.42%，较大输出长 K 仍慢约 1.23%，因此没有宣称全部用例都更快。与上轮记录相比，short_k 加速比从 1.018× 提高到 1.095×，两个小输出长 K 从约 0.85× 提高到 10.34×、11.46×；前后是分开的实验，不能当成直接配对的旧版/新版样本。参考是固定 CUTLASS 内核，不代表全局最优实现，结果也不代表通用 GEMM 的完整功能范围。

本轮计时窗口频率采样均为 SM 1800 MHz、显存 2619 MHz，clock reason mask 为 0，无外部计算 PID。只硬锁了 SM，显存观测频率不视为硬锁，最终验收状态仍为 inconclusive_memory_unlocked。测后已解除 SM 锁，空闲 SM 恢复为 345 MHz，监控服务保持 active，没有停止监控或重置 GPU。

证据入口：[数值汇总](evidence/results/opt-final-validation/summary.json)、[逐任务 manifest](evidence/results/opt-final-validation/manifest.json)、[性能汇总](evidence/results/opt-final-speed/summary.json)、[原始计时 CSV](evidence/results/opt-final-speed/sauto-ws0.csv)、[默认入口验证](evidence/results/opt-default-validation)、[全部证据归档](evidence.tgz)。本地归档排除二进制，服务器保留不可变二进制及 SHA256；本地含源码快照、构建日志、全部 CSV、sanitizer 日志、频率遥测和运行脚本。原始路径为 /home/fengyunhao/bare-hopper-qjh004-20261116/results/opt-final-* 与 opt-delivery-build。

复现实验先在 Linux CUDA 12.8 构建。默认策略无需设置环境变量：

```bash
cmake --build build -j 4
python3 scripts/run_coverage.py --gpu 0
python3 scripts/run_coverage.py --gpu 0 --suite long-k
```

本目录 measure.py 保留诊断计时入口，validate.py 对不可变二进制重跑 full、long、dispatch 三组套件，sanitize.sh 保留新增路径的 sanitizer 检查入口。它们记录原始 CSV 和 manifest；性能筛选中的中间二进制与最终版本分开归档。受其他 PID 检测影响的早期轮次不用于最终结论。代码与文档均未 commit 或 push。
