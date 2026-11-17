# 分段累加通过长 K 回归，GPU 诊断计时已补充

在 qjh004（124.174.78.73，H20 八卡，驱动 580.126.20）完成长 K 修复验证。设置 BARE_GEMM_STABLE_LONG_K=1 后，全部 9 条候选路径的 3024 项形状／分布／种子组合均满足原容差，每项刷新验证 3 次，共 9072 次候选 launch。最大归一化误差为 0.6149215353，阈值为 1。相同输入下，固定 CUTLASS 参考有 126 项失败，最大误差为 4.254370641，故 manifest 的整体 status 仍为 fail，candidate_status 为 pass，reference_status 为 fail。

这证明分段累加消除了本次长 K 套件中的候选数值失败。它不是任意形状、任意输入的误差保证，也没有把旧的 159 形状扫描宣称为新版本全量复跑。实验入口继续显式启用，默认关闭。随后补充的 [GPU 计时报告](SPEED.md) 显示：五个约定用例加速比为 0.982–1.101×，长 K 分段累加相对旧候选增加约 4.46%–5.37% 耗时；显存未硬锁，最终验收仍为 inconclusive。

## 数值修复与覆盖范围

候选仍为 BF16 输入输出、FP32 累加。K≥16384 时，每 256 个 K 元素形成一个 WGMMA 部分和；等待完成后，以普通 FP32 加法并入另一组总和，清零部分和并执行 WGMMA fence，然后继续下一段。最后一段和 K 尾块正常补零，最终只做一次 BF16 转换。没有增加 workspace 或 kernel launch。额外寄存器、段间加法和 fence 的总耗时代价已单独测量，各部分的归因仍待 profile。

14 个形状覆盖原失败 K=32768/65536，K=16376/16384/16392 的路径切换，16632/16640/16648 的分段尾部，32760/32768/32776 的长归约尾部，以及 M/N 的小矩阵和不完整 tile。每条路径检查 8 种分布、3 个种子、3 次刷新，共 336 项组合；向量加载、普通 TMA 1–4 级和独立分工 TMA 1–4 级全部通过。固定参考每条路径对应 14 项失败，未调整参考或容差。

校验基准为 BF16 舍入后输入的独立 FP64 DGEMM，结果转 FP32 后使用 |actual−expected|≤0.02+0.01|expected|。随机输入在同一个 prepare context 中刷新，输出先填 NaN，检查每个元素。解析输入另做精确基准检查。

长 K 与 sanitizer 使用上轮不可变二进制，SHA256 为 49dd7aeee3bc69d70899ccbc8a7b2c747878566fa4909feb63f6544f4f229b30。另在 qjh004 从相同源码、固定 CUTLASS 版本重新编译，原五个正式形状在全部 9 条配置下完成独立 FP64 校验：45 项、135 次刷新全部通过。本机二进制 SHA256 为 b101c10ba26d036337324a50de658418df4a8dd4f712415cdc08eb1f76677a38；没有把两份二进制的检查范围混为一谈。

## 内存、同步与环境证据

新增长 K 路径的 sanitizer 使用 65×136×16648 和 1×8×65536，混合幅度输入、种子 17、3 次刷新，全部 9 条配置均完成。保留默认 API 错误报告的 memcheck 全部为 0 errors，racecheck 全部为 0 errors / 0 warnings，synccheck 全部为 0 errors。此前也保留了关闭 API 报告的设备内存诊断，此次完整 memcheck 的通过是独立补跑结果。新机未复现旧机 cuBLAS 初始化中的 cuCtxGetLimit API 错误，但没有进一步隔离驱动和运行库差异的因果。

数值扫描开始前八卡空闲，先在 GPU 0 串行执行，随后停止本任务自己的串行驱动，保留已完成的结果，将剩余组合分配到空闲 GPU 0/3/4/5 并行执行。已核对 216 个任务标签无重复，每份 CSV 恰好 14 行、次序与形状清单一致，共 3024 行。完整 memcheck 和其他 sanitizer 使用 GPU 1，本机编译的正式形状复核使用 GPU 2。监控服务一直 active，其他作业未停止，未重置 GPU。

root 公钥登录被拒绝，本轮以已有 fengyunhao 公钥连接。正确性阶段的管理员认证用于检查 GPU 1 的即时显存锁频。即使在 580.126.20 驱动上，nvidia-smi -i 1 -lmc 2619,2619 仍明确回复不支持，要求 --lock-memory-clocks-deferred。后续诊断计时将 GPU 0 的 SM 硬锁在 1800 MHz，结束后解除；显存采样稳定在 2619 MHz，仍不视为硬锁。未设置 deferred、未重置 GPU。因此没有完成性能验收，不宣称候选满足完整锁频条件下的 1.03 耗时比要求。

原始文件在服务器 /home/fengyunhao/bare-hopper-qjh004-20261116。可用以下命令复跑长 K 数值检查；运行前要求 GPU 无其他计算进程：

```bash
BARE_GEMM_STABLE_LONG_K=1 python3 scripts/run_coverage.py --gpu 0 --suite long-k
```

本轮完整证据入口：[长 K 汇总](evidence/coverage-20261116T131616Z/summary.json)、[逐任务 manifest](evidence/coverage-20261116T131616Z/manifest.json)、[完整 memcheck](evidence/long-k-sanitizers/full-memcheck.json)、[racecheck 与 synccheck](evidence/long-k-sanitizers/manifest.json)、[本机五形状复核](evidence/native-formal/manifest.json)、[环境与锁频探查](evidence/node-evidence/manifest.json)。同目录包含全部逐项 CSV 和诊断日志，恢复并行执行的脚本也一并保存。

本地 ruff、pyright、15 项 pytest 和 diff 空白检查通过。代码、文档没有 commit 或 push。诊断计时后仍保留长 K 显式启用；下一步定位尾块回退和分段开销，并在满足完整锁频条件的环境完成最终验收。
