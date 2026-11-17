# bare-hopper

用 CUDA C++ 和 PTX 手写 Hopper GPU 算子，不依赖 CUTLASS/CuTe，通过复现高性能实现锻炼算子开发基本功。

从 BF16 GEMM 开始，逐步复现 FlashAttention-3 的前向和反向。在 H20 上保证数值正确，并以每个约定测试用例的耗时不超过参考实现的 103% 为目标。

## BF16 GEMM

候选使用 CUDA/PTX 实现，tile 为 64×128×64，BF16 输入输出、FP32 累加。A 是 row-major M×K，B 是 column-major K×N，D 是 row-major M×N；每次 launch 完整覆盖输出。CUTLASS 只用于固定参考内核，候选编译目标不获得 CUTLASS/CuTe include 路径。

默认按 K 选择 TMA 缓冲：K<16384 使用单级，长 K 使用三级，并启用每 256 项的 FP32 分段合并。长 K 且输出不超过 32 个 tile 时，采用 Split-K 并用第二个 kernel 归并 FP32 部分和。向量化加载、各级缓冲和独立 producer/consumer warpgroup 仍可显式选择。配置、描述符和临时空间分配在 prepare 阶段完成；候选计时包含 Split-K 的计算和归并两次 launch：

| 环境变量 | 取值 | 含义 |
| --- | --- | --- |
| BARE_GEMM_STAGES | 0 | 普通向量化加载，单级 WGMMA |
| BARE_GEMM_STAGES | 未设置（默认） | K<16384 选单级，长 K 选三级 |
| BARE_GEMM_STAGES | 1 | 单级 TMA + WGMMA |
| BARE_GEMM_STAGES | 2、3、4 | TMA 环形缓冲，分别占用 48、72、96 KiB 动态 shared memory |
| BARE_GEMM_WARP_SPECIALIZED | 1 | TMA 路径使用独立搬运和计算 warpgroup；默认关闭 |
| BARE_GEMM_STABLE_LONG_K | 1（默认）／0 | K≥16384 时分段累加；0 保留旧累加方式供对照 |
| BARE_GEMM_SPLIT_K | 1（默认）／0 | 长 K、小输出使用 K=256 的 Split-K；FP32 临时空间上限 64 MiB，超限回退串行分段路径 |
| BARE_GEMM_GROUP_M | 未设置（默认）／1／8 | 默认在 2048≤K<16384 且 M/N 分别对齐 64/128 时按 8 个 M tile 分组；1 关闭，8 强制分组 |

所有输入维度必须为正，N/K 必须是 8 的倍数；M/N/K 不必对齐 tile。普通加载显式补零，TMA 使用越界补零，输出按 M/N 边界写回。

在 Linux CUDA ≥12.3 环境构建：

```bash
bash scripts/setup_cutlass.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
python scripts/run_bench.py --gpu 0 --list-clocks
# 使用目标 GPU 实际支持的频率；锁频需要管理员权限。
python scripts/run_bench.py --gpu 0 --sm-clock SM_MHZ --memory-clock MEM_MHZ --sudo --compare
```

`scripts/run_bench.py` 需要 Python ≥3.12。每次运行在 results 下新建目录，保存二进制/源码 hash、配置、频率遥测和逐用例置信区间。容器内运行时使用 `--pid host`，使进程检查与 nvidia-smi 返回的宿主 PID 一致；只映射本次使用的 GPU。测量脚本保留锁频设置，实验操作者负责按运行前状态恢复。

内存检查可使用 `gemm_bench --compare --validate-only --cache hot`。该模式只执行数值验证，不生成计时样本，不能用于性能验收。`configs/gemm_edges.csv` 额外覆盖小矩阵、M/N 尾块和 K 尾块；正式性能用例仍由 `configs/gemm.csv` 固定。

扩展正确性扫描使用宿主 Python ≥3.10，无需管理员权限，也不修改频率或监控服务：

```bash
python3 scripts/run_coverage.py --gpu 0
bash scripts/check_gemm_sanitizers.sh 0
```

第一条命令对显式单级 TMA 扫描 159 个形状、8 种输入分布、3 个种子，每项重复 3 次。随机输入在同一已准备好的 context 上刷新，输出预填 NaN，逐元素对照独立 FP64 DGEMM；零矩阵、全一、矩形单位矩阵和成对消去还检查精确预期值。校验仍使用原容差，候选仍为 FP32 累加。其余 8 条实现路径各检查 24 个代表形状、4 种分布、2 个种子，每项同样重复 3 次。数值失败继续扫描后续形状，进程最终以非零退出，结果目录保存逐项 CSV、日志、二进制副本和源码 hash。

第二条命令对全部 9 条实现路径运行 memcheck、racecheck、synccheck，使用 7 个小型边界形状，最长 K=520，覆盖四级缓冲的多次复用与尾块。sanitizer 的非零退出或诊断错误均使脚本失败。它与广泛数值扫描分开执行，避免把大矩阵的诊断成本扩大到整套测试。

针对长 K 的数值失败，每 256 个 K 元素完成一次 WGMMA 部分和，再用普通 FP32 加法合并，最终只转换一次 BF16。串行分段路径增加一组 64 个 FP32 总和寄存器；Split-K 路径将部分和写入临时空间，随后按 K 顺序合并，避免原子加法。原实现记录见 [长 K 实验记录](experiments/h20-20261116/SEGMENTED.md)。可在空闲 GPU 上运行 `python3 scripts/run_coverage.py --gpu 0 --suite long-k`；该套件对全部 9 条路径扫描 14 个形状、8 种分布和 3 个种子，每项重复 3 次，并分别报告候选和参考状态。参考的既有失败仍保留为整体 fail，不用候选通过覆盖它。原五用例的旧版本计时与本轮优化分别保存，不跨版本合并结果。

该套件已在 qjh004 完成：候选 3024 项全通过，参考 126 项失败；新增长 K 路径的完整 memcheck、racecheck、synccheck 通过。详见 [qjh004 验证报告](experiments/qjh004-20261116/REPORT.md)。随后完成 GPU 诊断计时：五个约定用例加速比为 0.982–1.101×，长 K 分段累加相对旧候选增加约 4.46%–5.37% 耗时，见 [GPU 计时报告](experiments/qjh004-20261116/SPEED.md)。SM 已硬锁、测后恢复；即时显存锁频仍被拒绝，最终性能验收为 inconclusive。

2026-11-17 的后续优化已设为默认：BF16 成对写回、按 K 选择单级/三级 TMA，以及小输出长 K 的 Split-K。最终二进制完成 9888 项组合回归，候选全部通过，另完成默认入口检查和 51 项 sanitizer 任务。八个计时形状均满足诊断耗时比≤1.03；两个小输出长 K 相对固定参考为 10.34×、11.46×，显存硬锁限制仍保留。最新结果见 [优化报告](experiments/qjh004-opt-20261116/REPORT.md)，前述计时为旧版本历史记录。

同日进一步加入有条件的 M 方向 CTA 分组，同二进制开关对照中 square、small_m、m_tail 耗时分别下降约 0.37%、0.96%、0.46%。最新二进制完成新增 4864 项组合检查、16 项默认入口检查和 24 项 sanitizer 任务；原 9888 项属于前一版记录。8 个计时形状仍满足诊断耗时比≤1.03，见 [CTA 分组报告](experiments/qjh004-pipeline-20261117/REPORT.md)。

该 H20 主机的 535 驱动不支持立即锁定显存频率；nvidia-smi 即使返回 0，也可能在正文报告不支持。脚本会拒绝这种设置回执。脚本保留 `--memory-unlocked-reason "具体原因"` 作为诊断入口，其 acceptance_status 为 inconclusive。本次 H20 验收要求必须锁频，因此不使用豁免继续性能验收；监控服务保持运行。

2026-11-16 的 H20 实测、配置取舍和证据索引见 [实验报告](experiments/h20-20261116/REPORT.md)。`scripts/validate_h20.sh` 是该指定主机的复现实验入口，内含固定 GPU UUID 和镜像身份；换机器前必须核对这些参数。
