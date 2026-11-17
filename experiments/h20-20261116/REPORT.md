# H20 GEMM：采用单级 TMA，保留多级流水线作为实验路径

阶段 1–3 已在指定的 H20 主机实现并验证。默认单级 TMA 的五个正式用例均满足 candidate/reference ≤ 1.03，95% bootstrap 区间上界也均低于阈值。向量加载版、2/3/4 级环形缓冲和独立 producer/consumer warpgroup 均已实现，但本轮测量不支持将它们设为默认。

后续增加了 159 形状的正确性扫描，发现混合幅度输入在 K=32768/65536 时超出既定数值容差，候选与固定参考均有失败。不能把原有五个用例的通过外推为通用正确性；新增范围、FP64 复核与失败证据见 [扩展覆盖报告](COVERAGE.md)。

环境验收仍有一个明确限制：该主机驱动不支持立即锁定显存频率。本轮固定 SM 为 1800 MHz，显存遥测始终为 2619 MHz，但没有把观测稳定等同于硬件锁定。最终报告为 timing_status=pass、acceptance_status=inconclusive，表示计时达标、完整锁频验收未通过。

后续用户明确要求必须锁频，并禁止停止 cloud-monitor-agent.service。已检查无服务中断的替代方法：application clocks 请求显存 1593 MHz 后，配置查询显示 1593，但正式五矩阵 GEMM 运行期间显存实际保持 2619 MHz，不能作为有效锁频。直接调用 nvmlDeviceSetMemoryLockedClocks(1593,1593) 返回 code=3 / Not Supported；Auto Boost 状态查询同样不支持。DCGM 配置提供的是 application clocks 入口，未发现独立的即时显存硬锁机制。该限制不是仅由 nvidia-smi 命令封装造成。

deferred 设置可以写入，但需后续 GPU 初始化才生效；先前仅关闭 GPU 0 persistence mode 未使其生效，GPU 0 reset 又被平台监控的设备句柄拒绝。监控服务没有停止。探查后 deferred 请求已清除，persistence mode 恢复 Enabled，application clocks 恢复 1980/2619 MHz，SM 空闲 345 MHz；无 GEMM 进程残留。证据在 evidence/results/strict-clocks-20261116 和 evidence/results/clock-alternatives-20261116。

因此当前不满足“必须锁频”的环境条件，停止后续性能验收，不把豁免结果升级为 pass。复现实验脚本已撤掉显存豁免参数，锁频不成功即终止。若平台不能提供无需中断监控的 GPU 重新初始化途径，需要由平台安排维护窗口预先激活 deferred 锁频，或使用已具备有效硬锁的节点。[NVIDIA 官方 nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)明确说明 Hopper 显存使用 deferred 方式，下一次 GPU 初始化时生效。

## 默认候选的逐用例结果

最终运行是 results/20261116T025544Z-b276839f，每个用例 30 组 AB/BA 配对采样。表内耗时是独立中位数，比例是配对耗时比的中位数，因此两列中位数直接相除可能有微小差异。

| 用例 M×N×K | 参考 ms | 单级 TMA ms | 配对耗时比 | 耗时比 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| square：4096×4096×4096 | 1.053374 | 1.009499 | 0.958360 | [0.958324, 0.958404] |
| small_m：1024×4096×4096 | 0.304958 | 0.276866 | 0.907823 | [0.907684, 0.908001] |
| short_k：4096×4096×1024 | 0.279764 | 0.275189 | 0.983596 | [0.983455, 0.983683] |
| m_tail：3968×4096×4096 | 0.980687 | 0.999015 | 1.018636 | [1.018204, 1.018880] |
| mn_tail：4104×4088×4096 | 1.057143 | 1.009641 | 0.955091 | [0.955050, 0.955183] |

四个用例比参考快约 1.64%–9.22%，m_tail 慢约 1.86%，仍在 3% 预算内。3968 可被当前 M tile=64 以及参考 M tile=128 整除，因此这个 ID 不能作为“存在 M 方向不完整 tile”的证据；该用例的回归原因尚未单独定位。

## 实现路径和未采纳的方案

普通加载版使用 64×128×64 tile、128 个线程，一个 warpgroup 发出四条 K=16 的 WGMMA。A/B 沿连续 K 维做 16 字节向量加载，shared memory 使用 128-byte swizzle。每个线程持有 64 个 FP32 accumulator，最后转换成 BF16，按线程映射和输出边界写回。每次计算完成后等待 WGMMA，再允许下一轮覆盖操作数。该版本解决了布局、descriptor、输出映射和边界正确性，但性能明显落后。

单级 TMA 保持同一 tile 和计算路径，把普通加载替换成两个二维 TMA 事务。prepare 编码 tensor map，launch 只提交计算。一个 stage 的 A/B 合计 24 KiB；mbarrier 的期望事务字节数与该容量一致，等待 phase 完成后才发出 WGMMA。输出仍用普通全局写回，没有 TMA epilogue。

环形缓冲保留 1–4 级配置。每个 stage 复用前，先完成引用它的 WGMMA；同一 warpgroup 的搬运线程在其他 stage 计算期间提交后续 TMA。独立分工版本则使用 256 个线程，分成计算和搬运两个 warpgroup，以 full/empty mbarrier 协调生产、消费和复用。当前搬运组由一个线程发出事务，没有实现动态寄存器再分配。

| 版本 | square | small_m | short_k | m_tail | mn_tail | 计时判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 向量加载，最终二进制 | 2.6729 | 2.9663 | 2.2924 | 2.8743 | 2.6117 | fail |
| 单级 TMA，最终豁免运行 | 0.9584 | 0.9078 | 0.9836 | 1.0186 | 0.9551 | pass |
| 两级 TMA，早期对照 | 1.0106 | 0.9163 | 1.1397 | 1.0496 | 1.0371 | fail |
| 三级 TMA，早期对照 | 1.0116 | 0.9335 | 1.1418 | 1.0466 | 1.0038 | fail |
| 四级 TMA，早期对照 | 1.0218 | 0.9400 | 1.1597 | 1.0518 | 1.0409 | fail |
| 两级 TMA + 独立 warpgroup，最终二进制 | 1.0187 | 0.9342 | 1.1742 | 1.0516 | 1.0209 | fail |

表中数字均为相对同轮固定参考的配对耗时比，不是不同版本之间的直接配对比较。早期环形缓冲实验与最终二进制的源码组织有变化，不能把它们视为只改一个参数的最终二进制扫描；原始二进制 hash、源码 patch 和运行配置均留存。所有性能结论都受显存未硬锁限制，早期 summary 中的完整 pass 标记已被下述锁频问题更正。

Nsight Compute 对 4096³、hot cache 下的一次候选 launch 采集 SpeedOfLight，得到如下诊断指标：

| 版本 | Compute (SM) Throughput | DRAM Throughput |
| --- | ---: | ---: |
| 向量加载 | 34.69% | 2.41% |
| 单级 TMA | 94.97% | 6.99% |
| 两级 TMA | 90.48% | 7.12% |
| 两级 TMA + 独立 warpgroup | 89.77% | 11.53% |

这些 profile 不是 rotating cache 性能验收样本，也不是对所有 shape 的瓶颈证明。它们支持一个有限判断：4096³ 的单级 TMA 已有较高计算利用率，增加搬运流水线并未进一步提升该指标。更多 stages 会增加 shared memory 占用，独立搬运组还增加线程和寄存器预算；不能仅凭这些资源成本就断言它们解释了全部回归。

## 正确性、计时和锁频证据

四条主要路径——向量加载、单级 TMA、两级 TMA、两级独立分工——均通过 5 个正式矩阵和 4 个额外边界矩阵的 compute-sanitizer memcheck，错误数均为 0。边界矩阵覆盖 M=1、M/N 不完整 tile 和 K 不足 64。两级 TMA 和两级独立分工在四个边界矩阵上的 racecheck 均为 0 errors、0 warnings。三级/四级普通 TMA 通过正式计时运行内的数值检查；独立分工的 1/3/4 级配置只完成编译，未宣称动态验证通过。

正式数值校验逐元素比较独立 pedantic FP32 SGEMM 的结果，输入先精确按 BF16 表示，容差为 abs_error ≤ 0.02 + 0.01×abs(expected)。计时前检查完整写回，计时后检查实际留下的输出。validate-only 只用于数值和 sanitizer 检查，不产生性能结论。

性能采用普通 kernel launch、CUDA events、30 轮交替 AB/BA；每批按参考校准到至少约 250 ms。输入环超过 4 倍 L2，不使用 CUDA Graph。bootstrap 采样 4000 次，逐用例判断 95% 区间与 1.03 阈值，任一用例失败则整体失败。最终候选的相对 MAD 均小于 0.05%，遥测未发现外部计算进程或 SM 频率偏离。

本轮修复两个测量问题。首先，容器 PID 与宿主 nvidia-smi PID 不同会误判外部进程，复现容器改用 --pid host。最初两轮因此 inconclusive，未作为合格环境证据。其次，nvidia-smi 对 -lmc 返回 0，却输出“This option is not supported”，旧脚本只检查退出码会误报锁频成功。脚本现同时检查不支持/拒绝回执；真实回归验证已确认它会在 GPU benchmark 启动前报错。后续按显式 memory-unlocked-reason 重跑参考和候选，保留 inconclusive，未使用 deferred lock，也未重置 GPU 或重载驱动。

## 机器、复现与证据位置

主机为 fengyunhao@115.191.53.247（qjh000），8 张 NVIDIA H20，本轮仅使用 GPU 0：GPU-d1b595ce-0b57-cdee-c1c0-9b1051aaa07e。设备报告 78 个 SM、60 MiB L2；宿主驱动 535.161.08，nvcc 12.8.93。使用宿主 CUDA 12.8 编译和运行，容器提供 Python ≥3.12。容器镜像完整身份和二进制 hash 见 evidence 内 manifest 与 validation SHA256SUMS。

CUTLASS 固定在 ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e（3.9.2），参考 tile=128×128×64、cluster=1×1×1，实际自动 stages=6。参考不宣称是全部可选实现中的最优者；本结论仅适用于仓库约定的固定参考与五个测试矩阵。

服务器目录为 /home/fengyunhao/bare-hopper-20261116a，原始 telemetry、不可变二进制、源码快照和实验日志继续留在该目录。本地只归档紧凑证据，不下载全量原始数据：

- 最终候选：[summary.json](evidence/results/20261116T025544Z-b276839f/summary.json) 和同目录 manifest、samples、source.patch。
- 显存豁免参考：evidence/results/20261116T025454Z-7dc38c6c。
- 内存检查、racecheck、profile：evidence/results/validation-20261116T024441Z。
- GPU 设置恢复：evidence/gpu0-restored-final.txt、evidence/restore-clocks-final.log。

复现入口为 scripts/validate_h20.sh，硬编码了本轮指定 GPU 和镜像，换机器前必须重新核对。默认候选不需要环境变量；BARE_GEMM_STAGES=0 切回向量加载，1–4 选择 TMA stages，BARE_GEMM_WARP_SPECIALIZED=1 启用独立分工。脚本会保留锁频设置，操作者需负责恢复。本轮结束时 GPU 无计算进程，SM 恢复空闲 345 MHz，显存 2619 MHz，application clocks 与运行前同为 1980/2619 MHz；其他七张卡未改设置。

本地 pyright、ruff check、ruff format --check 和现有 15 项 pytest 均通过；CUDA 编译完成，所有候选均无 spill。文档和代码未在本地 commit 或 push。

后续 GEMM 优化应先定位 m_tail 的剩余回归，再决定是否需要调整 CTA 排布或 tile；已有数据不支持继续默认增加 stages。若要取得严格完整锁频 pass，需要在支持显存硬锁且可保持同样测量口径的环境重测；目前不据此启动 Attention 工作。
