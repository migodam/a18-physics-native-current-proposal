<!-- current-result -->
> 本文件保留实验前规格与预算；当前实测状态见 [GATE_DECISION.json](GATE_DECISION.json) 和 [中文报告](RESEARCH_REPORT_ZH.md)。

# GPU_JOB_ORDER：单台 RTX 4060 Laptop

**唯一 CUDA worker；所有 GPU 工作严格串行。以下是资源预算/估计，不是本轮实测，也不是执行完成承诺。**

## 1. 总上限

首轮所有 NN pilot：**240 GPU 分钟硬上限**，推荐路径约 70–155 分钟。包含 warm-up、validation、失败尝试和重启；不是仅记录成功 epochs。C：0 分钟。

共享 teacher 缺失时：先运行一个独占 physics-preparation job，最多 60 GPU 分钟；另记 CPU teacher active wall 上限 120 分钟。NN gate 通过后的唯一赢家 nonlinear transfer 最多 60 GPU 分钟，独立于 NN 预算。最坏的全部扩展上限合计 360 GPU 分钟；通常跳过大多数 optional jobs，不将它宣传成实测总耗时。

若已有 teacher 完整，GPU preparation 不运行。没有兼容缓存时先修数据，不启动网络空转。

## 2. 精确排队次序

|顺序/job|启动条件|参数目标|数据/epochs|计划 GPU 时间（上限）|peak VRAM 目标*|早停|
|---|---|---:|---|---|---|---|
|P0 physics cache|确实缺 labels且 backend 可用|无 NN|12-state preflight，再最多96states|按首2states实测外推；硬上限60min|<6GiB，实际记录|oracle/cost不通过就停，不生成全dataset|
|J0 smoke/calibration|CPU tests和schema通过|极小|2states，20steps|2–5min（10min）|<1.5GiB|NaN、packing/adjoint错、warmup超内存即停|
|J1 A1-MLP|A-lazy可随机访问|0.03–0.08M|batch16 states，40epochs|5–15min（20min）|<1.5GiB|epoch10无validation机会改善；patience10|
|J2 B-MLP|B representation/cost gate通过|0.2–0.5M|batch8 states，80epochs|15–35min（45min）|<2.5GiB|epoch20不优于no-NN；patience15|
|J3 A1-GNN|J1有headroom；边可廉价构造|0.1–0.25M|同J1，40epochs|10–20min（25min）|<2.5GiB|epoch15与MLP差<.01就停结构探索|
|J4 B-Set|J2通过且尚有>=.05headroom|0.2–0.5M|同J2，最多60epochs|15–30min（35min）|<3GiB|epoch20无优势停止，不加层|
|J5 winner seeds 2/3|只有已通过部署gate的一个模型|不变|每seed复用锁定epochs上限|20–50min（60min）|同赢家|任一seed失败如实保留，不重抽seed|
|J6 freeze/timing|停止所有优化器|不变|held-out一次；每state5次计时|5–10min（15min）|实测|超时记录missing，不隐性缩小难例|
|J7 contingency|已有任务失效的诊断，不准新模型|不变|至多一次修复重跑|余额内（30min）|按原job|超过240min NN总上限即全部停止|

* VRAM 数字是小 cached-feature 网络的 planning target，不是 GPU 容量承诺；启动时读取实际 free memory，保留至少 20% headroom。记录 allocated 与 reserved peak，另报进程RSS。FP64 teacher/cache 不全部搬进 GPU；小训练矩阵按 batch 搬运。

## 3. 强制校准而非猜测速度

J0 跑 20 个真实 minibatches，丢弃前 5 个 warm-up，记录每 step 的 median/p95、validation CPU 回传和 transfer。预测

$$T_{job}=N_{epochs}N_{batches}\,t_{step}+N_{validations}\,t_{validation}+t_{load}.$$

若预测超过 job hard cap，先减少 epochs 到 floor(cap/time)，仍不足最少10epochs则取消 job；不自动缩小物理 teacher、不静默改网络数值精度。Laptop 功耗/温度/CPU transfer 的差异由此测量吸收。

NN AdamW 用 float32；B 小矩阵训练可 complex64，validation complex128 CPU；每5epochs对缓存teacher做精度对照。若 CUDA 的 small complex solve 比CPU慢，允许 CPU-only neural pilot，GPU分钟记0、CPUwall照计，不为了“用了4060”而迁移。

## 4. 多 agent 协议

共享目录 `runs/gpu_queue/`。每 job 有 JSON：owner、route、source/data hash、status、max_wall、actual start/end、PID、checkpoint、exit code、failure reason、peak memory。一次只有一个锁持有者。

使用原子目录 `gpu.lock/` 作为互斥锁；持有者正常退出才删除。发现 stale lock 必须先确认对应PID/主机任务已停止；不得因为等待超时直接抢锁。CPU agent 只写自己分支和 `cpu_outputs/`，不得修改共同冻结cache。

建议三个开发角色：Integrator/Data（CPU）、Agent A（CPU实现A）、Agent B（CPU实现B）。只有 Integrator 启动队列。A/B agent 不直接 `cuda()`、不自己安装破坏共享环境、不设置后台长期训练。

## 5. 不可越过的停机规则

每 job 结束即写 immutable receipt，再决定下一 job。GNN/Set/C 默认 disabled，gate JSON 才能开启。训练失败和无结果也消耗预算；不能用“重跑不算”挤占别的路线。

J1 A1 如果 oracle bank coverage 不够，跳过 J1，B不必等待。J2如果manifold成本失败，B停止但A可继续。若二者都失败，直接交付负结果，GPU队列结束。
