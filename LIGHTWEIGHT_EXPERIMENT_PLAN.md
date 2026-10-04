# 轻量实验计划与共同评估

## 1. 本轮有三个不同的结果状态

- `ALGEBRA_TESTED`：本仓库可在 CPU 运行的 dense algebra tests；不等价于 Maxwell 实证。
- `SPEC_READY`：A/B 的训练与评估规格已固定；本交付不声称已训练网络或测得 4060 速度。
- `PHYSICS_PILOT_PASS/FAIL`：由 Codex 在真实 backend 与锁定数据上产生；未运行时必须为 `NOT_RUN`。

## 2. 公平 baseline 集合

|代号|作用|是否 NN|主要成本口径|
|---|---|---|---|
|R0 receiver-only|原有重建/step 基准|否|无 exchange 的共同 outer cost|
|D-full|完整原 bank + 原解析 ranking + 相同 verification budget|否|真正 acquisition、anchor/workspace、finalist 检查|
|D-recommended|A17 最新 cached-geometry top-1/cap-3 部署规则 [P2]|否|用于现实总成本对照，不与旧低效实现偷换|
|A0-replay|原数据 wiring 与历史 pool coverage|是/可不训练|若全 bank 已建，不算加速|
|A1-analytic|实际 lazy bank 的固定 family union/cheap analytic policy|否|与 A1 同 features 与 K|
|A1-MLP/GNN|lazy joint-action prior|是|没有完整 anchor/workspace|
|B-seed|degree-0 O/P/M|否|seed acquisition + 相同 K|
|B-poly|固定 low-order Neumann / chain-GMRES coefficients|否|与 B 相同 chain actions；不额外求 inverse|
|B-ridge/KNN|训练对象上的小型 coefficient prior|非神经学习|独立记录离线拟合/检索成本|
|B-MLP|本轮主路线|是|Q、features、selected endpoints、Jd 全计入|
|B-best-found|多起点连续参考|teacher-only|不作为可部署同成本算法|

精确 baseline 算式及 outgoing 规则见 [BASELINE_SPEC.md](BASELINE_SPEC.md)，不得自行给 baseline 增加免费 matvec。

B-poly 具体候选：每条 forward chain 的 degree 0、1、2 truncated Neumann response；以及以同 chain Arnoldi Hessenberg 求解的最小 residual polynomial。seed family 的 fixed order 与 K=4 的选择只在 validation 冻结。O-adjoint chain 只用于同构候选，不把它误当 forward solve。Ridge/KNN 的 phase-free target 用 teacher projector 或固定 gauge coefficient，不能直接平均相反相位。

## 3. 主指标定义

对于 state q，reference 中最佳正增益为 $g_q^*$；已生成动作中经 original FP64 verifier 确认的最佳正增益为 $g_q$。reference 无正机会的 state 单独统计 no-op/invalid/误触发，不放入 ratio 分母。

**Object-level opportunity coverage：**

$$\operatorname{Cov}(o)=\frac{\sum_{q\in o} g_q}{\sum_{q\in o}g_q^*},$$

再在独立 objects 上取 macro mean、median、worst-family；另报 per-state coverage 分布。分母接近 online significance scale 的 state 单列，绝不靠加很大 epsilon 美化。

相对原 dictionary 的 coverage 可大于 1（连续方向更好），不裁掉；相对 common evaluated union reference 不超过 1（除数值误差）。**这里的 coverage 是 gain coverage，不是分类准确率或“进入 top-m 的比例”。**同时报告 opportunity-hit：是否至少一个 $G>\tau$。

**Accepted finite gain：**固定同一 policy、最多三次 conditional exchange，报告每轮实际 $G$ 与累计 gain；后续 rounds 必须重算条件。不能把三个初始-state gains 加起来冒充 sequential gain。

**候选数量：**分别报告低维 records、实际构造的 unique current vectors、selected atomic actions、endpoint solves、full Jd directions、各 illumination RHS。K=4 指每轮四个 atomic finalists，不是四个 random seeds 后偷偷试 100 个动作。

**成本：**$T_{acq}$ 包括 state features、seed、Arnoldi、投影/QR、网络、CPU/GPU transfer、selected atom/current construction；$T_{inc}$ 为与 receiver-only 配对的额外完整耗时，包含 endpoint、verification、失败/fallback、cache rebuild。还报完整 $T_{total}$，不要只报预测延迟。

**被消掉的 machinery：**记录 `anchor64_builds`、`workspace128_builds`、`all_dictionary_current_builds`、全部 L/L* 次数与输出 bytes。新 manifold 的 Q 构造必须另列，不能用“避免 128”掩盖新建 120 列。

**泛化：**对象而非 action 的 split；near-contact、高 contrast、receiver geometry 变化独立分层。训练相同 material grid 不证明跨 resolution/frequency；未测则写 NOT_TESTED。

## 4. 四步实际执行

### E0：接通/锁定

先运行 dense tests，再对 2 个真实 states 做 original vs adapter endpoint/finite gain comparison。验证 current metric、polarizability derivative、real packing、same r/prior、singular-core fallback。任何关键 mismatch 不进入训练。

### E1：无 NN oracle 与费用，12 states

A1 lazy bank best reference vs 原完整 bank；B depth0/1/2、raw PFP/Schur、forward-only/two-sided、shared-root/per-core（后者 teacher-only）。记录 best-found coverage 与完整 acquisition wall。只对 depth2 有 oracle 与时间双重前景的表示继续。

### E2：NN tiny pilots，缓存后串行训练

共享 dataset 不超过 96 physical states。A1-MLP -> B-MLP -> 条件性的 A1-GNN -> 条件性的 B-Set -> 仅赢家追加两 seeds。默认 C 不在队列。超参不得根据 locked test 修改。

### E3：冻结后一次 held-out 与真实在线 timing

冷 cache、几何 warm cache 都测；共同 allowed geometry caching 对所有方法一致。每 state 的方法顺序轮换，至少 5 次 timing repetitions，median 与 p95 并列；这些不是新的物理样本。GPU 上不可并行任何别的 agent 工作。

Bootstrap 按 object 做 2,000 次 paired resamples，报告 coverage 与 $T_{inc}$ 差异的区间；只有 8 个 test objects 时如实提示区间宽，不把 288 state-budget queries 当独立 288 样本。

## 5. Pair 与非线性作为不同层次的测试

pair audit：12 states，预注册最多 6 个 drop pairs；联合 rank-2 endpoint 与两个 singleton 分开比较。若 pair-only opportunities 多，结论是需要 block-action proposal，不是自动需要 generative C。

nonlinear transfer：只允许 E3 通过的唯一赢家进入；使用原 constraints、line search、damping、forward residual。最小两个对象（smooth/near-contact），共享初始化与停机规则，报告 material error、held-out measurements、非线性 objective、完整 wall time。此阶段上限 60 GPU 分钟，若跑不完保留 incomplete，不能把 incomplete 当零改善。

通过 frozen gain gate 只获得运行非线性 transfer 的资格，不自动获得“更好成像、更快重建”的主张。A17 的 top-1/cap-2 反例使这一分离尤其重要。[P2]

## 6. GPU 与 CPU 分工

CPU 可并行写 adapter、做 source audit、生成小矩阵 teachers、Arnoldi cost microbenchmark、统计与画图；大规模 L action 若后端用 GPU 则必须入同一独占队列。训练期间不得另一个 agent 在 GPU 上生成 physics data。

所有 plot 从 sealed JSONL/CSV 生成；失败、timeout、NaN、invalid endpoint 作为状态码保留。code/model family 名字不能暗示未验证 physics equivalence。共享 encoder 只能作为单独 amortization branch，不污染独立训练 ablation。
