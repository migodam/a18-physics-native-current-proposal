# START HERE — A18 Physics-Native Current Proposal

**裁决：B 是有条件主线，A 是必须并行的实用对照，C 只保留理论。** 本包完成理论、文献审计、实施规格和 CPU 代数测试；真实 Maxwell/NN pilot 尚未运行。A17 主稿保持独立。

|路线|检验的核心假设|现在做什么|
|---|---|---|
|B / Primary|廉价 Schur-feedback current 空间足以覆盖好动作，任务先验能找到它们|CPU oracle/cost gate；通过后约0.3M MLP|
|A / Secondary|不生成全 bank，仅凭 recipe descriptors 能提出有用的 atom/drop 动作|lazy recipe audit；先MLP，GNN有条件|
|C / High-risk|同一物理空间存在 deterministic 多head仍漏掉的多region机会|不训练，不上Flow；保留projective分布规格|

## 最核心数学结构

$$P=I-UU^*,\qquad F_U^{eff}=PFP+PFU(I-U^*FU)^{-1}U^*FP.$$

O 用 dressed observation seed 与 adjoint chain；P/M 用 current-residual/material-injection seeds 与 forward chain。默认6 seeds、degree0/1/2，联合正交 rank≤18，hard cap24。

删去 u_i 后，网络给单位系数 t，提出 **c=[u_i,Q_U]t**，保留 partial rotation 与 no-op。由原 A17 endpoint solve 和同一 full frozen finite gain决定接受；不是以 NN 分数接受。degree惩罚对应嵌套层，而不是任意QR列号。

## Codex 应该先跑什么

Integrator 先运行 `python tools/run_algebra_tests.py`，再把真实A17 backend接到共享接口，在2个真实states验证原endpoint/gain一致。接着只做12-state无NN representation/cost preflight。给两个agent分别发送 [CODEX_ROUTE_A.md](CODEX_ROUTE_A.md) 和 [CODEX_ROUTE_B.md](CODEX_ROUTE_B.md)；两者共享数据、teacher、verifier和cost ledger，不共享有额外训练优势的checkpoint。

## 唯一4060怎么排队

CPU开发并行；GPU严格 **共享physics准备 → A-MLP → B-MLP → 条件性A-GNN/B-Set → 仅赢家追加seeds**。所有NN job合计最多240 GPU分钟，含失败；单job与显存预算见 [GPU_JOB_ORDER.md](GPU_JOB_ORDER.md)。这些是预算/估计，不是已测性能。所有训练epochs只读cache，不重跑Maxwell。

## 出现这些结果立即停止

A/B manifold或recipe oracle coverage不足，或者acquisition未明显省：不训练。最强同空间无NN已达.95 reference：停止NN。B不在coverage/acquisition上优于A：A升主线。GNN/Set不优于MLP：保留MLP，不claim physics决定网络架构。deterministic已充分覆盖：C继续关闭。frozen gain不代替最终成像验证。

完整理论与依据：[THEORY_OVERVIEW.md](THEORY_OVERVIEW.md)；执行配置：[config/pilot.json](config/pilot.json)；最终判据：[GO_NO_GO_GATES.md](GO_NO_GO_GATES.md)。

**交付状态：本地Git仓库已整理；远端尚未创建/推送，因为本会话GitHub连接仅提供读取操作。发布步骤见 [PUBLISH.md](PUBLISH.md)，不把准备完成称为已经上线。**
