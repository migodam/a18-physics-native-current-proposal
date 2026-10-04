# Route B：Schur-feedback current coordinates + deterministic neural prior

**定位：Primary，立即开发；先过无 NN oracle/cost gate，再取得 GPU 训练资格。**

## B1. 表示与 action space

按 [THEORY_OVERVIEW.md](THEORY_OVERVIEW.md) 和 [ORTHOGONAL_POLYNOMIAL_THEORY.md](ORTHOGONAL_POLYNOMIAL_THEORY.md) 构造 root $Q_U$，默认 rank 不超过 18，hard cap 24。对每个合法 outgoing slot $i$：

$$Z_i=[u_i,Q_U],\quad t\in\mathbb C^{r+1},\quad \|t\|=1,\quad c_i=Z_it.$$

每 slot 的网络输出 **两个 coefficient modes 和两个 proposal logits**：

$$\{(v_{i,h},\ell_{i,h}):h=1,2\},\quad
 t_{i,h}=v_{i,h}/\|v_{i,h}\|,\quad\pi_{i,h}=\mathrm{softmax}(\ell).$$

最多 $2k$ 个低维 action records；确定性挑 4 个，**只实例化与验证所挑的 4 个 current vectors**。另有 no-op logit。若 norm 小于 $10^{-8}$，该 mode 无效；不得除以小数后伪装合法。

此设计不枚举或构造原大 dictionary，不为每个 outgoing slot 重建 Krylov basis，也不输出高维 current image。对最终四个动作的 finite gain 仍逐个由 A17 backend 计算。

## B2. 种子与 phase convention

六 seed 的默认内容：

- O 两条：对已 phase-canonicalized 的各 illumination 残差做两列固定 sketch，再用 dressed $Y_U^*$ 回投。
- P 两条：对 $P(f_p-LR_Uf_p)$ 做同样两列 sketch。
- M 两条：$z_1$ 为已有 reduced material step 的单位方向，$z_2$ 为在固定 material gauge 下的 train-frozen Rademacher probe；对 $B_pz_j$ 做预定 illumination contraction，再应用 $P(I-LR_U)$。

illumination sketch 的列数、seed、源相位约定写入 manifest。不能任意重排 illumination 后沿用旧 sketch 权重。先按已知 incident-source phase canonicalize；若相位约定未知，保留完整 illumination seed span 作有收费的控制，不能宣称压缩 sketch 精确保持独立 illumination phase invariance。

默认 O 用 outer residual $r$，不是需要新 full tangent 才能得到的 $r+J_Fs_U$。后者仅在 A17 已经付费缓存时用于独立 ablation。$z_1=0$ 或某 residual seed 数值为零时记录 inactive，使用预注册的第二 probe，不随机救场。

## B3. 输入 feature contract

允许输入：旧 $U,LU,SU$ 的小摘要；contrast/polarizability 与 geometry 摘要；$r$、$\Lambda,\ell,s_U$ 的规范化摘要；chain Hessenberg；chain-to-joint 坐标；O/P/M projected seed coefficients；layer/missing masks；drop-slot descriptor。

默认每 state flatten 后不超过 2048 real numbers；每 outgoing slot 不超过 32 real numbers。feature 列由 [FEATURE_CONTRACT.md](FEATURE_CONTRACT.md) 与 `config/feature_schema.json` 固定，训练/验证/部署相同。保存 scale 和 masks，不能按 test labels 重新归一化。

**禁止输入：**full $J_F,H_F,s_F^*$、$Q$ 的 exact future gain、原 64-anchor features、需全 bank 才能得到的排序、真值材料、完整 projected oracle matrices。这些只能进入 teacher/evaluator 的隔离 namespace。

主网为共享 encoder `input -> 128 -> 128`，GELU、LayerNorm；与 outgoing descriptor 拼接后 `->128->2*(2*(rmax+1)+1)`，外加 no-op head。padding coefficient 必须乘 rank mask。目标 0.2–0.5M 参数，实际计数入账；不需要 voxel CNN、不用 transformer 编码整个 current grid。

可选 B-Set 用相同 O/P/M chain tokens 的两层 hidden-64 attention，只有 B-MLP 通过 gate 才跑。没有证据就不声称 neural architecture 被反馈唯一决定。

## B4. Teacher：exact endpoint algebra 可完全移到小矩阵

离线对每 frozen state 构造 $Z=[U,Q_U]$，仅 teacher 保存

$$A_Z=Z^*LZ,\quad O_Z=SZ,\quad N_{Z,p}=Z^*B_p.$$

任意 orthonormal endpoint coordinate $E$（大小 $(k+r)\times k$）满足

$$J_{E,p}=O_ZE(E^*A_ZE)^{-1}E^*N_{Z,p}.$$

这是 exact compression identity，不是 learned closure。所有 illumination realify 后解同一 regularized material quadratic。给定已缓存 $J_F$ 或 $H_F,h_F$，其 finite gain 也只涉及小 material 矩阵。**因此训练 epoch 不需要全 Maxwell，不需要 full current prediction。**

第一轮仅在可以完整缓存 full real material reference 的小 material cohort 上使用 direct-gain training（优先 24 real parameters）。大 voxel material task 不能用 16 probe 的 projected utility 偷换 full utility：那里只能做标签 distillation，并在部署验证时使用真实 full $Jd$。

每个 drop slot 的 teacher：no-op、projected best original-dictionary candidate、analytic seed/polynomial starts、随机 projective starts。preflight 每 slot 最多 8 starts × 40 steps；dataset 扩展每 slot最多 4 starts × 24 steps。projected dictionary start 只能离线使用，不能当在线 cheap feature。保留每 slot 至多 2 个 phase-separated（$|t^*t'|^2<.9$）、有效正增益的最好解及其原始 gain。

优化：complex coefficient 拆成实/虚，归一到 sphere；CPU complex128 autodiff；目标 $-G/g_{ref}$；Adam lr .03，每步 sphere retraction。endpoint 不稳定的 iterate 不记 label，沿上一步到当前步做最多 5 次半步回退；全部失败则停止该 start。禁止向 endpoint current matrix 偷加 ridge。最终用原 FP64 verifier 重算。

这些是 **best-found feasible continuous labels**，不称全球最优。若 oracle搜索无收获，先看原字典最优投影、phase距离、endpoint stability 与 rank，再判断表示 gate；不通过就不训练网络。

## B5. Loss：覆盖优先，不做纯 current MSE

对每 $(i,h)$ 使用 teacher cache 在训练端算 $G_{i,h}$，以及学到的 $\pi_{i,h}$。$V_K$ 使用 Route A 的 expected-best 公式；这里 gain 对 coefficients 是可微的（排序边界外）。reference 是冻结且 detached 的已知 best-found positive gain。

$$L_B=L_{cover}+0.25L_{list}+0.20L_{proj}
+0.01L_{div}+10^{-3}R_{order}.$$

- $L_{cover}=1-V_K/(g_{ref}+\epsilon_{ref})$；reference 为零时换 no-op loss，并保持原始 gain 日志。
- $L_{list}$ 的 target 是 per-outgoing teacher 最佳正增益的归一化质量；对同一 outgoing 的两个 modes 求和后比较，避免人为规定 mode 顺序。
- $L_{proj}$ 在同一 outgoing 内对 teacher modes 与预测 modes 做最佳二元匹配；距离为 $1-|\langle t,t_{\rm teach}\rangle|^2$，实现写成 `1 - abs(vdot(t, teacher))**2`。不同 phase 的同一 direction 不受惩罚。无 teacher mode 的 slot 不做该项。
- $L_{div}$ 只对同一 outgoing 的两 modes、且有多 mode label 证据时使用 `relu(abs(vdot(t1,t2))**2 - .9)**2`。不强行让每个 state 产生两个互斥好方向。
- $R_{order}$ 使用 layer operator；旧方向 $u_i$ 的权重为零。

AdamW 的 decoupled weight decay 为 $10^{-5}$，不另外叠加 L2 loss。正 utility 只计 feasible 且 $G>\tau_{\rm online}$ 的 action，signed raw G 同时保存。

当前方向的 exact gain 是 loss 的物理部分，不是再训练一个 Q surrogate。训练可以用 complex64 cached small solve 获取梯度，但 validation/labels/最终接受全部由 complex128 路径决定；32/64 差异超出阈值即停止 mixed-precision 分支，不把它升级为安全界。

默认 lr $10^{-3}$，AdamW，batch 8 states，max 80 epochs，gradient clip 1；每 5 epochs 进行真实 deterministic top-4 validation。所有超参先在 train/validation 锁定，test 只跑一次。

## B6. 完整在线 pseudocode

```text
B_PROPOSE_AND_VERIFY(state):
    assert original_endpoint_and_tolerance_adapter_is_frozen
    U, LU, reduced_step = get_already_paid_outer_state()
    seeds = build_OPM_seeds_without_new_inverse_solve()
    Q, layers, cheap_features = joint_feedback_flag(seeds, depth=2, cap=24)
    if root_stability_or_orthogonality_fails: return declared_fallback
    coefficients, logits = prior(cheap_features, outgoing_slot_features)
    mask zero/rank-invalid coefficient records; include no_op
    proposals = deterministic_top4(logits, stable_tie_break, unique_action_record)
    best = no_op
    for (drop_i, t) in proposals:
        c = [u_i, Q] @ t
        V = U without u_i
        assert V* c approximately 0 and norm(c) approximately 1
        # Obtain Lc directly; do not build all LQ for a small K.
        s_new = original_direct_endpoint([V, c], state, cached_LV, new_Lc)
        if endpoint fails original gates: log and continue
        d = s_new - s_old
        z = full_J_action(d)       # all illuminations, charged
        G = original_finite_gain(u, z, d, prior, damping)
        if G > original_tau and greater than best.G: best = action
    if best accepted:
        U = best.endpoint; s_old = best.step; u += best.z
        # changed U => changed Schur operator, seeds, coefficients
        repeat with a fresh cache key, up to 3 accepted exchanges
    else: abstain in this pool
```

A17 的 common-core Schur formula 可用于已稳定 core 的 endpoint 小算式；默认实现 direct endpoint 最容易避免奇异 core 误拒绝。

## B7. 不变性、diversity 与 pair synergy

$c\mapsto e^{i\phi}c$ 不改变 trial space，gain 必须不变。层内 $Q\mapsto QW$ 时 coefficients、projected features、degree matrix 必须一起变换；基础 MLP 不自动具备 arbitrary unitary equivariance。采用 deterministic QR gauge + phase/permutation augmentation，并报告 gauge stress test。输入 receiver unitary change 应对 $S,r$ 一起变换；不能只动一侧。

不同 outgoing 的两个 action 可以共享同一新方向，不能因此误判完全重复；它们的 endpoint 不同。最终 pair synergy 测试必须显式删除两列并加入一个 rank-2 block：$C=[A,Q]T$，$T^*T=I_2$。两个 singleton 均负而 pair 为正时，应原子性检查 joint endpoint，不能逐个接受/拒绝。

本轮训练 headline 是 singleton fixed-rank proposal；另保留 12 个 state 的 **无 NN pair oracle audit**，每 state 最多 6 个预定 drop pairs、每 pair 4 starts × 20 steps。只有 pair-only 机会占比显著，才另行申请 joint-block head；不通过“加 4 个独立 samples”冒充 pair 建模。

## B8. 复杂度与 kill conditions

六 chain、depth 2：理想 12 次 Feff/adjoint actions，另外 observation seed 最多 2 次 adjoint；再加最多 4 次 selected Lc、seed construction 与旧 LU 缓存。若在实际后端需额外 action，按真实值报告，不用理想数替代。

Gram/orthogonalization $O(nr^2)$，重建 K currents $O(nrK)$，网络与小 records 基本与 n 无关。完整 $Q$ 可保留在 CPU；在该环境下小网络 CPU inference 可能更快，必须与 GPU+transfer 实测比较。

直接停止 B：shared-root manifold 的 best-found coverage 不足；达到 coverage 需要过深/过宽导致 acquisition 不省；输入只能靠大 anchor 获取；no-NN 同空间 prior 已接近 reference；或训练获益只存在于历史同对象、不在 held-out object。所有失败都可以保留为负结果，不通过扩大网络抢救。
