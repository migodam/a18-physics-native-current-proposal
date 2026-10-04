# Route A：Lazy dictionary action prior

**定位：Secondary / 立即开发。先跑小型 MLP；GNN 是有条件的结构对照，不是默认主角。**

## A1. 科学假设

在不生成所有 state-dependent currents 的前提下，cheap descriptors 足以预测哪些 dictionary recipe 值得被实例化。此路线的优势必须来自 acquisition avoided，而不是“128 个向量已经建好以后挑出 4 个”。

A17 原 bank 有 receiver、internal propagation、Fourier、anchor primal/dual、anchor residual、random 六类。[P1] 后三种 state-dependent 工厂尤其不能假设可以随机访问。单凭现有 incoming labels 无法证明其在线 feature 免费。

因此区分两个实验，名称不可混用：

- **A0-replay：**按原 dictionary/原 label 接通训练与评估，仅用于正确性和历史 coverage 对照。若需要原 anchor/workspace，它不算 deployment success。
- **A1-lazy：**只保留可随机访问的 recipe，必要时以明确的新 recipe bank 替换不可 lazy 的原子，并重新生成 labels。必须同时报告相对原完整 A17 bank 与新 recipe bank 的 coverage。

## A2. 输入、输出与 action 定义

输入 $\xi$：当前 material/regularization 摘要、残差和几何摘要、O/P/M seed 小投影、当前 $U$ 的已付费信息；每个 recipe 的静态 descriptor $d_j$；每个可删除 slot 的在线 descriptor $o_i$。

输出是

$$\pi_\theta(j,i\mid\xi),\qquad a=(\text{remove }u_i,\text{instantiate recipe }j),$$

不是 $\widehat G(j,i)$。每轮选择最多 $K=4$ 个 joint action；因此最多实例化 4 个 incoming currents，可能更少（两个 action 共用一个 recipe）。所有 action 都要保留其 dropout context，不能把不同 outgoing 的 best label 偷换为任意一次替换。

将 selected raw atom $d_j^{\rm current}$ 对其 retained core 投影：

$$c_{ji}=\frac{(I-V_iV_i^*)d_j^{\rm current}}{\|(I-V_iV_i^*)d_j^{\rm current}\|}.$$

若投影后接近零，返回 invalid，不补随机 current。endpoint rank 始终等于原 $k$。可删除的只是不受用户/后端保护的列；原 A17 没有永久保护每一 receiver 列，只保护最终 rank。[P1]

## A3. 默认 lazy recipe bank（最多 122，不强凑 128）

|recipe family|上限|构造与在线费用|
|---|---:|---|
|receiver geometry modes|16|仅使用相同 geometry cache 中已存在的 singular vectors；新几何首次分解单独收费，超过真实 rank 的 slot inactive|
|vector Fourier atoms|32|按 reciprocal-grid shell、lexicographic frequency、Cartesian component 的固定顺序索引；单 atom 直接生成，不先建全矩阵|
|compact vector Haar atoms|32|固定笛卡尔网格的 level/tile/detail/component 索引；先按 coarse-to-fine，再 lexicographic 排序，前 32 个；网格不支持则该 family inactive|
|fixed random atoms|24|counter-based 固定 seed 生成 circular complex Rademacher，不能根据测试 gain 选 seed|
|seed-feedback recipes|18|六条 O/P/M seed 的 degree 0/1/2；只对 selected recipe 做对应 Feff/adjoint action，缓存同一链的前缀|

这里没有 anchor primal/dual 工厂，也没有新增 state eigensolve。新 bank 不等价于 A17 原 bank：A17 labels 不能直接按 family 名称搬过来。A0 与 A1 分开存文件，分别写 provenance。

若原 bank 已有经审计的 cheap random-access 实现，可以保持原 atoms，优先使用已有 labels，但必须导出 `recipe -> dependencies -> exclusive cost` 清单。

## A4. 不需要全 current bank 的 features

每 atom 用 32 维描述：family、支持/中心/尺度/方向、Fourier wavevector、degree、静态 receiver 覆盖摘要、当前 material 在支持上的均值/方差/contrast 范围、可廉价获得的 seed projection。missing values 用 mask，不用高成本值补齐。

Fourier projections 可对少量 seed 做 batched FFT 后随机访问频率；Haar projections 用局部积分/索引；random projections 必须计入完整 $O(snD_{random})$ 运算与内存访问，过贵就去掉该 feature。不得预先算全部 $Ld_j$、$S d_j$、$d_i^*F d_j$ 或 anchor gain。

每 outgoing slot 用 16–32 维 descriptor：已缓存的 $SU$、$LU$ 和 reduced normal 信息的小统计、$u_i$ 的 recipe/gauge 标识、合法性 mask。不能使用 full optimal step 或 true material image。

## A5. 三种编码器及物理边的诚实定义

**默认 A-MLP：**node MLP $[d_j,\xi]\to h_j\in\mathbb R^{64}$；pool mean/max；共享 action head $[h_j,\mathrm{pool}(h),o_i]\to\ell_{ji}$。参数目标 0.03–0.08M。

**A-Set：**相同输入、相近参数规模的两层 attention，hidden 64，4 heads，最多 16 inducing tokens。仅在 MLP 有正向 headroom 时作为控制。[R19]

**A-GNN：**两层 hidden-64 message passing，每 node 最多 8 条预定边。局部 atoms 之间可用支持中心、方向及 dyadic Green 的 cheap centroid approximation；全局 Fourier/receiver atoms 用 family/frequency/已有 sketch 连接。此边不是精确的 $d_i^*Fd_j$。保存 edge definition 和 approximation flag。

GNN 的实验问题是 cheap physics edges 是否优于等参数 MLP、无物理的几何边、随机置换边。若没有至少 0.03 absolute coverage 增益或同 coverage 下 10% acquisition 降低，不声称反馈代数决定了网络架构。精确 dense feedback-edge GNN 只可作离线诊断，不能借它宣称消掉了 full dictionary acquisition。

## A6. Teacher 与 loss

原 A0 使用 exact action-level A17 labels（合法性、drop slot、finite gain、当前 state、原 tolerance）。只有 best incoming label 时，不能训练 action head：需读取对应 outgoing 索引或重新生成。A1 在同一 frozen backend 上为 recipe/action 重新标注；pair labels 不能由两个 singleton gain 相加得到。

设 $g_j=G_j$ 当且仅当 action 合法且 $G_j>\tau_{j,\rm online}$，否则 $g_j=0$；另保留 raw positive gain 诊断。只对合法 action 定义 mass；reference $g_{ref}$ 为该 state 中所有已审计 baseline/reference 的最佳正增益，no-op 增益为零。target mass $p_j^*\propto g_j$；所有 gain 都非正时，target 为 no-op。

主 loss（另用 AdamW decoupled weight decay $10^{-5}$，不重复加入 L2 loss）：

$$L_A=L_{list}+L_{cover}+0.01L_{duplicate}.$$

$L_{list}=-\sum_jp_j^*\log\pi_j$。覆盖 surrogate 用从 $\pi$ 独立抽 K 次的 expected-best gain：将 $g_{(1)}\le\cdots\le g_{(M)}$ 排序，$g_{(0)}=0$，$T_l=\sum_{j:g_j\ge g_{(l)}}\pi_j$，

$$V_K=\sum_l[g_{(l)}-g_{(l-1)}]\{1-(1-T_l)^K\},\quad
L_{cover}=1-V_K/(g_{ref}+\epsilon_{ref}).$$

零 reference state 不用此比值，而使用 no-op cross entropy。该 surrogate 是 **iid sampling** 的精确期望，不是假称 deterministic top-K 的可微等价；validation 必须评估实际 deterministic without-replacement policy。

$L_{duplicate}$ 只抑制重复 recipe/drop action，不强行让不同 atoms 全正交。若 pair-only witnesses 存在，另行加 block-action teacher；第一轮 singleton headline 不覆盖此 claim。

## A7. 在线算法

```text
A_PROPOSE_AND_VERIFY(state, K=4):
    import frozen A17 endpoint/tolerance adapter
    xi = acquire_only_declared_cheap_features(state)
    logits = network(recipe_descriptors, outgoing_descriptors, xi)
    actions = deterministic_topK_legal(logits, K, stable_tie_break)
    instantiated = empty lazy cache
    best = no_op
    for (recipe, drop) in actions:
        raw = instantiate_once(recipe, instantiated)  # no whole bank
        V = retained_core(U, drop)
        C = project_normalize(raw, V)
        if invalid: log rejection; continue
        s_new = solve_original_endpoint([V, C], frozen_state)
        if endpoint_stability_or_normal_residual_fails: reject; continue
        d = s_new - s_old
        z = full_J_action(d, all_illuminations)
        G = exact_algebra_finite_gain(cached_u, z, d)
        if G > original_online_tolerance and G > best.G: best = action
    accept best or retain current space
    if accepted and accepted_count < 3: rebuild only state-dependent proposal data
```

网络在一次预测中完成 proposal allocation；没有 64-column ranking anchor。比较解析 baseline 时，同样可用这个 K-finalist wrapper。若需要更大 full fallback，单独作为 rescue policy 计时，不混入 K=4 主指标。

## A8. 复杂度与主要失败模式

node MLP 约 $O(Dh^2+Dkh)$；稀疏 GNN 加 $O(Eh^2)$。主物理费用是 selected recipe 的浅层 propagation 与 K 个 endpoint/Jd，而不是 $D$ 个 state-dependent atom 的构造。描述子提取、Fourier transforms、geometry cache 和多轮重建均计入 wall time。

主要失败：lazy bank 不覆盖 near-contact；原 labels 与新 recipe 不兼容；cheap features 没有预测信息；GNN 局部边漏掉长程反馈；max-gain target 集中过强造成重复；teacher 只见历史四对象导致过拟合。

**直接停止：**A1 仍需完整 anchor/workspace 才能得到 features；或 held-out 下无法在 .70 near-contact coverage 与至少 2x acquisition reduction 之间找到可行点。A0 的好结果不能覆盖这个失败。
