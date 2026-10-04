# A18：从 Maxwell 反馈代数到低成本 current proposal

**版本：2026-10-04。性质：理论推导、实施规格与可复核代数测试；不是已完成的 Maxwell / NN 实验。**

## 1. 最终研究裁决

主线 **Route B：Schur-feedback Krylov flag + deterministic coefficient prior**；对照 **Route A：lazy dictionary action prior**；高风险储备 **Route C：projective coefficient distribution**。A、B 同时开发，训练串行。C 本轮不训练，不提供启动训练指令。

A17 保留为独立的 mechanism + conditional exchange 工作；A18 不学习完整逆算子、不预测 material image、不替代有限增益检查。要检验的是：在付出很少物理 action 后，能否把少数 proposal 放到有用的 current-space 修改附近，而不是把已有 analytical Q scorer 换成网络。

优先级不是结果：B 若不能同时保住 opportunity coverage 与 acquisition cost，主线立即退回 A 或无学习方法。最后的主方法可以是 MLP；没有证据就不称“Maxwell 唯一决定了 GNN 架构”。

## 2. 已有证据与本轮假设必须分开

来源编号见 [SOURCES.md](SOURCES.md)。[P1–P4] 是项目公开材料，不是独立复现实验。

|证据|实际支持|不能推出|
|---|---|---|
|A17 原始四对象、35/36 改善；anchor top-1 捕获 98.1–99.6% 的 object-summed 局部机会 [P1]|好的解析排序已经很强，candidate coverage 是重要限制|每个 state 都达到同样比例；NN 已有价值|
|Stage 0：4 对象、12 states、36 queries；near-contact Cov12 从 .3528 提高到 .5178，仍低于 .70 [P3]|cheap pool 尚有缺口|有 47,712 个独立训练样本；新 NN 必然能填缺口|
|A17 efficiency：Jd 仅占额外 exchange 时间约 1.19%/3.82% [P2]|不要主要优化 finalist verification|可以不算新特征、Arnoldi、CPU/GPU 传输成本|
|A18-B：结构/整体精度通过，机制不确定，完整成本约为 GMRES 的 8.3 倍 [P4]|停止扩大该 full-feedback closure 实现|所有 neural solver 都不可能成功；degree-7 的成功证明 proposal coverage|
|A17 top-1/cap-2 保留约 99% frozen gain，却损害一个对象的非线性终值 [P2]|局部 GN utility 与最终重建质量须分别评估|通过本轮 frozen pilot 就可以宣传成像质量和端到端加速|

本轮新假设有两个：**低成本物理子空间覆盖机会**；**state/task-conditioned 学习比最强同预算解析 prior 更会分配 proposal**。两个都需要通过独立 gate。

## 3. 统一记号与物理模型

当前材料状态、几何、残差、正则项全部冻结。每个 illumination 有独立 current，但共用一个 material coordinate 向量和同一个空间 current basis：

$$L j_p=f_p,\qquad L=I-F,\qquad J_p=SL^{-1}B_p.$$

连续 contrast-source 解释中 $F=XG_D$；实际 DDA 后端必须使用其 polarizability $D_a$ 与 off-diagonal Green operator，即 $F=D_aG_{\rm off}$，并保留 $DX[Q_Ts]e_p^{\rm tot}$ 中的极化率导数与矢量局部场。[P1]

用正定 current metric $W$ 白化后再使用本文所有 $V^*V=I$ 公式：

$$\widetilde L=W^{1/2}LW^{-1/2},\quad\widetilde S=SW^{-1/2},\quad\widetilde B=W^{1/2}B.$$

以下省略波浪号。material coordinates 是实数，包含复 contrast 的实部、虚部。将所有 illumination 的复 $J_p$、$r_p$ **一起 realify** 后求 normal equations；不能把一列复 current 当作 rank-one real material update。

采用 A17 的符号 $r=\mathrm{predicted}-\mathrm{observed}$：

$$\Phi_F(s)=\tfrac12\|r+J_Fs\|^2+\tfrac12s^T\Lambda s+\ell^Ts,\quad \Lambda\succ0.$$

绝不在不同 candidate 间重算材料状态、改变 $B,r,\Lambda,\ell$，也不通过重新选 material gauge 制造改善。

## 4. 最核心的构造：消元后的 feedback，不是裸投影

令 $U$ 为当前保留的 $k$ 维空间，$P=I-UU^*$，$D_U=U^*LU$，$R_U=UD_U^{-1}U^*$。在 $\mathcal E=\operatorname{Range}P$ 上定义

$$T_U=(I-R_UL)P,$$
$$A_U^{\rm eff}=PL(I-R_UL)P\big|_{\mathcal E},$$
$$F_U^{\rm eff}=I_{\mathcal E}-A_U^{\rm eff}.$$

**命题 1（exact retained-feedback elimination）。** 若 $D_U$ 和 $L$ 可逆，则 $A_U^{\rm eff}$ 在 $\mathcal E$ 上可逆，而且

$$J_F=SR_UB+Y_U(A_U^{\rm eff})^{-1}B_U^{\rm eff},$$
$$Y_U=S(I-R_UL)P,\qquad B_U^{\rm eff}=P(I-LR_U)B.$$

证明：将 current 写成 $Uv+e$，先从 retained 方程解出 $v=D_U^{-1}(U^*B-U^*Le)$，代入 complement 方程得到 $A_U^{\rm eff}e=B_U^{\rm eff}$；再代回 receiver 输出。此处的逆只用于证明，**在线 proposal 不求这个完整逆**。

展开得到

$$\boxed{F_U^{\rm eff}=PFP+PFU(I-U^*FU)^{-1}U^*FP.}$$

第二项正是 unresolved → retained → unresolved 的反馈。单用 $PFP$ 会删除这条路径。因此主表示用 $F_U^{\rm eff}$；$PFP$ 是必做的便宜 ablation，而非默认等价替代。

更重要的是，$LU$ 已由旧 endpoint 付费时，每次 $A_U^{\rm eff}q$ 只需 **一次新的 L action**：

```text
q <- P q
l <- L(q)
h <- solve(D_U, U* l)
Aeff(q) <- P(l - (LU) h)
Feff(q) <- q - Aeff(q)
```

这消除了“Schur feedback 一定要两次大 matvec”的误解，但 $LU$、投影和小 solve 的成本仍须入账。完整 adjoint 见 [NONNORMAL_MAXWELL_ANALYSIS.md](NONNORMAL_MAXWELL_ANALYSIS.md)。

## 5. O/P/M 不再是三个相乘的分数

|轴|种子|表示/条件|正则与诊断|
|---|---|---|---|
|O：receiver return|$b_O=Y_U^*r$；默认两条 illumination sketch|用 $(F_U^{\rm eff})^*$ 传播；保存相位敏感的投影和 Arnoldi coupling|比较 raw $PS^*r$ 与 dressed seed；不强制高 receiver energy|
|P：内部传播|$b_P=P(f-LR_Uf)$|用 $F_U^{\rm eff}$ 传播；保存 retained coupling、链 Hessenberg、小 residual|检查近零 residual，不把 roundoff 归一成有效方向|
|M：材料激发|$b_M=P(I-LR_U)Bz$|$z$ 来自已有 reduced step 与固定 material probe；保留真实 DDA injection|任务条件、degree prior；不把材料梯度范数当 utility|

默认 O/P/M 各两条 seed，depth $d=0,1,2$，最多 18 列，硬上限 24。O 用 adjoint，P/M 用 forward。联合空间为

$$\mathcal K_d=\sum_{b\in\mathcal B_{PM}}\mathcal K_{d+1}(F_U^{\rm eff},b)
+\sum_{b\in\mathcal B_O}\mathcal K_{d+1}((F_U^{\rm eff})^*,b).$$

这是**两侧联合 feedback-Krylov flag**。不是把三个一般会相交的空间写成直和，也不是声称全部列都是同一 scalar polynomial 作用于同一个 seed。每条单独链满足相应 polynomial 表示；联合层级保持 degree filtration。[R15–R18]

## 6. “manifold” 的严格含义

固定状态且 rank 不变时，$Q=[Q_0,Q_1,Q_2]$ 是 $\mathcal K_2$ 的正交坐标。单位 current direction 模去全局相位后属于 $\mathbb{CP}^{r-1}$；$b$ 列 joint proposal 模去基变换后属于 complex Grassmannian。不同状态的 QR rank、pivot 可发生变化，所以整体更准确地说是 **state-dependent proposal bundle / subspace family**，不是已证明光滑的 Maxwell 解流形。

任意 $Qa$ 是可作为 Galerkin trial current 的方向，并不自动满足给定 forcing 的 Maxwell 方程。Maxwell 一致性由 endpoint solve 负责。

## 7. 连续 proposal 如何合法进入 fixed-rank exchange

删去旧空间中的列 $u_i$，保留 $V=U\setminus u_i$。只准 $c\perp U$ 会漏掉有价值的“部分旋转”：

$$\boxed{c_i(t)=Z_i t,\quad Z_i=[u_i,Q_U],\quad t\in\mathbb C^{r+1},\quad\|t\|=1.}$$

由于 $Z_i^*Z_i=I$ 且 $V^*Z_i=0$，$[V,c_i(t)]$ 总是 rank $k$；$t=e_1$ 就是 no-op。共享 $Q_U$ 只构造一次，避免每个 outgoing slot 重跑 Arnoldi。

**这是一项明确的近似选择：**$Q_U$ 不是逐个 leave-one-out core 的 $F_V^{\rm eff}$ Krylov basis。它的 coverage 要由 shared-root 与 per-core oracle 对照检验；endpoint 代数本身仍完全合法。

对保留 core $V$，

$$\bar C=(I-R_VL)C,\quad Y_C=S\bar C,\quad N_C=C^*(B-LR_VB),\quad\Gamma_C=C^*L\bar C,$$
$$J_{[V,C]}-J_{[V,A]}=Y_C\Gamma_C^{-1}N_C-Y_A\Gamma_A^{-1}N_A.$$

若 common core 奇异，但两个 endpoint 稳定，直接求两个 endpoint；不能误判该 action 非法。[P1] 网络不输出 dressed $\bar C$ 作为原始正交基。

## 8. 最终决策仍是同一个有限位移增益

对旧 endpoint step $x=s_U$、新 step $s_{U'}$，$d=s_{U'}-x$。缓存 $u=r+J_Fx$，付费求 $z=J_Fd$；首次 full $J_Fx$ 若尚不存在也必须付费，不能以 reduced $J_Ux$ 代替：

$$\boxed{G=-\operatorname{Re}\langle u,z\rangle-(\ell+\Lambda x)^Td
-\tfrac12(\|z\|^2+d^T\Lambda d).}$$

$G>\tau_{\rm online}$ 才接受；更新 $u\leftarrow u+z$，然后重新条件化下一轮 proposal。默认一轮最多 4 个 action，最多 3 次 accepted exchanges；停止表示当前 proposal 集中未找到可接受动作，不代表局部或全局最优。[P1,P2]

保持 A17 FP64 significance threshold、current stability 与 material normal residual cutoff；其数值常数/源码身份由 adapter 从固定上游导入。它是数值检查，不是已闭合的 deterministic output-error certificate。任何新增网络都不改变这一事实。

## 9. 严格定义 learned prior

令 $\xi$ 是允许在线获得的 cheap features，$a=(\text{drop set},\text{coefficients or atom recipe})$ 是 action。选择显式的探索基准 $q_0(a\mid\xi)$，定义目标

$$p_*(a\mid\xi,\mathrm{useful})\propto q_0(a\mid\xi)
\mathbb P\{G_x(a)>\tau_x\mid\xi,a\}.$$

也可把 indicator 换成有限、单调的 gain weight，得到 utility-tilted proposal prior。这个分布依赖任务、残差、正则与训练人群，也依赖 teacher 的探索基准；它不是 $p(\chi\mid y)$，不是物理定律，更不是校准好的成功概率。Route A 学离散 action mass，Route B 学有限个连续 modes，Route C 才拟合分布。

## 10. 为什么这不是重新学习 polynomial solver

固定 polynomial 用来近似 $L^{-1}b$；此处 polynomial/Krylov 只定义可以搜索的 trial space。网络决定 **在该空间里，为当前有限 material displacement 提出哪个 current-space rotation**。目标是 $G$ 与 acquisition cost，不是 $\|L c-b\|$。

但这项区别还不构成实验价值。必须与以下无 NN 控制比较：degree-0 seeds、低阶 Neumann/GMRES chain coefficients、固定 O/P/M union、同空间最强解析 proposal、nearest-neighbor/ridge coefficient prior。若其中一个已经达到至少 .95 的可见 oracle opportunity，取消 NN 训练，不为“学习”制造任务。

## 11. 三个不同的边界

1. **表示边界：**便宜 manifold 之外的好 current 再好的网络也提不出。
2. **先验边界：**在 manifold 内有好 action，不等于训练信息能预测它。必须做 label-free online-feature 审计与 held-out object 验证。
3. **决策边界：**exact algebra + numerical finite-gain check 限制冻结模型中的错误接受；它不保证噪声鲁棒真值恢复、不补全 rigorous solver bound、不保证下一轮 nonlinear 改善。

以上是本轮方案的可检验研究对象。具体实现见两份 Route 文档与两份 Codex 指令。
