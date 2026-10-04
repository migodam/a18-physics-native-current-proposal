# Novelty audit：哪些已经有人做，哪些仍值得检验

**结论：**“orthogonal polynomial + NN”“physics-informed GNN”“learned current”“physics decides”分别都不是新贡献。最有希望的是一个更窄、可被实证推翻的接口：

$$\text{Schur-resummed Maxwell feedback}
\longrightarrow\text{cheap O/P/M current proposal coordinates}
\longrightarrow\text{task-conditioned action prior}
\longrightarrow\text{unchanged full material finite-gain decision}.$$

这仍是**候选贡献**，不是检索已经证明“首次”。来源与全文可用性见 [SOURCES.md](SOURCES.md)。

## 1. 电磁 inverse/current 学习家族

|工作|已存在的核心思想|本轮不能 claim|本轮应检验的真正区别|
|---|---|---|---|
|SOM [R01]|用 exterior mapping 的 current subspace 组织逆问题|首次利用 receiver current modes|在固定 material state 下生成少量可改善完整 GN quadratic 的 current-space 修改|
|TSOM/FFT-TSOM [R02]|引入 internal current-to-field spectral structure、降低搜索规模|首次将内部传播加入 current representation|retained Schur feedback、当前 residual/regularization 与具体 finite-step utility 联合决定 proposal|
|SOM-Net [R03]|SOM-inspired unrolling 学习反演|首次把 SOM/current subspace 放进网络|网络不展开整个反演；仅提出 action，保留原 endpoint 与 full verification|
|Wei–Chen ICLM [R04]|学习 induced-current 的较小/难学部分，利用 basis expansion 与物理已知大分量|首次 current learning；首次用物理减少网络负担|不预测完整/遗漏场以直接重建材料，而是学习下一次 fixed-rank current rotation 的提议分布|
|Physics Embedded DNN [R05,R32]|一手题名明确覆盖 full-wave inverse 与 VIE solver 的物理嵌入|首次 physics-embedded EM DNN|输出/teacher/interface 的细节需全文补核；本轮不据二手摘要作排他性 novelty 声明|
|NeuralBIM [R06]|从 Born iteration 启发 field/contrast alternating update learning，支持监督与物理目标|首次 residual-conditioned inverse update learning|本轮不学习 field/material update rules，而是付费前的 current proposal allocation|
|Far-Field Approximation Learning [R07]|已确认是毫米波短距 imaging 学习方法|首次近似物理前处理+学习成像|primary full text 未取得；需核对其 learned variable、成本与3D建模，不把题名差异当充分证明|
|Learned-SVD inverse source [R28]|对相关 current distributions 引入数据先验，且有 distribution shift 风险|首次 learned current prior|目标是 full nonlinear material-step utility，且 current proposal 不直接作为 inverse-source 解|

ICLM 是最重要的电磁近邻之一。不能把“我们也让网络输出 current/coefficients”当作贡献；必须证明**为什么 A17 的有限材料效用需要新的可部署 proposal target**，并量化它相对不学习/已有解析 ranking 的增益。

## 2. 神经算子、数值结构与导数学习

|家族|已覆盖的内容|区别与强制 baseline|
|---|---|---|
|Neural Green’s Operators [R08]|coefficient-dependent、forcing-linear 的 Green-type representation；求解/预条件相关用途|不要重做 A18-B learned inverse；本轮 polynomial 只产生搜索坐标，不拟合整个 Green operator|
|VarMiON [R09]|用 variational discrete structure 组织 basis 与 coefficient networks|“数学结构决定网络”不是新口号；本轮须证明实际 current-action target 的必要性|
|OPNO/SNO [R10,R11]|正交多项式/谱系数与神经学习结合、非周期边界表示|空间 polynomial basis 与 $p(F)$ current-feedback space 不同；后者的non-normal稳定性要独立说明|
|2026 Chebyshev operator [R12]|Chebyshev spectral representation 与 physics-informed conditional modulation|不能claim Chebyshev+NN首次；不能把实域逼近性质直接搬到Maxwell L|
|LanczosNet [R13]|Krylov/Lanczos graph compression、多尺度powers、learned spectral filters|Hermitian graph Laplacian 与 non-normal Schur feedback 不同；必须比较普通 Krylov features，不靠名字区分|
|Neural Krylov Iteration [R14]|学习有利子空间来加速 linear system solve|“学习Krylov子空间”不是贡献；不同处是 material utility target、非solver output、fixed-rank exchange与取得成本|
|learned GNN preconditioners / NeuralIF [R18,R26]|矩阵结构启发的消息传递、低成本预条件构造与solver指标|本轮不以residual/iteration count为目标，也不输出可作用任意forcing的preconditioner|
|DINO / derivative-aware operators [R23]|用导数/Jacobian监督来改善算子在优化中的适用性|GN/task loss不是新贡献；本轮目标是具体finite-action gain，非泛化的Jacobian fidelity|
|DINO geometric MCMC [R24]|learned acceleration与高保真校正可联合使用|“NN proposes, physics corrects”作为通用原则已有先例；本轮不是材料posterior sampling|
|adaptive inverse ROM [R27]|用误差/模型管理与高保真acceptance控制逆问题ROM|高保真accept/reject不新；A17机制与本轮acquisition接口是更具体对象|

Faber / numerical range / Arnoldi [R15–R17] 为可靠计算提供工具，不属于本研究发明。没有查到可作为标准“Faber neural operator”代表的可靠一手条目，不代表文献不存在；论文不能凭此 claim first。

## 3. Graph current solvers：已有直接近邻

[R29] 的 CFIE physics-informed residual learning 已使用 GNN 处理 triangular meshes 上的 PEC EM scattering。它足以排除“图网络第一次学习电流/积分方程散射”的宽泛定位。应比较的是 learned variable 与决策目标，而不是将其归成 generic CNN 后忽略。

[R30] ScaGNN 于 **2026-09-29** 提交，学习 BEM boundary trace，并通过 dynamic edge sampling 选择长程交互；[R31] PIBNet 同样以图结构表示多个散射体与长程相互作用。因此“学习哪些反馈连接重要”也已是直接相关思想。它们的摘要没有证明与本轮完全相同的 3D vector-Maxwell/material-GN 设置，不能擅自扩展其适用范围。

[R25] 的 EM dynamics graph/Transformer surrogate 是另一相邻类别。三类都应在写作中说明，但本轮不需要耗4060复现它们的完整field solver——A18-B已经提供更直接的full-closure负对照。

## 4. Proposal、subset 与生成模型

Set Transformer [R19]、attention-based combinatorial heuristics [R20]、DIFUSCO [R21]、DPP subset learning [R22] 已覆盖 permutation-invariant set encoding、learned selection、conditional generation、quality/diversity trade-off。

所以 A 的 learned shortlist 本身不是 ML novelty；C 的低维 conditional mixture 或后续flow也不是 ML novelty。DPP偏好negative dependence，不能自动表达“两个单独无益、合起来有益”的 Maxwell/least-squares endpoint synergy；这必须由真正joint action teacher表达。

本轮无需证明一种新的通用机器学习架构。可以把ML贡献限定为一个资源可控、由物理目标监督的 amortized action-prior 方法，再把电磁specific部分讲清楚。

## 5. Generic ML 与 EM-specific contribution 分离

### Generic：只可作为已知工具/工程设计

低维coefficients、set/GNN encoder、多head、coverage/listwise loss、phase-invariant distance、sphere/Grassmann coordinates、degree regularization、conditional mixture、high-fidelity checking、cached small-matrix training。

### EM-specific：最值得争取的两项主贡献

**(E1) 低成本物理 proposal 坐标。**用实际DDA material injection、receiver-adjoint return 与 retained Schur feedback 构造两侧 O/P/M flag，能保存 weak-direct-radiation/strong-feedback-return 的候选；共享旧LU与少数actions而不构建大anchor。需要 raw PFP、receiver-only、forward-only、无真实injection等因果ablation支持。

**(E2) finite-material-utility proposal prior。**学习的不是 current solution、material image 或 Green inverse，而是针对当前 residual/prior 的 fixed-rank action。用A17 full finite gain保留决策权，并实证将candidate opportunity/cost向更优Pareto点移动。必须在held-out objects和near-contact上体现，否则只剩应用组合。

### 次贡献与不能升级的结论

layer-invariant depth penalty、leave-out lift、projective mixture可提高方案正确性，但Schur消元、Krylov flag与投影几何本身是标准数学。不要把独立推导误称为新定理的优先权。

若 GNN≈MLP，论文题目应强调 physics-generated representation / learned prior，而不是“feedback algebra determines a neural architecture”。若 B≈A且更贵，B不做headline。若 polynomial prior已足够，最有价值的结果可能是**不需要神经网络**。

## 6. 推荐论文级 claim（只有实验通过后才使用）

> We generate a small set of task-conditioned current-space modifications in a Schur-feedback Krylov representation, avoiding full candidate-bank acquisition while retaining the original full material finite-gain decision.

这句话仍需成本、coverage和held-out证据。现在只能写 “we propose and specify”，不能写 “we demonstrate”。

## 7. 审稿人最可能追问的实验

是否只是把64-column anchor换成了一个同样贵的24-column manifold？为何fixed polynomial/小GMRES不够？O与M是否真有因果贡献，还是network只识别物体family？continuous oracle是否只是局部优化器偏差？任意phase/gauge变化是否改变输出？近奇异endpoint是否制造虚假高gain？原A17state是否泄漏进teacher？GNN收益是否来自更多训练/特征？冻结GN改进能否转移到非线性质量？

本仓库的gate、costledger与pairedcontrols逐项对应这些问题。
