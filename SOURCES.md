# Sources / 文献与项目来源

检索与项目读取日期：**2026-10-04**。本表按一手 publisher/author/institution 页面建立；摘要级核查与全文核查明确分开。没有复现实验、没有完整全文的新颖性排除，不写成已经证明的优先权结论。本文自推命题属于代数推导；不把引用文献误写成已经证明本轮 coverage。

## P0

用户附件：`粘贴的文本 (1).txt`，1210 行；作为研究背景与候选思想。原文件不公开复制到新仓库。重点采纳 current-task prior、O/P/M seeds、Arnoldi coordinates、A17 verification；修正 chain direct sum、Born-order 解释、pair synergy 推出 generative necessity 等表述。

## P1

A17 public evidence / theory：

- https://github.com/migodam/a17-maxwell-certified-exchange
- https://github.com/migodam/a17-maxwell-certified-exchange/blob/main/closure_final/START_HERE.md
- https://github.com/migodam/a17-maxwell-certified-exchange/blob/main/closure_final/manuscript/CORE_THEORY.tex
- https://github.com/migodam/a17-maxwell-certified-exchange/blob/main/closure_final/manuscript/CONDITIONAL_EXCHANGE_THEORY.tex
- https://github.com/migodam/a17-maxwell-certified-exchange/blob/main/closure_final/manuscript/FINAL_EXCHANGE_METHOD.tex

读取了上述文本。原始 snapshot commit：`56097653e6a814399581c3759fc3e3e8f0c2284c`；final scientific baseline 在 efficiency report 中标为 `f5bd7d0aa0e229110c425f1becb6384e09663196`。`main` 链接可变，执行时必须解析并保存具体 commit。FINAL_EXCHANGE_METHOD.tex 本次返回 blob SHA `f8c0b399f702972ec1d22d616073ebda1aa8ff80`。

## P2

A17 efficiency follow-up：

- https://github.com/migodam/a17-maxwell-certified-exchange/blob/main/efficiency_followup/GPT_REVIEW_CONTEXT.md
- release tag: `a17-efficient-verification-2026-10-04`

读取完整 review context；其 cost、nonlinear 与条件性范围按原文保留。不是独立重复运行。四十二个 external/vendor modules 未包含在公开证据包，真实 backend 重放仍需用户原环境。

## P3

A18 Stage 0：

- https://github.com/migodam/a18-nn-native-maxwell-stage0/blob/main/START_HERE.md

读取完整 entry；它没有训练 NN，48k action 不代表48k独立objects。原 upstream A17 commit 为 `56097653e6a814399581c3759fc3e3e8f0c2284c`。

## P4

A18-B stopped closure：

- https://github.com/migodam/a18-b-maxwell-feedback-closure/blob/main/START_HERE.md
- release identity in public entry: `v1.0.0`

读取完整 entry；48 scenes /96 states、部署成本与 rigorous-bound 未闭合按该研究报告理解。其“route C”历史裁决标签与**本仓库 Route C probabilistic prior 没有编号继承关系**。公开 entry 不是全部原始数据，本次未下载/重算其 NPZ。

## Literature ledger

每项带 access depth；primary metadata-only 的文献不会被用于细节性的排除 claim。

## R01

X.-D. Chen, Subspace-Based Optimization Method for Solving Inverse-Scattering Problems, TGRS 48, 42–49 (2010).

Primary: https://doi.org/10.1109/TGRS.2009.2025122

核查层级：author-publication-list + A17 references。SOM bibliographic identity and current-subspace framing; not a fresh full-paper replication.

## R02

Y. Zhong and X. Chen, Twofold Subspace-Based Optimization Method, Inverse Problems 25, 085003 (2009); FFT-TSOM, TAP 59, 914–927 (2011).

Primary: https://doi.org/10.1088/0266-5611/25/8/085003

核查层级：author/institution abstract。Additional primary identity: https://doi.org/10.1109/TAP.2010.2103027 ; author abstract: https://www.math.uci.edu/node/33832

## R03

SOM-Net: Unrolling the Subspace-based Optimization for Solving Full-wave Inverse Scattering Problems.

Primary: https://arxiv.org/abs/2209.03567

核查层级：author abstract。Full HTML retrieval failed; compare stated unrolling target, not uninspected implementation details.

## R04

Z. Wei and X. Chen, Physics-Inspired Convolutional Neural Network for Solving Full-Wave Inverse Scattering Problems, TAP 67, 6138–6148 (2019).

Primary: https://www.ece.nus.edu.sg/stfpage/elechenx/Papers/Wei_TAP_2019.pdf

核查层级：author full-text parsed。DOI 10.1109/TAP.2019.2922779. PDF screenshot retrieval failed; textual architecture discussion inspected, no figure-based numerical claims.

## R05

Physics Embedded Deep Neural Network for Solving Full-Wave Inverse Scattering Problems, TAP 70, 6148–6159 (2022; online 2021).

Primary: https://ieeexplore.ieee.org/document/9509361/

核查层级：primary metadata only。DOI 10.1109/TAP.2021.3102135. Primary full text/abstract not available in this session; detailed architectural exclusion remains open.

## R06

Neural Born Iteration Method For Solving Inverse Scattering Problems: 2D Cases, TAP 71, 818–829 (2023).

Primary: https://arxiv.org/abs/2112.09831

核查层级：author abstract。DOI 10.1109/TAP.2022.3217333; preprint first submitted 2021. Alternating field/contrast update learning.

## R07

T. Yin, K. Tan and X. Chen, Far-Field Approximation Learning Method for Millimeter-Wave Short-Range Imaging (2023).

Primary: https://ieeexplore.ieee.org/document/10041854

核查层级：primary identity; access gap。DOI 10.1109/TAP.2023.3242424. Primary detailed content not retrieved; do not base a priority claim on a secondary abstract.

## R08

Neural Green’s Operators for Parametric Partial Differential Equations.

Primary: https://arxiv.org/abs/2406.01857

核查层级：author abstract and HTML text。https://arxiv.org/html/2406.01857v1 ; coefficient-dependent/forcing-linear Green representation is prior art.

## R09

VarMiON: A variationally mimetic operator network.

Primary: https://arxiv.org/abs/2209.12871

核查层级：author abstract。Variational-discretization-inspired basis/coefficient architecture.

## R10

Render unto Numerics: Orthogonal Polynomial Neural Operator for PDEs with Non-periodic Boundary Conditions.

Primary: https://arxiv.org/abs/2206.12698

核查层级：author abstract。OPNO; boundary-aware spatial spectral representation, not automatic stability of p(L).

## R11

Spectral Neural Operators.

Primary: https://arxiv.org/abs/2205.10573

核查层级：author abstract。Fourier/Chebyshev coefficient maps already combine spectral numerics and neural learning.

## R12

Physics-Informed Chebyshev Polynomial Neural Operator for Parametric Partial Differential Equations (submitted 2026-02-02).

Primary: https://arxiv.org/abs/2602.01737

核查层级：author abstract。Current research entry verified on 2026-10-04; no empirical results reproduced here.

## R13

LanczosNet: Multi-Scale Deep Graph Convolutional Networks, ICLR 2019.

Primary: https://arxiv.org/abs/1901.01484

核查层级：author abstract。Graph-Laplacian low-rank decomposition and learned spectral filters.

## R14

Neural Krylov Iteration for Accelerating Linear System Solving, NeurIPS 2024.

Primary: https://proceedings.neurips.cc/paper_files/paper/2024/hash/e88870ec82f2469b0ddf32c817920c68-Abstract-Conference.html

核查层级：primary proceedings abstract。Learned invariant/subspace structures for linear solving; not novel to learn Krylov-related subspaces.

## R15

M. Crouzeix and C. Palencia, The Numerical Range is a (1+sqrt(2))-Spectral Set, SIAM 2017.

Primary: https://epubs.siam.org/doi/10.1137/17M1116672

核查层级：primary abstract and author preprint。https://arxiv.org/abs/1702.00668 ; used for complete numerical-range polynomial norm bound.

## R16

B. Beckermann and L. Reichel, Error Estimates and Evaluation of Matrix Functions via the Faber Transform.

Primary: https://epubs.siam.org/doi/10.1137/080741744

核查层级：primary abstract。Faber transform, field of values and Arnoldi/rational approximation error analysis.

## R17

S. Pozza and V. Simoncini, Inexact Arnoldi residual estimates and decay properties for functions of non-Hermitian matrices, BIT 59, 969–986 (2019).

Primary: https://doi.org/10.1007/s10543-019-00763-6

核查层级：primary abstract/metadata。Non-Hermitian Arnoldi/Faber analysis reference; no transfer of unverified approximation bounds.

## R18

J. Chen, Graph Neural Preconditioners for Iterative Solutions of Sparse Linear Systems, ICLR 2025.

Primary: https://research.ibm.com/publications/graph-neural-preconditioners-for-iterative-solutions-of-sparse-linear-systems--1

核查层级：author/institution abstract。General sparse-system learned preconditioners; measured construction and solve overhead are relevant controls.

## R19

J. Lee et al., Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks, ICML 2019.

Primary: https://proceedings.mlr.press/v97/lee19d.html

核查层级：primary proceedings abstract。Set permutation invariance/attention is generic ML prior art.

## R20

W. Kool, H. van Hoof and M. Welling, Attention, Learn to Solve Routing Problems!, ICLR 2019.

Primary: https://arxiv.org/abs/1803.08475

核查层级：author abstract。Learned constructive combinatorial heuristics; not a new general paradigm here.

## R21

DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization.

Primary: https://arxiv.org/abs/2302.08224

核查层级：author abstract。Conditional/generative combinatorial solution proposals are established.

## R22

Low-Rank Determinantal Point Processes, AAAI 2017.

Primary: https://ojs.aaai.org/index.php/AAAI/article/view/10869

核查层级：primary proceedings abstract。DOI 10.1609/aaai.v31i1.10869; diversity/subset learning control, not a model of all positive pair synergy.

## R23

Derivative-Informed Neural Operator: An Efficient Framework for High-Dimensional Parametric Derivative Learning.

Primary: https://arxiv.org/abs/2206.10745

核查层级：author abstract。DINO; JCP 496, 112555 (2024), DOI 10.1016/j.jcp.2023.112555.

## R24

Derivative-informed neural operator acceleration of geometric MCMC for infinite-dimensional Bayesian inverse problems, JMLR 26 (2025).

Primary: https://jmlr.org/beta/papers/v26/24-0745.html

核查层级：primary abstract。Author preprint https://arxiv.org/abs/2403.08220 ; learned proposal plus high-fidelity correction is not a new generic idea.

## R25

Physics-informed surrogates for electromagnetic dynamics using Transformers and graph neural networks (2024).

Primary: https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/mia2.12463

核查层级：primary abstract。EM dynamics surrogate; adjacent to but not identical with current-space exchange.

## R26

P. Hausner, O. Oktem and J. Sjolund, Neural incomplete factorization: learning preconditioners for the conjugate gradient method (2023).

Primary: https://arxiv.org/abs/2305.16368

核查层级：author abstract。NeuralIF; matrix-inspired GNN structure and preconditioner loss are prior art.

## R27

Adaptive Reduced Basis Trust Region Methods for Parameter Identification Problems.

Primary: https://arxiv.org/abs/2309.07627

核查层级：A17 cited primary identity。DOI 10.1007/s44207-024-00002-z; high-fidelity model-management/acceptance not claimed as new.

## R28

A Learned-SVD Approach to the Electromagnetic Inverse Source Problem, Sensors 24, 4496 (2024).

Primary: https://www.mdpi.com/1424-8220/24/14/4496

核查层级：primary abstract and discussion。DOI 10.3390/s24144496; learned current-distribution prior in a scalar inverse-source setting.

## R29

Solving Combined Field Integral Equations with Physics-informed Residual Learning, ACES 2023.

Primary: https://ieeexplore.ieee.org/document/10114780/

核查层级：primary abstract。DOI 10.23919/ACES57841.2023.10114780; GNN on triangular meshes for PEC electromagnetic scattering.

## R30

ScaGNN: a Graph Neural Network for Multiple Scattering Simulations (submitted 2026-09-29).

Primary: https://arxiv.org/abs/2609.37509

核查层级：author abstract。Dynamic adaptive edge sampling for boundary-trace prediction; do not infer 3D vector-Maxwell scope from the abstract.

## R31

PIBNet: a Physics-Inspired Boundary Network for Multiple Scattering Simulations.

Primary: https://arxiv.org/abs/2512.02049

核查层级：author abstract。Graph-based long-range interaction and boundary-trace learning; adjacent multiple-scattering reference.

## R32

Physics Embedded Deep Neural Network for Solving Volume Integral Equation: 2D Case.

Primary: https://www.eledia.org/eledia-unitn/news/physics-embedded-deep-neural-network-for-solving-volume-integral-equation-2d-case/

核查层级：research-center primary identity。DOI 10.1109/TAP.2021.3070152; primary detailed architecture not retrieved.

## 明确保留的检索缺口

1. R05 与 R07 的 publisher 题名/身份可识别，但本轮未取得可核验的一手完整内容。二手索引中出现的详细描述不作为强 novelty exclusion 的依据。提交论文前需补作者稿全文比对。
2. 找到了 Faber 数值分析（R16），没有确认一个能作为该领域标准代表的“Faber neural operator”一手条目；不据此断言此方向不存在。
3. 最新结构化复数 surface-current GNN 的二手索引出现 DOI `10.1109/TAP.2026.3703511`，但本轮未一手核验；作为待查线索，不写入已确认比较结论。
4. 本次是面向所列家族的 targeted novelty audit，不是有检索召回率保证的系统综述。所有“可能的新贡献”仍需 reviewer-level 全文对照及实验支持。
