# Codex 最终文献边界与原创性审查

本轮 ScholarQA 检索受 429 限流影响。Integrator 的四个检索操作没有返回可用新论文；A 的检索缺口独立保留。不能将这种访问不足解释为没有近邻方法。本文仅基于已经取得的 primary evidence 与本仓库实现作有限判断。

1. **Neural Krylov Iteration (NeurIPS2024)** 已提出学习子空间并用于 Krylov 迭代，包含子空间正交化和 projector 型损失。因此“首次学习 Krylov 空间”“首次采用 projective loss”均不成立。本轮目标是输出少量 current exchange action，由原有限增益 verifier 接受；它与学习子空间预处理完整线性求解存在任务区别，但任务区别本身不证明全球原创性。来源：[会议原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e88870ec82f2469b0ddf32c817920c68-Abstract-Conference.html)，正文 Section4。
2. **Subspace-Based Optimization Neural Network / SOM-Net** 已学习 inverse scattering 的 induced currents，并结合解析材料更新。不能声称首次学习 current、首次将物理和网络联合。其 inspected method 为2D TM，本轮使用3D full-vector 离散 Maxwell 并输出低维 action records；该实验目标和 verified finite utility 的区别可以明确描述，不能转成未经对照的精度或速度优势。来源：[arXiv2209.03567](https://arxiv.org/abs/2209.03567)，v2 PDF SectionII.B、Algorithm1；A 的 locator ledger 保留。
3. **Wei/Chen Physics-Inspired CNN for Full-Wave Inverse Scattering, TAP2019** 的题名与 DOI `10.1109/TAP.2019.2922779` 已核验。作者全文链接在本轮未成功取得，不能用仅有摘要/元数据证明它没有包含某种具体机制。[DOI](https://doi.org/10.1109/TAP.2019.2922779)。未核验的 IEEE document ID 不用于正文证据。

Schur complement、Arnoldi/MGS、GMRES、子空间旋转和 listwise 学习均为既有数学或算法构件。本轮可检验的研究主张是：在当前 U 上构造付费的 O/P/M Schur-feedback 表示，预测合法系数/drop，仅 materialize 选中 currents，并在不更改原 endpoint/tau 的条件下改善 coverage-cost。是否有实用贡献由 G3/G4/G6/G8 决定，不能以构件组合较复杂代替实证。

本轮没有证明新核心定理，也没有取得足以支持全球优先权的穷尽检索。允许的表述是“受已有 Schur/Krylov 与 current-learning 方法启发的有限预算研究方案及其可复核实验”；若后续部署门槛失败，仍可公开表示和学习负结果，不升级为普适不可能性。

全文 PDF 不随 Git 或 Release 再分发。现有检索原始回执、claim ledger、429 和 access gaps 均保存；公开入口应先读最终 `GATE_DECISION.json` 与中文研究报告。
