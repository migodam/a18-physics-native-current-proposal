# GO / NO-GO：预注册裁决表

**以下数值是本轮设计阈值，不是既有 A17/A18 实证推导出的普适常数。**所有 gate 在看 locked test 前冻结；`HOLD` 表示证据不足，不自动授权加大网络/数据。

## G0：物理与 provenance，硬门

必须：真实backend可运行；polarizability derivative/real packing/current metric一致；原endpoint/current stability/normal residual/tau不变；2真实states的adapter与原实现相对差<=1e-9（在非零reference尺度上）；dense algebra tests通过；objectsplit无泄漏。

任何失败：`NO_GO_BACKEND`。只能修兼容性并保留失败记录，不能跑NN后用好曲线反证正确。

## G1：无 NN 表示与 acquisition

preflight12states，不使用lockedtest。完整bank/共同参考有正机会的分组分别算coverage。

- A-lazy：recipe bank best-found/enumerated coverage macro>=.85，near-contact>=.70；online features 不依赖全 bank/anchor。
- B-manifold：best-found continuous macro>=.90，near-contact>=.75；shared-root相对per-core参考的macro loss<=.05；需要更深/更宽才达标则重测成本，不免费升级。
- 两条实用路线均要求 $\mathrm{median}(T_{acq}/T_{acq,Dfull})\le.50$，且P95不超过1.0；所有新增operator actions计入。

不通过：停止相应NN训练。连续优化失败不证明全局不存在好方向，但足以终止本次有限预算方案；正确结论是 `NO_GO_TESTED_REPRESENTATION_OR_SEARCH`。

## G2：学习 headroom

用同样 cheap representation 的最强 B-poly/B-ridge/KNN 或 A-analytic 进行 K=4 验证。若其相对 best-found reference 的 coverage>=.95，停止NN路线；保留该解析/非神经方法作为结果。

若 oracle覆盖高但任何 cheap feature 条件模型都无法预测，结论为 `FEATURE_INFORMATION_LIMIT`，不能直接上Flow。若 feature acquisition 本身吃掉节省，结论为 `COST_LIMIT`。

## G3：tiny validation gain

A epoch10、B epoch20 检查；最后与最强同预算非神经基线比较。至少达到一项：

1. macro coverage absolute +.05，near-contact不下降超过.02，acquisition不恶化超过10%；或
2. 候选数从4降到2仍保留至少95% accepted finite gain，且完整incremental wall至少下降20%。

此外全流程 median $T_{inc}$ 不高于 D-recommended，同批真实计时不得只比较NN inference。训练集上PASS但validation无改善：`NO_GO_LEARNING`；不能调test再宣布通过。

## G4：B 相对 A 的最终主线裁决

B保留Primary，需满足一个真正的coverage-cost Pareto优势：

- $Cov_B\ge Cov_A+.05$ 且 $T_{acq,B}\le T_{acq,A}$；或
- $Cov_B\ge Cov_A-.02$ 且 $T_{acq,B}\le.80T_{acq,A}$。

两者都需满足前述G0-G3。若A自身失败，B与最强可部署非神经baseline比较。若B无Pareto优势，**A升为Primary，B归档为representation negative/control**；不得因连续/多项式更“新”而强留。

## G5：架构 claim

相同数据、输入信息、训练budget、参数规模下，GNN/Set相对MLP若没有>=.03 coverage增益或>=10%实用成本下降，采用MLP，不 claim physics-native neural architecture 的额外优势。

即使GNN更好，random-edge/geometry-edge/physics-edge三者必须区分。若physicaledge获取需要全current bank，GNN可以有研究结果，但不属于本次acquisition解决方案。

## G6：held-out 与统计

模型/超参/degree/seedfeature在validation后冻结，test一次。要求point estimate保住原门槛；paired object bootstrap的coverage差异95%区间若跨明显劣化区域，记HOLD，不宣称强泛化。

8个testobjects只能构成tiny feasibility evidence；大量correlatedactions不能补充独立样本数。新geometry或near-contact缺失要写NOT_TESTED，不以全局均值代替。

## G7：pair与C

若12-state pair audit中超过20%的positive-opportunity states仅有joint-positive/singleton-negative action，或pair机会构成reference gain的>=10%，单独标记 `NEED_BLOCK_PROPOSAL`。这停止“singleton方案已覆盖关键机制”的claim，而不必杀死singleton实用路线。

C默认NO-GO。启动需满足其文档中多region、独立object和deterministic多head控制条件；deterministic已达.95 reference直接杀死C。没有genuine full-physics weak-material evidence，material generative prior继续排除。

## G8：最终成像/部署 claim

只有赢家nonlinear transfer完整跑完并满足相同line-search/constraints协议，才讨论最终material/heldout-measurement质量。frozen gain通过不代表此gate通过。cost不省则不能宣传加速，即使成像质量改善。

## G9：资源封顶

任一job超限或NN累计240GPUmin：立即停止新训练；已完成结果照交付。出现不稳定连续teacher、memoryoverflow、混精度错误、outer-cache污染不能用扩大网络处理。所有失败记录保留。

## 最终允许的结论用语

`PRIMARY_B / PRIMARY_A / ANALYTIC_ONLY / NO_GO_BOTH / HOLD` 五选一，附每个gate的PASS/FAIL/NOT_RUN。当前仓库状态为 **SPEC_READY；真实Maxwell/NN gates全部NOT_RUN**，不是预先宣布PRIMARY_B已实证成功。
