# 无 NN baseline 的完整实施规格

所有 baseline 使用相同冻结 state、K4 selected atomic actions、同一 A17 endpoint/verifier。D-recommended 另外保留原 top1/cap3，作为真实部署 cost 对照，不能称同 K 控制。任何选择 degree、family顺序或drop规则的试验只用development/validation，最多下面这组固定配置。

## B-poly：不多花一次 matvec

每单链给出 q1,q2,q3 与 Feff（或 O 的 adjoint）Arnoldi H 的前两列，Hbar为3×2。它来自两次 operator action。六链给24个最多的低维records：degree0、degree1、degree2 truncated Neumann和两维GMRES；不提前实例化或验算全部 physical currents。

Neumann控制为 b、b+Fb、b+Fb+F²b。用已有 H、chain columns 表示这三个向量并归一，不新算 powers。O chain 此处的 F 是 Feff*，所以其 Neumann current 是 adjoint-chain proposal，不声称求解 forward current。

两维GMRES：

$$y=\arg\min_{y\in\mathbb C^2}\|\|b\|e_1-(\bar I_{3\times2}-\bar H_F)y\|,$$
$$c=[q_1,q_2]y/\|[q_1,q_2]y\|.$$

**不能在没有 Fq3 的情况下声称获得 K3 空间的完整 GMRES 解。**发生提前breakdown则用实际合法行/列；零向量返回invalid。O链相应最小化adjoint effective equation residual，仅作proposal控制。

将选中 chain coefficient 通过 chain-to-joint 坐标映射到 Q。为了控制partialrotation因素，每current record另有两个预注册mix选项：c_new 或 normalized(u_i+c_new)。不生成连续角度网格；选择该二元配置仅由validation决定，不能在线试两倍endpoints。

## 固定的 proposal/drop policy

三组 candidate family order：[M1,O1,P1,M2]、[O1,M1,P1,O2]、[P1,M1,O1,P2]。统一degree从0/1/2/GMRES中选一个；mix从pure/half中选一个。只允许从这些24个固定策略中用validation选择一个，不用test调换family。

drop规则提供两组：当前U中norm(Su_i)最小的4个legal slots，或按原receiver序号从尾至首的4个slots。把family顺序与drop顺序一一配对，K4而不是4×k。共48个有限策略可离线用现有teacher评估，online只运行最终冻结的一个。成本记录48个策略的离线选择费，部署没有全局current打分。

A1-analytic 用同一两个drop规则，incoming按family-balanced轮转：receiver/Fourier/Haar/random/seed，或seed/receiver/Fourier/Haar/random；各family内部按冻结recipe ID排序，取前四个合法可随机访问recipe。degree选择为0/1/2，不按test增益挑recipe。原A17的完整解析ranking另列D-full，不能混用该廉价policy的名字。

## Ridge 与 nearest-neighbor 控制

B-ridge 输入与B-MLP完全相同（state1552+outgoing32）；target为teacher最佳正增益direction的Hermitian projector tt*，拆实虚和mask，no-positive slot单独标记。用dual ridge：`W=X.T solve(X X.T+lambda I,Y)`；lambda候选1e-4、1e-2、1，仅validation选择。对预测矩阵做Hermitian symmetrization，取最大特征向量并mask/normalize。outgoing priority用另一ridge回归teacher positive opportunity mass，非正priority可no-op。训练数据及offline成本与NN独立计费。

B-KNN：同一train-normalizedfeatures，L2最近的3个training object-state-slot records，只从不同parent objects各取至多一个；inverse-distance权重平均其teacher projectors，再取top eigenvector；priority为同一权重下positive mass。distance=0时只平均零距离neighbors。不跨test访问teacher。

这两个控制是在相同坐标gauge下的非神经条件先验，非纯物理solver。raw coefficient直接平均会相位抵消，因此不可替代projector平均。任何ridge/KNN完整实现若已达到.95可见reference，应依G2停止NN扩展。

## Ablation预算与baseline选择泄漏

oracle/baseline策略选择在development/validation的小矩阵cache上CPU进行，不重复fullMaxwell；最后冻结在线policy后，实测真实feature/acquisition/endpoint cost。不能把teachercache的免费查表时间当成部署速度。

如果48个有限analytic策略在12statepreflight中已反复挑最优，该preflight只能筛选，不能再称其coverage为unbiased generalization。最终test只用冻结赢家。
