# Codex 数学与科学审查

审查对象为有限预算内的 current proposal 与原有限增益 verifier。本文不把数值一致性升级成连续 Maxwell 误差证书、全局最优性或最终成像质量结论。

设 U 的列正交，L=I-F，P=I-UU*，D=U*LU。只在 D 满足注册的稳定性阈值时构造 root。消去 U 子空间得到

```
Aeff(q) = P[L(Pq) - LU D^{-1} U* L(Pq)]
Feff(q) = Pq - Aeff(q)
Aeff*(q) = P L*[Pq - U D^{-*} (LU)* Pq]
```

把 L=I-F 代入，Feff=PFP+PFU D^{-1} U*FP。LU 已付款并缓存时，前向与伴随各只新增一次大 L/L*；小 LU solve 的成本也保留。F 与材料 injection 的定义来自离散后端，不能将此公式当作任意材料参数下的免费算子。

六条 O/P/M 链先用 complex128 两遍 MGS，再按 degree 联合正交。rank 默认上限18，硬上限24；零 seed 不补噪声。层权重作用于嵌套层空间：删除列的权重0，degree0/1/2 的权重0/3/8。在每层内换正交基时二次罚项不变；混合不同层则必须同步变换权重矩阵。固定 gauge 的 MLP 本身没有被证明对任意基旋转等变。

对 drop i，c=[u_i,Q]t 保留 partial rotation。t=e0 对应原空间。只删除 i 并加入 c，rank 保持 k。对 joint rank2，系数在 [u_i,u_j,Q] 的 Stiefel 空间内，必须求一个联合 endpoint；不能相加两个 singleton gains。

所有 endpoint 保持原 residual sign、材料实坐标、Lambda、稳定性和 normal residual 阈值。有限增益通过所有 illumination 的真实 Jd 计算：若 d=x_new-x_old，u=r+Jx_old，h=ell+Lambda x_old，则

```
g = -<Jd,u> - <d,h> - (||Jd||^2 + <d,Lambda d>)/2.
```

该值等于同一冻结线性二次目标的下降量。原 A17 tau 是针对这些求和项的数值显著性阈值；它不是对迭代线性求解误差、离散化误差或真实 nonlinear objective 的完整确定性证书。只有 verifier 原始 g>tau 的结果可以接受；调用方改写返回标记不能改变这一条件。

teacher 是有注册 starts/steps 上限的 best-found。原字典、A bank、连续 B teacher 与注册多项式策略共同构成经验 singleton reference，未包含所有连续方向或 rank2 动作。coverage>1 可以出现于另一个参考分母或部署发现更好的方向，不截断，也不说明达到全局最优。

对象级 coverage 先在一个对象内对两状态的 gain/reference 分别求和，再取对象之间的宏平均。状态均值、action 均值和该统计量不等价。12-state 历史 development 已暴露，不能作为独立泛化证据；新 pilot 与旧24/6/18 划分隔离。

缓存有效性依赖当前材料、残差、先验、U/LU/SU/D、source phase、geometry 和 gauge。接受后配置 hash 改变，U 相关 root 与特征重建并计费。旧 A18 anchor/closure 不进入在线特征。NN epoch 仅访问离线小矩阵，物理回调被 guard 阻止；部署时先选择低维 records，再物理实例化最多四个合法动作。

G0、表示覆盖、学习收益、成本、锁定测试、nonlinear 成像与公开发布均独立判定。预算耗尽或门槛失败必须保留原始记录；没有实际证据的条件使用 NOT_RUN。
