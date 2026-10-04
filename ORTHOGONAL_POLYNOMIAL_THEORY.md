# 正交多项式、Krylov flag 与正则化

## 1. 合理的 polynomial 表示是什么

对固定复矩阵 $F$ 和非零 seed $b$，在无 breakdown 的前提下，Arnoldi 给出

$$FQ_m=Q_mH_m+h_{m+1,m}q_{m+1}e_m^T,\qquad Q_m^*Q_m=I,$$
$$q_j=p_{j-1}(F)b.$$

所谓“正交”是 $[p,q]_{F,b}:=(p(F)b)^*q(F)b$ 下的正交；这可能只是半内积，因为 minimal polynomial 会产生零向量。它一般不是实区间上一种固定正权 measure 下的 Chebyshev/Legendre/Jacobi 正交性。多 seed 用 block polynomial；两侧联合用两类 operator 的空间之和。

Arnoldi 不要求 Hermitian、normal 或 $\rho(F)<1$。它提供数值稳定坐标，不自动给出低阶逼近误差小、utility 大或递增改善的定理。[R15,R17,R18]

## 2. L、I-L 与尺度变化

**命题 2（affine Krylov invariance）。** 对 $\alpha\ne0$，

$$\mathcal K_m(F,b)=\mathcal K_m(\alpha F+\beta I,b).$$

证明：二项展开把右侧每个 power 表为左侧低阶 powers 的三角组合，非零对角 $\alpha^j$ 使变换可逆。故同 seed 下对 $L$ 或 $I-L$ 构造相同完整 Krylov 空间；差别是解释、缩放和浮点误差，不是新表达能力。

默认使用消元后的 $F_U^{\rm eff}$，因为它直接暴露 feedback 路径。可除以 $\alpha_s>0$ 防止巨大数值，但应保存尺度，且只称坐标稳定措施；不能由经验尺度宣称 operator norm 小于 1。

固定、线性的右预条件 $M$ 可以定义 $I-LM$ 的 Krylov space，再用 $M$ 映射到 current；这已经是另一种、预条件依赖的表示。左预条件作用于 $ML$ 时，forcing、output 与 trial/test space 必须一起推导。可变/非线性 preconditioner 的 flexible Krylov vectors 一般不能写成固定 $p(F)b$。第一轮不引入新 learned preconditioner。

## 3. 两侧、多个物理 seed 的联合算法

默认每个 seed 建一条长度 3 的 Arnoldi chain：O 两条用 adjoint；P、M 共四条用 forward。每条链使用 complex128 MGS 两遍，后续再按 degree 逐层联合正交化。不能先把全部列做 pivoted QR、打乱层级后仍用列号当 degree。

```text
CHAINS = []
for family, b in ordered_active_seeds:
    if norm(b) <= seed_relative_floor * source_scale: record inactive; continue
    op = Feff_adjoint if family == O else Feff
    q = b / norm(b)
    chain = [q]
    for degree in [1, 2]:
        w = op(chain[-1])
        w = MGS_twice(w, chain)
        if norm(w) <= breakdown_floor * recorded_action_scale: break
        chain.append(w / norm(w))
    CHAINS.append(chain)
Q = empty
for degree in [0, 1, 2]:
    block = all available CHAINS[*][degree]
    block = project_twice_against(block, [U, Q])
    Qdegree = rank_revealing_orth(block, deterministic_tie_break)
    append Qdegree to Q; record exact layer boundaries
return Q, layers, chain_Hessenbergs, chain_to_joint_coordinates, masks, costs
```

初始相对 seed floor 为 $10^{-10}$，breakdown 为 $10^{-12}$（FP64）；最终 $\|Q^*Q-I\|_F$、$\|U^*Q\|_F$ 都应低于 $10^{-10}$。这些是 pilot 的数值 gate，不是物理误差界。若 source scale 为零，该 seed 必须 inactive。不要用任意小数补出随机方向。

对于同层严重秩亏，用 SVD/QR 可合法更换层内坐标；保存变换与 layer projector。保留固定 seed family provenance，但联合列不能硬贴“百分之几属于 O”的唯一标签：重叠部分没有唯一归属。

## 4. 真正 gauge-invariant 的 degree prior

令 $\Pi_d$ 是 $\mathcal K_d$ 的正交 projector，$\Pi_{-1}=0$，$Q_d$ 张成新增层 $\mathcal K_d\ominus\mathcal K_{d-1}$。对 $c\in\mathcal K_D$，

$$\mathcal R_{\rm order}(c)=\sum_{d=0}^D w_d\|Q_d^*c\|^2
=c^*\left[\sum_d w_d(\Pi_d-\Pi_{d-1})\right]c,$$

等价地

$$\mathcal R_{\rm order}(c)=w_0\|c\|^2+
\sum_{d=1}^D(w_d-w_{d-1})\|(I-\Pi_{d-1})c\|^2.$$

证明：对每一正交层的能量展开，交换两重求和；第 j 层累计权重为 $w_j$。

因此该正则对层内 arbitrary unitary rotation 不变。实现中的 $\sum_j w_j|a_j|^2$ 只有当 $j$ 的权重来自 layer label 时才正确。换成任意全局基 $QW$ 后，必须同时变换 degree matrix $D\mapsto W^*DW$，不能继续按新列号加惩罚。

默认 $w_d=(d+1)^2-1$，即 0、3、8；$\lambda_{\rm order}=10^{-3}$ 乘在已按 reference gain 归一化的 loss 上。控制组 $\lambda=0$ 必做。leave-out lift 的旧列 $u_i$ 权重为 0。不可硬截断高阶 coefficient，也不能只因为近接触高阶能量大就宣布过拟合。

**物理解释：**它偏好低层 feedback complexity。在裸 $XG_D$ powers 中可联系 successive scattering paths；在 Schur $F_U^{\rm eff}$ 中，每一阶已包含 retained resolvent，因此是 unresolved feedback excursion 层数，不等于精确 Born scattering order。不同 seed 本身也可能已经包含全场、retained solve 或 adjoint 反馈。

## 5. 各 polynomial 家族裁决

|家族|能否代入任意矩阵|其熟悉稳定性/正交性何时可用|本轮裁决|
|---|---|---|---|
|Monomials|能|高次非正规 powers 易线性相关/瞬态放大|仅 depth-2 原始控制，不作主坐标|
|Chebyshev|能|Hermitian spectrum 经正确缩放到实区间；或另有有效 complex-domain bounds|不直接套原始 Maxwell L；空间 Cheb neural operator 不给这里的保证|
|Legendre/Jacobi|能|指定实 measure/边界条件下的 spectral representation|不把实空间基的正交性误当作 current-feedback 正交性|
|Faber|能|有合适 compact complex set、外共形映射、可控制的 approximation domain|理论比较；第一轮不实现复杂外域拟合|
|Arnoldi-generated|能|无需 normal；数值重正交与 breakdown 处理必须正确|主坐标|
|Lanczos|一般不能直接用 Hermitian 三项版|Hermitian/适当结构；bi-Lanczos 另有 breakdown 和双正交问题|不在一般 Maxwell L 上误用三项版|
|Cheb on $F^*F$|能，proxy 为 PSD|仍须有效上界与缩放|仅 controlled proxy；会改变空间、丢失相位定向信息并增加 adjoint 成本|

OPNO、SNO、Chebyshev neural operator 已覆盖“多项式 + NN”；LanczosNet 与 neural Krylov 也覆盖学习 spectral/subspace representation。[R09–R14] 本研究不能把基函数名称作为 novelty。

## 6. 一条有用但有条件的 utility 连续性结论

固定 embedding $Z$，令 $A_Z=Z^*LZ$、$O_Z=SZ$、$N_Z=Z^*B$。两个正交 endpoint coordinates $E,E'$ 满足 $\|E-E'\|\le\delta$，且两 endpoint 的最小奇异值都至少 $\sigma_0>0$。则

$$\|J_E-J_{E'}\|\le\|O_Z\|\|N_Z\|
\left(\frac{2}{\sigma_0}+\frac{2\|A_Z\|}{\sigma_0^2}\right)\delta.$$

证明：展开 $O_Z E(E^*A_ZE)^{-1}E^*N_Z$ 的两端与中间差，使用 $\|E^*A_ZE-E'^*A_ZE'\|\le2\|A_Z\|\delta$ 和 inverse difference identity。

若 $\Lambda\succeq\lambda_0I$，相应 real material step 满足

$$\|s'-s\|\le\lambda_0^{-1}\|J'-J\|
[\|r\|+(\|J'\|+\|J\|)\|s\|].$$

同一 full quadratic 上

$$|\Phi_F(s')-\Phi_F(s)|\le
(\|H_F\|\|s\|+\|h_F\|)\|s'-s\|
+\tfrac12\|H_F\|\|s'-s\|^2.$$

上述单 illumination 记号在多 illumination 情况下，必须先用共享 material 的 block lifting / realification 组成联合算子，再计算这些范数常数；不可直接照搬某一 illumination 的常数。

因此 phase/subspace distance 可作为 teacher distillation 的辅助损失；但接近 singular compression 时常数会爆炸，不能只凭 current reconstruction MSE 保证 gain。这也是 exact gain 与 stability gate 不能省的原因。

## 7. 表示误差与学习误差分离

对固定行动规则与预算，记 dictionary 和 manifold 的真实最优局部机会为 $G_D^*,G_M^*$。有恒等式

$$G_D^*-G(P_\theta)=(G_D^*-G_M^*)+(G_M^*-G(P_\theta)).$$

第一项可为负，连续空间可能更好；第二项只有对真实最优才非负。多起点连续优化仅给出 $G_M^{\rm found}\le G_M^*$，不是可认证上界。因此所有图、gate 必须写 **best-found manifold reference**，不能把它标为 global oracle。

A18-B 的 polynomial inverse baseline 很好，只说明该求解任务有强解析控制；它没有指定本轮 seeds 是否覆盖最佳 current rotation，也没有证明上述第二项必须靠 NN 缩小。
