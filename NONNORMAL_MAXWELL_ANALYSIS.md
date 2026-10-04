# Non-normal Maxwell：operator、稳定性与实现边界

## 1. 不做的三个假设

不假定 $L=L^*$、不假定 $F$ normal、不假定 $\rho(F)<1$。Reciprocity 可能带来复杂对称/加权关系，但不自动得到 Euclidean Hermitian 正定性；代码必须用 conjugate adjoint 而非 transpose。current metric 与 real material metric 分别声明。

例：$F=\begin{bmatrix}0&M\\0&0\end{bmatrix}$ 的全部特征值为零，$\|F\|=M$ 却任意大。仅靠 eigenvalue/Ritz 点看起来“靠近零”，不能推断 powers 或 polynomial action 稳定。

## 2. 数值域与 Faber 的正确使用

Crouzeix–Palencia 的结果给出

$$\|p(F)\|\le (1+\sqrt2)\max_{z\in W(F)}|p(z)|,$$

其中 $W(F)=\{x^*Fx:\|x\|=1\}$ 是 **完整 operator** 的 numerical range。[R15] Faber 方法可基于合适 complex compact set/数值域组织 matrix-function 近似与误差估计。[R16]

但 $W(Q^*FQ)\subseteq W(F)$ 通常只是内近似，不是完整域的上包络；小 Hessenberg 的 spectrum 更不是完整 pseudospectrum。只采若干角度/随机向量也没有自动获得严格外界。第一轮只把这些量当 diagnostic，不拿来认证稳定性或 acceptance。

为缩放计算可用已知/可验证 norm bound，或保存经验 scale 再由 action norm 检查防 overflow。后者不称 rigorous bound。

## 3. Schur action 与其 adjoint

以 $U^*U=I$、$D=U^*LU$、$L_U=LU$ 为基础。所有输入输出都在 complement：

```text
P(q) = q - U(U* q)
Aeff(q):
    q = P(q)
    l = L(q)
    return P(l - L_U solve(D, U* l))
Aeff_adjoint(q):
    q = P(q)
    h = solve(D*, L_U* q)
    return P(L*(q - U h))
Feff(q) = P(q) - Aeff(q)
Feff_adjoint(q) = P(q) - Aeff_adjoint(q)
```

adjoint 版本在已缓存 $LU$ 时同样只需一次 $L^*$ action，不必为每个 state 额外生成整块 $L^*U$。必须测试 $\langle x,Aeff(y)\rangle=\langle Aeff^*(x),y\rangle$；近似 Green action 也要通过它自己的离散 adjoint 测试。

observation seed：令 $v=S^*r$，

$$b_O=P[v-L^*U D^{-*}U^*v].$$

可以用一次 $L^*$ 对线性组合 $UD^{-*}U^*v$ 的 action 计算。不能把它简化成 $PS^*r$ 却仍称 dressed observation；若 $S^*r$ 已在 receiver backbone 内，后者可能完全为零。

material/forcing seed：令 $v=Bz$ 或 $f$，

$$b=P[v-(LU)D^{-1}U^*v].$$

仅需旧 $LU$ 与小 solve；这里不需要 $L^{-1}v$。

## 4. 为什么 retained loop 很重要：可复核二阶例子

$$F=\begin{bmatrix}.2&.6\\.4&.1\end{bmatrix},\quad U=e_1,\quad S=[1,0].$$

对 $c=e_2$，直接 radiation 为零，$S\bar c=.6/.8=.75$；complement bare feedback 是 .1，而 Schur feedback 为 $.1+.4(.8)^{-1}.6=.4$。这说明先投影后传播会删掉当前机制要保留的 receiver return 和闭环。它只是代数 witness，不是新的 Maxwell 实验。

## 5. 低成本 feature 不能偷偷付出完整 workspace

在线默认只保存：各链已算出的 Arnoldi Hessenberg、joint QR 坐标变换、O/P/M seed 投影、旧 $LU,SU,U^*Bz$ 与 task/geometry 小摘要。**不要求为了 GNN edge 额外计算所有 $Q^*FQ$ 或 $Z^*LZ$。**

每个 selected continuous candidate 可直接付一次 $Lc$，与已缓存 $LV$ 构成小 endpoint 矩阵。只有在同一 outer state 的多轮使用中，实测缓存 $LQ$ 更省时，才允许一个预注册 cache branch；其首次取得成本必须完整计入。离线 teacher 的 $Z^*LZ$ 不得注入在线 features。

这使 Route B 真正有可能省去原 64-column anchor / 128-workspace，而不是给新 workspace 换名字。

## 6. 稳定性 gate

原 A17 在 admissible endpoints 上使用 relative current-stability cutoff $10^{-10}$，material normal residual cutoff $10^{-10}$。[P1] Adapter 必须确认相同定义、相同 normalization 与 FP64 路径。

额外的 proposal 检查：有限数值；weighted norm；$\|V^*C\|$；block rank；重复/near-no-op 标记。near-no-op 可以合法返回，但不能计为成功 proposal。

若 $D_U$ 本身不满足旧 endpoint 的 stability 条件，停止 Schur manifold 分支并交回固定 fallback/no-op，不为构造 proposal 放松稳定性。对于某个 leave-out $V$ 的奇异 core 则不同：其 endpoint 可能仍然可逆，必须走 direct endpoint。

若在连续 teacher 中靠接近 singular compression 取得巨大“gain”，应按同一 endpoint gate 删除；绝不能通过 train-time ridge $\epsilon I$ 改写 endpoint 后把它当 A17 label。

## 7. 预条件、物理约束与 singular states

通量/连续性约束应由原离散模型与合法边界处理实现。不要把所有 contrast currents 误投影成 divergence-free；电荷可导致非零散度。若后端已有严格定义的 admissible current subspace，应先在该空间白化并构造 operator。

不因为 Maxwell 的 passive 材料就假设任意 Galerkin compression 可逆。full $L$ 可逆也不保证每个 $V^*LV$ 可逆。极端 non-normal、near-contact、collective resonance、强损耗/强 contrast 均应作为独立 failure strata。

双侧 Arnoldi 与单侧 Arnoldi 的收益是实验问题：如果 cheap adjoint 和 dressed O seed 无提升，就用更便宜的 forward-only 版本。不能保留多余结构来维护“物理味道”。

## 8. 必报 diagnostics

保留 orthogonality residual、各层 surviving rank、seed inactivity、每链 action norm/scale、旧 D 的条件估计、selected endpoint 最小奇异值、teacher boundary rejection、receiver geometry holdout、近接触分组。小矩阵 pseudospectrum 和 numerical range 只标为 projected diagnostics。

单张 GPU 的争用会污染 wall time：计时时禁止其他 agent 训练、调用 CUDA Maxwell 或做 GPU plotting；所有 backend action 与数据传输的时间均有独立字段。
