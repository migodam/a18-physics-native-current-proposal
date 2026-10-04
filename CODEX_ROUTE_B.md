# CODEX ROUTE B — 实施指令，可直接粘贴给独立 agent

你负责 Schur-feedback polynomial current proposal，不学习 L^{-1}，不做 neural solver，不学 Q scorer，不预测 material image。另一个 agent 负责 A。以本仓库 THEORY_OVERVIEW、ORTHOGONAL_POLYNOMIAL_THEORY、NONNORMAL_MAXWELL_ANALYSIS、ROUTE_B_ORTHOGONAL_FEEDBACK_PROPOSAL、FEATURE_CONTRACT、共享数据与 gate 文档为固定规格。

## 0. 所有权与顺序

branch `route-b`；你拥有 `route_b/`、`tests/test_route_b_*`、`results/route_b/`。Integrator 所有 shared backend、manifest、teacher 和 GPU queue。与 A 并行写 CPU代码；GPU必须 J1 A-MLP 释放锁之后才进入 J2 B-MLP。C不训练。

原 A17 不改，特别不改 prior/damping/residual sign、polarizability derivative、material gauge、current rank、endpoint stability 或 tau。无法取得完整 Maxwell backend 时报告 BACKEND_UNAVAILABLE；不要以本仓库 dense随机矩阵结果代替 Maxwell pilot。

## 1. 先实现不用 NN 的物理表示

`route_b/feedback.py` 实现矩阵自由 Feff 和其精确 adjoint。使用已缓存 LU：

```text
P(x)=x-U(U*x)
D=U*LU
Aeff(q)=P(L(Pq)-LU solve(D,U*L(Pq)))
Feff(q)=Pq-Aeff(q)
Aeff_adj(q)=P L* (Pq-U solve(D*,(LU)*Pq))
Feff_adj(q)=Pq-Aeff_adj(q)
```

每 action 必须实际只调用一次大 L 或 L*；缓存 miss 的 LU 构造另外计费。D不稳定时不强行 inverse，不以 ridge 改问题。对 frozen合法U，执行dot-product adjoint test；与显式 block Schur 在小系统对齐。`src/algebra.py` 是参考公式，不是实际FFT-DDA后端。

`route_b/seeds.py`：O dressed Y* r 两条；P retained current residual 两条；M effective injection 两条。四illumination的sourcephase/sketch固定；Bz保留真实材料导数。O走adjoint chain，PM走forward。source floor/phase convention与FEATURE_CONTRACT一致。默认root从六seed、degree0/1/2构造，最多18rank、hard cap24。

`route_b/flag.py`：每条链两遍complex128 MGS，再按degree联合正交；不可先打乱列再按index加degree权重。记录H、chain-to-joint、seedcoords、layers、masks、norms、operators counters、gauge hash。零seed为inactive，不补噪声。Q⊥U，Q*Q=I，误差<1e-10。

## 2. 无 NN coverage/cost preflight

先12个developmentstates，k8。比较 degree0、raw PFP、Schur Feff、单侧与OPM联合；root depths0/1/2只在development选择。并比较 shared-root 与逐drop-core oracle；后者只离线诊断，不免费送给在线B。

continuation action 对drop i为 `c=[u_i,Q]t`，t单位norm；不要强制c⊥全部U，因为那会漏掉partialrotation。endpoint `[U without ui,c]` rank始终k。t=e0为no-op。旧 u_i 的order权重0，新层0/3/8。

Oracle先用 frozen Z=[U,Q] 压缩 AZ/OZ/NZp，按Route B小矩阵identity求J、真实real-materialstep、full frozenG。缓存中含fullJ/H，只能在teacher进程。B最优teacher是best-found，不是globaloracle。

preflight: 每drop最多8 starts×40 steps，lr.03，complex128、sphere retraction、5次半步稳定回退；扩展96states每drop4starts×24steps。全部共享CPUteacher累计120min封顶；不允许由两个agent重复生成。最终保留至多2个有效且projective separated的正增益teacher。

若G1B coverage/cost或G2 no-NN headroom不通过，立刻冻结结果，不训练。必须报告原full dictionary与共同union reference两个分母。projection of original best dictionary只允许teacher初始化，不能作为onlinefeature。

## 3. 小模型与loss

输入 FEATURE_CONTRACT 的固定1552 real；state encoder1552→128→128，GELU、LayerNorm。与32-dimensional outgoing拼接：160→128→102，其中102=2×(2×25+1)，表示每drop两个complex25系数向量与两个logits；no-op head128→1。rankmask使padding为零，再归一。保存实际参数量，约0.3M，不用CNN或高维currentdecoder。

实现 `route_b/loss.py`：使用cached small AZ/OZ/NZ求每个预测mode的真实endpointstep和G，不求fullMaxwell。采用Route B的cover/listwise/projective/diversity/order组合。positive utility由可行且G>原tau定义，保留signedrawG。invalid endpoint不偷偷加ridge；赋固定invalid penalty=1并通过teacher projective term提供学习信号。没有teacher的slot不伪造target。

projective loss=`1-abs(vdot(t,teacher_t))**2`；两个modes取最佳二元匹配，teacher只有1个时对预测最接近者计算。list target对两个modes按outgoing求概率和，避免label-order任意性。diversity只在该slot确有两个separated有效teacher时启用。order用layer matrix，不是全局QR列号。

AdamW lr1e-3、weight_decay1e-5、batch8、max80epochs、clip1；每5epochs跑真实deterministic top4 validation。epoch20 gate，patience15；训练小solve可complex64，但FP64 validation为准。complex64失败不能转成更松接受阈值。每epoch断言Maxwell actioncounter为0。

在同一train/validation预算下跑lambda_order=0控制。可用固定loss ablation在CPU缓存上做短轮，不另开无限GPUcampaign；所有epochs含失败都计费。

## 4. 部署实现

`route_b/proposal.py`先输出2k个低维action records与logits，选4个，再只实例化4个physical currents。不要为了所有records预先算LQ、SQ或所有candidate endpoints。只对selected c求Lc、Sc、c*B，以及真实endpoint所需的数据；B也不能假设能全存，用原后端的material-adjoint/dual路径。

所有selected动作都由共享A17 verifier检查 full all-illumination Jd 和finiteG。某commoncore奇异不意味着stableendpoint非法：使用directendpointfallback。接受后重建依赖U的root/cache，最多三次接受。把多轮重建费加入Tinc。

记录`n_coefficient_records`、`n_current_vectors_materialized`、`n_endpoint_solves`、`n_full_Jd`、`n_illumination_rhs`；不能只报K4却不报2krecords，也不能把全部端口数当成currentrank。

## 5. Pair审计与Route C隔离

只做12state无NN pair audit：每state最多6个预定drop pairs、每pair4starts×20steps、rank2 Stiefel coordinates。pair gain来自joint endpoint，不求两个singletons的和。pair-only显著说明actionspace需要block，不自动说明需要generative model。

deterministic接近oracle则停止学习扩展。C仅理论；不要安装Flow/Diffusion或在同一个GPUjob中偷偷加入随机head。B-Set只有G3通过且拿到J4配额后才实现/训练，仍需matched-MLP与gauge/degree ablation。

## 6. 待新增CLI与输出

```text
python -m route_b.audit --config config/pilot.json
python -m route_b.preflight --split development --teacher-budget-minutes 120
python -m route_b.train --config config/pilot.json --job J2 --seed 1801
python -m route_b.evaluate --checkpoint <frozen_path> --split locked_test
```

这些是本agent要新增的接口，不是当前仓库已有可运行训练代码。训练只能经integrator的tools/gpu_queue.py启动。

必须交付：所有接口与CPUtests、feature/cost dependency图的机器可读表、oracle-versus-analytic-versus-NN coverage、original-dictionary comparison、gauge/order/Schur ablation、峰值VRAM与完整jobledger、GATE_DECISION.json、失败原始记录。

B的直接终止条件：shared-root不覆盖好action；为补coverage必须深到没有acquisition收益；bestnoNN已达到.95 reference；对held-out object无提升；或A在coverage/acquisition Pareto上支配B。不要为了“continuous更有novelty”保留失败路线。结果统一提交本仓库，不再拆新仓库。
