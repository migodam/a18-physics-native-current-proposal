# Route C：Projective coefficient distribution（理论储备）

**定位：High-risk。本轮训练预算 0；不启动 Flow、diffusion 或 material prior。**

## C1. 它额外检验什么

Route B 的两个 mode/head 是确定性的少量 useful action 预测。Route C 只有在下列额外事实成立时才有理由：在相同 cheap features 下存在多个分离的 useful coefficient regions，deterministic top-4 在对象泛化中系统性漏掉其中一些，而且增加同规模 deterministic heads 或无学习随机补充不能修复。

“两个 singleton 为负、joint pair 为正”只证明 non-additive block utility，**不证明 singleton useful distribution 多峰**；应先用 joint-block proposal，而不是据此上 generative model。

## C2. 概率对象与可解释 target

对每 outgoing slot $i$，$t\in\mathbb{CP}^{r}$ 表示 $[u_i,Q]$ 中的一条 current direction。目标是

$$p_*(i,t\mid\xi,\mathrm{useful})\propto
q_0(i,t\mid\xi)\,\mathbb P(G_x(i,t)>\tau_x\mid\xi,i,t).$$

teacher 的探索分布 $q_0$ 必须记录。不能只保存优化器找到的最大 mode，再把它们称为 unbiased posterior samples。用于 weighted likelihood 的样本来自预注册探索/teacher pool；importance weighting 只有在抽样概率已知时才使用。

这里的概率是 proposal policy，不是材料后验，也不提供 Bayesian credibility 或 inverse-problem uncertainty 的自动解释。

## C3. 首个合理模型不是 Flow：complex angular Gaussian mixture

定义相对于复单位球均匀测度的 angular central Gaussian density：

$$f(t;\Sigma)=\det(\Sigma)^{-1}(t^*\Sigma^{-1}t)^{-m},
\qquad m=r+1,\quad\Sigma\succ0,\quad\|t\|=1.$$

这是本方案使用的解析密度：从 $z\sim\mathcal{CN}(0,\Sigma)$ 取 $t=z/\|z\|$，对 radial coordinate 积分即可得到；它对 $t\mapsto e^{i\phi}t$ 不变，也对 $\Sigma\mapsto a\Sigma$ 不变。因此固定 $\operatorname{tr}\Sigma=m$，避免不可识别的尺度。

令

$$p_\theta(t\mid i,\xi)=\sum_{h=1}^{H}\pi_h f(t;\Sigma_h),\quad H=2\text{ 或 }4,$$

$\Sigma_h=\epsilon I+\lambda_h\mu_h\mu_h^*$ 后 trace normalize。限制 eigenvalue ratio 不超过 100，防止训练集上的 delta collapse；这是一项设计 regularization，不是概率校准定理。

可从每分量取一个方向（deterministic mean-axis control），也可按同样总预算 K=4 抽样。**必须比较这两个版本**，否则随机改善可能只是多 mode head 的收益。复杂维度只有 19–25，网络无需输出高维 material image。

## C4. Teacher、loss、invariance

用 Route B 同一 cache、同一 action set、同一 split。探索 coefficient 样本加标签 $G$ 与合法性；用 bounded monotone utility weights 训练 mixture NLL：

$$L_C=-\sum_j\bar w_j\log p_\theta(i_j,t_j\mid\xi)
+\lambda_cL_{coverage}+\lambda_dR_{order}+\lambda_eR_{collapse}.$$

计算 NLL 时用真实 angular density，不能把归一到 sphere 的 coefficient 仍当 ordinary unconstrained Gaussian。zero-gain state 学 no-op mass。phase augmentation 对 density 应严格不变；basis rotation 后同时变换 $\Sigma$ 与 features。

pair block 的分布在 Grassmannian 上，不可把上式直接换成矩阵。若确有 pair target，先在已采样第一列的正交补中给第二列条件分布，并用 joint endpoint utility；单列 density 相乘不是无条件的 joint-pair novelty。

## C5. 未来算法（尚未授权训练）

```text
encode exactly the Route B cheap features
predict outgoing mass and H trace-normalized covariance factors
draw at most K coefficient directions with a frozen RNG seed policy
decode only selected currents; reject numerical degeneracies
evaluate exact A17 endpoints and full finite gains
accept the best legal positive action; otherwise no-op
```

训练不跑 Maxwell；测试 K、Jd 数量、seed 数量和 total acquisition wall 与 deterministic B 严格匹配。分别报告单次部署期望、尾部失败率和随机 seed variability，不能只报告“多试 100 次取最好”。

## C6. 资源、失败模式与启动 gate

未来模型上限 0.3M，cached dataset 与 B 相同；首次 mixture pilot 不超过 20 epochs/20 GPU minutes/2 GiB target，仍需单卡队列。**这些是未来预算，不属于本轮训练授权。**

启动需要：B/A 至少一条已通过实用成本 gate；B 的 best-found manifold reference 仍留出至少 .10 coverage headroom；在不少于 8 个独立 validation objects 上发现两个 phase/subspace-separated useful regions；确定性 4-head/8-head 与 fixed random supplement 控制仍不足；四样本随机策略有可预见同成本优势。

若 deterministic coverage 已达 reference 的 .95，则 C 直接 NO-GO。若差距来自 manifold 缺失、廉价 features 不含信息、近 singular endpoints 或 pair-only utility，C 不是默认补救。Flow/diffusion 只有在这个简单 mixture 明确因分布几何而失败时才另立研究任务。

material generative prior 的启动条件仍独立：必须有 genuine full-physics material weak-subspace 的 oracle evidence，并排除 current approximation defect；本仓库不提供该证据，也不训练该模型。
