# Online feature contract：禁止让 agent 自己补出昂贵特征

版本 v1；为第一轮固定四 illumination、标量 isotropic DDA 设置。其他 illumination 数、各向异性材料、接收 packing 的迁移需另建 schema 与测试，不静默截断。所有 current 数据在已声明的正定 metric 中白化。

## 1. 共享 state vector：192 个实数

`xi = [scalars64, residual_sketch64, material_sketch64]`。train-only z-score；离散 flags、masks 不 z-score。对正范数采用 `log1p(value / train_scale)`；scale 只由 training objects 的非零中位数确定，空列 scale=1。不按 test outcome 再选尺度。

**scalars64：**下面每组恰好 16 维，顺序固定。

1. 材料：Re χ 的 mean/std/min/max，Im χ 的同四项，abs χ 的同四项；非零 support fraction；abs polarizability 的 mean/max；有效体积。统计 current estimate，不读 truth。
2. 几何：计算盒 side lengths 3 项、中心 3 项、receiver centroid 相对计算盒中心 3 项；receiver RMS distance；复 measurement channels/illumination 数；illumination 数；wavenumber；grid spacing；active cells；geometry-cache-present flag。
3. residual/source：log norm r、mean/std/max abs r；四 illumination 的 log norm r_p；四 illumination 的 log norm f_p；四 illumination 的 log norm retained current residual。不存在的 residual 不填 full-solver residual，缺 cache 时按成本 contract 明确构造；本 pilot 固定四 illumination。
4. 旧 endpoint：k、real material dimension、scalar λ、norm ell、norm s_U、σ_min(D_U)/σ_max(D_U)、normal residual、mean/max abs D_U、Re/Im trace D_U、Frobenius norm SU、Frobenius norm LU、已有 reduced gradient norm、accepted-exchange count、cached-full-output-present flag。

**residual_sketch64：**对已知源相位 canonicalize 后、按固定 receiver/illumination ID 排序的 complex r，做 32 个固定 complex Rademacher 内积，权重 modulus=1/sqrt(m_complex)，拆 Re/Im。seed=180018，counter key=`receiver_geometry_hash, channel_id, sketch_id`。同一几何所有 policy 共用。receiver permutation 时索引和 S/r 同步变化；不能只移动 r 的行。独立改变源相位而不 canonicalize 不是不变操作。

**material_sketch64：**当前 complex χ 与 32 个体积加权 Fourier test functions 的内积，test frequencies 按整数 lattice shell norm、lexicographic 排序（含零频），坐标用固定计算盒归一。只输出 32 complex numbers；不生成全 Fourier current bank。不同 object 可以有相同计算盒；不得用 truth support 重定义盒子。该特征不声称旋转不变或未知平移不变。

## 2. A 的固定 schema

state condition 为上述 192 维加 6 个 seed log norms、6 个 active masks，合计 **204 real**。A 使用同一 O/P/M seed 工厂，但不为了网络输入先做全 Arnoldi chain。全部 seed 成本仍计入 acquisition。

每 recipe **32 real**，顺序：family one-hot 5；support center 3；support scale 3；Cartesian orientation 3；Fourier wavevector 3；degree 1；O/P/M family one-hot 3；静态 raw atom norm 1；support fraction 1；支持上 Re χ mean、Im χ mean、abs χ std 3；seed ID 1；receiver geometry-mode index 1；support/frequency/material-summary/geometry-summary 四 valid flags。无此定义的字段填 0 并置对应 flag=0。global atom 的材料摘要用全盒，不按结果寻找局部 support。

第一轮 **关闭所有 atom–seed projection feature**，避免 random atom dot products 或全 bank radiation 藏入输入费用。打开必须另标 `A_feature_plus`、计费并重新对照。

每 outgoing slot **32 real**：one-hot slot ID 16（k<=16）；norm Lu_i、norm Su_i；Re/Im D_ii；mean/max abs row_i(D)；mean/max abs col_i(D)；Re/Im `<Su_i, sum_p r_p>`；Re/Im mean of row_i(U*B Zprobe)（仅已有时）；原 receiver-mode index；protected flag；两项有效性 flag（injection row、slot valid）。不存在 U*B Zprobe 缓存不新求，置 0/mask；非 receiver 原始列的 mode index 为 -1，source correlation 对本四 illumination pilot 必须可用。以上恰为 16+16；norm current 已经为 1，无需重复填。

A0 原 labels 的 node ID 与 A1 recipe ID 不共用。任何新 recipe 必须重算 label。

## 3. B 的固定 schema：1552 real

|块|形状|real 个数|
|---|---|---:|
|shared xi|192|192|
|六 chain 的 Arnoldi H|6×3×2 complex|72|
|chain-to-joint coordinates|24×18 complex|864|
|六原 seed 的 joint coordinates|24×6 complex|288|
|6 seed norms + 12 action norms|18|18|
|joint rank mask|24|24|
|joint layer one-hot|24×3|72|
|seed active mask|6|6|
|root diagnostics|16|16|
|合计||1552|

18 chain columns 顺序固定为 O1/O2/P1/P2/M1/M2，每条 degree 0/1/2。joint basis 按 degree 层构造；缺列补零，使用 masks。chain H 由已发生的 Arnoldi inner products 产生，不能为了补齐 H 的最后一列再调用一次 F。

root diagnostics16：rank；三个 layer ranks；six seed-to-original-source norm ratios；max orthogonality defect；max backbone overlap；min active Arnoldi residual/action ratio；max chain action norm；relative σ_min(D_U)；unused-rank fraction。缺 active residual 的 min 设 0，并从 masks 解释。

`chain_to_joint = Q*chain_columns`；`seed_coordinates = Q*seeds` 只需 CPU 内积。**不输入** `Q*LQ`、`SQ`、full `Q*B`、全局真 gain；这些属于 teacher。输入动态 dropout descriptor 与 A 相同，32 real。

实现中 coefficient dimension rmax+1=25：第 0 项对应旧 outgoing u_i，其余对应 padded Q；输出前乘 `[1, rank_mask]`。无效 padding 不参与归一化或 diversity。

## 4. GNN 的边：明确近似，不伪装 full feedback matrix

A-GNN 每个 node 最多 8 incoming neighbors。compact local atoms 用 support centroid 距离建立邻居（同距按 recipe ID）；global atoms 按 family、degree/frequency distance 排序连接。无 overlap/local support 的 global atom 不使用假 centroid dyadic coupling。

每边 6 real：Re/Im centroid Green coupling、归一化距离、k·distance、same-family indicator、degree difference。local coupling 由同一 DDA Green kernel 在两个 centroid 上给出，投影到 Cartesian atom orientations，并乘 receiving centroid 的当前 polarizability；这是 **centroid approximation**，不是 d_i* F d_j。self/dist=0 边的该 coupling 置零，以单独 residual/self-channel 表示。Green 子程序不可调用全 n×n matrix constructor。

message=`MLP134to64([h_i,h_j,edge6])`，输入 134；两层 affine 134→64→64、GELU。聚合取 neighbor mean，`h_new=LayerNorm(h+aggregate)`；两轮。用于同参数随机边控制时保持 degree sequence、edge-feature marginal distribution，只置换端点。几何边控制去掉 complex Green 两列，补零保持网络规模。

## 5. Gauge、seed floors 与 cache key

新 Arnoldi 列先两遍 MGS，再将最大 modulus 的 current component 旋转到非负实数（并列取首个 grid ID）；记录相位并同步更新 H、坐标和 labels。底层代数不依赖此 gauge；神经输入采用该约定只为重复性，不声称自动 unitary-equivariant。

P source scale=`norm(f)+norm(LU solve(D,U*f))`；M 用同一公式把 f 换成 Bz；O source scale 为 dressed adjoint 的两个实际项范数之和。零 scale 则 inactive；相对 seed floor=1e-10，Arnoldi breakdown=1e-12。相位 canonicalization 属于待实现 extractor；仓库的 NumPy algebra tests 检查代数 gauge invariance，并非已验证一个 NN 的 gauge robustness。

cache key 必含 state hash、当前 U hash、B/gauge hash、recipe-schema hash、seed-schema hash、所有 numerical cutoffs、source phase convention。accepted exchange 必然产生新 U key。首次几何 preprocessing 与 warm-cache 周期分别计费。
