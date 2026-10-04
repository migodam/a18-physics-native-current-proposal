# 共享数据、teacher 与 backend contract

## 1. 三层数据，不混淆独立性

**历史接线层：**A17 / Stage 0 的原始四对象、12 states、36 queries 与已有 action labels。47,712 moves 或 5,208 incoming labels 不是独立 objects。严格保留原 smooth-train / near-contact-test 等 fold 的含义；已用于方向判断的历史 test 不能重新宣传成 blind holdout。[P1,P3]

**共享 pilot 层：**优先读取 A18-B 的 48 scenes / 96 frozen states 和 full-precision teachers。[P4] 只有在离散物理、材料 gauge、forcing、receiver packing、polarizability derivative 与 A17 adapter 均核对后，才复用这些 state/teacher。它们不是现成的 A17 exchange labels：需新建 receiver U、recipe/action 与连续 proposal labels。

**独立 generalization 层：**若要宣称新对象泛化，需未用于设计的 geometry/object seeds。目标规模为 48 个 objects、每对象 2 states：32 train / 8 validation / 8 locked test；这个比例仅适用于尚未冻结的新 manifest。已有数据已固定 split 时不得随意重分配，只把历史分析作为 exposed-data pilot，另留新对象做最终测试。

每个 proposal query 还必须有 `configuration_id=current_U_hash`：同一物理 state 在接受一次 exchange 后，已经是不同的条件决策问题。第一轮 tiny training 默认使用初始 receiver U；最多三轮 sequential exchange 是必须单独报告的 current-configuration transfer，不假称训练已覆盖这些访问状态。只有development/train对象可追加analytic-policy产生的post-exchange配置，且仍受原teacher预算约束；禁止把held-out rollout回填训练。

所有 outer states、noise variants、budgets、illumination views、动作、current configurations 和派生 teacher 都随 parent object 进入同一 split。object family 分组还需记录，尤其 smooth、near-contact、layered、asymmetric/high-contrast。

## 2. 最小 preflight 与升级条件

先做最多 6 objects / 12 states 的 representation-and-cost preflight，全部标为 development，不动 locked test。默认 k=8；k=4/16 仅作 evaluation transfer，不把三个 budget 视为三倍独立样本。

只有无 NN 的 oracle/cost gate 通过，才扩至最多 96 physical states、288 state-budget queries。若缺失外部 vendor modules、完整 teacher 或 source hashes，停止并报告 `BACKEND_UNAVAILABLE`；不得用 scalar Born、随机矩阵或缩小物理模型偷偷替代后报 Maxwell PASS。

## 3. 文件布局与具体 schema

```text
data/manifest.jsonl               # one frozen physical state per row
data/splits.json                  # object-level, immutable hash
cache/online_features/*.npz       # deployment-allowed only
cache/bases/*.npz                 # U, Q, layers, source scales (CPU)
cache/teacher_only/*.npz          # full-J/H, compressed matrices, truth
labels/action_A0.jsonl            # original dictionary, historical replay
labels/action_A1.jsonl            # actual lazy recipe bank
labels/continuous_B.jsonl         # best-found directions + gains
labels/pair_audit.jsonl           # explicit joint endpoint labels
results/policy_actions.jsonl      # every attempted/invalid/accepted action
results/costs.jsonl               # exclusive stage times and counters
```

每个 manifest row 必含：

```json
{
  "object_id": "immutable-parent-id",
  "state_id": "object-id:outer-000",
  "family": "near_contact",
  "split": "train",
  "historical_exposure": true,
  "backend_commit": "resolved-upstream-commit",
  "state_sha256": "actual-file-hash",
  "material_gauge_sha256": "actual-gauge-hash",
  "receiver_geometry_sha256": "actual-geometry-hash",
  "current_metric": "whitened-positive-volume-metric",
  "n_current_complex": 0,
  "n_material_real": 24,
  "n_illuminations": 4,
  "residual_sign": "predicted_minus_observed",
  "teacher_status": "not_generated"
}
```

上面 0 和字符串是 schema 示例，不是可运行假数据；loader 遇到未解析字段必须拒绝。B 每条 label 另含 `drop_indices, coefficient_real, coefficient_imag, coefficient_gauge_hash, gain_fp64, tau_online, feasible, sigma_min_relative, normal_residual, start_id, optimizer_steps, reference_type=best_found_not_global`。记录 raw seed 和原始 endpoint，不只保留最优表格。

## 4. backend 最小接口

```text
freeze_state(object_id, state_id) -> immutable State
apply_L(v), apply_L_adjoint(v) -> complex current, cost receipt
apply_S(v), apply_S_adjoint(y) -> output/current, cost receipt
apply_B(real_material_direction, illumination) -> current
get_receiver_basis(k, geometry_cache) -> U, provenance
get_old_endpoint(U, frozen_state) -> step, LU, cached_reduced_objects
solve_endpoint(current_basis, frozen_state, cached_actions) -> Endpoint
full_J_action(real_material_direction) -> all-illumination real data
finite_gain(old_step, new_step, cached_full_output) -> GainRecord
online_tolerance(GainRecord) -> original A17 significance threshold
```

`Endpoint` 返回 current compression condition、material normal residual、material step、所有 solve 成本和失败原因。上述接口是 Codex 要实现的 adapter，不是本仓库已包含的完整 Maxwell solver。`src/algebra.py` 仅提供独立 dense 代数验证实现。

### 第一次 verifier cache 不是免费的

若没有真实 full output cache，先对当前旧 step x=s_U 支付一次 `full_J_action(x)`（全部 illumination），得到 u=r+J_F x。不能用 reduced r+J_U x、outer state 的 field residual、或已付费的 step-zero gradient 替代。初始化费用计入 Tinc 和 full-direction/RHS counters；只有后续已接受 d 才能用 u←u+J_Fd 精确更新。同一个 frozen state/同一 policy 内可复用；方法之间的比较必须使用相同允许缓存规则。

### 当前模型状态与 material injection

$B_p$ 必须来自当前 $\chi$、当前真实 total field、实际 polarizability derivative。不能按 candidate 分别重算 total field，不能将 A18-B 的 learned closure 输出拿来替代本轮 exact endpoint labels。

### complex / real packing

统一 `pack(y_p)=[Re y_1, Im y_1, ..., Re y_P, Im y_P]`。真实 material 的 adjoint 是此 packing 下的实 transpose。train-time complex current 内积仍用 conjugate transpose。提供 packing/adjoint unit test 与 reference dot-product test。

## 5. teacher acquisition 只做一次

同一 state 共享 full J/H 或必要 full directional reference、LU、receiver geometry、online seeds、Q、candidate cache。Route A 与 B 不能由两个 agent 重复跑 full teacher。

A0 直接读取有效已有 labels；A1 用真正 lazy bank 重算 finite gain，不能复用不一致 recipe 的 label。B 用离线 $Z=[U,Q]$ 的 compressed matrices 生成连续 labels。pair audit 用 rank-2 endpoint 直接算，不能用 singleton 表拼凑。

资源封顶：preflight 12 states；CPU teacher 全阶段最多 120 分钟 active wall；必须使用 GPU 生成缺失物理数据时最多 60 GPU 分钟，并在 NN 队列之前独占执行。超过封顶则缩小为 development feasibility report，不能偷偷删掉难例或把缺失当失败/零 gain。

这些是预算，不是硬件速度预测。冻结 cache 后 epochs 禁止调用 `apply_L`、`apply_L_adjoint`、full solve 或 FFT Maxwell；小矩阵训练算式不受此禁令影响。

## 6. 数据隔离与审计

online feature loader 只允许 schema 白名单，禁止 `teacher_only/` 文件路径。每个 feature 必须带 `origin`、`dependencies`、`online_cost`。训练时可以读 teacher 作 loss，但 feature extractor 进程不能读取 full J/H/truth。

feature 标准化均值/方差只用 training objects；zero mask 单独编码。B 的 layer/QR gauge 与 coefficient label 一起 hash，不能用上一 state 的 coefficient 解释下一 state 的 Q。任何 accepted exchange 后 U 变化，cache key 也必须变化。

offline train encoder 可以跨 A/B 共用同一输入定义，但公平对照必须从同一随机初值策略独立训练；不能把 B 更长训练的 encoder 作为 A 的免费起点或反之。共享的是数据/后端/格式，不是隐藏的额外监督。

## 7. 教师不完美时怎么报告

多起点 continuous best-found 只是下界。dictionary 如果全枚举且每个 endpoint 通过原验算，可称 `exact_enumerated_dictionary_reference`（指有限集合，不指 rigorous floating-point certificate）。未求全的 dictionary 只能称 `audited_pool_reference`。

每个状态的 common reference 用原字典、lazy bank、连续 best-found、无 NN controls 的所有有效已检查动作取最大；报告各 reference 单独值，防止某方法靠较弱分母看起来优秀。

若 source 当前只能提供 16-dimensional material sketch，它只能做投影诊断，不能填 full utility 字段。若真实 nonlinear reference 缺失，frozen 指标仍可报告，但不得得出最终 imaging quality 结论。
