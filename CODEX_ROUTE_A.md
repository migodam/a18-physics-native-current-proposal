# CODEX ROUTE A — 实施指令，可直接粘贴给独立 agent

你负责本仓库 Route A 的实现，不负责重写 A17、训练完整 Maxwell solver、拟合 Q scorer 或 material image。另一个 agent 实现 Route B。请先读 START_HERE、THEORY_OVERVIEW、ROUTE_A_DICTIONARY_PROPOSAL、FEATURE_CONTRACT、SHARED_DATA_AND_TEACHER_SPEC、GPU_JOB_ORDER、GO_NO_GO_GATES。config/pilot.json 是数值配置；冲突先暂停并记录，不自行选有利版本。

## 0. 分工与硬边界

仅在 branch `route-a` 工作。你拥有 `route_a/`、`tests/test_route_a_*`、`results/route_a/`；共享 backend adapter、数据 manifest、teacher generator、GPU lock 由 integrator 所有。提交修改共享接口的 patch 给 integrator，不在自己分支私造一个不同 Maxwell simulator。

原 A17 / Stage 0 / A18-B 仓库只读，按 sources provenance 固定 commit。若外部 42 个 vendor modules 无法解析，返回 BACKEND_UNAVAILABLE；本仓库随机矩阵 unit test 不可替代 full vector Maxwell evidence。

## 1. 第一个交付：无 GPU dependency audit

实现 `route_a/recipes.py`：`Recipe(id, family, static_descriptor, dependency_ids)`；`describe(state)` 不返回 current；`materialize(state, lazy_cache)` 只生成选中的 raw current。按 Route A 的 16/32/32/24/18 family 上限，得到至多 122 个 recipe，少于上限照实记录。

实现 `route_a/features.py`：严格使用 FEATURE_CONTRACT 的 204-dimensional state、32-dimensional node、32-dimensional outgoing。默认关闭 atom–seed projections。所有当前 χ 数据来自 state estimate，不读 ground truth。为每 feature 输出 source/dependency/exclusive timing，遇到 `anchor64`、`workspace128`、teacher-only path 立即失败。

实现 `route_a/audit.py`：插桩计数 all-bank materialization、full inverse action、full SVD、L/L* calls、anchor/workspace allocations。A0-replay 可读取历史大 machinery，但所有结果标 REPLAY_ONLY；A1 不可偷偷降格为 A0。

## 2. 共享接口接线

Integrator 提供 `BackendProtocol` 与 `FrozenState`，详见 SHARED_DATA_AND_TEACHER_SPEC。你调用的接口必须输出 cost receipt；离线 teacher 单独进程只读 labels，不让在线 extractor 访问 `teacher_only/`。

`propose(features,K)->list[ActionSpec]` 只返回 `(recipe_id, drop_index, prior_logit)`，不返回 predicted gain。`instantiate(action)->current` 对 retained core 投影、归一化；相对 norm floor 不通过则 invalid。不得补 random candidate。

最终 action 都交 `verified_exchange.evaluate(action)`：原 endpoint solve、原 1e-10 gates、原 tau_online、full all-illumination Jd。不自行写较松阈值。接受后由 integrator 更新 U、step、cached output，并使 state-dependent cache 失效；最多三次 accepted exchanges。

## 3. CPU preflight 和 labels

历史 A0 验证 original node IDs/outgoing IDs 与 gain 表一一对应。A1 必须取得新 recipe 的 action-level finite gain；只有 best incoming label 不足以训练 drop-conditioned action head。

在 integrator 分配的最多12个 development states 上：运行无 NN cheap fixed recipes、family-balanced OPM、同成本 analytic prior；以完整原 dictionary 以及 A1 枚举参考分别评估。提交 opportunity coverage、near-contact coverage、Tacq、Tinc、missing/invalid/zero-reference 分母和 all-machinery counters。

G1A 未过，不申请 GPU；G2 显示最强无 NN 已达 0.95 reference，也不训练。禁止以 ordinary classification accuracy 申请扩大实验。

## 4. 默认模型与训练

`route_a/model.py`：node affine (32+204)→64→64，GELU；mean/max pool 得128；action head concat node64、pool128、outgoing32，224→64→1；另以 pool128→1 作 no-op。加 LayerNorm、padding masks，记录精确参数数。主网无 GNN。

`route_a/loss.py`：正机会 g=G 当且仅当该 action 可行且 G>tau_online，否则 g=0；保留 raw G 另行报告。target positive mass∝g，all-zero state 为 no-op。实现 listwise cross entropy、exact iid expected-best K=4 surrogate 和 duplicate mass；duplicate 项为 canonical-equivalent action groups 的 squared total probability，单一 group 不惩罚自身 mass。不要把概率 MSE 当 gain loss。

AdamW lr=1e-3、weight_decay=1e-5、batch16 states、max40epochs、clip1、每5epoch validate actual deterministic top4 without replacement。epoch10若无正向 headroom 或 patience10无改善，停止。训练只用 cached features/labels，禁止调用 Maxwell。内置 profiler 断言 physics action count 在 epoch 中为零。

teacher likelihood/coverage surrogate 的分母使用固定 detached reference；reference 为零时不用 ratio。epsilon 仅作 machine-safe division：正 reference 分支 max(g_ref,tiny64)，不要用任意大 epsilon 隐藏 small-gain cases。训练保留 numerical-floor diagnostic；最终使用 FP64 gain acceptance。

## 5. 结构对照的启动资格

仅通过 G3 的 A-MLP 可以申请 J3 的 A-GNN。按 FEATURE_CONTRACT 的两轮64维/8邻居 message passing；物理 centroid-edge、几何-edge、degree-preserving shuffled-edge 与等参数 MLP 分别计费。GNN improvement<0.03 coverage且无10%成本收益，则最终保留MLP，停止 physics-native architecture claim。A-Set 不属于第一轮默认新增 job；需从相同 optional budget 中替换而非叠加。

共享 encoder 是定义与数据共享，不是拿另一条 route 的更长训练 checkpoint 免费 warm-start。原 train/validation/test 划分不动；test 在 integrator 冻结 checkpoint/config 后只运行一次。

## 6. 你要新增的 CLI（这些目前尚未实现）

```text
python -m route_a.audit --config config/pilot.json --split development
python -m route_a.preflight --manifest data/manifest.jsonl --output results/route_a/preflight
python -m route_a.train --config config/pilot.json --job J1 --seed 1801
python -m route_a.evaluate --checkpoint <frozen_path> --split locked_test
```

训练命令只能由 integrator 经 `tools/gpu_queue.py` 包装启动。不要自己 SSH 后台 nohup；不要与 B 的 GPU job 同时运行。CPU开发可并行，CPU线程总额遵循 integrator。

## 7. 必须写的 tests 与结果

测试 recipe 单独访问不触发全 bank；A0/A1 IDs 不可互读；feature loader 拒读teacher；complex phase不改变 endpoint/gain；receiver置换同时变换S/r；protected slots不可删除；zero norm返回invalid；exact sameaction不重复；endpoint失败不隐式加ridge；no-positive states不产生fakecoverage。

输出 `MODEL_CARD.md`、`GATE_DECISION.json`、`ACTION_LEDGER.jsonl`、`COST_LEDGER.jsonl`、object-level bootstrap/coverage表、参数量/epochs/失败日志。至少给图的原始数据，不只给一张漂亮曲线。全部结果放回同一研究仓库，不能另建孤立仓库。

直接杀死 A1 的结果：oracle coverage不足；lazy feature其实要求anchor/workspace；held-out对象收益消失；或与同预算解析先验相比无coverage/cost改进。失败时冻结完整结果，不增加网络规模或数据直到重新授权。
