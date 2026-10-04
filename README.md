# A18 — Physics-Native Current Proposal

**Amortized task-conditioned current-space proposals from Maxwell feedback algebra.**

研究规格日期：2026-10-04。本包 Route B 指新的 continuous proposal 路线，不是已停止的 A18-B full-feedback closure；路线字母不继承旧实验结论。默认语言：中文；数学和接口采用英文术语。先读 [START_HERE.md](START_HERE.md)。本仓库是一份统一的研究/实施交接包，不是已训练模型或完整 Maxwell backend。

## 交付内容

|文件|用途|
|---|---|
|[THEORY_OVERVIEW.md](THEORY_OVERVIEW.md)|研究裁决、Schur有效反馈、OPM、合法连续替换与prior定义|
|[ROUTE_A_DICTIONARY_PROPOSAL.md](ROUTE_A_DICTIONARY_PROPOSAL.md)|lazy dictionary action prior；MLP主控制与有条件GNN|
|[ROUTE_B_ORTHOGONAL_FEEDBACK_PROPOSAL.md](ROUTE_B_ORTHOGONAL_FEEDBACK_PROPOSAL.md)|联合feedback flag、连续系数、cached exact endpoint teacher|
|[ROUTE_C_PROBABILISTIC_PRIOR.md](ROUTE_C_PROBABILISTIC_PRIOR.md)|projective分布；理论保留、训练关闭|
|[ORTHOGONAL_POLYNOMIAL_THEORY.md](ORTHOGONAL_POLYNOMIAL_THEORY.md)|Arnoldi/各多项式裁决、层级正则、连续性界|
|[NONNORMAL_MAXWELL_ANALYSIS.md](NONNORMAL_MAXWELL_ANALYSIS.md)|非正规风险、adjoint、numerical range与反例|
|[NOVELTY_AUDIT.md](NOVELTY_AUDIT.md)|逐家族比较、generic ML与Maxwell贡献分离、访问缺口|
|[SOURCES.md](SOURCES.md)|32项文献/理论来源与项目原始证据入口|
|[SHARED_DATA_AND_TEACHER_SPEC.md](SHARED_DATA_AND_TEACHER_SPEC.md)|统一backend、数据schema、对象split、teacher隔离|
|[FEATURE_CONTRACT.md](FEATURE_CONTRACT.md)|逐块固定1552/204维online inputs，避免隐藏昂贵特征|
|[BASELINE_SPEC.md](BASELINE_SPEC.md)|同预算polynomial/GMRES、analytic、ridge/KNN具体算式|
|[LIGHTWEIGHT_EXPERIMENT_PLAN.md](LIGHTWEIGHT_EXPERIMENT_PLAN.md)|baseline、主指标、对象泛化、费用和非线性测试|
|[GPU_JOB_ORDER.md](GPU_JOB_ORDER.md)|单卡串行job、epochs/VRAM/时间预算、早停|
|[GO_NO_GO_GATES.md](GO_NO_GO_GATES.md)|硬停止和最终Primary裁决|
|[CODEX_ROUTE_A.md](CODEX_ROUTE_A.md)|给A agent的直接实施指令|
|[CODEX_ROUTE_B.md](CODEX_ROUTE_B.md)|给B agent的直接实施指令|
|[INTEGRATOR.md](INTEGRATOR.md)|共享代码/teacher所有权、queue、发布与合并顺序|
|[PUBLISH.md](PUBLISH.md)|远端发布状态与用户环境中的受控发布命令|

C当前不具备启动训练资格，因此没有 `CODEX_ROUTE_C.md`。

## 已有可运行内容

```sh
python -m pip install -r requirements.txt
python tools/run_algebra_tests.py
python tools/validate_delivery.py
```

`src/algebra.py` 和 `tests/test_algebra.py` 是小规模 dense NumPy reference。17项测试覆盖 exact Schur/algebra identities、adjoint、phase、metric whitening、多illumination realification、连续replacement与非加性反例。实际回执见 [results/algebra_test_report.json](results/algebra_test_report.json)。它们没有采样真实Maxwell对象，也没有训练NN。

`tools/gpu_queue.py` 提供 cooperative exclusive lock、子进程time cap与失败计费。它不是已验证的4060集群管理系统，也不能阻止不遵守同一锁的外部程序。`tools/publish_github.py` 仅在用户已认证的本地GitHub CLI环境下执行，不在本会话运行。

## 待实施内容

两条 route 的 `route_a/`、`route_b/`、真实 A17 adapter、教师数据生成和训练命令，都由 Codex 按实施文档新增。文档中的 CLI 明确标为待实现，不能把未存在的训练入口当作本包已经运行的代码。当前真实物理/NN结果见 [results/EXPERIMENT_STATUS.json](results/EXPERIMENT_STATUS.json)：全部 NOT_RUN。

## 研究边界

A17仍是独立机制与conditional exchange论文。A18-B full learned feedback solver的已测负结果不被重包装成正结果。本研究不是material generative prior、不以分类准确率作为主指标、没有完整确定性safety certificate。主假设能否成立由12-state preflight和held-out objects决定，不能由这里的正面理论措辞替代。

上游仓库commit与用户附件hash见 [sources/project_provenance.json](sources/project_provenance.json)。未复制私人输入文本、第三方论文全文、vendor modules、训练数据或访问令牌；权属说明见 [RIGHTS_AND_PROVENANCE.md](RIGHTS_AND_PROVENANCE.md)。
