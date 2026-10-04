# A18 physics-native current proposal

实现并检验 Schur-feedback current proposals，集成共享 Maxwell verifier、Route A recipe控制和Route B连续系数MLP。

**最终科学裁决：NO_GO_BOTH；G6/G7 claim 为HOLD。** B发展集表示空间可覆盖好动作，但本轮两个学习模型未通过validation停止门。没有部署赢家或nonlinear成像结论。

从 [START_HERE.md](START_HERE.md) 阅读。[中文研究报告](RESEARCH_REPORT_ZH.md) 给出真实G0、preflight、96-state共同pilot、训练早停、1280行冻结实体计时、rank迁移、失败和成本。 [GATE_DECISION.json](GATE_DECISION.json) 与 [RESULTS_SUMMARY.json](RESULTS_SUMMARY.json) 是机器入口。

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
python -m tools.validate_delivery
```

大数据、checkpoint和依赖见 [Release](https://github.com/migodam/a18-physics-native-current-proposal/releases/tag/pilot-2026-10-04)。真实实验还需[重现说明](REPRODUCE.md)中的原数值依赖；dense tests不代替Maxwell证据。原理论/路线规格保持为研究历史，C为THEORY_ONLY。许可与来源见[RIGHTS_AND_PROVENANCE.md](RIGHTS_AND_PROVENANCE.md)。
