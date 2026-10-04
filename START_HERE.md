# A18_C: physics-native current proposal — public research archive

**科学裁决：NO_GO_BOTH。** 共享真实物理后端及B发展集表示门通过，A表示门失败，B两个模型均在epoch20未超过validation无NN控制，停止扩展。冻结test的正差值区间跨越明显劣化区域，G6 HOLD；pair口径未消除，G7 HOLD。没有最终nonlinear成像质量claim。

网页GPT／研究者阅读顺序：

1. [中文研究报告](RESEARCH_REPORT_ZH.md)
2. [简明机器结果](RESULTS_SUMMARY.json) 与 [逐项gate裁决](GATE_DECISION.json)
3. [覆盖／成本表](tables/locked_k8_coverage_cost.csv)、[真实五次计时与counters](tables/locked_k8_raw_costs.csv)
4. [数学核验](MATHEMATICAL_REVIEW.md)、[claim ledger](CLAIM_LEDGER.json)、[文献审查](research/literature/CODEX_NOVELTY_REVIEW.md)
5. [复核与重现](REPRODUCE.md)、[权属与来源](RIGHTS_AND_PROVENANCE.md)、[发布说明](PUBLISH.md)

原研究规格与预定门槛继续保留；其中旧的SPEC_READY/NOT_RUN描述是制定时历史状态，当前实测以根目录报告和GATE_DECISION.json为准。old A18-B24/6/18为历史暴露，新48对象32/8/8独立split。数据初始化为真值附近局部扰动，teacher为BEST_FOUND，没有全局最优／全局反演能力保证。

Git提供代码、报告、表图和manifests；较大合成数据、checkpoint、缓存及数值依赖在[Release](https://github.com/migodam/a18-physics-native-current-proposal/releases/tag/pilot-2026-10-04)。先核对[公开manifest](PUBLIC_MANIFEST.json)中的SHA256，再解压。科学判定与远端发布核验分别记录，公开归档不等于peer-reviewed acceptance。
