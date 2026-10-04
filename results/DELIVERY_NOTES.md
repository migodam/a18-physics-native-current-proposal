# Final delivery notes

科学结论NO_GO_BOTH；G6和G7的相应主张HOLD。根目录GATE_DECISION.json、RESULTS_SUMMARY.json及RESEARCH_REPORT_ZH.md为当前裁决；历史SPEC_READY文件、A路线独立preflight card和未经纠正的旧aggregate保留provenance，不是本轮最后状态。

本轮实际实现A/B及共享真实物理后端，生成96新状态，在同等预算下训练两个B模型20epochs，完成16 locked states、8方法、每方法5次cold/warm实体测量，以及rank4/16冻结FP64quality transfer。训练epochs Maxwell action count均0。单元测试与真实G0分别记录；没有winner nonlinear transfer、额外seed、架构扩展或C训练。

较大数据和checkpoint通过Release交付；Git只存代码、报告、表图、JSON结果和manifests。PUBLIC_MANIFEST.json记录hash和范围，匿名远端核验另见publication receipt。85 tests通过只是实现检查，不是科学gate或peer review acceptance。
