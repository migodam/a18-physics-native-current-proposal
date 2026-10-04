# Publication

本次用户明确授权创建统一公开仓库migodam/a18-physics-native-current-proposal，并发布成功、失败及预算不足结果。GitHub CLI身份和目标可用性已实测；实际创建、push和匿名读取成功以results/publication_receipt.json及Release中的FINAL_PUBLICATION_QA.json为准。仓库与五个Release归档现已上线；初次匿名核验固定commit为c8386612208cf1857cb4f7cdaedb3f702627afd6，6个Git文件和5个ZIP共182,908,753bytes均HTTP200且SHA256一致。最终main核验另保存于Release的FINAL_PUBLICATION_QA.json。

目标入口：[仓库](https://github.com/migodam/a18-physics-native-current-proposal)、[证据包Release](https://github.com/migodam/a18-physics-native-current-proposal/releases/tag/pilot-2026-10-04)。网页版GPT从固定commit的原始START_HERE.md进入，再读取中文报告、RESULTS_SUMMARY.json、GATE_DECISION.json与代表性结果；具体固定链接由publication receipt给出。

Git保存代码、报告、表图、manifests和原始JSON结果。Release提供pilot-data.zip、frozen-caches-checkpoints.zip、historical-development-states.zip、a17-numerical-dependency.zip和raw-execution-logs.zip。PUBLIC_MANIFEST.json给出每个归档的SHA256、字节数与成员来源。Release也提供冻结数值输入的逐文件manifest；检查hash后解压到项目根，保留各相对目录。原G0输入没有捆绑，不能声称独立重跑其全部历史验证。

发布前检查可达Git历史与归档成员，保留原许可状态，不上传私有profile、凭据、SSH argv或第三方全文。发布后读取远端main commit，并以无认证HTTP核对固定commit的START_HERE、报告与机器结果的HTTP状态、SHA256；较大Release资产核对API size/digest和匿名HTTP下载可达性。科学结论始终NO_GO_BOTH，发布成功不改变gate状态，也不是期刊录用。
