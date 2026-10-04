# 复核与重现

Git 保存代码、Markdown、表格、图和 manifests。数据、checkpoint、拟合控制权重与数值缓存通过 GitHub Release 提供。先核对 Release 的 SHA256 清单，再解压到相同相对目录。每个缓存记录也保存自己的文件 hash 与 dataset fingerprint。

本轮数值环境是 Windows Python3.10.9、NumPy2.2.6、SciPy1.15.3、PyTorch2.6.0+cu124，RTX4060 Laptop 8GB。FFT Maxwell、共享 teacher 和实际 online verification 使用 native CPU；是否使用 CUDA 训练以 J0 实测为准。Mac 上的单元测试使用独立本地环境。不要把两个机器的运行时间直接混合。

```sh
python -m unittest discover -s tests -q
python -m tools.validate_delivery
python -m route_b.audit --help
python -m route_b.preflight --help
python -m route_b.train --help
python -m route_b.evaluate --help
```

代数单元测试可独立运行，真实实验需要原 A17 数值依赖。原 A17 项目保持独立：[公开项目](https://github.com/migodam/a17-maxwell-certified-exchange)，原发布背景 pin `56097653e6a814399581c3759fc3e3e8f0c2284c`。实际运行的 source manifest hash 及原 tolerance hash 位于 `results/shared/G0.json`。物理生成器的3个原文件以原字节保留在 `vendor_physics/`，逐文件来源和 hash 位于其 `SOURCE_MANIFEST.json`。

共享后端不需要远程服务器。`tools/remote_transport.py` 是本次复用已授权私有通道的可选执行适配器，仓库不包含该通道的凭据、profile 或 SSH argv。公开重现应在自己的计算环境直接运行数值命令，而不是尝试恢复本次私有连接。

实际数据生成和 teacher 命令：

```sh
python -m shared_backend.pilot_data --plan manifests/new_cohort_plan.json --physical-config config/physical_generation.json --original-root vendor_physics --gate-receipt results/preflight_review/FINAL_INTEGRATED_PREFLIGHT.json --output data/pilot --minutes-cap 55
python -m shared_backend.pilot_cache --cohort-root data/pilot --output cache/pilot/train_validation --a17-root PATH_TO_PINNED_A17 --minutes-cap REMAINING_CPU_MINUTES
```

禁止静默覆盖已有 dataset/cache，禁止重分旧24/6/18 对象。locked test teacher 和 evaluation 需要 FROZEN receipt；重新训练应使用自己的新审计目录与预算，不把本次 sealed test 用于超参数选择。NN job 需要共享队列 owner 与真实 gate receipt。

实验所得最终结论与 Release 资产范围由 `GATE_DECISION.json`、`RESEARCH_REPORT_ZH.md` 和 `PUBLIC_MANIFEST.json` 说明；重现命令存在不代表某 gate 已通过。

## 本轮公开资产解压与离线回放

从Release下载五个zip并先核对PUBLIC_MANIFEST.json。解压到仓库根：pilot-data会创建assets_v002/pilot，cache归档创建cache/pilot、checkpoints/route_b和拟合权重，数值依赖创建A17_CURRENT_EXCHANGE，历史development归档创建assets_v001/development。原私有机器的绝对D:/路径只属于实际运行receipt，重现时必须改为本机路径，不能声称原freeze receipt仍与改写的rules相同。

原始冻结receipt与rules保持不改。可用tools/frozen_cached_transfer.py直接读取artifact_path的便携fallback，核对所有checkpoint/rule SHA，然后对已发布paid cache作quality replay：

```sh
python -m tools.frozen_cached_transfer --freeze-receipt results/route_b/FREEZE_RECEIPT.json --cache-root cache/pilot/locked_test --ranks 8 4 16 --output replay_cached_quality.json
python -m tools.summarize_evaluation --root results/frozen_evaluation --reference-root cache/pilot/locked_test --baseline B-poly --candidate B-main --output replay_summary.json
```

原windows环境PyTorch2.6.0+cu124为本次native结果；Mac PyTorch2.14回放存在约1e-7量级macro差异，保留独立LOCAL回执，不混作同环境计时。复现实体现时先复制rules至自己的审计目录，把a17_root、checkpoint、control_weights改成本机路径并生成新的freeze receipt，后运行route_b.evaluate；这是新独立复核，不覆盖本次sealed结果。物理依赖runtime根路径已便携化，原numerical模块和42-file source manifest保持hash不变。原G0两输入未包含，需上游原输入另行重放。

已公开locked test不能继续作为本轮优化选择数据。成本和bootstrap可直接从raw rows离线重建，无需重跑Maxwell。tools/build_final_report生成科学汇总而非远端publication receipt；重建后publication status需重新核验。
