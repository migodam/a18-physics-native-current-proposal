# Integrator：一个仓库、一个teacher、一个GPU队列

## 接线责任

你拥有 shared_backend、schema、manifest、teacher_only cache、A17 verifier、global cost ledger；A/B agents只拥有各自route目录。上游只读，按 sources/project_provenance.json 固定版本；下载/访问失败必须记录，不用相邻实验替换。

先运行本包17个dense tests。然后在2个真实frozen states接通A17：material/current metric、DDA极化率导数、实材料packing、同一r/Λ/ell、sharedcomplexrank、singular-core fallback与原tau全部对齐。两条路线共用的 `verified_exchange` 不允许各自复制并改容差。

## 合并与数据顺序

先合并feature/recipe/flag接口与cost插桩，再生成12-state preflight cache。只有G1/G2通过的route才能进入扩展dataset与training。teacher生成一次，两条route共享；不得把原incoming labels当新lazyrecipe的action labels，不得把16-probe material projection当full utility。

所有state/噪声/视角/query跟随parentobject split。读取fullH/J/真值的teacher进程与online extractor隔离；features只由白名单schema载入。阶段结束写schema/config/source hashes。

## 启动单卡job

只有integrator启动queue。示例（route训练CLI由agent实现后才可运行）：

```sh
python tools/gpu_queue.py --config config/pilot.json --job J1 --receipt-dir results/gpu_jobs --lock-dir .gpu.lock -- python -m route_a.train --config config/pilot.json --job J1 --seed 1801
```

J1完成或早停后才能J2。P0、J0–J7、N0所有GPU调用均用同一物理机器上的同一个lock路径；跨worktree时请使用绝对共享路径。脚本拒绝已有锁，不自动判旧/删锁；机器崩溃后由integrator核实进程状态，记录incident后处理。预算按实际occupied wall收费，timeout/failed attempts也收费。

job wrappers不验证科研gate本身；integrator必须在ledger写明G0/G1/G2授权状态。未授权C没有job ID。CPU teacher上限120min要由teacher generator单独强制；它不在GPU wall里冒充免费。

## 冻结与评估

在validation选模型/degree/drop规则后，保存不可变checkpoint/config manifest。test一次，bootstrap单位object。冷/warmcache、selectedmaterialization、重新建Q、fullJd、fallback全部计费。K记录同时拆成lowdimrecords/physicalcurrents/atomicendpoints/RHS。

全流程主结论只能是PRIMARY_B、PRIMARY_A、ANALYTIC_ONLY、NO_GO_BOTH、HOLD之一；附每gate证据。只有赢家进入独立nonlineartransfer。编写最终report时保留所有失败与不完整对象，不制造“多数成功所以剩余默认成功”。

## Git提交与发布

A/B通过pull/merge方式回到同一个本地仓库。不要把三个路线拆成三个失去共享context的repo。`.gitignore` 排除默认原始cache、数据、checkpoint；公开数据前另做权属/隐私/大文件release审计。

本交付没有在远端创建仓库。先确认要用的owner/name与visibility，再按PUBLISH执行；不得把用户已有A17 repo的origin覆盖为新项目，也不得force-push旧历史。
