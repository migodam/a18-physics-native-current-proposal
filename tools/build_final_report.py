"""Render final decision from immutable measured evidence; no physical calls."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from route_b.preflight import cov

ROOT=Path(__file__).resolve().parents[1]
def load(name):return json.loads((ROOT/name).read_text())
def write(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def md(name,text):(ROOT/name).write_text(text,encoding='utf-8')
def evidence(name):return {'path':name,'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}
def csvwrite(name,rows):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build():
    pre=load('results/preflight_review/FINAL_INTEGRATED_PREFLIGHT.json');frozen=load('results/frozen_evaluation/SUMMARY.json');transfer=load('results/route_b/FROZEN_CACHED_QUALITY_NATIVE.json');pair=load('results/preflight_review/G7_INTERPRETATION.json');generation=load('assets_v002/pilot/generation_manifest.json')
    main=load('results/route_b/training/main/training_summary.json');order=load('results/route_b/training/order0/training_summary.json');baseline=load('results/route_b/J0/G2_PILOT.json');vm=load('results/route_b/training/main/best_validation.json');vo=load('results/route_b/training/order0/best_validation.json');metrics=frozen['summaries']['8'];bs=frozen['paired_bootstrap']['8']
    jobs=[json.loads(l) for l in (ROOT/'results/gpu_jobs/jobs.jsonl').read_text().splitlines() if l.strip()];bucket={b:sum(r['occupied_seconds'] for r in jobs if r['bucket']==b) for b in ('physics','teacher','nn','nonlinear')};limits={'physics':60.,'teacher':120.,'nn':240.,'nonlinear':60.}
    budgets={b:{'charged_wall_minutes':s/60,'limit_minutes':limits[b],'status':'PASS' if s/60<=limits[b]+.02 else 'FAIL'} for b,s in bucket.items()}
    # CPU preparation is a dual attribution, not a second execution.
    unique=sum(r['occupied_seconds'] for r in jobs if r['status']!='CPU_PREPARATION_BUDGET_ATTRIBUTION')
    cost={'schema':'a18.final_cost_summary.v1','budgets':budgets,'queue_rows':len(jobs),'unique_charged_wall_minutes_including_local_bound':unique/60,'cpu_teacher_overlap_seconds':sum(r['occupied_seconds'] for r in jobs if r['status']=='CPU_PREPARATION_BUDGET_ATTRIBUTION'),'local_aux_seconds':120.,'local_aux_scope':'Conservative charged upper bound, not measured wall','cuda_occupied_seconds_measured_or_reserved':sum(r['gpu_occupied_seconds'] for r in jobs if r.get('compute')=='cuda' and r.get('gpu_occupied_seconds') is not None),'legacy_gpu_labels':'Early P0 generator was native CPU but old wrapper conservatively labeled 751.198s GPU occupied; immutable raw row retained. First P0 omitted compute classification. No GPU kernel-utilization claim.','memory_gaps':{'J0_peak_allocated_cuda_bytes':82355712,'J0_peak_reserved_cuda_bytes':'NOT_MEASURED','process_peak_RSS':'NOT_MEASURED'},'initial_state_cost_scope':'Ttotal includes measured backend/frozen-state construction wall. Incremental operator/RHS traces start after shared state creation; initial-state operator counter snapshot was not persisted.','materialization_scope':'Raw current_materializations counts incoming atoms/selected requests, not every normalized batched endpoint vector or heap allocation. B basis rank and D bank sizes reported separately.','method_costs':{}}
    raw=[]
    for p in sorted((ROOT/'results/frozen_evaluation').glob('*_k8.json')):raw.extend(json.loads(p.read_text())['rows'])
    counters=sorted({key for r in raw for key in r['cost']['counters']});crows=[]
    for method in metrics:
        cost['method_costs'][method]={}
        for mode in ('cold','warm_same_configuration'):
            rows=[r for r in raw if r['method']==method and r['cache_mode']==mode]
            cm={c:float(np.median([r['cost']['counters'].get(c,0) for r in rows])) for c in counters};cost['method_costs'][method][mode]={'rows':len(rows),'Tinc_median_over_all_repetitions':float(np.median([r['Tinc'] for r in rows])),'Tinc_P95_over_all_repetitions':float(np.percentile([r['Tinc'] for r in rows],95)),'Ttotal_median_over_all_repetitions':float(np.median([r['Ttotal'] for r in rows])),'operator_RHS_counter_medians':cm}
            for row in rows:crows.append({**{k:row[k] for k in ('object_id','state_id','method','repetition','cache_mode','Tacq','Tverify','Tinc','Ttotal','gain','selected_count')},**{c:row['cost']['counters'].get(c,0) for c in counters}})
    csvwrite('tables/locked_k8_raw_costs.csv',crows);write('COST_SUMMARY.json',cost)
    csvwrite('tables/locked_k8_coverage_cost.csv',[{'method':m,**v} for m,v in metrics.items()]);csvwrite('tables/locked_k8_state_metrics.csv',frozen['states'])
    abc=load('results/preflight_review/ABLATIONS.json')['coverage'];csvwrite('tables/development_ablations.csv',[{'variant':m,**v} for m,v in abc.items()])
    csvwrite('tables/rank_transfer_quality.csv',[{'rank':int(k),'method':m,**v} for k,q in transfer['ranks'].items() for m,v in q['metrics'].items()])
    comparisons=[]
    for rank,q in transfer['ranks'].items():
        for candidate in q['metrics']:
            differences=[]
            for oid in sorted({r['object_id'] for r in q['rows']}):
                group=[r for r in q['rows'] if r['object_id']==oid];differences.append(cov(group,candidate)-cov(group,'B-poly'))
            rng=np.random.default_rng(2026100418);samples=rng.choice(differences,(2000,len(differences)),replace=True).mean(1);ci=np.percentile(samples,[2.5,97.5]).tolist();comparisons.append({'rank':int(rank),'candidate':candidate,'baseline':'B-poly','point_difference':float(np.mean(differences)),'CI95':ci,'objects':8,'replicates':2000,'selection_role':'DESCRIPTIVE_ONLY_PRIMARY_PAIR_PRE_FROZEN','familywise_correction':'NONE_NO_MULTI_METHOD_SUPERIORITY_CLAIM'})
    write('results/frozen_evaluation/PAIRED_BOOTSTRAP_ALL.json',{'comparisons':comparisons,'aggregation':'Two states summed per parent before resampling; seed2026100418; 2000 resamples','primary_pair':bs})
    gates={**pre['gates'],'G3':{'status':'FAIL','reason':'MANDATORY_EPOCH20_LEARNING_STOP','main_macro':vm['macro'],'main_near':vm['near'],'order0_macro':vo['macro'],'order0_near':vo['near'],'baseline_macro':baseline['macro'],'baseline_near':baseline['near'],'coverage_branch':'FAIL','K2_cost_branch':'NOT_RUN_AFTER_MANDATORY_STOP','matched_validation_physical_Tinc':'NOT_RUN_AFTER_COVERAGE_STOP','epochs_each':20,'same_registered_epoch_cap':35},'G4':{'status':'FAIL','reason':'Prerequisite G3 fails; B-main also below strongest validation non-NN by more than0.02','B_minus_nonNN':vm['macro']-baseline['macro']},'G5':{'status':'NOT_RUN','reason':'No G3-passing MLP, no architecture extension'},'G6':{'status':'HOLD','primary_pair':bs,'frozen_primary_physical_rows':1280,'cached_transfer_ranks':[4,16],'full_transfer_deployment_timing':'NOT_RUN_AFTER_STOP'},'G7':{'status':'HOLD','interpretation':pair},'G8':{'status':'NOT_RUN','reason':'No deployment winner; no nonlinear transfer or final material quality claim'},'G9':{'status':'PASS','budgets':budgets,'J6':'TIME_CAP_ENFORCED','J7':'ONE_UNCHANGED_CONTINUATION_COMPLETED','all_failures_charged':True}}
    evidence_paths=['results/preflight_review/FINAL_INTEGRATED_PREFLIGHT.json','results/route_b/J0/G2_PILOT.json','results/route_b/J0/J2_SCHEDULE.json','results/route_b/FREEZE_RECEIPT.json','results/route_b/training/main/training_summary.json','results/route_b/training/order0/training_summary.json','results/frozen_evaluation/SUMMARY.json','results/route_b/FROZEN_CACHED_QUALITY_NATIVE.json','results/shared/CAP3_TRAIN_IMPLEMENTATION_AUDIT.json','results/preflight_review/G7_INTERPRETATION.json','results/gpu_jobs/jobs.jsonl']
    decision={'schema':'a18.integrated_gate_decision.v1','scientific_conclusion':'NO_GO_BOTH','publication_status':'LOCAL_EVIDENCE_COMPLETE_REMOTE_VERIFICATION_PENDING','scope':'Tested registered A bank and B deterministic learning pilot only, not global impossibility','gates':gates,'evidence':[evidence(p) for p in evidence_paths],'not_measured':['A0 historical ID/gain replay','successful T0/T1 per-action details','full deployment timing at k4/16','nonlinear winner transfer','extra NN seeds','J0 peak reserved VRAM and process RSS','initial shared-state operator snapshot'],'retuning_after_test':False,'deployment_winner':None,'route_C':'THEORY_ONLY_NO_TRAINING'};write('GATE_DECISION.json',decision)
    summary={'scientific_conclusion':'NO_GO_BOTH','publication_status':decision['publication_status'],'implementation':{'real_G0':'PASS','new_two_state_physics':'PASS','cap3_train_audit':'PASS','neural_parameters_each':249703,'A_neural_status':'UNTRAINED_34306_PARAMETERS'},'development':{k:{key:val for key,val in v.items() if key!='evidence'} for k,v in pre['gates'].items()},'validation':{'B-main':{'macro':vm['macro'],'near':vm['near']},'B-order0':{'macro':vo['macro'],'near':vo['near']},'strongest_nonNN':{'method':'B-poly36','macro':baseline['macro'],'near':baseline['near']}},'locked_k8':metrics,'primary_paired_bootstrap':bs,'transfer':{k:{'metrics':q['metrics'],'primary_CI95':q['paired_bootstrap_main_minus_poly_CI95'],'scope':'cached_exact_FP64_only'} for k,q in transfer['ranks'].items()},'budgets':budgets,'failure_count':sum(r['status'] in ['FAILED','TIME_CAP','INTERRUPTED'] for r in jobs),'dataset':{'master_seed':2026100418,'objects':48,'states':96,'split_objects':{'train':32,'validation':8,'locked_test':8},'initialization':'TRUTH_SEEDED_LOCAL_PERTURBATION_NOT_GLOBAL_RECOVERY'},'limitations':['Eight locked parents only','BEST_FOUND continuous reference, not global optimum','G7 overlap interpretation unresolved','No nonlinear material quality claim','Literature429 and incomplete full-text coverage']};write('RESULTS_SUMMARY.json',summary)
    j6=next(r for r in jobs if r['job']=='J6');j7=next(r for r in jobs if r['job']=='J7');write('results/raw_failures/J6_TIME_CAP.json',{'status':'TIME_CAP','receipt':j6,'continuation_receipt':j7,'same_freeze_sha256':load('results/frozen_evaluation/index.json')['freeze_sha256'],'restarts':1,'optimizer_restarted':False,'atomic_completed_cases_reused':True,'interrupted_partial_case':'Last uncommitted cold/warm pair may be lost and rerun; all occupied wall charged','final_completion':'16 states x8 methods x5 cold/warm=1280 committed rows'})
    claims=[('C1','PASS_WITH_SCOPE','Schur forward/adjoint and original discrete endpoint/finite gain agree in the tested audits','MATHEMATICAL_REVIEW.md; results/shared/G0*.json'),('C2','PASS_DEVELOPMENT_ONLY','B shared degree2 representation covers best-found development opportunities at0.993 macro','results/preflight_review/FINAL_INTEGRATED_PREFLIGHT.json'),('C3','FAIL','The registered A bank passes the common-reference representation threshold','results/route_a/GATE_DECISION.json'),('C4','FAIL','The registered B neural proposal passes epoch20 validation learning gate','results/route_b/training/'),('C5','HOLD','B has a reliable positive held-out gain advantage over its frozen B-poly comparator','results/frozen_evaluation/SUMMARY.json'),('C6','HOLD','Singleton proposals cover the important block mechanism','results/preflight_review/G7_INTERPRETATION.json'),('C7','NOT_RUN','The proposal improves final nonlinear material recovery or end-to-end deployment','GATE_DECISION.json G8'),('C8','NOT_CLAIMED','This is the first learned Krylov space, projective loss, or learned current proposal','research/literature/CODEX_NOVELTY_REVIEW.md'),('C9','NOT_CLAIMED','Finite discrete gain/tau is a continuum or complete safety certificate','MATHEMATICAL_REVIEW.md')]
    write('CLAIM_LEDGER.json',{'claims':[{'id':i,'status':s,'claim':c,'evidence':e} for i,s,c,e in claims],'novelty_scope':'Limited retrieved primary literature only, no exhaustive priority conclusion','scholarqa_status':'429_NO_USABLE_RETRIEVAL','full_text_gaps':['Wei Chen2019 author PDF timeout'],'literature_ledger':'research/literature/CODEX_NOVELTY_REVIEW.md'})
    write('results/EXPERIMENT_STATUS.json',{'status':'RESEARCH_COMPLETED_WITH_NEGATIVE_GATES','scientific_conclusion':'NO_GO_BOTH','publication_status':decision['publication_status'],'authoritative_decision':'GATE_DECISION.json','raw_timing_rows':1280,'all_NN_epochs_Maxwell_calls':0,'C':'THEORY_ONLY','nonlinear':'NOT_RUN_NO_WINNER'})
    # Standalone scientific figures; no generated decorative imagery.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.dpi':180});(ROOT/'figures').mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(10.5,5.1),sharey=True);names=['B-main','B-order0','B-poly','A-analytic','control:ridge_0.01_m1','control:KNN3_objects_m1','D-recommended','D-full'];yy=np.arange(len(names))
    axs[0].barh(yy-.16,[metrics[m]['macro_union'] for m in names],height=.3,label='macro');axs[0].barh(yy+.16,[metrics[m]['near_union'] for m in names],height=.3,label='near');axs[0].set_yticks(yy,[m.replace('control:','') for m in names]);axs[0].invert_yaxis();axs[0].set(xlabel='Coverage / common reference',xlim=(0,1),title='Frozen k=8: eight locked parents');axs[0].legend(fontsize=8)
    axs[1].barh(yy-.16,[metrics[m]['Tinc_median'] for m in names],height=.3,label='cold');axs[1].barh(yy+.16,[metrics[m]['Tinc_warm_median'] for m in names],height=.3,label='warm');axs[1].set(xlabel='Incremental wall (s), median of state medians',xlim=(0,7),title='Includes selected physical verification');axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'figures/coverage_cost.png');fig.savefig(ROOT/'figures/coverage_cost.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4));ks=list(abc);x=np.arange(len(ks));ax.bar(x,[abc[k]['macro_common'] for k in ks],label='macro');ax.scatter(x,[abc[k]['near_common'] for k in ks],color='black',s=25,label='near');ax.axhline(.9,color='red',ls='--',lw=1,label='G1B macro threshold');ax.set_xticks(x,ks,rotation=25);ax.set(ylabel='Best-found coverage / common reference',ylim=(0,1.08),title='Exposed development screening; bounded search, not global oracle');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/development_ablations.png');fig.savefig(ROOT/'figures/development_ablations.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.8))
    for tag in ['main','order0']:
        ep=[json.loads(l) for l in (ROOT/f'results/route_b/training/{tag}/epochs.jsonl').read_text().splitlines()];v=[r for r in ep if 'validation_macro' in r];axs[0].plot([r['epoch'] for r in v],[r['validation_macro'] for r in v],marker='o',label='B-'+tag)
    axs[0].axhline(baseline['macro'],color='black',ls='--',label='B-poly36');axs[0].set(xlabel='Epoch',ylabel='Validation coverage',title='Both stop at epoch20');axs[0].legend(fontsize=8)
    for m in ['B-main','B-order0','B-poly']:
        axs[1].plot([4,8,16],[transfer['ranks'][str(k)]['metrics'][m]['macro_union'] for k in [4,8,16]],marker='o',label=m)
    axs[1].set(xlabel='Current rank k',ylabel='Locked coverage (cached FP64)',title='Frozen transfer; deployment timing not run');axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/training_transfer.png');fig.savefig(ROOT/'figures/training_transfer.svg');plt.close(fig)
    table='\n'.join(f"|{m}|{v['macro_union']:.4f}|{v['macro_original']:.4f}|{v['near_union']:.4f}|{v['Tacq_median']:.3f}|{v['Tinc_median']:.3f}|{v['Tinc_warm_median']:.3f}|" for m,v in metrics.items())
    gate_table='\n'.join(f"|{g}|{v['status']}|" for g,v in gates.items())
    report=f'''# A18_C Route B 与 A/B 集成：中文研究报告

最终科学裁决为 **NO_GO_BOTH**。这针对本轮注册的 A recipe bank 和 B 轻量学习方案；没有证明所有 current proposal 不可能。共享真实物理后端通过 G0，B 的发展集表示能力通过 G1B，但表示空间中有好方向没有转化为足够好的廉价预测。A 的共同参考表示门槛失败；两套 B 模型均在 epoch20 停止。公开归档状态与科学裁决分别记录。

## 实现与数学核验

共享接口导出 BackendProtocol、FrozenState、SeedBundle、ActionSpec、CostReceipt、VerifiedExchange。在线路线读取当前材料、残差、forcing、U/LU/SU/D、旧 step、先验和几何，不读取 full J/H、真值、teacher、旧 anchor/closure。材料采用显式多照射 real packing；材料 adjoint、L/L*、S/S*、矩阵自由 Schur、原 endpoint、gain/tau、奇异 core fallback 和 cache 失效均有独立检查。

矩阵自由 $F_{{eff}}=PFP+PFU(I-U^*FU)^{{-1}}U^*FP$ 和精确 adjoint 保留已付费的 LU。六条 O/P/M 链使用 complex128、两遍 MGS、按 degree 联合正交，默认 rank≤18、hard cap24，退化 seed 保持 inactive。每次反馈 action 仅调用一次大 L 或 L*，其他低维运算另记。B 网络固定1552→128→128，32维 outgoing，每 drop 两个 padded complex25 modes 和 no-op，共249,703参数。最高合法 logit 不高于 no-op 时返回空列表，否则至多四个合法、projectively不重复的动作；只有选中 currents 进入 verifier。

原 A17 两真实状态 G0 最大 endpoint 误差约9.14e−13，finite gain/tau保持原实现；FFT CPU/CUDA及 batched dictionary 另有实际回执。它们与 dense unit tests分别报告。新的 separated/near 两状态物理审计通过；两个 TRAIN 状态 cap-3 审计实际接受了1、2、3次，确认 configuration 更新与 root/cache重建。该审计固定材料，只改变 current subspace，不是 nonlinear transfer。见 [数学审查](MATHEMATICAL_REVIEW.md)、[新状态审计](results/shared/NEW_PILOT_PHYSICS_AUDIT.json)、[cap-3审计](results/shared/CAP3_TRAIN_IMPLEMENTATION_AUDIT.json)。

## 数据、参考与公平性

旧 A18-B 保留24/6/18划分及历史暴露标记；12-state preflight来自六个已暴露对象，仅用于 development。新 master seed2026100418一次生成48对象、每对象两状态：32 train／8 validation／8 locked test。派生配置随 parent划分；teacher共享一次生成。locked test固定 separated2、near2、geometry2、high-contrast2，其中一个 high-contrast近接触。真实全矢量 Maxwell 使用原A8/A17数值源码和既有物理参数。

初始化是围绕真值的受控局部扰动，state1由预定原始 full-GN/line-search生成；这是局部冻结状态 pilot，不能外推成未知初始化的全局反演能力。共同 reference 是原字典、A bank、连续 B及 per-core搜索所得的审计 union；continuous teacher始终称 **BEST_FOUND**，不是全局 oracle。每对象coverage先以两状态 gain总和除reference总和，再对有正reference的对象等权平均。原字典分母另报；coverage超过1表示该方法超过旧字典，不表示超过全局最优。详细来源见 [生成 manifest](assets_v002/pilot/generation_manifest.json)。

## Preflight、训练与停止

B共同参考 macro={pre['gates']['G1B']['coverage_macro_union']:.6f}，near={pre['gates']['G1B']['coverage_near_union']:.6f}，shared-root损失={pre['gates']['G1B']['shared_root_loss_union']:.6f}；Tacq/D-full median={pre['Tacq_ratios']['B-poly']['median']:.4f}、P95={pre['Tacq_ratios']['B-poly']['P95']:.4f}，通过G1B。负的shared-root损失来自有限搜索结果，不是它理论上支配per-core。A共同参考 macro={pre['gates']['G1A']['coverage_macro_union']:.6f}、near={pre['gates']['G1A']['coverage_near_union']:.6f}，G1A失败；原字典口径 A={pre['coverage_original_dictionary']['A']:.6f}不能替代共同参考口径。

新 validation 最强无NN为B-poly36：macro={baseline['macro']:.6f}、near={baseline['near']:.6f}，未达到.95，允许B训练。ridge/KNN只拟合32个train parents，m1/m2相同validationgain时预先选择较便宜m1。B-main最佳macro={vm['macro']:.6f}、near={vm['near']:.6f}；lambda_order=0控制macro={vo['macro']:.6f}、near={vo['near']:.6f}。两个模型 seed1801、相同1350s预算与35epoch上限，每5epochs FP64验证，均在20epochs因无headroom停止；每个epoch Maxwell action count为0。

J0真实20minibatches丢弃前5：CPU median1.552s，CUDA4.464s，小复矩阵路径CPU更快，故训练用nativeCPU。35是根据J0在训练前冻结的同等上限，低于原80epoch硬上限。G3 coverage分支失败，K2部署分支因强制停止未运行；没有把未测分支写成完整否证。A-MLP/GNN、B-Set、追加seeds、C训练和winner nonlinear transfer均未启动。

![训练与迁移](figures/training_transfer.png)

## 冻结测试与完整 incremental 成本

冻结模型、权重、normalizer、方法列表、B-poly36及baseline pair后才打开locked test。k=8共有16状态×8方法×5次cold/warm，共1280行实际物理测量。每状态轮换方法顺序；cold从共同固定状态开始，warm只复用同configuration已付费root/inference/records和旧full Jx，重新物化所有选中currents、endpoint以及新full Jd。D-recommended按原口径top1，其他方法top4，数量差异明确保留。

|方法|Cov共同|Cov原字典|near共同|Tacq冷(s)|Tinc冷(s)|Tinc暖(s)|
|---|---:|---:|---:|---:|---:|---:|
{table}

时间为每状态五次中位数再取跨状态中位数；Tinc包括acquisition和所有selected验证，Ttotal另包括实际共同backend/state构造。原始五次行、全部incremental operator/RHS counters和action traces保存在 [frozen_evaluation](results/frozen_evaluation) 与 [成本表](tables/locked_k8_raw_costs.csv)。初始共享state的operator快照未持久化，不能宣称已具备全部end-to-end operator分解；其wall仍计入Ttotal。current_materializations计数为原子/选中请求，不是所有batched规范化向量的内存分配次数。D完整bank与B反馈basis本身的构造成本均包含在冷Tacq。

B-main−B-poly对象级coverage差={bs['point_difference']:.6f}，2000次paired parent bootstrap 95%区间[{bs['CI95'][0]:.6f}, {bs['CI95'][1]:.6f}]；下界≤−.02，G6 HOLD。只有8个parents；16状态/大量动作不能增加独立对象数。测试上较好的点估计不推翻validation停止决定，不选择测试表现更好的新baseline。

k=4/16使用冻结模型及rank-specific已付费reference作精确FP64小endpoint回放，Windows环境k8缓存gain与实体gain一致；这是quality transfer，不是完整部署计时。B-main macro分别为{transfer['ranks']['4']['metrics']['B-main']['macro_union']:.6f}／{transfer['ranks']['16']['metrics']['B-main']['macro_union']:.6f}，near分别为{transfer['ranks']['4']['metrics']['B-main']['near_union']:.6f}／{transfer['ranks']['16']['metrics']['B-main']['near_union']:.6f}。k4 order0及ridge无accepted positive gain是保留的失败结果。完整transfer Tinc、最终成像、held-out measurement质量均NOT_RUN。

![覆盖与成本](figures/coverage_cost.png)

## 消融、pair机制与文献

degree0/1/2 development共同macro分别.892270／.965940／.993372；rawPFP=.985400，O-only=.872960、P-only=.867877、M-only=.895436。筛选消融来自有限teacher搜索，不能把小差值解释为Schur必然优越或有统计显著性。原始记录及共同/原字典双分母见 [消融表](tables/development_ablations.csv)。

![发展集消融](figures/development_ablations.png)

12-state×6 pair的真实rank2 joint endpoint audit：joint/reference macro=.925246，正额外gain=max(joint−singleton reference,0) macro=0；optimized joint-only state比例0。原G7的“pair机会构成reference gain≥10%”没有规定重叠分配，两个口径会给出不同判断。因此正式机制覆盖结论为HOLD，保留原输出并以[解释回执](results/preflight_review/G7_INTERPRETATION.json)限定。任意QR列投影产生的16个joint-only pairs依赖frame gauge，不能作为gate。

按scholarqa-research执行检索，429没有得到可用新结果，保留失败和全文缺口。取得的primary evidence表明[Neural Krylov](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e88870ec82f2469b0ddf32c817920c68-Abstract-Conference.html)已有学习subspace／projector训练思路，[SOM-Net](https://arxiv.org/abs/2209.03567)已有学习current与解析材料更新；Wei/Chen2019 [DOI](https://doi.org/10.1109/TAP.2019.2922779)仅部分元数据核验，author PDF访问失败。不能宣称首次learned Krylov、首次projective loss或首次learned current；本轮具体组合也未获得部署支持。原创性判断限于取得证据，详见[文献与主张核验](research/literature/CODEX_NOVELTY_REVIEW.md)和[claim ledger](CLAIM_LEDGER.json)。

## 裁决、资源与失败记录

|Gate|最终状态|
|---|---|
{gate_table}

共享teacher累计{budgets['teacher']['charged_wall_minutes']:.2f}/120min；physics bucket {budgets['physics']['charged_wall_minutes']:.2f}/60min；NN bucket（含J0、失败计时和freeze/timing）{budgets['nn']['charged_wall_minutes']:.2f}/240min；nonlinear0/60min。CPU准备751.198s同时归入teacher与physics预算，是重叠归属，非第二次执行。旧wrapper误把CPU准备标为GPU占用的原记录保留，派生成本表明确分类。120s local auxiliary为保守计费上界，不冒充实测时间。J0只记录allocated VRAM82,355,712bytes；reserved peak与process RSS未测。

J6在15min硬限被终止，J7用同freeze唯一续跑完成剩余病例，合计{(j6['occupied_seconds']+j7['occupied_seconds'])/60:.2f}min，失败及重复工作均计费，没有重启optimizer或更改规则。T1终端numpy.bool序列化失败发生在有效原子结果写入之后；G7终端类型错误、早期background SSH失败与J6超时均保留[原始失败记录](results/raw_failures)。旧T0/T1成功action details和A0历史ID/gain重放缺失明确记录，新的冻结test有完整selected traces，不逆向补写历史。

最终 **NO_GO_BOTH**；G6统计与G7机制claim各自HOLD。保留便宜解析控制作为比较结果，但不将其重新命名为通过部署门槛的赢家。后续若研究其他feature、初始化、预算或block proposal，应新建独立预注册和未暴露test；本轮无权用locked test继续调参。

代码、报告、表图和manifests在Git；合成物理数据、checkpoint、拟合权重、缓存、依赖及原日志放Release，并核对SHA256。原许可状态保留，不擅自添加许可证或重分发论文全文。入口：[START_HERE](START_HERE.md)，重现：[REPRODUCE](REPRODUCE.md)，机器裁决：[GATE_DECISION](GATE_DECISION.json)，来源：[RIGHTS_AND_PROVENANCE](RIGHTS_AND_PROVENANCE.md)。公开归档不等于期刊录用或同行评审。
'''
    md('RESEARCH_REPORT_ZH.md',report)
    md('START_HERE.md','''# A18_C: physics-native current proposal — public research archive

**科学裁决：NO_GO_BOTH。** 共享真实物理后端及B发展集表示门通过，A表示门失败，B两个模型均在epoch20未超过validation无NN控制，停止扩展。冻结test的正差值区间跨越明显劣化区域，G6 HOLD；pair口径未消除，G7 HOLD。没有最终nonlinear成像质量claim。

网页GPT／研究者阅读顺序：

1. [中文研究报告](RESEARCH_REPORT_ZH.md)
2. [简明机器结果](RESULTS_SUMMARY.json) 与 [逐项gate裁决](GATE_DECISION.json)
3. [覆盖／成本表](tables/locked_k8_coverage_cost.csv)、[真实五次计时与counters](tables/locked_k8_raw_costs.csv)
4. [数学核验](MATHEMATICAL_REVIEW.md)、[claim ledger](CLAIM_LEDGER.json)、[文献审查](research/literature/CODEX_NOVELTY_REVIEW.md)
5. [复核与重现](REPRODUCE.md)、[权属与来源](RIGHTS_AND_PROVENANCE.md)、[发布说明](PUBLISH.md)

原研究规格与预定门槛继续保留；其中旧的SPEC_READY/NOT_RUN描述是制定时历史状态，当前实测以根目录报告和GATE_DECISION.json为准。old A18-B24/6/18为历史暴露，新48对象32/8/8独立split。数据初始化为真值附近局部扰动，teacher为BEST_FOUND，没有全局最优／全局反演能力保证。

Git提供代码、报告、表图和manifests；较大合成数据、checkpoint、缓存及数值依赖在[Release](https://github.com/migodam/a18-physics-native-current-proposal/releases/tag/pilot-2026-10-04)。先核对[公开manifest](PUBLIC_MANIFEST.json)中的SHA256，再解压。科学判定与远端发布核验分别记录，公开归档不等于peer-reviewed acceptance。
''')
    md('README.md','''# A18 physics-native current proposal

实现并检验 Schur-feedback current proposals，集成共享 Maxwell verifier、Route A recipe控制和Route B连续系数MLP。

**最终科学裁决：NO_GO_BOTH；G6/G7 claim 为HOLD。** B发展集表示空间可覆盖好动作，但本轮两个学习模型未通过validation停止门。没有部署赢家或nonlinear成像结论。

从 [START_HERE.md](START_HERE.md) 阅读。[中文研究报告](RESEARCH_REPORT_ZH.md) 给出真实G0、preflight、96-state共同pilot、训练早停、1280行冻结实体计时、rank迁移、失败和成本。 [GATE_DECISION.json](GATE_DECISION.json) 与 [RESULTS_SUMMARY.json](RESULTS_SUMMARY.json) 是机器入口。

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
python -m tools.validate_delivery
```

大数据、checkpoint和依赖见 [Release](https://github.com/migodam/a18-physics-native-current-proposal/releases/tag/pilot-2026-10-04)。真实实验还需[重现说明](REPRODUCE.md)中的原数值依赖；dense tests不代替Maxwell证据。原理论/路线规格保持为研究历史，C为THEORY_ONLY。许可与来源见[RIGHTS_AND_PROVENANCE.md](RIGHTS_AND_PROVENANCE.md)。
''')
    for name in ['GO_NO_GO_GATES.md','GPU_JOB_ORDER.md']:
        p=ROOT/name;txt=p.read_text()
        if '<!-- current-result -->' not in txt:p.write_text('<!-- current-result -->\n> 本文件保留实验前规格与预算；当前实测状态见 [GATE_DECISION.json](GATE_DECISION.json) 和 [中文报告](RESEARCH_REPORT_ZH.md)。\n\n'+txt)
    print(json.dumps({'status':'FINAL_ARTIFACTS_BUILT','conclusion':'NO_GO_BOTH','budgets':budgets,'bootstrap':bs}))

if __name__=='__main__':build()
