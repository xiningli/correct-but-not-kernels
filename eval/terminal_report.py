"""K1–K4: evidence-linked completion/failure report, written before archive and shutdown."""
import argparse
import json
from pathlib import Path
from tools.gatelib import ROOT, timestamp, write_json
from tools.replication import sha256

def generate(study, controller, root=ROOT):
    study,controller,root=map(Path,(study,controller,root))
    outcome=json.loads((controller/'outcome.json').read_text())
    protocol=sha256(study/'frozen.json')
    gates={}
    for gate in ['C0','C1','C2','C3','R0','R1','R2','R3']:
        records=[]
        for path in sorted((root/'runs/gates').glob(gate.lower()+'-*.json')):
            record=json.loads(path.read_text())
            if gate.startswith('R') and record.get('protocol_sha256')!=protocol:continue
            records.append((path,record))
        if records:
            path,record=records[-1]
            gates[gate]={'status':record['status'],'record':str(path.relative_to(root)),
                'checks':record['checks'],'artifacts':record.get('artifacts')}
        else:gates[gate]={'status':'not run'}
    reports={}
    for stage in ['reevaluation','confirmation']:
        path=study/stage/'REPORT.md'
        if path.exists():reports[stage]={'path':str(path.relative_to(root)),'sha256':sha256(path)}
    finished=outcome['status']=='complete' and gates['R3']['status']=='PASS' and 'confirmation' in reports
    record={'generated_at':timestamp(),'protocol_sha256':protocol,'outcome':outcome,
        'formal_study_complete':finished,'gates':gates,'reports':reports}
    lines=['# 阶段运行报告','',
        '状态：'+('正式实验完成，结果见下方确认报告。' if finished else '正式实验尚未完成；本报告保留当前进度与阻塞证据。'),
        '',f"控制器状态：`{outcome['status']}`；执行阶段：`{outcome.get('stage','unknown')}`。",
        '', '| 检查 | 状态 | 证据 |','|---|---|---|']
    for gate,value in gates.items():
        lines.append(f"| {gate} | {value['status']} | `{value.get('record','—')}` |")
    if outcome.get('error'):lines+=['','阻塞原因：`'+outcome['error']+'`。']
    for gate,value in gates.items():
        if value['status']=='FAIL':
            for check in value.get('checks',[]):
                if check.get('error'):lines+=['',gate+'：`'+check['error']+'`。']
            if value.get('artifacts'):
                for log in sorted((root/value['artifacts']).glob('*.log')):
                    tail=log.read_text(errors='replace')[-2000:]
                    if 'Traceback' in tail:
                        lines+=['',f'错误日志：`{log.relative_to(root)}`。','', '```text',tail,'```']
    for stage,value in reports.items():lines+=['',f"{stage} 结果：`{value['path']}`；SHA-256 `{value['sha256']}`。"]
    if not finished:
        lines+=['','这次运行不能证明新的收益，也不能替代独立复评、多训练种子或 SFT 对照结果。']
    lines+=['','统计解释：任务集沿用已观察过的原划分；任务聚类区间条件于已保存的策略，'
        '不能替代训练种子总体不确定性。检测标记和源码分类仍需人工核查。',
        '',f'协议 SHA-256：`{protocol}`。',
        '', '报告生成时间记录在同名 JSON 中；关机是否成功须另查 shutdown-return.json 或平台状态。','']
    write_json(controller/'TERMINAL_REPORT.json',record)
    path=controller/'TERMINAL_REPORT.md'
    with path.open('x') as stream:stream.write('\n'.join(lines))
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--controller',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    print(generate(args.study,args.controller,args.root))

if __name__=='__main__':main()
