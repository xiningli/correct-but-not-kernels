"""K1–K4: generate gate/numerical status from immutable records under runs/."""
import argparse
import json
from tools.gatelib import ROOT

def render():
    lines = ['<!-- BEGIN GENERATED GATES -->', '| Gate | Status | Record |', '|---|---|---|']
    latest = {}
    for i in range(7):
        paths = sorted((ROOT/'runs/gates').glob(f'c{i}-*.json'))
        if paths:
            data = json.loads(paths[-1].read_text())
            latest[i] = data
            lines.append(f"| C{i} | {data['status']} | `{paths[-1].relative_to(ROOT)}` |")
        else:
            lines.append(f'| C{i} | not run | — |')
    if (ROOT/'experiments/replication-v1.yaml').exists():
        for gate in ['R0','R1','R2','R3']:
            paths=sorted((ROOT/'runs/gates').glob(f'{gate.lower()}-*.json'))
            if paths:
                record=json.loads(paths[-1].read_text())
                lines.append(f"| {gate} | {record['status']} | `{paths[-1].relative_to(ROOT)}` |")
            else:lines.append(f'| {gate} | not run | — |')
    lines += ['']
    if 2 in latest:
        checks = {c['name']:c for c in latest[2]['checks']}
        if 'reference_noise' in checks:
            c = checks['reference_noise']
            lines += [f"C2: CV across 30 median estimates = {100*c['pair_median_cv']:.4f}%; "
                      f"maximum within-pair raw CV = {100*c['maximum_within_pair_cv']:.2f}%. "
                      'Raw samples were retained, including outliers.']
    if 3 in latest:
        controls = latest[3]['checks']
        cheats = [c for c in controls if c['name'] in ['precomputed','default_distribution','input_alias','async_timer']]
        lines += [f"C3: {sum(c['status']=='PASS' for c in cheats)}/{len(cheats)} deliberate cheats "
                  'pass naive verification and are rejected by the hardened evaluator.']
    if 4 in latest:
        for check in latest[4]['checks']:
            for name, summary in check.get('summaries',{}).items():
                lines += [f"C4 {name}: mean P_signal = {summary['mean_p_signal']:.4f}; "
                          f"tasks with nonzero pass@8 = {summary['signal_tasks']}."]
    lines += ['<!-- END GENERATED GATES -->']
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-status',action='store_true',help='K1–K4: update generated status block')
    a=p.parse_args()
    text=render()
    if a.write_status:
        path=ROOT/'REPORT_STATUS.md'; old=path.read_text()
        begin,end='<!-- BEGIN GENERATED GATES -->','<!-- END GENERATED GATES -->'
        if begin in old:
            left,rest=old.split(begin,1); _,right=rest.split(end,1)
            path.write_text(left+text+right.lstrip('\n'))
        else:
            path.write_text(old.rstrip()+'\n\n'+text)
    print(text)

if __name__ == '__main__':
    main()
