"""Check every quantitative claim in paper/writeup.md against its receipt.

Each claim pairs a string that must occur verbatim in the writeup with the value
recomputed from a committed receipt. Run from the repository root:
  python -m research.paper_v1.claims
Numbers quoted from earlier reports without a machine-readable receipt are listed
in MANUAL with their source, so a reviewer can check those by hand.
In a public checkout the runs/ receipts come from the data bundle:
  python -m research.paper_v1.claims --data release-data/data
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = None  # data-bundle root holding runs/...; local receipts take precedence
PAPER = ROOT / 'paper/writeup.md'
EV = ROOT / 'evidence/paper-shortcut-audit-20261025'
R3 = 'runs/full-audit-20261018/replication-archive/selected/runs/replication-v1.1/confirmation/REPORT.json'
PORT = 'runs/portability-v1-20261020/run-r3/REPORT.json'
TANH = 'evidence/paper-probes-20261025/tanh-boundary.json'
REPAIR_V5 = 'runs/diagnostics-v5/terminal-audit-20261016/diagnostics/metrics.jsonl'
REPAIR_V6 = 'runs/diagnostics-v6-resume31b/terminal-audit-20261017'
PUBLIC = 'evidence/paper-public-audit-20261026'

# Quoted from REPORT.md / REPORT_STATUS.md, whose underlying receipts predate this package.
MANUAL = {
    'CV across 30 median estimates 0.0145%': 'REPORT.md C2',
    'between 0.988 and 1.011': 'REPORT_STATUS.md 速度方面 (C/S/SFT per-seed geomean ratios)',
    '(3 of 57 PyTorch-only)': 'Hari et al. arXiv:2603.29010 Section 6.5',
    '0 successes': 'REPORT_STATUS.md 36-program strict-prompt pilot',
}


def source(p):
    local = ROOT / p
    return local if local.exists() or DATA is None else DATA / p


def jsonl(p):
    return [json.loads(line) for line in source(p).read_text().splitlines()]


def load(p):
    return json.loads(source(p).read_text())


def pct(x, d=1):
    return f'{100 * x:.{d}f}'


def signed(x, d=2):
    return f'{100 * x:+.{d}f}'.replace('-', '−')


def ci(lo, hi, d=2):
    return f'[{100 * lo:.{d}f}, {100 * hi:.{d}f}]'.replace('-', '−')


def frac(a, b):
    return f'{a}/{b}'


def total(d, prefix, key):
    return sum(v.get(key, 0) for k, v in d.items() if k.startswith(prefix))


def claims():
    r3, port = load(R3), load(PORT)
    sa = json.loads((EV / 'shortcut-audit.json').read_text())
    ta = json.loads((EV / 'training-shortcut-audit.json').read_text())
    tanh = load(TANH)
    c = r3['comparisons']
    out = []
    add = lambda text, why: out.append((text, why))

    # 4.1 main table
    l2 = lambda pol: r3['policies'][pol]['levels']['level2']['correctness']['value']
    for arm, label in [('C', 'C (GRPO), 3 seeds'), ('S', 'S (speed-shaped), 3 seeds'), ('SFT', 'SFT (RFT), 3 seeds')]:
        mean = sum(l2(p) for p in r3['policies'] if p.split('-')[0] == arm) / 3
        add(f'| {label} | {pct(mean, 3)}% |', f'R3 {arm} mean')
    add(f'| Base | {pct(l2("base"), 3)}% |', 'R3 base')
    add(f'from {pct(l2("base"))}% to', 'abstract base')
    programs = sum(sum(v['failure_components'].values()) for v in r3['policies'].values())
    add(f'three seeds, {programs:,} programs', 'abstract program count')
    for k, row in [('C-base/level2', 'C − base'), ('S-base/level2', 'S − base'), ('C-SFT/level2', 'C − SFT')]:
        p = c[k]['paired']
        add(f'| {row} | {signed(p["value"], 3)} | {ci(*p["ci"], 3)} |', f'R3 {k}')
    add(f'({signed(c["C-base/level2"]["paired"]["value"], 1)} points, 98.3% CI {ci(*c["C-base/level2"]["paired"]["ci"], 1)}',
        'abstract C-base')
    p = c['C-SFT/level2']['paired']
    add(f'C − SFT {signed(p["value"])} {ci(*p["ci"], 1)}', 'abstract C-SFT')

    # Consumer-GPU replication
    pol = port['policies']
    rate = lambda arm: (sum(v['local_numeric_pass'] for k, v in pol.items() if k.split('-')[0] == arm)
                        / sum(v['attempts'] for k, v in pol.items() if k.split('-')[0] == arm))
    add(f'base {pct(rate("base"), 2)}%, C {pct(rate("C"), 2)}%, SFT {pct(rate("SFT"), 2)}%', 'portability rates')
    e = port['effects']
    add(f'C − base {signed(e["C-base/local"]["estimate"])} {ci(*e["C-base/local"]["interval"])}', 'portability C-base')
    add(f'C − SFT {signed(e["C-SFT/local"]["estimate"])} {ci(*e["C-SFT/local"]["interval"])}', 'portability C-SFT')
    add(f'over {port["attempts_recorded"]:,} attempts', 'portability attempts')
    add(f'{port["complete_tasks"]} tasks passed', 'portability tasks')
    add(f'the other {len(port["tasks_excluded"])} hit', 'portability exclusions')

    # 4.2 decomposition
    eff, cnt = sa['paired_task_effects'], sa['counts']
    rows = [('correct', '**Correct, total**'), ('correct_no_declared_kernel', 'Correct, plain PyTorch'),
            ('correct_declared_not_launched', 'Correct, decorative'), ('correct_launched', 'Correct, launched')]
    for key, label in rows:
        cells = []
        for comp in ['C-base', 'SFT-base', 'C-SFT']:
            x = eff[comp][key]
            v = f'{signed(x["estimate"])} {ci(*x["ci95"])}'
            if key == 'correct' and comp != 'C-SFT':
                v = f'**{signed(x["estimate"])}** {ci(*x["ci95"])}'
            cells.append(v)
        add(f'| {label} | ' + ' | '.join(cells) + ' |', f'decomposition {key}')
    add(f'0 of {sum(v["attempts"] for v in cnt.values()):,} programs are both correct and compliant',
        'strict successes') if all(v['strict_success'] == 0 for v in cnt.values()) else None
    add(f'base {frac(cnt["base"]["correct_by_class"]["launched"], cnt["base"]["correct"])},', 'launched correct base')
    add(f'C {frac(cnt["C"]["correct_by_class"]["launched"], f"{cnt["C"]["correct"]:,}")}, '
        f'SFT {frac(cnt["SFT"]["correct_by_class"]["launched"], f"{cnt["SFT"]["correct"]:,}")}', 'launched correct C/SFT')
    n_launched = sum(v['correct_by_class']['launched'] for v in cnt.values())
    add(f'All {n_launched} still run', 'launched total')
    dno = lambda k: (cnt[k]['correct_declared_not_launched_name_appears_only_at_definition'],
                     cnt[k]['correct_by_class']['declared_not_launched'])
    add(f'{frac(*dno("base"))} (base), {frac(*dno("C"))} (C) and {frac(*dno("SFT"))} (SFT)', 'definition-only share')
    add(f'({signed(eff["C-base"]["correct_no_declared_kernel"]["estimate"], 1)})',
        'abstract plain PyTorch')
    add(f'(+{100 * eff["C-base"]["correct_declared_not_launched"]["estimate"]:.1f})', 'abstract decorative')
    x = eff['C-base']['correct_launched']
    add(f'({signed(x["estimate"])} points {ci(*x["ci95"])})', 'abstract launched')

    # 3.4 static classifier validation
    sv = ta['static_vs_runtime_on_v4_replay']
    d_only = sv['definition_only'].get('declared_not_launched', 0)
    assert sum(sv['definition_only'].values()) == d_only
    add(f'{d_only:,}/{d_only:,} agreement', 'static validation intro')
    add(f'all {d_only:,} "definition only" programs', 'static validation methods')
    add(f'{sv["reachable"]["launched"]} of {sum(sv["reachable"].values())} "reachable"', 'reachable validation')

    # 4.3 training-task table
    base = ta['base_candidates_layernorm']
    n = total(base, 'SFT', 'n')
    dec, dec_ok = total(base, 'SFT', 'definition_only'), total(base, 'SFT', 'correct_definition_only')
    pt, pt_ok = total(base, 'SFT', 'no_declared_kernel'), total(base, 'SFT', 'correct_no_declared_kernel')
    rk, rk_ok = total(base, 'SFT', 'reachable'), total(base, 'SFT', 'correct_reachable')
    grpo = {s: b for s, v in ta['grpo_rollouts_layernorm'].items() if s.startswith('C-')
            for s, b in [(s, v['buckets'])]}
    g = lambda key: sum(b.get(key, 0) for bs in grpo.values() for name, b in bs.items() if name != 'step_0')
    gn = g('n')
    sft = ta['sft_datasets_layernorm']
    sn = total(sft, 'SFT', 'n')
    add('(' + '/'.join(str(v['n']) for _, v in sorted(sft.items())) + ' per seed)', 'RFT kept samples')
    p0 = lambda a, b: f'{round(100 * a / b)}%'
    add(f'Base samples ({n}) | GRPO-C, steps 1–40 ({gn}) | RFT data ({sn})', 'table header')
    add(f'| Decorative (definition only) | {p0(dec, n)}; **{p0(dec_ok, dec)} pass** | '
        f'{p0(g("definition_only"), gn)}; {p0(g("correct_definition_only"), g("definition_only"))} pass | '
        f'{p0(total(sft, "SFT", "definition_only"), sn)} |', 'decorative row')
    add(f'| Plain PyTorch | {p0(pt, n)}; {p0(pt_ok, pt)} pass | '
        f'{p0(g("no_declared_kernel"), gn)}; {p0(g("correct_no_declared_kernel"), g("no_declared_kernel"))} pass | '
        f'{p0(total(sft, "SFT", "no_declared_kernel"), sn)} |', 'plain row')
    add(f'| Calls its own kernel | {p0(rk, n)}; **{p0(rk_ok, rk)} pass ({rk_ok}/{rk})** | '
        f'{p0(g("reachable"), gn)}; {p0(g("correct_reachable"), g("reachable"))} pass '
        f'({g("correct_reachable")}/{g("reachable")}) | '
        f'{100 * total(sft, "SFT", "reachable") / sn:.1f}% |', 'kernel row')
    add(f'pass {round(100 * dec_ok / dec)}% of the time', 'abstract decorative pass')
    add(f'pass {rk_ok} times in {rk}', 'abstract kernel pass')
    add(f'{p0(total(sft, "SFT", "definition_only"), sn)} decorative', 'abstract SFT decorative')
    add(f'from {p0(dec, n)} to {p0(g("definition_only"), gn)}', 'decorative share rise')
    add(f'the {sum(v["rows"] for v in ta["grpo_rollouts_layernorm"].values()):,} GRPO rollouts', 'rollout count')
    cc = ta['base_crash_causes_sft_20261022']['counts']
    pt_crash = sum(v for k, v in cc.items() if k.startswith('no_declared_kernel/'))
    add(f'{cc["no_declared_kernel/nonexistent_triton_attribute"]} of the {pt_crash} plain-PyTorch crashes', 'crash causes')
    add(f'Only {rk_ok} of {n} base LayerNorm samples', 'discussion success count')
    add(f'probability ≈{100 * (1 - (1 - rk_ok / n) ** 8):.1f}%', 'group success probability')
    add(f'({g("correct_reachable")} of {g("reachable")} kernel-calling rollouts passed)', 'GRPO kernel-calling passes')
    declared = dec + rk + total(base, 'SFT', 'unreachable_other')
    add(f'{declared} define a\n`@triton.jit` kernel', 'declared kernels in base samples')
    add(f'kernel and {rk} call it', 'reachable in base samples')

    # 4.4 issue-style repair
    v5 = jsonl(REPAIR_V5)
    arms = {}
    for r in v5:
        a = arms.setdefault(r['policy'].split('-')[0], {'n': 0, 'resolved': 0, 'wrong': 0, 'violation': 0})
        a['n'] += 1
        a['resolved'] += bool(r['resolved'])
        a['wrong'] += r['component'] == 'incorrect' and r['suites']['visible']['contract_compliant'] is True
        a['violation'] += r['component'] == 'contract_violation'
    for arm, label in [('base', 'Base'), ('C', 'C, 3 checkpoints'), ('SFT', 'SFT, 3 checkpoints')]:
        a = arms[arm]
        add(f'| {label} | {a["resolved"]}/{a["n"]} | {a["wrong"]} | {a["violation"]} |', f'repair row {arm}')
    tuned = [a for k, a in arms.items() if k != 'base']
    add(f'resolve {sum(a["resolved"] for a in tuned)} of {sum(a["n"] for a in tuned)} attempts', 'abstract repair')
    add(f'({sum(a["resolved"] for a in tuned)}/{sum(a["n"] for a in tuned)} resolved after fine-tuning)', 'contribution repair')
    add(f'Each policy made {arms["base"]["n"]} single-shot attempts', 'repair attempts per policy')
    violations = [r for r in v5 if r['component'] == 'contract_violation']
    assert all(r['suites']['visible']['numerically_correct'] is False for r in violations)
    assert {op for r in violations for c in r['suites']['visible']['contract_checks'] for op in c['ops']} == {'aten.contiguous.default'}
    add(f'the {["zero", "one", "two", "three", "four"][len(violations)]} violations are host-side\n`.contiguous()` calls in wrong programs', 'repair violations')
    resolved_tasks = {(r['task'], r['condition']) for r in v5 if r['resolved']}
    assert len(resolved_tasks) == 1, 'paper says both base successes share defect and feedback form'
    it = {m: jsonl(f'{REPAIR_V6}/{m}/episodes.jsonl') for m in ['gemma-4-12B-it', 'gemma-4-31B-it']}
    ok = lambda m: f'{sum(e["resolved"] for e in it[m])}/{len(it[m])}'
    add(f'resolved {ok("gemma-4-12B-it")} episodes with\nbase Gemma-4-12B-it and {ok("gemma-4-31B-it")} with Gemma-4-31B-it', 'iterative repair')
    by = {}
    for e in it['gemma-4-31B-it']:
        by.setdefault(e['condition'], []).append(e['resolved'])
    order = ['independent', 'revision_verdict', 'revision_diagnostic']
    add('(' + ', '.join(f'{sum(by[c])}/{len(by[c])}' for c in order[:2]) + f' and {sum(by[order[2]])}/{len(by[order[2]])}',
        '31B by feedback form')
    nodes = {n['id']: n for n in jsonl(f'{REPAIR_V6}/gemma-4-31B-it/nodes.jsonl')}
    distinct = {nodes[e['selected']]['source'] for e in it['gemma-4-31B-it'] if e['resolved']}
    add(f'from only {["zero", "one", "two", "three", "four"][len(distinct)]} distinct successful programs', 'distinct 31B successes')
    gain = c['C-base/level2']['paired']['value']
    add(f'checkpoints that gained\n{round(100 * gain)} points', 'discussion gain')

    # 4.5 public corpora
    sk = load(f'{PUBLIC}/sakana-static.json')['levels']
    kb = load(f'{PUBLIC}/kernelbook-static.json')
    lv = lambda l, key: sk[l]['counts'].get(key, 0)
    classes = ['launch_site', 'definition_only', 'no_declared_kernel', 'unresolved']
    sk_correct = sum(sk[l]['correct'] for l in sk)
    for l in sk:
        assert sum(lv(l, f'class/{k}') for k in classes) == sk[l]['correct']
        gemm = sum(lv(l, f'class/{k}/host_gemm_or_conv=True') for k in classes)
        add(f'| AI CUDA Engineer, L{l} ({sk[l]["correct"]:,}) | {lv(l, "class/definition_only")} | '
            f'{lv(l, "class/no_declared_kernel")} | {gemm:,} |', f'sakana row L{l}')
    assert set(kb['static_class']) == {'reachable'}
    add(f'| KernelBook ({kb["rows"]:,}) | 0 | 0 | {kb["reachable_with_extern_gemm_or_conv"]["True"]:,} |', 'kernelbook row')
    add(f'({sk_correct:,} CUDA kernels its harness marked\ncorrect)', 'sakana correct total')
    add(f'KernelBook [18] ({kb["rows"]:,} PyTorch-to-Triton pairs', 'kernelbook rows')
    add(f'public corpora ({sk_correct + kb["rows"]:,} kernels)', 'abstract corpus total')
    agree = []
    for l in sk:
        a = {k: v for k, v in sk[l]['counts'].items() if k.startswith('profile/agreement/')}
        same = a.get('profile/agreement/static=True/profile=True', 0) + a.get('profile/agreement/static=False/profile=False', 0)
        agree.append(f'{round(100 * same / sum(a.values()))}%')
    add(f'on {agree[0]},\n{agree[1]} and {agree[2]} of Level 1–3 kernels', 'static vs profile agreement')
    add(f'({lv("1", "class/launch_site/host_gemm_or_conv=False"):,} of\n{sk["1"]["correct"]:,} launch a custom kernel',
        'L1 genuine')
    fast2 = lv('2', 'faster_than_native')
    fast3 = lv('3', 'faster_than_native')
    deleg2 = sum(lv('2', f'faster_than_native/{k}/host_gemm_or_conv=True') for k in classes)
    add(f'{deleg2:,} of the {fast2:,} archive kernels reported faster', 'L2 delegated speedups')
    add(f'in {lv("2", "faster_than_native/launch_site/host_gemm_or_conv=True"):,} the custom kernel handles only',
        'L2 delegated with own kernel')
    deleg3 = sum(lv('3', f'faster_than_native/{k}/host_gemm_or_conv=True') for k in classes)
    assert deleg2 > fast2 / 2 and deleg3 > fast3 / 2, 'contribution 5 says "most Level-2/3"'
    nokernel3 = sum(lv('3', f'faster_than_native/{k}/host_gemm_or_conv={h}')
                    for k in ['no_declared_kernel', 'definition_only'] for h in [True, False])
    add(f'{nokernel3} of {fast3:,} "faster" kernels launch no custom kernel', 'L3 kernel-free speedups')

    # 4.4 tanh probe
    res = tanh['results']
    ok = {r['magnitude'] for r in res if r['pass']}
    bad = {r['magnitude'] for r in res if r['nonfinite_rows'] > 0}
    assert max(ok) < min(bad) and all(not r['pass'] for r in res if r['magnitude'] >= min(bad))
    add(f'|x| ≥ {min(bad):.0f} in float32', 'tanh failure boundary')
    add(f'correct at |x| ≤ {max(ok):.0f}', 'tanh pass boundary')
    return out


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, help='data-bundle root (the bundle\'s data/ directory)')
    DATA = ap.parse_args().data
    # Compare modulo line wrapping.
    text = ' '.join(PAPER.read_text().split())
    failed = 0
    for want, why in claims():
        hit = ' '.join(want.split()) in text
        failed += not hit
        print('ok  ' if hit else 'MISS', why, '|', want)
    for want, source in MANUAL.items():
        print('ok  ' if want in text else 'MISS', 'manual:', source, '|', want)
        failed += want not in text
    print(f'{failed} missing')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
