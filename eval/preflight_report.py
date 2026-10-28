"""K1–K4: local preparation numbers from runs/, explicitly separate from research metrics."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from tools.gatelib import ROOT


def render(directory):
    rec = json.loads((directory / "preflight.json").read_text())
    inventory = json.loads((directory / "inventory.json").read_text())
    tests_path = directory / "tests.xml"
    suites = list(ET.parse(tests_path).getroot().iter("testsuite")) if tests_path.exists() else []
    tests, failures, errors = (sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors"))
    tasks = rec.get("tasks", {})
    lines = ["<!-- BEGIN GENERATED PREFLIGHT -->", "## Local preparation evidence", "",
             f"Source: `{directory.relative_to(ROOT) / 'preflight.json'}`.", "",
             f"Preparation checks: **{'PASS' if rec['local_checks_passed'] else 'FAIL'}**, not a C0 verdict.",
             f"Pytest: {tests} tests, {failures} failures, {errors} errors.",
             f"Locked tasks: {tasks.get('total')}; Level-1 train: {tasks.get('train_level1')}; "
             f"Level-1 held-out: {tasks.get('heldout_level1')}; Level-2 held-out: {tasks.get('heldout_level2')}.", "",
             f"GPU inventory: `{inventory['commands']['gpu'].get('stdout', 'unavailable')}`.",
             f"Logical CPUs: {inventory['logical_cpus']}; free disk: {inventory['disk_bytes']['free'] / 2**30:.1f} GiB.", "",
             "| Gate | Status | JSON record |", "|---|---|---|"]
    for i in range(7):
        paths = sorted((ROOT / "runs/gates").glob(f"c{i}-*.json"))
        if paths:
            gate = json.loads(paths[-1].read_text())
            lines.append(f"| C{i} | {gate['status']} | `{paths[-1].relative_to(ROOT)}` |")
        else:
            lines.append(f"| C{i} | not run | — |")
    ref = directory / "reference.json"
    lines += ["", "### Numbers and limits", ""]
    if ref.exists():
        data = json.loads(ref.read_text())
        lines.append(f"Trusted CUDA reference smoke: {len(data['results'])} modules executed with finite outputs; "
                     f"torch {data['torch']} / CUDA {data['torch_cuda']}.")
        lines.append("These executions validate reference plumbing only; no candidate correctness or speedup was measured.")
    lines += ["Timing noise floor: not measured. Detector coverage: not measured (no cheats run).",
              "Base-model pass@1, pass@8, P_signal and rollouts/hour: not measured.", "",
              "### Reachability", ""]
    for item in inventory["network"]:
        lines.append(f"- {item['url']}: reachable={item['reachable']}, HTTP={item.get('http_status')}, "
                     f"error={item.get('error', 'none')} (does not prove weight-download access).")
    lines += ["", "<!-- END GENERATED PREFLIGHT -->"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("record_dir", type=Path, help="K1–K4: preflight directory under runs/")
    p.add_argument("--write-status", action="store_true", help="K1–K4: replace only the generated status block")
    a = p.parse_args()
    directory = a.record_dir.resolve()
    directory.relative_to(ROOT / "runs")
    text = render(directory)
    if a.write_status:
        path = ROOT / "REPORT_STATUS.md"
        old = path.read_text()
        begin, end = "<!-- BEGIN GENERATED PREFLIGHT -->", "<!-- END GENERATED PREFLIGHT -->"
        if begin in old:
            left, rest = old.split(begin, 1)
            _, right = rest.split(end, 1)
            path.write_text(left + text + right.lstrip("\n"))
        else:
            path.write_text(old.rstrip() + "\n\n" + text)
    print(text)


if __name__ == "__main__":
    main()
