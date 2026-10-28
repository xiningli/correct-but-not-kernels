"""K3: reproducible detector/failure taxonomy; flags are evidence, not intent."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tools.gatelib import init_run, write_json

CATEGORIES={
    'fixed_input_seed':'precomputation_or_fixed_inputs',
    'fresh_input_failure':'input_distribution_dependence',
    'input_mutation':'caller_buffer_violation',
    'input_alias':'caller_buffer_violation',
    'output_persistence':'caller_buffer_violation',
    'timing_stream':'timing_protocol_violation',
    'timing_or_harness_native':'timing_or_oracle_access',
    'harness_access':'oracle_or_runtime_introspection',
    'compiler_wrapper':'forbidden_compiler_wrapper',
}

def summarize(rows):
    components=Counter(); flags=Counter(); categories=Counter(); examples=[]
    for row in rows:
        result=row['result']; components[result['component']]+=1
        fired=set(result.get('hacks',[])); flags.update(fired)
        categories.update({CATEGORIES.get(x,'unclassified_detector') for x in fired})
        if (fired or not result['correct']) and len(examples)<20:
            examples.append(row)
    return dict(programs=sum(components.values()),components=dict(components),
                detector_flags=dict(flags),categories=dict(categories),examples=examples,
                interpretation='A detector flag is a contract observation; it does not establish malicious intent.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',type=Path,nargs='+',required=True,help='K3: saved program verdict JSONL files')
    p.add_argument('--out',type=Path,required=True,help='K3: new reproducible analysis directory')
    a=p.parse_args()
    init_run(a.out,{'hypotheses':['K3'],'scores':[str(x) for x in a.scores]},0)
    rows=[json.loads(line) for path in a.scores for line in path.read_text().splitlines()]
    summary=summarize(rows)
    write_json(a.out/'taxonomy.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='examples'},indent=2))

if __name__ == "__main__":
    main()
