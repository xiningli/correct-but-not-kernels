"""Current-GPU checks that compiler errors become bounded visible feedback."""
import argparse
import json
from pathlib import Path
from research.diagnostics_v2.common import ROOT, load_prepared, source_seal, save, digest, utc
from research.diagnostics_v1.controls import validate_controls
from research.diagnostics_v2.feedback import visible_feedback
from research.reliability_v2.strict_evaluator import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','controls','out']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();plan,tasks,_=load_prepared(a.prepared)
    validate_controls(a.controls,ROOT/plan['controls']['prepared_v5'])
    import torch
    torch.set_num_threads(4)
    a.out.mkdir(parents=True,exist_ok=False)
    inputs=[[torch.arange(3,dtype=torch.float32)]]; expected=[inputs[0][0]+1]
    source='import torch\nimport triton.language as tl\nclass ModelNew(torch.nn.Module):\n    def forward(self, x):\n        return tl.defeq(x)\n'
    result=evaluate(source,inputs,expected,a.out/'missing-api',seed=plan['seeds']['evaluator'])
    feedback=visible_feedback('development/control',source,inputs,expected,result)
    wanted={'kind':'missing_attribute','identifiers':['triton.language','defeq']}
    if result['component']!='crash' or wanted not in feedback['detail'].get('compiler_errors',[]):
        raise RuntimeError('Current-runtime API diagnostic control failed')
    save(a.out/'visible-feedback.json',feedback)
    save(a.out/'receipt.json',{'status':'PASS','utc':utc(),'source_sha256':source_seal(),
        'prepared_sha256':digest(a.prepared/'receipt.json'),
        'base_controls_sha256':digest(a.controls/'receipt.json'),
        'feedback_sha256':digest(a.out/'visible-feedback.json')})
    print('Current-runtime bounded API feedback PASS',flush=True)


if __name__=='__main__': main()
