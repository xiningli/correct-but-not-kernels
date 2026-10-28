"""K1–K4: a failed or foreign protocol must not be reported as completed research."""
import json
from eval.terminal_report import generate
from tools.replication import sha256

def setup(tmp_path):
    study=tmp_path/'runs/study';controller=study/'controllers/attempt'
    controller.mkdir(parents=True)
    (study/'frozen.json').write_text('{}')
    gates=tmp_path/'runs/gates';gates.mkdir()
    return study,controller,gates

def test_failed_audit_and_foreign_success_are_not_completed_study(tmp_path):
    study,controller,gates=setup(tmp_path)
    (controller/'outcome.json').write_text(json.dumps({'status':'blocked','error':'Worker exited 1','stage':'all'}))
    (gates/'r1-1.json').write_text(json.dumps({'status':'FAIL','protocol_sha256':sha256(study/'frozen.json'),
        'checks':[{'name':'audit','status':'FAIL','error':'BatchEncoding plus list'}]}))
    (gates/'r3-2.json').write_text(json.dumps({'status':'PASS','protocol_sha256':'unrelated','checks':[]}))
    path=generate(study,controller,tmp_path)
    record=json.loads((controller/'TERMINAL_REPORT.json').read_text())
    assert not record['formal_study_complete']
    assert record['gates']['R1']['status']=='FAIL'
    assert record['gates']['R3']['status']=='not run'
    assert 'BatchEncoding plus list' in path.read_text()

def test_completed_stage_is_not_full_confirmation_without_result_report(tmp_path):
    study,controller,gates=setup(tmp_path)
    (controller/'outcome.json').write_text(json.dumps({'status':'complete','error':None,'stage':'audit'}))
    (gates/'r1-1.json').write_text(json.dumps({'status':'PASS','protocol_sha256':sha256(study/'frozen.json'),'checks':[]}))
    generate(study,controller,tmp_path)
    assert not json.loads((controller/'TERMINAL_REPORT.json').read_text())['formal_study_complete']

def test_completed_confirmation_preserves_report_hash(tmp_path):
    study,controller,gates=setup(tmp_path)
    (controller/'outcome.json').write_text(json.dumps({'status':'complete','error':None,'stage':'all'}))
    (gates/'r3-1.json').write_text(json.dumps({'status':'PASS','protocol_sha256':sha256(study/'frozen.json'),'checks':[]}))
    report=study/'confirmation/REPORT.md';report.parent.mkdir();report.write_text('measured results')
    generate(study,controller,tmp_path)
    record=json.loads((controller/'TERMINAL_REPORT.json').read_text())
    assert record['formal_study_complete']
    assert record['reports']['confirmation']['sha256']==sha256(report)
