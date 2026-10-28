"""K1–K4: an interrupted segment must evaluate its newest completed optimizer state."""
import json
from tools.gate_c6 import final_checkpoint

def test_new_checkpoint_beats_prior_segment_completion(tmp_path):
    (tmp_path/'completion-step-10.json').write_text(json.dumps({'step':10,'adapter':'old-policy'}))
    current=tmp_path/'checkpoints/checkpoint-12';current.mkdir(parents=True)
    (current/'trainer_state.json').write_text('{}');(current/'optimizer.pt').touch()
    incomplete=tmp_path/'checkpoints/checkpoint-13';incomplete.mkdir()
    got=final_checkpoint(tmp_path)
    assert got['step']==12 and got['adapter']==str(current)
