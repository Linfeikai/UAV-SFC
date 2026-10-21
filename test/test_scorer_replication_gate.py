"""A claimed pass alone must not authorize an independent replication run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.scorer_replicate import FROZEN_SHA, verify_fresh_gate
from scripts.audit_scorer_replicates import compare


class ReplicationGateTests(unittest.TestCase):
    def evidence(self):
        metric = dict(episodes=100, success_rate_pct=53., crash_episode_rate_pct=0., mean_episode_reward=650.)
        evaluations={name:dict(metric) for name in ('learned','static_heuristic','heuristic','obs_heuristic')}
        evaluations['learned'].update(success_rate_pct=53.6, mean_episode_reward=659.)
        return dict(status='complete', model_sha256=FROZEN_SHA, episodes=100, seed_start=90000,
                    comparison=dict(passes_this_cohort=True), evaluations=evaluations)

    def test_recompute_gate_instead_of_trusting_pass_label(self):
        state = self.evidence()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'status.json'
            path.write_text(json.dumps(state))
            verify_fresh_gate(path)
            for key,value in [('success_rate_pct',53.4), ('crash_episode_rate_pct',6.), ('mean_episode_reward',649.)]:
                modified = copy.deepcopy(state)
                modified['evaluations']['learned'][key]=value
                path.write_text(json.dumps(modified))
                with self.assertRaises(AssertionError):
                    verify_fresh_gate(path)

    def test_audit_uses_unrounded_margin_and_rejects_nonfinite_metrics(self):
        metrics=self.evidence()['evaluations']
        metrics['learned']['success_rate_pct']=53.499999999
        self.assertFalse(compare(metrics,100)['passes'])
        metrics['learned']['success_rate_pct']=53.5
        self.assertTrue(compare(metrics,100)['passes'])
        metrics['learned']['mean_episode_reward']=float('nan')
        with self.assertRaises(AssertionError):
            compare(metrics,100)

    def test_short_cohort_or_changed_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'status.json'
            for field,value in [('episodes',50),('model_sha256','changed')]:
                state=self.evidence()
                state[field]=value
                path.write_text(json.dumps(state))
                with self.assertRaises(AssertionError):
                    verify_fresh_gate(path)


if __name__ == '__main__':
    unittest.main()
