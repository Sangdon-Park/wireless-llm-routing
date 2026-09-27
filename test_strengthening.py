"""Tests for paired resampling, expected shortfall, and observed radio state."""
import unittest
import gzip
import json
import numpy as np
from qbr.data import Study,DATA
from qbr.strengthening import mean_risks,task_mapping,resampled_trace,radio_correction,RadioController

class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.study=Study()

    def test_mean_and_distribution_obey_jensen(self):
        _,trace=self.study.requests('primary',23924000)
        mean=mean_risks(self.study,trace)
        distribution=self.study.risks('calibrated',trace)
        mask=np.isfinite(mean)
        self.assertTrue(np.all(distribution[mask]>=mean[mask]-1e-12))
        self.assertGreater(float(np.max(distribution[mask]-mean[mask])),.01)
        # Equal means need not imply equal shortfall: the manuscript example.
        self.assertEqual(float(np.maximum(.5-np.array([0.,1.]),0).mean()),.25)
        self.assertEqual(max(.5-np.mean([0.,1.]),0.),0.)

    def test_task_pairs_stay_together_and_only_composition_changes(self):
        _,trace=self.study.requests('primary',23924000)
        mapping=task_mapping(self.study,0)
        other=task_mapping(self.study,1)
        np.testing.assert_array_equal(mapping,task_mapping(self.study,0))
        self.assertFalse(np.array_equal(mapping,other))
        self.assertLess(len(np.unique(mapping)),225)
        for i,k in enumerate(mapping):
            self.assertEqual(self.study.tasks[i]['service'],self.study.tasks[k]['service'])
        changed=resampled_trace(trace,mapping)
        for key in ('arrival','service','quality','user'):
            np.testing.assert_array_equal(getattr(trace,key),getattr(changed,key))
        np.testing.assert_array_equal(changed.task,mapping[trace.task])

    def test_radio_uses_prefix_backlog_and_observed_capacity(self):
        state=dict(users=np.array([0,1,0]),visible=np.array([100.,70.,20.]),delivered=np.array([80.,20.,20.]))
        controller=object.__new__(RadioController)
        controller.sc=type('Scenario',(),{'users':2})()
        controller.radio_state(1.,state)
        np.testing.assert_array_equal(controller.radio_backlog,[20.,50.])
        payload=np.array([[10.,10.],[2.,2.]])
        serving=np.ones((2,2))
        capacity=np.array([10.,100.])
        # 2.5 s observed airtime debt; edge/cloud add 1/0.2 s for this user.
        np.testing.assert_allclose(radio_correction(controller.radio_backlog,capacity,payload,serving,0),[[2.5,2.5],[1.7,1.7]])
        np.testing.assert_allclose(radio_correction(np.zeros(2),capacity*100,payload,serving,0),0.)
        # The hook stores a new backlog array, not mutable simulator buffers.
        state['visible'][:]=1e9
        np.testing.assert_array_equal(controller.radio_backlog,[20.,50.])

    def test_released_controls_reproduce(self):
        archive=DATA/'strengthening_results.json.gz'
        if not archive.exists():self.skipTest('Supplementary run archive has not yet been generated')
        from strengthen import initialize,episode
        with gzip.open(archive,'rt',encoding='utf-8') as f:records=json.load(f)
        initialize()
        for method in ('mean','radio_mean','radio_calibrated'):
            result=episode(('primary',23924000,method,-1))
            expected=next(r for r in records if r['condition']=='primary' and r['seed']==23924000
                          and r['method']==method and r['replicate']==-1)
            self.assertEqual(result['choices_sha256'],expected['choices_sha256'])
            for key in ('objective','delay','delay_p95','shortfall','money'):
                self.assertAlmostEqual(result['metrics'][key],expected['metrics'][key],places=8)

if __name__=='__main__':unittest.main()
