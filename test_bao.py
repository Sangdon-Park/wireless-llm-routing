"""Tests for Bao score arithmetic, tie handling, and controller inputs."""
from types import SimpleNamespace
import unittest
import numpy as np
from qbr.bao import BaoController
from qbr.bao_rule import bao_rule_transfer, EstimatedLatencies, FusionParameters
from qbr.radio import Scenario
from qbr.information_controls import release_correction


class BaoTests(unittest.TestCase):
    def decision(self,p,edge,cloud,alpha,theta):
        return bao_rule_transfer(p,EstimatedLatencies(edge,cloud),FusionParameters(theta,alpha))

    def test_strict_tie(self):
        self.assertEqual(self.decision(.5,1,1,1,.5).cloud,0)
        self.assertEqual(self.decision(.51,1,1,1,.5).cloud,1)

    def test_seconds_direction_unclipped(self):
        self.assertEqual(self.decision(.75,1,5,.1,.5).cloud,0)
        self.assertGreater(self.decision(.9,10,1,1,.5).fused_score,1)
        self.assertLess(self.decision(.1,1,10,1,.5).fused_score,0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):self.decision(1.1,1,1,1,.5)
        with self.assertRaises(ValueError):self.decision(.5,1,1,-1,.5)
        with self.assertRaises(ValueError):self.decision(.5,-1,1,1,.5)

    def controller(self,method,service=0,target=.2):
        profile=SimpleNamespace(mu=np.full((3,2,2),2.),enabled=np.ones((3,2,2),bool),
                                money=np.full((3,2,2),9e20),second=np.full((3,2,2),8e20))
        forecast=SimpleNamespace(decision=profile,mean_bits=np.full((3,2,2),5.))
        c=BaoController(forecast,Scenario(users=2),[None],'fixed',(1,1),
                        probabilities=[.6],method=method,alpha=.1,theta=.5)
        c.compute_state=lambda now:(np.array([10.,0.]),[],[])
        c.radio_state(5.,dict(users=np.array([0,1]),visible=np.zeros(2),delivered=np.zeros(2)))
        route,_,times=c.choose(5.,0,service,target,0,np.ones(2),None,None)
        return c,route,times

    def test_queue_seconds_and_budgets(self):
        c,route,times=self.controller('bao_queue')
        np.testing.assert_array_equal(times,[12.,2.])
        self.assertEqual(c.choices[0],(route,1))
        for service in (1,2):self.assertEqual(self.controller('bao_queue',service)[0].choices[0][1],0)

    def test_archived_debt_is_not_release_aware(self):
        c,_,times=self.controller('bao_radio')
        np.testing.assert_array_equal(times,[12.,5.])
        release=release_correction(np.zeros(2),np.ones(2),np.full((2,2),5.),
                                    np.array([10.,0.]),np.full((2,2),2.),0)
        np.testing.assert_array_equal(release[:,1],[3.,3.])
        self.assertEqual(c.actions[0]['radio_excess_s'],[0.,3.])

    def test_target_not_inserted(self):
        _,a,x=self.controller('bao_queue',target=0)
        _,b,y=self.controller('bao_queue',target=1)
        self.assertEqual(a,b);np.testing.assert_array_equal(x,y)


if __name__=='__main__':unittest.main()
