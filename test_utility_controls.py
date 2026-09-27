"""Tests for utility and shortfall scores and controller-visible state."""
import unittest
import numpy as np
from qbr.utility_controls import ComparisonController
from qbr.policy import ActionProfile,BudgetForecast,best
from qbr.radio import Scenario

class Rules(unittest.TestCase):
    def make(self,means=(.4,.8),method='shortfall_queue',target=.7,waits=(.2,.4),mu=(1.,2.),second=(1.,4.)):
        profile=ActionProfile(np.tile(np.array(mu)[:,None],(3,1,2)),
            np.tile(np.array(second)[:,None],(3,1,2)),np.zeros((3,2,2)),np.ones((3,2,2),bool),
            {key:np.array([.5]) for key in np.ndindex(3,2,2)})
        forecast=BudgetForecast(profile,{k:np.array([profile.mu[k]]) for k in np.ndindex(3,2,2)},np.ones((3,2,2)),())
        controller=ComparisonController(forecast,Scenario(),[None],'fixed',(1,1),qualities=np.array([means]),method=method)
        controller.compute_state=lambda now:(np.array(waits),[],[])
        route,_,values=controller.choose(0.,0,0,target,0,np.ones(16),None,None)
        return route,values,controller.actions[0]
    def test_utility_exact(self):
        _,v,a=self.make(method='utility_queue')
        np.testing.assert_allclose(v,[.2+1+12*(1-.4),.4+2+12*(1-.8)])
        self.assertEqual(a['target'],.7);self.assertEqual(a['score_target'],1.)
    def test_shortfall_exact(self):
        _,v,a=self.make()
        np.testing.assert_allclose(v,[.2+1+12*(.7-.4),.4+2])
    def test_workload_increment(self):
        _,a,_=self.make();_,b,_=self.make(method='cq_neutral')
        np.testing.assert_allclose(b-a,.5*(np.array([.2,.4])*np.array([1.,2.])+np.array([1.,4.])/2))
    def test_target_one_identity(self):
        for q in ((0.,1.),(.5,.5),(.2,.7)):
            a,av,_=self.make(means=q,target=1.);b,bv,_=self.make(means=q,target=1.,method='utility_queue')
            self.assertEqual(a,b);np.testing.assert_array_equal(av,bv)
    def test_clipping_difference(self):
        for means in ((.2,.3),(.4,.9),(.9,.8)):
            _,a,_=self.make(means=means);_,b,_=self.make(means=means,method='utility_queue')
            np.testing.assert_allclose(a-(b-12),12*np.maximum(.7,means))
    def test_tie_cloud(self):
        for method in ('cq_neutral','utility_queue','shortfall_queue'):
            route,_,_=self.make(means=(.5,.5),waits=(0.,0.),mu=(1.,1.),second=(1.,1.),method=method)
            self.assertEqual(route,1)

if __name__=='__main__':unittest.main()
