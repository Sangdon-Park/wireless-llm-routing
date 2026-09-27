"""Tests for value-of-information scores using synthetic controller fixtures."""
import unittest
import numpy as np
from qbr.information_value import InformationController, spec, fpi_kappa
from qbr.policy import ActionProfile, BudgetForecast, scores
from qbr.radio import Scenario, MIX


class Rules(unittest.TestCase):
    def parts(self):
        profile = ActionProfile(np.ones((3, 2, 2)), np.ones((3, 2, 2)), np.zeros((3, 2, 2)), np.ones((3, 2, 2), bool),
                                {k: np.array([.5]) for k in np.ndindex(3, 2, 2)})
        forecast = BudgetForecast(profile, {k: np.array([1.]) for k in np.ndindex(3, 2, 2)}, np.ones((3, 2, 2)), ())
        return forecast, Scenario(), [None], 'fixed', (1, 1)

    def make(self, method, means=(.4, .8), target=.7, waits=(.2, .4), mu=(1., 2.), second=(1., 4.), kappa=None):
        profile = ActionProfile(np.tile(np.array(mu)[:, None], (3, 1, 2)), np.tile(np.array(second)[:, None], (3, 1, 2)),
                                np.zeros((3, 2, 2)), np.ones((3, 2, 2), bool), {k: np.array([.5]) for k in np.ndindex(3, 2, 2)})
        forecast = BudgetForecast(profile, {k: np.array([profile.mu[k]]) for k in np.ndindex(3, 2, 2)}, np.ones((3, 2, 2)), ())
        s = spec(method)
        k = kappa if kappa is not None else np.full(2, 0. if s['kappa'] == 'fpi' else float(s['kappa']))
        c = InformationController(forecast, Scenario(), [None], 'fixed', (1, 1), qualities=np.array([means]), spec=s, kappa=k)
        c.compute_state = lambda now: (np.array(waits), [], [])
        route, _, values = c.choose(0., 0, 0, target, 0, np.ones(16), None, None)
        return route, values, profile

    def test_queue_matches_policy_scores(self):
        _, v, profile = self.make('cq')
        risk = np.maximum(.7-np.array([.4, .8]), 0)[:, None]*np.ones((1, 2))
        ref = scores(profile, 0, .7, np.array([.2, .4]), risk=risk, required=np.array([[False, True], [False, True]]))
        np.testing.assert_array_equal(v, ref.min(1))

    def test_utility_and_intrinsic(self):
        _, v, _ = self.make('utility')
        np.testing.assert_allclose(v, [.2+1+12*(1-.4), .4+2+12*(1-.8)])
        _, v, _ = self.make('intrinsic')
        np.testing.assert_allclose(v, [1+12*.3, 2.])

    def test_rule_weight_suffix(self):
        _, v, _ = self.make('shortfall@b8')
        np.testing.assert_allclose(v, [.2+1+8*.3, .4+2])

    def test_per_route_kappa(self):
        _, a, _ = self.make('shortfall'); _, b, _ = self.make('fpi', kappa=np.array([.3, .7]))
        np.testing.assert_allclose(b-a, np.array([.3, .7])*(np.array([.2, .4])*np.array([1., 2.])+np.array([1., 4.])/2))

    def test_fixed_routes(self):
        self.assertEqual(self.make('always_edge')[0], 0); self.assertEqual(self.make('always_cloud')[0], 1)

    def test_fpi_formula(self):
        class S: arrival_mu = np.array([[2., 3.], [1., 2.], [4., 1.]])
        k = fpi_kappa(S, .65)
        np.testing.assert_allclose(k, .65/(.35*(MIX @ S.arrival_mu)))

    def test_static_and_noisy_specs(self):
        s = spec('static_p0.75')
        self.assertEqual((s['static'], s['kappa'], s['wait']), (.75, 0., False))
        self.assertEqual(spec('static_prop')['static'], 'prop')
        s = spec('noisy0.3'); self.assertEqual((s['quality'], s['sigma']), ('noisy', .3))

    def test_static_routes_follow_given_list(self):
        s = spec('static_p0.5')
        for fixed in (0, 1):
            c = InformationController(*self.parts(), qualities=np.array([[.4, .8]]), spec=dict(s, routes=np.array([fixed])), kappa=np.zeros(2))
            c.compute_state = lambda now: (np.array([100., 0.]), [], [])
            self.assertEqual(c.choose(0., 0, 0, .7, 0, np.ones(16), None, None)[0], fixed)

    def test_unknown_method(self):
        with self.assertRaises(ValueError): spec('nonsense')


if __name__ == '__main__':
    unittest.main()
