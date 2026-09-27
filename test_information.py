"""Independent fluid integration and exact decision-boundary checks."""
from fractions import Fraction as F
import unittest
import numpy as np
from qbr.information_controls import release_correction


class InformationChecks(unittest.TestCase):
    def test_completion_against_segment_integration(self):
        rng=np.random.default_rng(23926024)
        for _ in range(1000):
            debt=F(int(rng.integers(0,100)),7)
            waits=[F(int(v),7) for v in rng.integers(0,40,2)]
            duration=[[F(int(v),7) for v in row] for row in rng.integers(1,40,(2,2))]
            airtime=[[F(int(v),7) for v in row] for row in rng.integers(0,40,(2,2))]
            observed=release_correction([float(debt)],[1.],np.array(airtime,float),np.array(waits,float),np.array(duration,float),0)
            for m in (0,1):
                for b in (0,1):
                    w,s,a=waits[m],duration[m][b],airtime[m][b]
                    if a==0:expected=F(0)
                    else:
                        backlog=max(F(0),debt-w)
                        production=a/s
                        expected=max(F(0),backlog+(production-1)*s)
                    self.assertAlmostEqual(float(expected),observed[m,b],places=12)

    def test_radio_premium_and_ties(self):
        rng=np.random.default_rng(23926025)
        for _ in range(2000):
            D=F(int(rng.integers(0,50)),7)
            w=[F(int(v),7) for v in rng.integers(0,30,2)]
            mu=[F(int(v),7) for v in rng.integers(1,30,2)]
            a=[F(int(v),7) for v in rng.integers(1,30,2)]
            delta=[max(0,max(D,w[m])+a[m]-w[m]-mu[m]) for m in (0,1)]
            z=[max(0,a[m]-mu[m]) for m in (0,1)]
            k=[max(w[m],w[m]+mu[m]-a[m]) for m in (0,1)]
            sign=(1 if k[0]>k[1] else -1 if k[0]<k[1] else 0)
            g=z[1]-z[0]+sign*min(max(0,D-min(k)),abs(k[0]-k[1]))
            self.assertEqual(g,delta[1]-delta[0])
            A=F(int(rng.integers(-20,20)),7)
            for benefit in (A,A+g,(A+A+g)/2,A-10,A+g+10):
                self.assertEqual((benefit>=A)!=(benefit>=A+g),min(A,A+g)<=benefit<max(A,A+g))

    def test_release_counterexample(self):
        correction=release_correction([0.],[1.],np.full((2,2),2.),[2.,2.],np.ones((2,2)),0)
        np.testing.assert_array_equal(3+correction,np.full((2,2),4.))
        self.assertEqual(max(3,0+2),3)  # Ignoring the generation start underestimates completion by one second.


if __name__=='__main__':unittest.main()
