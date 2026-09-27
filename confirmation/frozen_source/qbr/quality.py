"""Prompt-conditioned quality distributions and target-integrated calibration."""
import re
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import betainc
from scipy.integrate import quad
from scipy.stats import beta
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

class PromptDistribution:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
            max_features=4000, sublinear_tf=True, dtype=np.float32)

    @staticmethod
    def lengths(prompts):
        return csr_matrix(np.log1p([[len(p), len(p.split()), len(re.findall(r'\d', p)),
                                     p.count('\n')] for p in prompts]), dtype=np.float32)

    def features(self, prompts, fit=False):
        text = self.vectorizer.fit_transform(prompts) if fit else self.vectorizer.transform(prompts)
        return hstack((text, self.lengths(prompts)), format='csr', dtype=np.float32)

    def fit(self, prompts, seconds):
        self.seconds = np.asarray(seconds, dtype=float)
        assert len(prompts) == len(self.seconds) and np.all(self.seconds > 0)
        x = self.features(prompts, fit=True)
        self.forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
            max_features=1.0, random_state=20260920, n_jobs=4)
        self.forest.fit(x, np.log(self.seconds))
        leaves = self.forest.apply(x)
        self.members = [{int(leaf): np.flatnonzero(leaves[:, tree] == leaf)
                         for leaf in np.unique(leaves[:, tree])}
                        for tree in range(leaves.shape[1])]
        return self

    def weights(self, prompts):
        leaves = self.forest.apply(self.features(prompts))
        weights = np.zeros((len(prompts), len(self.seconds)))
        for i, row in enumerate(leaves):
            for tree, leaf in enumerate(row):
                members = self.members[tree][int(leaf)]
                weights[i, members] += 1/(len(row)*len(members))
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
        assert np.all(weights >= 0)
        return weights

    def moments(self, prompts):
        weights = self.weights(prompts)
        mean, second = weights @ self.seconds, weights @ self.seconds**2
        assert np.all(second >= mean**2-1e-9)
        return mean, second

class PromptQuality(PromptDistribution):
    def fit(self, prompts, scores):
        self.scores = np.asarray(scores, dtype=float)
        assert self.scores.shape == (len(prompts), 2)
        assert np.all((0 <= self.scores) & (self.scores <= 1))
        # The inherited weight builder uses this only to determine sample count.
        self.seconds = np.ones(len(prompts))
        x = self.features(prompts, fit=True)
        self.forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
            max_features=1., random_state=20260920, n_jobs=4)
        self.forest.fit(x, self.scores)
        leaves = self.forest.apply(x)
        self.members = [{int(leaf): np.flatnonzero(leaves[:, tree] == leaf)
                         for leaf in np.unique(leaves[:, tree])}
                        for tree in range(leaves.shape[1])]
        return self


def hinge_product(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.maximum(a, b)
    zeroth = 1 - betainc(7, 2.5, m)
    first = 7/9.5 * (1 - betainc(8, 2.5, m))
    second = 7*8/(9.5*10.5) * (1 - betainc(9, 2.5, m))
    return second - (a+b)*first + a*b*zeroth


def benefit_gram(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 144 * (hinge_product(a[:, 0, None], b[None, :, 0])
                  - hinge_product(a[:, 0, None], b[None, :, 1])
                  - hinge_product(a[:, 1, None], b[None, :, 0])
                  + hinge_product(a[:, 1, None], b[None, :, 1]))


def integrated_loss(training, weights, outcomes):
    weights = np.asarray(weights)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
    assert np.all(weights >= 0) and weights.shape == (len(outcomes), len(training))
    gram = benefit_gram(training, training)
    cross = benefit_gram(training, outcomes)
    diagonal = np.diag(benefit_gram(outcomes, outcomes))
    loss = np.einsum('ij,jk,ik->i', weights, gram, weights) - 2*np.einsum('ij,ji->i', weights, cross) + diagonal
    assert np.all(np.isfinite(loss)) and np.min(loss) >= -1e-10
    return np.maximum(loss, 0.)


def check_integrals():
    training = np.array([[0., 1.], [.7, .9], [.95, .2], [1., 1.]])
    truth = np.array([[1., 0.], [.4, .5], [.7, .9]])
    weights = np.array([[.25]*4, [.1, .2, .3, .4], [0., 1., 0., 0.]])
    exact = integrated_loss(training, weights, truth)
    observed = []
    for row, w in zip(truth, weights):
        def integrand(q):
            predicted = 12*np.dot(w, np.maximum(q-training[:, 0], 0)-np.maximum(q-training[:, 1], 0))
            actual = 12*(max(q-row[0], 0)-max(q-row[1], 0))
            return (predicted-actual)**2*beta.pdf(q, 7, 2.5)
        points = np.unique(np.r_[training.ravel(), row])
        observed.append(quad(integrand, 0., 1., points=points, epsabs=1e-10)[0])
    np.testing.assert_allclose(exact, observed, atol=1e-9, rtol=1e-10)
    assert exact[2] < 1e-10
    return {'hinge_integral_cases': 3, 'independent_quadrature_passed': True, 'exact_identity_zero_loss': True}


def fit_mixture(forest, uniform, quality):
    """Fit Eq. (9) from out-of-fold weights and paired calibration scores."""
    forest, uniform, quality = map(np.asarray, (forest, uniform, quality))
    n = len(quality)
    assert forest.shape == uniform.shape == (n, n)
    assert np.all(forest >= 0) and np.all(uniform >= 0)
    np.testing.assert_allclose(forest.sum(1), 1, atol=1e-12)
    np.testing.assert_allclose(uniform.sum(1), 1, atol=1e-12)
    assert np.all(np.diag(forest) == 0) and np.all(np.diag(uniform) == 0)
    gram = benefit_gram(quality, quality)
    delta, residual = forest - uniform, np.eye(n) - uniform
    denominator = float(np.einsum('ij,jk,ik->', delta, gram, delta))
    numerator = float(np.einsum('ij,jk,ik->', delta, gram, residual))
    assert denominator >= -1e-8
    return 0. if denominator <= 1e-12 else float(np.clip(numerator / denominator, 0, 1))
