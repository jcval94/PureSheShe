import logging
import numpy as np

from deldel import _numeric_utils
from deldel._logging_utils import verbosity_to_level
from deldel.engine import _predict_labels


def test_verbosity_to_level_matches_legacy_behavior():
    assert verbosity_to_level(-1) == verbosity_to_level(0) == logging.WARNING
    assert verbosity_to_level(1) == logging.INFO
    assert verbosity_to_level(2) == verbosity_to_level(3) == logging.DEBUG


def _legacy_standardize_matrix(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    return Z, mu, sd


def _legacy_destandardize_quadratic(Qz, rz, cz, mu, sd):
    D = np.diag(1.0 / sd)
    Qx = D @ Qz @ D
    r0 = D @ rz
    rx = r0 - 2.0 * (Qx @ mu)
    cx = float(mu.T @ Qx @ mu - r0.T @ mu + cz)
    return Qx, rx, cx


def _legacy_unpack_quadratic_parameters(theta, idx, d):
    Qz = np.zeros((d, d))
    for i in range(d):
        Qz[i, i] = theta[idx["diag"][i]]
    for k, (i, j) in enumerate(idx["pairs"]):
        Qz[i, j] = Qz[j, i] = 0.5 * theta[idx["off"][k]]
    rz = theta[idx["lin"][0] : idx["lin"][0] + d]
    cz = theta[idx["c"]]
    return Qz, rz, float(cz)


def test_numeric_helpers_match_legacy_implementations():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(10, 3))
    Z_new, mu_new, sd_new = _numeric_utils.standardize_matrix(X)
    Z_old, mu_old, sd_old = _legacy_standardize_matrix(X)

    np.testing.assert_allclose(Z_new, Z_old)
    np.testing.assert_allclose(mu_new, mu_old)
    np.testing.assert_allclose(sd_new, sd_old)

    Qz = rng.normal(size=(3, 3))
    Qz = 0.5 * (Qz + Qz.T)  # ensure symmetry
    rz = rng.normal(size=3)
    cz = float(rng.normal())

    destandardized_new = _numeric_utils.destandardize_quadratic(Qz, rz, cz, mu_new, sd_new)
    destandardized_old = _legacy_destandardize_quadratic(Qz, rz, cz, mu_old, sd_old)
    for new, old in zip(destandardized_new, destandardized_old):
        np.testing.assert_allclose(new, old)

    idx = {
        "diag": [0, 1, 2],
        "pairs": [(0, 1), (0, 2), (1, 2)],
        "off": [3, 4, 5],
        "lin": [6],
        "c": 9,
    }
    theta = rng.normal(size=10)
    unpacked_new = _numeric_utils.unpack_quadratic_parameters(theta, idx, d=3)
    unpacked_old = _legacy_unpack_quadratic_parameters(theta, idx, d=3)

    for new, old in zip(unpacked_new, unpacked_old):
        np.testing.assert_allclose(new, old)


def test_predict_labels_verbosity_controls_output(capsys):
    class _DummyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = np.zeros((2, 3))
    model = _DummyModel()

    _predict_labels(model, X, verbosity=0)
    captured = capsys.readouterr()
    assert captured.out == ""

    _predict_labels(model, X, verbosity=1)
    captured = capsys.readouterr()
    assert "Usando modelo" in captured.out
