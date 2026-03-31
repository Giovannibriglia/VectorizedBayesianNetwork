import time

import numpy as np
import pytest
from stress_test.inference import ExactInferenceEngine
from stress_test.metrics import js_divergence_normalized, kl_divergence, wasserstein_1d
from stress_test.queries import label_seen
from stress_test.synthetic_ground_truth import SyntheticRLGroundTruth
from stress_test.timeout_utils import run_with_timeout


def test_reward_probs_sum_to_one():
    gt = SyntheticRLGroundTruth(state_dim=2, cardinality=5, seed=0)
    states = np.array([[0, 1], [2, 3]], dtype=np.int64)
    actions = np.array([1, 4], dtype=np.int64)
    probs = gt.get_reward_probs(states, actions)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_forward_sampling_bounds():
    gt = SyntheticRLGroundTruth(state_dim=3, cardinality=7, seed=1)
    df = gt.sample(n_samples=50, seed=2)
    assert (df.values >= 0).all()
    assert (df.values < 7).all()


def test_metrics_finite():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.1, 0.4, 0.5])
    assert np.isfinite(kl_divergence(p, q))
    assert np.isfinite(js_divergence_normalized(p, q))
    assert np.isfinite(wasserstein_1d(p, q))


def test_seen_unseen_labeling():
    gt = SyntheticRLGroundTruth(state_dim=1, cardinality=3, seed=3)
    df = gt.sample(n_samples=20, seed=4)
    row = df.iloc[0]
    evidence = {"S_0": int(row["S_0"]), "A": int(row["A"])}
    seen, count = label_seen(df, evidence)
    assert seen
    assert count >= 1


def test_exact_reward_inference_matches_cpd():
    gt = SyntheticRLGroundTruth(state_dim=2, cardinality=4, seed=5)
    engine = ExactInferenceEngine(gt)
    evidence = {"S_0": 1, "S_1": 2, "A": 3}
    probs_direct = gt.reward_probs_single([1, 2], 3)
    probs_inf = engine.infer("R", evidence)
    assert np.allclose(probs_direct, probs_inf, atol=1e-8)


def test_timeout_triggers():
    def sleeper():
        time.sleep(0.2)

    _, timed_out = run_with_timeout(sleeper, timeout_sec=0.05)
    assert timed_out


def test_vbn_batch_inference_order():
    torch = pytest.importorskip("torch")
    from stress_test.dag import build_rl_dag
    from stress_test.models import VBNModelWrapper

    gt = SyntheticRLGroundTruth(state_dim=1, cardinality=3, seed=7)
    df = gt.sample(n_samples=200, seed=8)
    dag = build_rl_dag(1)
    wrapper = VBNModelWrapper(
        dag=dag,
        cardinality=3,
        seed=7,
        device="cpu",
        inference_method="importance_sampling",
        inference_samples=64,
        fit_epochs=2,
        fit_batch_size=64,
        hidden_dims=[16],
        cpd_sample_n=128,
    )
    wrapper.fit(df)
    ev1 = {"S_0": 0, "A": 0}
    ev2 = {"S_0": 2, "A": 1}

    torch.manual_seed(123)
    batch_a = wrapper.infer_batch("R", [ev1, ev2])
    torch.manual_seed(123)
    batch_b = wrapper.infer_batch("R", [ev2, ev1])
    assert np.allclose(batch_a[0], batch_b[1], atol=0.3)
    assert np.allclose(batch_a[1], batch_b[0], atol=0.3)
