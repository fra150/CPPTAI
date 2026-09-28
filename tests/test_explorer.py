import sys
import os
import unittest
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpptai.exploration import ExplorerEngine, TrajectoryAnalyzer
from cpptai.types import ExplorationTrajectory, AnalyzerPrepared


def _make_fake_trajectories(n: int = 3) -> List[ExplorationTrajectory]:
    """Create n deterministic fake trajectories for testing."""
    results = []
    for i in range(n):
        traj = ExplorationTrajectory(
            id=f"T{i+1}",
            interpretation=f"Interpretation {i+1}: framing the problem as {['optimization', 'constraint', 'tradeoff'][i % 3]}"
                if i < 3 else f"Interpretation {i+1}: alternative framing with different keywords",
            confidence=0.3 + (i % 5) * 0.15,
            reasoning_path=[f"Step {j}" for j in range(3)],
            noise_seed=i * 42,
        )
        results.append(traj)
    return results


class TestExplorerEngine(unittest.TestCase):

    def test_explore_returns_trajectories(self):
        engine = ExplorerEngine(num_trajectories=3, seed=0)
        result = engine.explore("test problem")
        self.assertEqual(len(result), 3)
        for traj in result:
            self.assertIsInstance(traj, ExplorationTrajectory)
            self.assertGreaterEqual(traj.confidence, 0.0)
            self.assertLessEqual(traj.confidence, 1.0)

    def test_select_best_returns_highest_confidence(self):
        engine = ExplorerEngine(num_trajectories=3, seed=0)
        result = engine.explore("test problem")
        best = engine.select_best(result)
        self.assertEqual(best.confidence, max(t.confidence for t in result))

    def test_deterministic_with_seed(self):
        engine1 = ExplorerEngine(num_trajectories=3, seed=42)
        engine2 = ExplorerEngine(num_trajectories=3, seed=42)
        result1 = engine1.explore("deterministic problem")
        result2 = engine2.explore("deterministic problem")
        for t1, t2 in zip(result1, result2):
            self.assertEqual(t1.interpretation, t2.interpretation)
            self.assertEqual(t1.confidence, t2.confidence)

    def test_diverse_interpretations(self):
        engine = ExplorerEngine(num_trajectories=4, noise_level=1.0, temperature=1.0, seed=0)
        result = engine.explore("diverse test")
        noise_seeds = [t.noise_seed for t in result]
        unique_seeds = len(set(noise_seeds))
        self.assertEqual(unique_seeds, 4, "Each trajectory must use a different noise seed")
        diverse_id = len(set(t.id for t in result))
        self.assertEqual(diverse_id, 4, "Each trajectory must have a unique id")

    def test_trajectory_summary_dict(self):
        engine = ExplorerEngine(num_trajectories=1, seed=0)
        result = engine.explore("summary test")
        d = result[0].to_summary_dict()
        self.assertIn("id", d)
        self.assertIn("interpretation", d)
        self.assertIn("confidence", d)
        self.assertIn("reasoning_steps", d)
        self.assertIn("novelty", d)

    def test_inject_noise_uses_different_lens(self):
        engine = ExplorerEngine(seed=0)
        framing1 = engine._inject_noise("noise test", seed=1)
        framing2 = engine._inject_noise("noise test", seed=999)
        self.assertNotEqual(framing1, framing2)


class TestTrajectoryAnalyzer(unittest.TestCase):

    def test_analyze_returns_prepared(self):
        trajectories = _make_fake_trajectories(3)
        analyzer = TrajectoryAnalyzer()
        prepared = analyzer.analyze(trajectories, "original problem")
        self.assertIsInstance(prepared, AnalyzerPrepared)
        self.assertTrue(prepared.enriched_problem)
        self.assertGreaterEqual(prepared.aggregate_confidence, 0.0)
        self.assertLessEqual(prepared.aggregate_confidence, 1.0)

    def test_top_interpretations_count(self):
        trajectories = _make_fake_trajectories(6)
        analyzer = TrajectoryAnalyzer(ensemble_size=2)
        prepared = analyzer.analyze(trajectories, "original problem")
        self.assertLessEqual(len(prepared.top_interpretations), 2)

    def test_empty_trajectories_fallback(self):
        analyzer = TrajectoryAnalyzer()
        prepared = analyzer.analyze([], "fallback problem")
        self.assertEqual(prepared.enriched_problem, "fallback problem")
        self.assertEqual(prepared.trajectories_used, 0)

    def test_enriched_problem_contains_original(self):
        trajectories = _make_fake_trajectories(3)
        analyzer = TrajectoryAnalyzer()
        prepared = analyzer.analyze(trajectories, "unique original fragment")
        self.assertIn("unique original fragment", prepared.enriched_problem)

    def test_confidence_is_weighted_average(self):
        trajectories = _make_fake_trajectories(3)
        analyzer = TrajectoryAnalyzer()
        prepared = analyzer.analyze(trajectories, "original problem")
        confs = [t.confidence for t in trajectories]
        self.assertGreaterEqual(prepared.aggregate_confidence, min(confs))
        self.assertLessEqual(prepared.aggregate_confidence, max(confs))

    def test_identify_patterns(self):
        trajectories = [
            ExplorationTrajectory(
                id="T1", interpretation="shared keyword analysis pattern",
                confidence=0.5, reasoning_path=[], noise_seed=0,
            ),
            ExplorationTrajectory(
                id="T2", interpretation="shared keyword different framing",
                confidence=0.5, reasoning_path=[], noise_seed=1,
            ),
            ExplorationTrajectory(
                id="T3", interpretation="shared keyword third trajectory",
                confidence=0.5, reasoning_path=[], noise_seed=2,
            ),
        ]
        analyzer = TrajectoryAnalyzer()
        prepared = analyzer.analyze(trajectories, "original problem")
        self.assertIn("shared", prepared.patterns_identified)
        self.assertIn("keyword", prepared.patterns_identified)


class TestExplorerScale(unittest.TestCase):

    def test_adaptive_trajectory_count_short(self):
        engine = ExplorerEngine(num_trajectories=5, adaptive=True)
        n = engine._compute_optimal_trajectories("Short problem.")
        self.assertLessEqual(n, 5)

    def test_adaptive_trajectory_count_long(self):
        engine = ExplorerEngine(num_trajectories=20, adaptive=True)
        long_problem = "word " * 300
        n = engine._compute_optimal_trajectories(long_problem)
        self.assertGreaterEqual(n, 8)

    def test_batch_explore(self):
        engine = ExplorerEngine(num_trajectories=3, adaptive=False)
        problems = ["Problem A", "Problem B"]
        results = engine.explore_batch(problems)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 3)
        self.assertEqual(len(results[1]), 3)

    def test_cache_hit(self):
        engine = ExplorerEngine(num_trajectories=3, seed=42, cache_results=True)
        r1 = engine.explore("Cached test")
        r2 = engine.explore("Cached test")
        self.assertEqual(
            [t.interpretation for t in r1],
            [t.interpretation for t in r2],
        )

    def test_cache_miss_with_different_problem(self):
        engine = ExplorerEngine(num_trajectories=3, seed=42, cache_results=True)
        r1 = engine.explore("First problem")
        r2 = engine.explore("Second problem")
        self.assertNotEqual(
            [t.interpretation for t in r1],
            [t.interpretation for t in r2],
        )

    def test_cache_clear(self):
        engine = ExplorerEngine(num_trajectories=3, seed=42, cache_results=True)
        r1 = engine.explore("Cache clear test")
        engine.clear_cache()
        r2 = engine.explore("Cache clear test")
        self.assertEqual(
            [t.interpretation for t in r1],
            [t.interpretation for t in r2],
        )

    def test_adaptive_ignores_when_disabled(self):
        engine = ExplorerEngine(num_trajectories=10, adaptive=False)
        n = engine._compute_optimal_trajectories("word " * 300)
        self.assertEqual(n, 10)

    def test_adaptive_caps_at_num_trajectories(self):
        engine = ExplorerEngine(num_trajectories=4, adaptive=True)
        n = engine._compute_optimal_trajectories("word " * 300)
        self.assertLessEqual(n, 4)

    def test_different_noise_levels_per_trajectory(self):
        engine = ExplorerEngine(num_trajectories=5, noise_level=0.6, adaptive=False)
        noise_levels = []
        for i in range(5):
            traj = engine._generate_single_trajectory("test problem", i)
            noise_levels.append(traj.metadata.get("noise_level", 0.0))
        self.assertEqual(len(set(noise_levels)), 5)
        self.assertTrue(all(0.3 <= nl <= 0.9 for nl in noise_levels))

    def test_sixteen_lenses_available(self):
        from cpptai.exploration import EXPLORER_LENSES
        self.assertEqual(len(EXPLORER_LENSES), 16)

    def test_explore_batch_empty(self):
        engine = ExplorerEngine(num_trajectories=3)
        results = engine.explore_batch([])
        self.assertEqual(len(results), 0)

    def test_diversity_weight_scoring(self):
        engine_low = ExplorerEngine(num_trajectories=3, diversity_weight=0.0)
        engine_high = ExplorerEngine(num_trajectories=3, diversity_weight=1.0)
        r_low = engine_low.explore("Diversity test problem")
        r_high = engine_high.explore("Diversity test problem")
        for t_low, t_high in zip(r_low, r_high):
            self.assertIsInstance(t_low.confidence, float)
            self.assertIsInstance(t_high.confidence, float)


if __name__ == "__main__":
    unittest.main()
