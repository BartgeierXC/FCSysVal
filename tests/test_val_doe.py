import numpy as np
import pandas as pd
import pytest

from src.utils.val_doe import (
    CoverageResult,
    DistanceResult,
    GPRUncertaintyResult,
    cal_val_cov,
    cal_val_dist,
    cal_val_gpr_uncertainty,
    extract_bounds,
    generate_lhs,
    generate_sobol,
    propose_new_val_experiments,
    propose_val_experiments,
)


class TestCalValDist:
    def _make_frames(self):
        rng = np.random.default_rng(0)
        application_df = pd.DataFrame(
            {
                "temperature": rng.uniform(0.0, 100.0, size=200),
                "load": rng.uniform(0.0, 1.0, size=200),
            }
        )
        validation_df = pd.DataFrame(
            {
                "temperature": rng.uniform(0.0, 100.0, size=30),
                "load": rng.uniform(0.0, 1.0, size=30),
            }
        )
        return validation_df, application_df

    def test_returns_coverage_result_with_expected_shape(self):
        validation_df, application_df = self._make_frames()
        result = cal_val_dist(validation_df, application_df)

        assert isinstance(result, DistanceResult)
        assert result.r.shape == (len(application_df),)
        assert result.columns == ["temperature", "load"]

    def test_distances_are_non_negative(self):
        validation_df, application_df = self._make_frames()
        result = cal_val_dist(validation_df, application_df)
        assert np.all(result.r >= 0.0)

    def test_d_max_and_q95_consistent_with_distances(self):
        validation_df, application_df = self._make_frames()
        result = cal_val_dist(validation_df, application_df)

        assert result.d_max == pytest.approx(result.r.max())
        assert result.q95 == pytest.approx(np.quantile(result.r, 0.95))
        assert result.q95 <= result.d_max
        assert result.eta > 0.0

    def test_zero_distance_when_application_equals_validation(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0]})
        result = cal_val_dist(validation_df=df, application_df=df, scale=None)
        assert result.d_max == pytest.approx(0.0)
        assert np.allclose(result.r, 0.0)
        assert result.eta == pytest.approx(0.0)  # d_max=0 → eta defined as 0

    def test_known_distance_without_scaling(self):
        validation_df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        application_df = pd.DataFrame({"x": [3.0], "y": [4.0]})
        result = cal_val_dist(validation_df, application_df, scale=None)
        assert result.d_max == pytest.approx(5.0)
        # n_val (1) >= n_app (1): the empirical optimal reference is degenerate
        # (no room to place an optimal design), so eta is defined as 0.
        assert result.eta == pytest.approx(0.0)

    def test_minmax_scaling_uses_box_and_envelope_bounds(self):
        validation_df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        application_df = pd.DataFrame({"x": [10.0], "y": [6.0]})
        bounds = {
            "x": (0.0, 20.0),
            "y": np.array([[0.0, 0.0, 2.0], [5.0, 0.0, 10.0]]),
        }

        result = cal_val_dist(validation_df, application_df, bounds=bounds)

        assert result.d_max == pytest.approx(np.hypot(10.0 / 20.0, 6.0 / 10.0))

    def test_std_scaling_remains_derived_from_application(self):
        validation_df, application_df = self._make_frames()
        bounds = {
            "temperature": (0.0, 1000.0),
            "load": (0.0, 100.0),
        }

        result_with_bounds = cal_val_dist(
            validation_df, application_df, scale="std", bounds=bounds
        )
        result_without_bounds = cal_val_dist(
            validation_df, application_df, scale="std"
        )

        assert np.array_equal(result_with_bounds.r, result_without_bounds.r)

    def test_minmax_bounds_must_match_frame_columns(self):
        validation_df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        application_df = pd.DataFrame({"x": [1.0], "y": [1.0]})
        with pytest.raises(ValueError, match="bounds must contain exactly"):
            cal_val_dist(
                validation_df,
                application_df,
                bounds={"x": (0.0, 1.0)},
            )

    def test_eta_in_unit_interval_for_dense_application(self):
        validation_df, application_df = self._make_frames()
        result = cal_val_dist(validation_df, application_df)
        # Empirical efficiency d_max_opt / d_max is a ratio of covering radii
        # measured on the same grid; the actual design cannot beat the optimal
        # reference, so eta lies in (0, 1].
        assert 0.0 < result.eta <= 1.0 + 1e-9

    def test_eta_near_one_for_near_optimal_validation(self):
        from src.utils.val_doe import _farthest_first_indices

        rng = np.random.default_rng(3)
        application_df = pd.DataFrame(
            {
                "temperature": rng.uniform(0.0, 100.0, size=400),
                "load": rng.uniform(0.0, 1.0, size=400),
            }
        )
        # Build a validation set that is itself a near-optimal k-center design
        # drawn from the application grid: eta should then be close to 1.
        app_scaled = (
            application_df.to_numpy(float)
            / (application_df.max(axis=0) - application_df.min(axis=0)).to_numpy(float)
        )
        seed = int(np.argmin(np.linalg.norm(app_scaled - app_scaled.mean(axis=0), axis=1)))
        idx = _farthest_first_indices(app_scaled, 25, seed)
        validation_df = application_df.iloc[idx].reset_index(drop=True)

        result = cal_val_dist(validation_df, application_df)
        assert result.eta == pytest.approx(1.0, abs=0.05)

    def test_column_order_is_aligned(self):
        validation_df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        application_df = pd.DataFrame({"y": [4.0], "x": [3.0]})
        result = cal_val_dist(validation_df, application_df, scale=None)
        assert result.columns == ["x", "y"]
        assert result.d_max == pytest.approx(5.0)

    def test_mismatched_columns_raise(self):
        validation_df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        application_df = pd.DataFrame({"x": [1.0], "z": [2.0]})
        with pytest.raises(ValueError):
            cal_val_dist(validation_df, application_df)

    def test_empty_frame_raises(self):
        validation_df = pd.DataFrame({"x": [], "y": []})
        application_df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(ValueError):
            cal_val_dist(validation_df, application_df)

    def test_non_dataframe_input_raises(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(ValueError):
            cal_val_dist([[1.0, 2.0]], df)

    def test_unknown_scale_raises(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        with pytest.raises(ValueError):
            cal_val_dist(df, df, scale="bogus")


class TestCalValCov:
    def test_returns_coverage_result_with_expected_fields(self):
        rng = np.random.default_rng(0)
        application_df = pd.DataFrame(
            {
                "temperature": rng.uniform(0.0, 100.0, size=200),
                "load": rng.uniform(0.0, 1.0, size=200),
            }
        )
        validation_df = pd.DataFrame(
            {
                "temperature": rng.uniform(0.0, 100.0, size=30),
                "load": rng.uniform(0.0, 1.0, size=30),
            }
        )
        result = cal_val_cov(validation_df, application_df)

        assert isinstance(result, CoverageResult)
        assert result.vol_val > 0.0
        assert result.vol_app > 0.0
        assert result.eta > 0.0

    def test_eta_matches_analytic_quarter_square_ratio(self):
        # application_df spans the unit square [0, 1] x [0, 1] -> area 1.
        application_df = pd.DataFrame(
            {"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]}
        )
        # validation_df spans the quarter square [0, 0.5] x [0, 0.5] -> area 0.25.
        validation_df = pd.DataFrame(
            {"x": [0.0, 0.5, 0.0, 0.5], "y": [0.0, 0.0, 0.5, 0.5]}
        )
        result = cal_val_cov(validation_df, application_df, scale=None)

        assert result.vol_app == pytest.approx(1.0)
        assert result.vol_val == pytest.approx(0.25)
        assert result.eta == pytest.approx(0.25)

    def test_eta_is_one_when_validation_equals_application(self):
        df = pd.DataFrame(
            {"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]}
        )
        result = cal_val_cov(validation_df=df, application_df=df, scale=None)
        assert result.eta == pytest.approx(1.0)

    def test_one_dimensional_volume_is_range(self):
        validation_df = pd.DataFrame({"x": [1.0, 3.0, 5.0]})
        application_df = pd.DataFrame({"x": [0.0, 10.0]})
        result = cal_val_cov(validation_df, application_df, scale=None)

        assert result.vol_val == pytest.approx(4.0)  # 5 - 1
        assert result.vol_app == pytest.approx(10.0)  # 10 - 0
        assert result.eta == pytest.approx(0.4)

    def test_eta_invariant_to_scale(self):
        rng = np.random.default_rng(1)
        application_df = pd.DataFrame(
            {
                "temperature": rng.uniform(0.0, 500.0, size=100),
                "load": rng.uniform(0.0, 1.0, size=100),
            }
        )
        validation_df = pd.DataFrame(
            {
                "temperature": rng.uniform(0.0, 500.0, size=20),
                "load": rng.uniform(0.0, 1.0, size=20),
            }
        )
        result_minmax = cal_val_cov(validation_df, application_df, scale="minmax")
        result_std = cal_val_cov(validation_df, application_df, scale="std")
        result_none = cal_val_cov(validation_df, application_df, scale=None)

        assert result_minmax.eta == pytest.approx(result_none.eta)
        assert result_std.eta == pytest.approx(result_none.eta)
        # Absolute volumes do depend on the scaling.
        assert result_minmax.vol_app != pytest.approx(result_none.vol_app)

    def test_column_order_is_aligned(self):
        validation_df = pd.DataFrame(
            {"x": [0.0, 0.5, 0.0, 0.5], "y": [0.0, 0.0, 0.5, 0.5]}
        )
        application_df = pd.DataFrame(
            {"y": [0.0, 0.0, 1.0, 1.0], "x": [0.0, 1.0, 0.0, 1.0]}
        )
        result = cal_val_cov(validation_df, application_df, scale=None)
        assert result.eta == pytest.approx(0.25)

    def test_mismatched_columns_raise(self):
        validation_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        application_df = pd.DataFrame({"x": [0.0, 1.0], "z": [0.0, 1.0]})
        with pytest.raises(ValueError):
            cal_val_cov(validation_df, application_df)

    def test_empty_frame_raises(self):
        validation_df = pd.DataFrame({"x": [], "y": []})
        application_df = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]})
        with pytest.raises(ValueError):
            cal_val_cov(validation_df, application_df)

    def test_non_dataframe_input_raises(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]})
        with pytest.raises(ValueError):
            cal_val_cov([[1.0, 2.0]], df)

    def test_too_few_points_for_dimensionality_raises(self):
        # 2-D hull needs at least 3 non-collinear points.
        validation_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        application_df = pd.DataFrame(
            {"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]}
        )
        with pytest.raises(ValueError):
            cal_val_cov(validation_df, application_df)

    def test_collinear_points_raise(self):
        validation_df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
        application_df = pd.DataFrame(
            {"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]}
        )
        with pytest.raises(ValueError):
            cal_val_cov(validation_df, application_df)


class TestProposeValExperiments:
    def test_proposes_requested_number(self):
        rng = np.random.default_rng(1)
        application_df = pd.DataFrame(
            {"a": rng.uniform(0, 1, 100), "b": rng.uniform(0, 1, 100)}
        )
        # pool must be at least as large as the requested number
        validation_df = pd.DataFrame(
            {"a": rng.uniform(0, 1, 20), "b": rng.uniform(0, 1, 20)}
        )
        proposals = propose_val_experiments(
            application_df, validation_df, n=3
        )
        assert len(proposals) == 3
        assert list(proposals.columns) == ["a", "b"]

    def test_bounds_scale_by_bound_range(self):
        # With bounds, distances are normalized by the bound range instead of
        # the range of application_df.
        rng = np.random.default_rng(4)
        application_df = pd.DataFrame(
            {"a": rng.uniform(0, 1, 100), "b": rng.uniform(0, 1, 100)}
        )
        validation_df = pd.DataFrame(
            {"a": rng.uniform(0, 1, 5), "b": rng.uniform(0, 1, 5)}
        )
        bounds = {"a": (0.0, 1.0), "b": (0.0, 4.0)}
        proposals = propose_val_experiments(
            application_df, validation_df, n=4, bounds=bounds
        )
        expected = propose_val_experiments(
            application_df / [1.0, 4.0], validation_df / [1.0, 4.0], n=4, scale=None
        )
        np.testing.assert_allclose(
            proposals.to_numpy(), expected.to_numpy() * [1.0, 4.0]
        )

    def test_selected_subset_minimises_coverage(self):
        # New semantics: proposals come from application_df (not validation_df).
        # validation_df = 4 corners (already done); application_df = 3×3 grid.
        # With n=1 the worst-covered app point is the centre (0.5, 0.5),
        # which is sqrt(0.5)≈0.707 from every corner — farther than any
        # edge midpoint (distance 0.5). The algorithm must propose (0.5, 0.5).
        application_df = pd.DataFrame({
            "x": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "y": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
        })
        validation_df = pd.DataFrame({
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        })
        proposals = propose_val_experiments(
            application_df, validation_df, n=1, scale=None
        )
        assert proposals.iloc[0]["x"] == pytest.approx(0.5)
        assert proposals.iloc[0]["y"] == pytest.approx(0.5)

    def test_first_proposal_covers_worst_region(self):
        # propose_val_experiments selects from application_df (the new experiment
        # pool) the n points that best cover the application domain given the
        # already-done experiments in validation_df.
        # validation_df = centre (5, 5); application_df = 4 corners.
        # All corners are equidistant (sqrt(50)) from the centre.
        # With n=1 the algorithm picks the first-encountered corner (argmax
        # tiebreak on index 0 → (0, 0)).
        application_df = pd.DataFrame({
            "x": [0.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 10.0, 10.0],
        })
        validation_df = pd.DataFrame({"x": [5.0], "y": [5.0]})
        proposals = propose_val_experiments(
            application_df, validation_df, n=1, scale=None
        )
        assert proposals.iloc[0]["x"] == pytest.approx(0.0)
        assert proposals.iloc[0]["y"] == pytest.approx(0.0)

    def test_invalid_n_proposals_raises(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        with pytest.raises(ValueError):
            propose_val_experiments(df, df, n=0)

    def test_add_df_pool_reduces_dmax_on_application_df(self):
        rng = np.random.default_rng(3)
        application_df = pd.DataFrame(
            {"a": rng.uniform(0, 1, 200), "b": rng.uniform(0, 1, 200)}
        )
        validation_df = pd.DataFrame({"a": [0.1, 0.2], "b": [0.1, 0.2]})
        add_df = pd.DataFrame(
            {"a": rng.uniform(0, 1, 50), "b": rng.uniform(0, 1, 50)}
        )
        proposals = propose_val_experiments(
            application_df, validation_df, n=3, add_df=add_df
        )
        assert len(proposals) == 3
        assert not proposals.duplicated().any()
        merged = proposals.merge(add_df, how="left", indicator=True)
        assert (merged["_merge"] == "both").all()
        before = cal_val_dist(validation_df, application_df).d_max
        after = cal_val_dist(
            pd.concat([validation_df, proposals], ignore_index=True), application_df
        ).d_max
        assert after < before

    def test_add_df_clipped_and_validated(self):
        application_df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
        validation_df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        add_df = pd.DataFrame({"x": [2.0, 1.0], "y": [2.0, 1.0]})
        out = propose_val_experiments(
            application_df, validation_df, n=5, scale=None, add_df=add_df
        )
        assert len(out) == 2
        assert out.iloc[0]["x"] == pytest.approx(2.0)
        with pytest.raises(ValueError):
            propose_val_experiments(
                application_df, validation_df, add_df=add_df.rename(columns={"x": "z"})
            )
        with pytest.raises(ValueError):
            propose_val_experiments(
                application_df, validation_df, add_df=add_df.iloc[0:0]
            )


class TestGenerateLhs:
    def test_box_lhs_shape_and_bounds(self):
        bounds = {"temperature": (0.0, 100.0), "load": (0.0, 1.0)}
        df = generate_lhs(bounds, n=50, seed=0)

        assert list(df.columns) == ["temperature", "load"]
        assert len(df) == 50
        assert df["temperature"].between(0.0, 100.0).all()
        assert df["load"].between(0.0, 1.0).all()

    def test_box_lhs_is_reproducible_with_seed(self):
        bounds = {"a": (0.0, 1.0), "b": (10.0, 20.0)}
        df1 = generate_lhs(bounds, n=20, seed=42)
        df2 = generate_lhs(bounds, n=20, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_box_lhs_spreads_over_unit_intervals(self):
        # A Latin hypercube must place exactly one point per stratum, so the
        # sorted samples land in distinct equal-width bins.
        bounds = {"a": (0.0, 1.0)}
        df = generate_lhs(bounds, n=10, seed=1)
        strata = np.floor(df["a"].to_numpy() * 10).astype(int)
        assert len(np.unique(strata)) == 10

    def test_constraint_lhs_respects_interpolated_envelope(self):
        # 'load' is allowed between 0 and an upper limit that grows with
        # temperature: 0.2 at T=0 up to 1.0 at T=100.
        envelope = np.array(
            [
                [0.0, 0.0, 0.2],
                [100.0, 0.0, 1.0],
            ]
        )
        bounds = {"temperature": (0.0, 100.0), "load": envelope}
        df = generate_lhs(bounds, n=60, seed=3)

        assert list(df.columns) == ["temperature", "load"]
        assert len(df) == 60

        upper = 0.2 + (1.0 - 0.2) * (df["temperature"] / 100.0)
        assert (df["load"] >= -1e-9).all()
        assert (df["load"] <= upper + 1e-9).all()

    def test_constraint_lhs_preserves_independent_marginal(self):
        # The feasible band is very narrow at low temperature (width 0.1) and
        # very wide at high temperature (width 9.0). Plain rejection sampling
        # would cluster samples where the band is wide; the conditional
        # inverse-transform must instead keep the independent variable's
        # marginal near-uniform (constant density along the independent axis).
        envelope = np.array(
            [
                [0.0, 0.0, 0.1],
                [10.0, 0.0, 9.0],
            ]
        )
        bounds = {"temperature": (0.0, 10.0), "load": envelope}
        n = 2000
        df = generate_lhs(bounds, n=n, seed=3)

        counts, _ = np.histogram(df["temperature"], bins=np.linspace(0.0, 10.0, 11))
        expected = n / 10
        # A proper LHS marginal places exactly one point per stratum, so every
        # equal-width bin holds ~expected points. A generous tolerance still
        # rules out the strong wide-band bias of plain rejection sampling
        # (which produced roughly a 6x low/high imbalance).
        assert np.all(np.abs(counts - expected) <= 0.25 * expected)

    def test_constraint_lhs_fills_local_band_at_both_ends(self):
        # With an envelope and no rules the dependent variable must reach across
        # its full local band at both ends of the independent range.
        envelope = np.array(
            [
                [0.0, 0.0, 0.1],
                [10.0, 0.0, 9.0],
            ]
        )
        bounds = {"temperature": (0.0, 10.0), "load": envelope}
        df = generate_lhs(bounds, n=2000, seed=11)

        low = df[df["temperature"] < 0.1]["load"]
        high = df[df["temperature"] > 9.9]["load"]
        # Low-temperature band upper bound is < 0.19 here, so values stay tiny.
        assert low.max() < 0.2
        # High-temperature band is [0, ~8.91]; values must span most of it.
        assert high.max() > 6.0
        assert high.min() < 2.0

    def test_constraint_lhs_feeds_coverage_functions(self):
        envelope = np.array([[0.0, 0.0, 0.5], [100.0, 0.0, 1.0]])
        bounds = {"temperature": (0.0, 100.0), "load": envelope}
        application_df = generate_lhs(bounds, n=80, seed=4)
        validation_df = generate_lhs(bounds, n=10, seed=5)

        result = cal_val_dist(validation_df, application_df)
        assert result.r.shape == (80,)

        proposals = propose_val_experiments(
            application_df, validation_df, n=3
        )
        assert len(proposals) == 3

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            generate_lhs({"a": (0.0, 1.0)}, n=0)

    def test_unsupported_spec_raises(self):
        with pytest.raises(ValueError):
            generate_lhs({"a": "not-a-bound"}, n=5)

    def test_wrong_envelope_shape_raises(self):
        bad = np.array([[0.0, 1.0], [1.0, 2.0]])  # only 2 columns
        bounds = {"t": (0.0, 1.0), "x": bad}
        with pytest.raises(ValueError):
            generate_lhs(bounds, n=5)

    def test_multiple_independent_vars_with_constraint_raises(self):
        envelope = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0), "x": envelope}
        with pytest.raises(ValueError):
            generate_lhs(bounds, n=5)

    def test_reversed_box_bound_raises(self):
        with pytest.raises(ValueError):
            generate_lhs({"a": (1.0, 0.0)}, n=5)

    def test_rules_only_box_bounds_respected(self):
        # a and b are both box bounds; rule enforces a > b
        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        df = generate_lhs(bounds, n=100, seed=7, rules=[lambda d: d["a"] > d["b"]])
        assert (df["a"] > df["b"]).all()

    def test_rules_with_envelope_respected(self):
        envelope = np.array([[0.0, 0.0, 1.0], [100.0, 0.0, 1.0]])
        bounds = {"temperature": (0.0, 100.0), "lo": envelope, "hi": envelope}
        df = generate_lhs(
            bounds,
            n=60,
            seed=8,
            rules=[lambda d: d["hi"] > d["lo"]],
        )
        assert (df["hi"] > df["lo"]).all()

    def test_rules_impossible_constraint_raises(self):
        # a > a is never true → cannot collect enough feasible points
        bounds = {"a": (0.0, 1.0)}
        with pytest.raises(ValueError, match="Could only draw"):
            generate_lhs(
                bounds, n=10, seed=0,
                rules=[lambda d: d["a"] > d["a"] + 1],
                max_oversample=8,
            )

    def test_rules_none_behaves_like_no_rules(self):
        bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        df1 = generate_lhs(bounds, n=30, seed=99, rules=None)
        df2 = generate_lhs(bounds, n=30, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    # --- space_filling tests ---------------------------------------------------

    def test_space_filling_rules_still_respected(self):
        # Rules must be satisfied for every returned row even with space_filling.
        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        df = generate_lhs(
            bounds, n=50, seed=5,
            rules=[lambda d: d["a"] > d["b"] + 0.2],
            space_filling=True,
        )
        assert len(df) == 50
        assert list(df.columns) == ["a", "b"]
        assert (df["a"] > df["b"] + 0.2).all()

    def test_space_filling_improves_spread(self):
        # The minimum pairwise distance (maximin) over the selected points is
        # strictly larger with space_filling=True than with the default first-n
        # selection, demonstrating better coverage quality.
        from sklearn.metrics import pairwise_distances

        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        rule = [lambda d: d["a"] + d["b"] < 1.3]  # cuts off one corner
        n = 40

        df_default = generate_lhs(bounds, n=n, seed=3, rules=rule)
        df_sf = generate_lhs(
            bounds, n=n, seed=3, rules=rule, space_filling=True, pool_factor=30
        )

        def maximin(df):
            D = pairwise_distances(df.to_numpy())
            np.fill_diagonal(D, np.inf)
            return D.min(axis=1).min()

        assert maximin(df_sf) > maximin(df_default)

    def test_space_filling_default_unchanged(self):
        # space_filling=False must reproduce the current first-n behavior.
        bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        rules = [lambda d: d["x"] > d["y"]]
        df1 = generate_lhs(bounds, n=30, seed=17, rules=rules)
        df2 = generate_lhs(bounds, n=30, seed=17, rules=rules, space_filling=False)
        pd.testing.assert_frame_equal(df1, df2)

    def test_space_filling_reproducible_with_seed(self):
        # Two identical seeded calls with space_filling=True must return the
        # same frame.
        bounds = {"a": (0.0, 2.0), "b": (0.0, 2.0)}
        rules = [lambda d: d["a"] + d["b"] < 3.0]
        df1 = generate_lhs(
            bounds, n=25, seed=42, rules=rules,
            space_filling=True, pool_factor=20,
        )
        df2 = generate_lhs(
            bounds, n=25, seed=42, rules=rules,
            space_filling=True, pool_factor=20,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_space_filling_with_envelope_and_rules(self):
        # Both interpolated-limit envelope and rules together with space_filling.
        envelope = np.array([[0.0, 0.0, 1.0], [100.0, 0.0, 1.0]])
        bounds = {"temperature": (0.0, 100.0), "lo": envelope, "hi": envelope}
        df = generate_lhs(
            bounds, n=30, seed=9,
            rules=[lambda d: d["hi"] > d["lo"] + 0.05],
            space_filling=True,
        )
        assert len(df) == 30
        assert (df["hi"] > df["lo"] + 0.05).all()


class TestProposeNewValExperiments:
    def test_from_scratch_returns_requested_count(self):
        rng = np.random.default_rng(0)
        app = pd.DataFrame({"x": rng.uniform(0, 1, 200), "y": rng.uniform(0, 1, 200)})
        result = propose_new_val_experiments(app, n=10)
        assert len(result) == 10
        assert list(result.columns) == ["x", "y"]

    def test_from_scratch_rows_come_from_application_df(self):
        rng = np.random.default_rng(1)
        app = pd.DataFrame({"x": rng.uniform(0, 1, 100), "y": rng.uniform(0, 1, 100)})
        result = propose_new_val_experiments(app, n=5)
        for _, row in result.iterrows():
            match = ((app["x"] == row["x"]) & (app["y"] == row["y"])).any()
            assert match, "proposed row not found in application_df"

    def test_from_scratch_improves_spread_over_random(self):
        # Farthest-first should produce a larger maximin distance than a random
        # sample of the same size from the same pool.
        from sklearn.metrics import pairwise_distances

        rng = np.random.default_rng(7)
        app = pd.DataFrame({"x": rng.uniform(0, 1, 500), "y": rng.uniform(0, 1, 500)})
        n = 20
        result = propose_new_val_experiments(app, n=n)
        random_sample = app.sample(n, random_state=7)

        def maximin(df):
            D = pairwise_distances(df.to_numpy())
            np.fill_diagonal(D, np.inf)
            return D.min(axis=1).min()

        assert maximin(result) > maximin(random_sample)

    def test_with_existing_validation_augments_correctly(self):
        rng = np.random.default_rng(2)
        app = pd.DataFrame({"x": rng.uniform(0, 1, 100), "y": rng.uniform(0, 1, 100)})
        val = pd.DataFrame({"x": [0.1], "y": [0.1]})
        result = propose_new_val_experiments(app, n=3, validation_df=val)
        assert len(result) == 3

    def test_empty_application_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            propose_new_val_experiments(pd.DataFrame({"x": [], "y": []}), n=1)

    def test_invalid_n_raises(self):
        app = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        with pytest.raises(ValueError):
            propose_new_val_experiments(app, n=0)

    def test_mismatched_columns_raises(self):
        app = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        val = pd.DataFrame({"x": [0.5], "z": [0.5]})
        with pytest.raises(ValueError):
            propose_new_val_experiments(app, n=1, validation_df=val)

    def test_bounds_scale_by_bound_range(self):
        # With bounds, distances are normalized by the bound range instead of
        # the range of app: same selection as unscaled data divided by it.
        rng = np.random.default_rng(3)
        app = pd.DataFrame({"x": rng.uniform(0, 1, 200), "y": rng.uniform(0, 1, 200)})
        bounds = {"x": (0.0, 2.0), "y": np.array([[0.0, 0.0, 5.0], [2.0, 1.0, 10.0]])}
        result = propose_new_val_experiments(app, n=8, bounds=bounds)
        expected = propose_new_val_experiments(app / [2.0, 10.0], n=8, scale=None)
        np.testing.assert_allclose(result.to_numpy(), expected.to_numpy() * [2.0, 10.0])

    def test_bounds_with_wrong_columns_raises(self):
        app = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        with pytest.raises(ValueError, match="bounds"):
            propose_new_val_experiments(app, n=1, bounds={"x": (0.0, 1.0)})


class TestExtractBoundsFromData:
    def _make_df(self):
        rng = np.random.default_rng(0)
        n = 300
        i_s = rng.uniform(0.0, 600.0, size=n)
        return pd.DataFrame(
            {
                "I.S [A]": i_s,
                "p.Si.A [barg]": 0.5 + i_s / 1000.0 + rng.uniform(-0.05, 0.05, size=n),
                "p.So.A [barg]": 0.3 + i_s / 1500.0 + rng.uniform(-0.03, 0.03, size=n),
                "T.Si.A [°C]": rng.uniform(20.0, 80.0, size=n),
            }
        )

    def test_returns_box_bound_for_indep_col(self):
        df = self._make_df()
        bounds = extract_bounds(df, indep_col="I.S [A]")
        lo, hi = bounds["I.S [A]"]
        assert lo == pytest.approx(df["I.S [A]"].min())
        assert hi == pytest.approx(df["I.S [A]"].max())

    def test_returns_array_for_dep_cols(self):
        df = self._make_df()
        bounds = extract_bounds(
            df, indep_col="I.S [A]", dep_cols=["p.Si.A [barg]"]
        )
        table = bounds["p.Si.A [barg]"]
        assert isinstance(table, np.ndarray)
        assert table.ndim == 2
        assert table.shape[1] == 3

    def test_default_dep_cols_excludes_indep(self):
        df = self._make_df()
        bounds = extract_bounds(df, indep_col="I.S [A]")
        assert "I.S [A]" not in bounds or isinstance(bounds["I.S [A]"], tuple)
        for col in ["p.Si.A [barg]", "p.So.A [barg]", "T.Si.A [°C]"]:
            assert col in bounds
            assert isinstance(bounds[col], np.ndarray)

    def test_lower_leq_upper_in_all_bins(self):
        df = self._make_df()
        bounds = extract_bounds(
            df, indep_col="I.S [A]", dep_cols=["p.Si.A [barg]", "p.So.A [barg]"]
        )
        for col in ["p.Si.A [barg]", "p.So.A [barg]"]:
            table = bounds[col]
            assert (table[:, 1] <= table[:, 2]).all(), f"lower > upper in '{col}'"

    def test_output_is_valid_input_for_generate_lhs(self):
        df = self._make_df()
        bounds = extract_bounds(
            df, indep_col="I.S [A]", dep_cols=["p.Si.A [barg]", "p.So.A [barg]"]
        )
        sample = generate_lhs(bounds, n=50, seed=1)
        assert list(sample.columns) == ["I.S [A]", "p.Si.A [barg]", "p.So.A [barg]"]
        assert len(sample) == 50

    def test_missing_indep_col_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError, match="not found"):
            extract_bounds(df, indep_col="nonexistent")

    def test_missing_dep_col_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError, match="not found"):
            extract_bounds(df, indep_col="I.S [A]", dep_cols=["ghost"])

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            extract_bounds(pd.DataFrame(), indep_col="I.S [A]")

    def test_n_bins_controls_table_rows(self):
        df = self._make_df()
        bounds_10 = extract_bounds(
            df, indep_col="I.S [A]", dep_cols=["p.Si.A [barg]"], n_bins=10
        )
        bounds_30 = extract_bounds(
            df, indep_col="I.S [A]", dep_cols=["p.Si.A [barg]"], n_bins=30
        )
        # More bins → at most more rows (non-empty bins only)
        assert bounds_30["p.Si.A [barg]"].shape[0] >= bounds_10["p.Si.A [barg]"].shape[0]



class TestCalValGPRUncertainty:
    def _make_frames_with_outputs(self):
        """Create validation and application frames with synthetic outputs."""
        rng = np.random.default_rng(42)
        # Validation data
        validation_df = pd.DataFrame(
            {
                "temperature": rng.uniform(0.0, 100.0, size=20),
                "load": rng.uniform(0.0, 1.0, size=20),
            }
        )
        # Synthetic output: y = T/10 + L*5 + small noise
        validation_y = (
            validation_df["temperature"] / 10.0
            + validation_df["load"] * 5.0
            + rng.normal(0, 0.1, size=20)
        )
        
        # Application data (denser)
        application_df = pd.DataFrame(
            {
                "temperature": rng.uniform(0.0, 100.0, size=100),
                "load": rng.uniform(0.0, 1.0, size=100),
            }
        )
        return validation_df, validation_y, application_df

    def test_returns_gpr_result_with_expected_shape(self):
        validation_df, validation_y, application_df = self._make_frames_with_outputs()
        result = cal_val_gpr_uncertainty(validation_df, validation_y, application_df)

        assert isinstance(result, GPRUncertaintyResult)
        assert result.sigma.shape == (len(application_df),)
        assert result.columns == ["temperature", "load"]
        assert result.length_scales.shape == (2,)

    def test_sigma_non_negative(self):
        validation_df, validation_y, application_df = self._make_frames_with_outputs()
        result = cal_val_gpr_uncertainty(validation_df, validation_y, application_df)
        assert np.all(result.sigma >= 0.0)

    def test_sigma_max_and_q95_consistent(self):
        validation_df, validation_y, application_df = self._make_frames_with_outputs()
        result = cal_val_gpr_uncertainty(validation_df, validation_y, application_df)

        assert result.sigma_max == pytest.approx(result.sigma.max())
        assert result.sigma_q95 == pytest.approx(np.quantile(result.sigma, 0.95))
        assert result.sigma_mean == pytest.approx(result.sigma.mean())
        # Ordering check
        assert result.sigma_max >= result.sigma_q95
        assert result.sigma_q95 >= result.sigma_mean or result.sigma_mean >= result.sigma_q95 * 0.9

    def test_sigma_near_zero_at_validation_points(self):
        # When application points match validation points, sigma should be ~0
        validation_df = pd.DataFrame(
            {
                "a": [0.0, 0.5, 1.0],
                "b": [0.0, 0.5, 1.0],
            }
        )
        validation_y = np.array([1.0, 2.0, 3.0])
        
        # Application = validation (should have sigma � 0)
        application_df = validation_df.copy()
        
        result = cal_val_gpr_uncertainty(
            validation_df, validation_y, application_df, alpha=1e-10
        )
        
        # With very small alpha, sigma should be near zero at training points
        assert np.allclose(result.sigma, 0.0, atol=1e-3)

    def test_sigma_increases_with_distance(self):
        # Points far from validation data should have higher uncertainty
        validation_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        validation_y = np.array([0.0, 1.0])
        
        # Application points: some near, some far
        application_df = pd.DataFrame(
            {
                "x": [0.1, 0.9, 5.0],  # First two near validation, last far
                "y": [0.1, 0.9, 5.0],
            }
        )
        
        result = cal_val_gpr_uncertainty(
            validation_df, validation_y, application_df, scale=None
        )
        
        # Far point should have higher uncertainty than near points
        assert result.sigma[2] > result.sigma[0]
        assert result.sigma[2] > result.sigma[1]

    def test_learned_length_scales_reasonable(self):
        validation_df, validation_y, application_df = self._make_frames_with_outputs()
        result = cal_val_gpr_uncertainty(
            validation_df, validation_y, application_df, length_scale="auto"
        )
        
        # Length scales should be positive
        assert np.all(result.length_scales > 0.0)
        # With auto optimization and reasonable data, expect length scales in [0.1, 10]
        assert np.all(result.length_scales >= 0.05)
        assert np.all(result.length_scales <= 15.0)

    def test_scale_none_vs_minmax(self):
        validation_df, validation_y, application_df = self._make_frames_with_outputs()
        
        result_none = cal_val_gpr_uncertainty(
            validation_df, validation_y, application_df, scale=None
        )
        result_minmax = cal_val_gpr_uncertainty(
            validation_df, validation_y, application_df, scale="minmax"
        )
        
        # Both should produce valid results
        assert result_none.scale is None
        assert result_minmax.scale == "minmax"
        # Sigma values may differ due to scaling, but structure should be consistent
        assert result_none.sigma.shape == result_minmax.sigma.shape

    def test_raises_on_mismatched_columns(self):
        validation_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        validation_y = np.array([0.0, 1.0])
        application_df = pd.DataFrame({"a": [0.5], "b": [0.5]})
        
        with pytest.raises(ValueError, match="same columns"):
            cal_val_gpr_uncertainty(validation_df, validation_y, application_df)

    def test_raises_on_wrong_y_length(self):
        validation_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
        validation_y = np.array([0.0, 1.0, 2.0])  # Wrong length!
        application_df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        
        with pytest.raises(ValueError, match="validation_y must have length"):
            cal_val_gpr_uncertainty(validation_df, validation_y, application_df)

    def test_raises_on_empty_dataframe(self):
        validation_df = pd.DataFrame({"x": [], "y": []})
        validation_y = np.array([])
        application_df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        
        with pytest.raises(ValueError, match="must not be empty"):
            cal_val_gpr_uncertainty(validation_df, validation_y, application_df)

    def test_known_simple_case(self):
        # 1D case with known linear relationship
        validation_df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        validation_y = np.array([0.0, 1.0, 2.0])  # y = x
        application_df = pd.DataFrame({"x": [0.5, 1.5]})
        
        result = cal_val_gpr_uncertainty(
            validation_df,
            validation_y,
            application_df,
            scale=None,
            length_scale=1.0,
            alpha=1e-10,
        )
        
        # With fixed length_scale, results should be deterministic
        assert result.sigma.shape == (2,)
        assert np.all(result.sigma >= 0.0)
        # Interpolated points should have low uncertainty
        assert result.sigma_max < 0.5

    def test_fixed_length_scale_parameter(self):
        validation_df, validation_y, application_df = self._make_frames_with_outputs()
        
        # Test fixed isotropic length scale
        result_fixed = cal_val_gpr_uncertainty(
            validation_df, validation_y, application_df, length_scale=1.0
        )
        assert np.allclose(result_fixed.length_scales, 1.0)
        
        # Test per-dimension length scales
        length_scales_custom = np.array([0.5, 2.0])
        result_custom = cal_val_gpr_uncertainty(
            validation_df,
            validation_y,
            application_df,
            length_scale=length_scales_custom,
        )
        assert np.allclose(result_custom.length_scales, length_scales_custom)

    def test_accepts_series_for_validation_y(self):
        validation_df, _, application_df = self._make_frames_with_outputs()
        # Use pd.Series instead of np.array
        validation_y = pd.Series([1.0] * len(validation_df))
        
        result = cal_val_gpr_uncertainty(validation_df, validation_y, application_df)
        assert isinstance(result, GPRUncertaintyResult)
        assert result.sigma.shape == (len(application_df),)


class TestGenerateSobol:
    def test_box_sobol_shape_and_bounds(self):
        bounds = {"temperature": (0.0, 100.0), "load": (0.0, 1.0)}
        df = generate_sobol(bounds, n=64, seed=0)

        assert list(df.columns) == ["temperature", "load"]
        assert len(df) == 64
        assert df["temperature"].between(0.0, 100.0).all()
        assert df["load"].between(0.0, 1.0).all()

    def test_box_sobol_is_reproducible_with_seed(self):
        bounds = {"a": (0.0, 1.0), "b": (10.0, 20.0)}
        df1 = generate_sobol(bounds, n=32, seed=42)
        df2 = generate_sobol(bounds, n=32, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_n_not_power_of_two_raises(self):
        bounds = {"a": (0.0, 1.0)}
        with pytest.raises(ValueError, match="power of 2"):
            generate_sobol(bounds, n=50)

    def test_n_zero_raises(self):
        with pytest.raises(ValueError):
            generate_sobol({"a": (0.0, 1.0)}, n=0)

    def test_unsupported_spec_raises(self):
        with pytest.raises(ValueError):
            generate_sobol({"a": "not-a-bound"}, n=8)

    def test_reversed_box_bound_raises(self):
        with pytest.raises(ValueError):
            generate_sobol({"a": (1.0, 0.0)}, n=8)

    def test_constraint_sobol_respects_interpolated_envelope(self):
        # 'load' is constrained between 0 and an upper limit that grows linearly
        # with temperature: 0.2 at T=0 up to 1.0 at T=100.
        envelope = np.array(
            [
                [0.0, 0.0, 0.2],
                [100.0, 0.0, 1.0],
            ]
        )
        bounds = {"temperature": (0.0, 100.0), "load": envelope}
        df = generate_sobol(bounds, n=64, seed=3)

        assert list(df.columns) == ["temperature", "load"]
        assert len(df) == 64

        upper = 0.2 + (1.0 - 0.2) * (df["temperature"] / 100.0)
        assert (df["load"] >= -1e-9).all()
        assert (df["load"] <= upper + 1e-9).all()

    def test_constraint_sobol_independent_marginal_not_biased(self):
        # With a wildly asymmetric band the conditional inverse-transform must
        # not cluster points where the band is wide.  Sobol' does not have the
        # LHS one-per-stratum guarantee, so we use a generous tolerance: every
        # equal-width bin should contain at least 20 % of the expected count.
        envelope = np.array(
            [
                [0.0, 0.0, 0.1],
                [10.0, 0.0, 9.0],
            ]
        )
        bounds = {"temperature": (0.0, 10.0), "load": envelope}
        n = 2048
        df = generate_sobol(bounds, n=n, seed=3)

        counts, _ = np.histogram(df["temperature"], bins=np.linspace(0.0, 10.0, 11))
        expected = n / 10
        assert np.all(counts >= 0.2 * expected), (
            "Marginal is badly biased; conditional inverse-transform may not be "
            f"applied. Bin counts: {counts}"
        )

    def test_constraint_sobol_fills_local_band_at_both_ends(self):
        envelope = np.array(
            [
                [0.0, 0.0, 0.1],
                [10.0, 0.0, 9.0],
            ]
        )
        bounds = {"temperature": (0.0, 10.0), "load": envelope}
        df = generate_sobol(bounds, n=2048, seed=11)

        low = df[df["temperature"] < 0.1]["load"]
        high = df[df["temperature"] > 9.9]["load"]
        assert low.max() < 0.2
        assert high.max() > 6.0
        assert high.min() < 2.0

    def test_constraint_sobol_feeds_coverage_functions(self):
        envelope = np.array([[0.0, 0.0, 0.5], [100.0, 0.0, 1.0]])
        bounds = {"temperature": (0.0, 100.0), "load": envelope}
        application_df = generate_sobol(bounds, n=64, seed=4)
        validation_df = generate_sobol(bounds, n=16, seed=5)

        result = cal_val_dist(validation_df, application_df)
        assert result.r.shape == (64,)

        proposals = propose_val_experiments(application_df, validation_df, n=3)
        assert len(proposals) == 3

    def test_multiple_independent_vars_with_constraint_raises(self):
        envelope = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0), "x": envelope}
        with pytest.raises(ValueError):
            generate_sobol(bounds, n=8)

    def test_wrong_envelope_shape_raises(self):
        bad = np.array([[0.0, 1.0], [1.0, 2.0]])  # only 2 columns
        bounds = {"t": (0.0, 1.0), "x": bad}
        with pytest.raises(ValueError):
            generate_sobol(bounds, n=8)

    def test_scramble_false_is_deterministic(self):
        # Unscrambled Sobol' is deterministic and seed has no effect.
        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        df1 = generate_sobol(bounds, n=32, seed=0, scramble=False)
        df2 = generate_sobol(bounds, n=32, seed=99, scramble=False)
        pd.testing.assert_frame_equal(df1, df2)

    def test_rules_only_box_bounds_respected(self):
        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        df = generate_sobol(
            bounds, n=128, seed=7, rules=[lambda d: d["a"] > d["b"]]
        )
        assert (df["a"] > df["b"]).all()

    def test_rules_with_envelope_respected(self):
        envelope = np.array([[0.0, 0.0, 1.0], [100.0, 0.0, 1.0]])
        bounds = {"temperature": (0.0, 100.0), "lo": envelope, "hi": envelope}
        df = generate_sobol(
            bounds,
            n=64,
            seed=8,
            rules=[lambda d: d["hi"] > d["lo"]],
        )
        assert (df["hi"] > df["lo"]).all()

    def test_rules_impossible_constraint_raises(self):
        bounds = {"a": (0.0, 1.0)}
        with pytest.raises(ValueError, match="Could only draw"):
            generate_sobol(
                bounds, n=8, seed=0,
                rules=[lambda d: d["a"] > d["a"] + 1],
                max_oversample=8, rejection=True,
            )

    def test_rules_none_behaves_like_no_rules(self):
        bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        df1 = generate_sobol(bounds, n=32, seed=99, rules=None)
        df2 = generate_sobol(bounds, n=32, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    # --- space_filling tests ---------------------------------------------------

    def test_space_filling_rules_still_respected(self):
        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        df = generate_sobol(
            bounds, n=64, seed=5,
            rules=[lambda d: d["a"] > d["b"] + 0.2],
            space_filling=True,
        )
        assert len(df) == 64
        assert list(df.columns) == ["a", "b"]
        assert (df["a"] > df["b"] + 0.2).all()

    def test_space_filling_improves_spread(self):
        from sklearn.metrics import pairwise_distances

        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        rule = [lambda d: d["a"] + d["b"] < 1.3]
        n = 64

        df_default = generate_sobol(bounds, n=n, seed=3, rules=rule)
        df_sf = generate_sobol(
            bounds, n=n, seed=3, rules=rule, space_filling=True, pool_factor=30
        )

        def maximin(df):
            D = pairwise_distances(df.to_numpy())
            np.fill_diagonal(D, np.inf)
            return D.min(axis=1).min()

        assert maximin(df_sf) > maximin(df_default)

    def test_space_filling_reproducible_with_seed(self):
        bounds = {"a": (0.0, 2.0), "b": (0.0, 2.0)}
        rules = [lambda d: d["a"] + d["b"] < 3.0]
        df1 = generate_sobol(
            bounds, n=32, seed=42, rules=rules,
            space_filling=True, pool_factor=20,
        )
        df2 = generate_sobol(
            bounds, n=32, seed=42, rules=rules,
            space_filling=True, pool_factor=20,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_space_filling_with_envelope_and_rules(self):
        envelope = np.array([[0.0, 0.0, 1.0], [100.0, 0.0, 1.0]])
        bounds = {"temperature": (0.0, 100.0), "lo": envelope, "hi": envelope}
        df = generate_sobol(
            bounds, n=32, seed=9,
            rules=[lambda d: d["hi"] > d["lo"] + 0.05],
            space_filling=True,
        )
        assert len(df) == 32
        assert (df["hi"] > df["lo"] + 0.05).all()
