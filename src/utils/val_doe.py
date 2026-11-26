from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


@dataclass
class DistanceResult:
    """Result of a nearest-neighbor validation coverage analysis.

    Attributes
    ----------
    r : numpy.ndarray
        Distance of every application point to its nearest validation point,
        measured in the (optionally) scaled input space.
    d_max : float
        Worst-case coverage distance ``max(r)`` (minimax radius). No application
        point is farther than ``d_max`` from a validation point.
    q95 : float
        95-percent quantile of ``r``: 95 % of the application domain lies within
        this distance of a validation point.
    mean : float
        Mean coverage distance ``mean(r)``: average distance of an application
        point to its nearest validation point.
    std : float
        Standard deviation of ``r``: spread of per-point coverage distances
        around the mean.
    eta : float
        Empirical coverage efficiency ``d_max_opt / d_max``.  ``d_max_opt`` is
        the minimax covering radius of a near-optimal ``n_val``-point design
        selected from ``application_df`` (the discretized domain ``F``) via a
        greedy farthest-first / k-center heuristic:

        .. math::

            d_{\\max}^{opt} = \\max_{x \\in X_N}\\ \\min_{c \\in C}\\ \\|x - c\\|

        where ``C`` is the selected reference design and ``X_N`` is
        ``application_df``.  Unlike an analytic optimal radius, this needs no
        volume of ``F`` and is therefore valid for **arbitrary non-box domains**
        with dependent / nonlinear bounds.  Interpretation:

        * ``eta ≈ 1`` – validation points cover ``F`` nearly as well as an
          optimal design of the same size.
        * ``eta << 1`` – the domain has avoidable under-covered regions.

        The metric is meaningful only when ``application_df`` is substantially
        denser than ``validation_df`` (``n_app >> n_val``); otherwise the
        reference design drawn from ``application_df`` is too coarse.  Greedy
        farthest-first is a 2-approximation of exact k-center, so ``d_max_opt``
        is slightly conservative (``eta`` marginally pessimistic).
    columns : list[str]
        Dimensions of ``X`` (column order used for the distance computation).
    scale : str | None
        Scaling method applied to the inputs before computing distances.
    """

    r: np.ndarray
    d_max: float
    q95: float
    mean: float
    std: float
    eta: float
    columns: list[str]
    scale: str | None


@dataclass
class CoverageResult:
    """Result of a convex-hull-based validation coverage analysis.

    Attributes
    ----------
    vol_val : float
        Volume of the convex hull spanned by ``validation_df``, in the
        (optionally scaled) input space.
    vol_app : float
        Volume of the convex hull spanned by ``application_df``, in the same
        (optionally scaled) input space.
    eta : float
        Geometric coverage ratio ``vol_val / vol_app``. Because the same
        per-dimension scaling factors divide both point sets, this ratio is
        invariant to the ``scale`` argument of :func:`cal_val_cov` (only the
        absolute ``vol_val`` / ``vol_app`` values depend on it).

        * ``eta ≈ 1`` – validation points span nearly the same convex region
          as the application domain.
        * ``eta << 1`` – validation covers only a small geometric fraction of
          the application domain; large parts of it are extrapolation.

        ``eta`` says nothing about how *uniformly* the covered region is
        sampled – pair with :func:`cal_val_dist` (``d_max`` / ``q95``) for
        that.
    """

    vol_val: float
    vol_app: float
    eta: float


@dataclass
class GPRUncertaintyResult:
    """Result of a GPR-based validation coverage analysis.

    Attributes
    ----------
    sigma : numpy.ndarray
        Predictive standard deviation at every application point, measured
        from a Gaussian Process trained on validation data. Higher values
        indicate greater uncertainty / poorer coverage.
    sigma_max : float
        G-criterion (worst-case uncertainty): ``max(sigma)``. The application
        point with the highest predictive uncertainty. Minimizing this metric
        is the goal of G-optimal experimental design.
    sigma_q95 : float
        95-percent quantile of ``sigma``: 95 % of the application domain has
        predictive uncertainty below this value.
    sigma_mean : float
        Average predictive uncertainty across the application domain. Related
        to the A-criterion (integrated variance) in optimal design theory.
    length_scales : numpy.ndarray
        Learned length-scale parameters of the RBF kernel, one per dimension.
        Small values indicate the output varies rapidly in that dimension;
        large values indicate smooth variation.
    columns : list[str]
        Dimensions of ``X`` (column order used for the GPR model).
    scale : str | None
        Scaling method applied to the inputs before training the GPR.
    """

    sigma: np.ndarray
    sigma_max: float
    sigma_q95: float
    sigma_mean: float
    length_scales: np.ndarray
    columns: list[str]
    scale: str | None


def _scaling_factors(application_df: pd.DataFrame, scale: str | None) -> np.ndarray:
    """Return per-dimension scaling factors derived from the application domain.

    The application frame defines the domain ``X_app``, so its extent is used as
    the natural reference for making distances comparable across dimensions.
    """
    if scale is None:
        return np.ones(application_df.shape[1], dtype=float)

    if scale == "minmax":
        span = application_df.max(axis=0) - application_df.min(axis=0)
        factors = span.to_numpy(dtype=float).copy()
    elif scale == "std":
        factors = application_df.std(axis=0, ddof=0).to_numpy(dtype=float).copy()
    else:
        raise ValueError(f"Unknown scale '{scale}'. Use 'minmax', 'std' or None.")

    # Avoid division by zero for constant dimensions.
    factors[factors == 0.0] = 1.0
    return factors


def _bound_scaling_factors(
    bounds: dict[str, tuple[float, float] | np.ndarray],
    frame_columns: list[str],
    columns: list[str],
) -> np.ndarray:
    """Return per-dimension scaling factors from the overall bound range.

    The bound range is the extent of the bounding hyperrectangle ``B`` (box
    limits, or minimum lower and maximum upper limit of an envelope table), so
    the scaling does not depend on the sampled frame.
    """
    if set(bounds) != set(frame_columns):
        raise ValueError(
            "bounds must contain exactly the frame columns. "
            f"Expected {frame_columns}, got {list(bounds)}."
        )
    _, _, _, lower, upper, _ = _prepare_bound_arrays(
        {column: bounds[column] for column in columns}
    )
    factors = upper - lower
    factors[factors == 0.0] = 1.0
    return factors


def _validate_val_app_frames(
    validation_df: pd.DataFrame, application_df: pd.DataFrame
) -> None:
    """Shared input validation for validation/application frame pairs.

    Checks that both arguments are DataFrames, share the same set of columns
    (dimensions of ``X``), and are non-empty. Used by :func:`cal_val_dist` and
    :func:`cal_val_cov`.

    Raises
    ------
    ValueError
        If either input is not a DataFrame, the column sets differ, or a
        frame is empty.
    """
    if not isinstance(validation_df, pd.DataFrame) or not isinstance(
        application_df, pd.DataFrame
    ):
        raise ValueError("Both validation_df and application_df must be DataFrames.")

    if set(validation_df.columns) != set(application_df.columns):
        raise ValueError(
            "Validation and application frames must share the same columns "
            f"(dimensions of X). Got {list(validation_df.columns)} vs "
            f"{list(application_df.columns)}."
        )

    if validation_df.empty or application_df.empty:
        raise ValueError("Validation and application frames must not be empty.")


def cal_val_dist(
    validation_df: pd.DataFrame,
    application_df: pd.DataFrame,
    scale: str | None = "minmax",
    exclude_cols: list[str] | None = None,
    bounds: dict[str, tuple[float, float] | np.ndarray] | None = None,
) -> DistanceResult:
    """Evaluate how well validation data covers an application domain ``X``.

    For every point of the application frame the Euclidean distance to the
    nearest validation point is computed in a (optionally) normalized input
    space. Small distances mean the point is well covered by validation data;
    large distances indicate extrapolation, where no validated statement about
    prediction accuracy can be made.

    See Also
    --------
    cal_val_gpr_uncertainty : Model-based coverage via GPR predictive uncertainty.
        Use when validation outputs (y) are available and you want uncertainty
        estimates that account for correlation structure.

    Parameters
    ----------
    validation_df : pandas.DataFrame
        Validation points. Each column is a dimension of ``X``.
    application_df : pandas.DataFrame
        Sample of the application domain ``X_app`` (e.g. a grid or Latin
        hypercube sample). Must have the same columns as ``validation_df``.
    scale : {"minmax", "std", None}, default "minmax"
        Per-dimension scaling applied before computing distances so that the
        distances are interpretable across dimensions with different units.
        By default, scaling factors are derived from the application domain.
    exclude_cols : list[str] | None, default None
        Column names to exclude from the distance computation. Use this to
        drop dimensions that are constrained to a narrow corridor by operating
        rules (e.g. a pulse-time parameter whose range is fully determined by
        the current setpoint). Excluded dimensions do not contribute to
        ``d_max``, ``q95``, or ``eta``. The returned ``DistanceResult.columns``
        reflects only the columns actually used.

        A ``ValueError`` is raised if any name in ``exclude_cols`` is not
        present in the frames.
    bounds : dict[str, tuple | numpy.ndarray] | None, default None
        Optional bounds specification using the same format as
        :func:`generate_lhs`: each frame column maps to a ``(lower, upper)``
        tuple or an ``(m, 3)`` envelope array with columns
        ``[independent_value, lower, upper]``. When ``scale="minmax"``, the
        overall bound range for each dimension is used instead of the range
        observed in ``application_df``. For ``scale="std"`` or ``None``, this
        argument has no effect.

    Returns
    -------
    DistanceResult
        Object holding the per-point distances ``r``, the worst-case distance
        ``d_max``, the 95-percent quantile ``q95``, and the coverage efficiency
        ratio ``eta``.  ``columns`` lists the dimensions that were included.

    Raises
    ------
    ValueError
        If the column sets of the two frames differ, if a frame is empty, or if
        any name in ``exclude_cols`` is not a column of the frames, or if
        ``bounds`` does not match the frame columns or contains invalid
        specifications (when ``scale="minmax"``).
    """
    _validate_val_app_frames(validation_df, application_df)
    frame_columns = list(validation_df.columns)

    if exclude_cols:
        unknown = set(exclude_cols) - set(validation_df.columns)
        if unknown:
            raise ValueError(
                f"exclude_cols contains names not present in the frames: {sorted(unknown)}"
            )
        keep = [c for c in validation_df.columns if c not in exclude_cols]
        validation_df = validation_df[keep]
        application_df = application_df[keep]

    columns = list(validation_df.columns)
    # Align column order so dimensions match position-wise.
    application_df = application_df[columns]

    if scale == "minmax" and bounds is not None:
        factors = _bound_scaling_factors(bounds, frame_columns, columns)
    else:
        factors = _scaling_factors(application_df, scale)
    val = validation_df.to_numpy(dtype=float) / factors
    app = application_df.to_numpy(dtype=float) / factors
    # app = normalize(application_df.to_numpy(dtype=float).T, norm='l2').T
    # val = normalize(validation_df.to_numpy(dtype=float).T, norm='l2').T

    # scaler = MinMaxScaler()
    # app = scaler.fit_transform(application_df.to_numpy(dtype=float))
    # val = scaler.transform(validation_df.to_numpy(dtype=float))

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(val)
    dist, _ = nn.kneighbors(app)
    r = dist.ravel()

    # --- eta: empirical coverage efficiency d_max_opt / d_max ----------------
    # The optimum is measured empirically *inside* F instead of assuming a unit
    # box (vol = 1). application_df is the discretized domain X_N; a near-optimal
    # n_val-point design is selected from it via farthest-first (k-center). This
    # makes eta valid for arbitrary, non-box (dependent-bound) domains.
    #   eta ≈ 1  -> validation covers F nearly as well as an optimal design
    #   eta << 1 -> avoidable coverage gaps
    n_val = len(validation_df)
    d_max_val = float(r.max())
    d_max_opt = _empirical_opt_radius(app, n_val)
    eta = d_max_opt / d_max_val if d_max_val > 0.0 else 0.0

    return DistanceResult(
        r=r,
        d_max=float(r.max()),
        q95=float(np.quantile(r, 0.95)),
        mean=float(r.mean()),
        std=float(r.std()),
        eta=eta,
        columns=columns,
        scale=scale,
    )


def _convex_hull_volume(points: np.ndarray) -> float:
    """Volume of the convex hull spanned by *points*.

    Parameters
    ----------
    points : numpy.ndarray, shape ``(n, d)``
        Point set in ``d`` dimensions.

    Returns
    -------
    float
        Lebesgue volume of ``conv(points)``. For ``d == 1`` this is the range
        ``max(points) - min(points)``, since :class:`scipy.spatial.ConvexHull`
        does not support one-dimensional input.

    Raises
    ------
    ValueError
        If the hull cannot be computed, e.g. fewer than ``d + 1`` points or
        degenerate (collinear / coplanar) input.
    """
    if points.shape[1] == 1:
        return float(points.max() - points.min())

    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        raise ValueError(
            "Could not compute the convex hull: the point set has too few "
            f"points ({points.shape[0]}) for {points.shape[1]} dimensions, or "
            "the points are degenerate (collinear / coplanar)."
        ) from exc

    return float(hull.volume)


def cal_val_cov(
    validation_df: pd.DataFrame,
    application_df: pd.DataFrame,
    scale: str | None = "minmax",
) -> CoverageResult:
    """Evaluate validation coverage via convex-hull volumes.

    Computes the convex hull of ``validation_df`` and of ``application_df``
    in a (optionally) normalized input space and compares their volumes. This
    provides a purely geometric coverage metric: it answers *how much of the
    application domain's extent is spanned by validation data*, independent
    of how densely that region is sampled.

    See Also
    --------
    cal_val_dist : Nearest-neighbor coverage distance. Use this to assess how
        *densely/uniformly* the application domain is sampled by validation
        data; ``cal_val_cov`` only measures the spanned geometric extent, so
        the two metrics are complementary.
    cal_val_gpr_uncertainty : Model-based coverage via GPR predictive
        uncertainty.

    Parameters
    ----------
    validation_df : pandas.DataFrame
        Validation points. Each column is a dimension of ``X``.
    application_df : pandas.DataFrame
        Sample of the application domain ``X_app`` (e.g. a grid or Latin
        hypercube sample). Must have the same columns as ``validation_df``.
    scale : {"minmax", "std", None}, default "minmax"
        Per-dimension scaling applied before computing the convex hulls so
        that dimensions with different units contribute comparably to the
        volume. Scaling factors are derived from the application domain (see
        :func:`cal_val_dist`). Because the same scaling factors divide both
        point sets, ``eta`` is invariant to this choice; only the absolute
        ``vol_val`` / ``vol_app`` magnitudes depend on it.

    Returns
    -------
    CoverageResult
        Object holding the validation hull volume ``vol_val``, the
        application hull volume ``vol_app``, and the coverage ratio
        ``eta = vol_val / vol_app``.

    Raises
    ------
    ValueError
        If the column sets of the two frames differ, a frame is empty, or a
        convex hull cannot be computed (fewer than ``n_dims + 1`` points, or
        degenerate/collinear/coplanar points).

    Notes
    -----
    ``eta`` is a geometric coverage ratio, not a density measure: two very
    differently distributed point sets can span the same convex hull. Pair
    ``cal_val_cov`` with :func:`cal_val_dist` (``d_max`` / ``q95``) to also
    assess sampling density/uniformity within the covered region.

    See ``docs/cal_val_cov.md`` for the full mathematical derivation,
    including a proof that ``eta`` does not depend on ``scale``.
    """
    _validate_val_app_frames(validation_df, application_df)

    columns = list(validation_df.columns)
    # Align column order so dimensions match position-wise.
    application_df = application_df[columns]

    factors = _scaling_factors(application_df, scale)
    val = validation_df.to_numpy(dtype=float) / factors
    app = application_df.to_numpy(dtype=float) / factors

    vol_val = _convex_hull_volume(val)
    vol_app = _convex_hull_volume(app)
    eta = vol_val / vol_app if vol_app > 0.0 else 0.0

    return CoverageResult(vol_val=vol_val, vol_app=vol_app, eta=eta)


def cal_val_gpr_uncertainty(
        validation_df: pd.DataFrame,
        validation_y: np.ndarray | pd.Series,
        application_df: pd.DataFrame,
        scale: str | None = "minmax",
        length_scale: float | np.ndarray | str = "auto",
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 5,
) -> GPRUncertaintyResult:
    """Evaluate validation coverage via GPR-based predictive uncertainty.

    Trains a Gaussian Process Regressor on the validation data (inputs and
    outputs) and computes the predictive standard deviation at every point in
    the application domain. High uncertainty indicates regions far from
    validation data where predictions are unreliable (poor coverage).

    This provides a model-based complement to the geometric coverage metric
    :func:`cal_val_dist`. While :func:`cal_val_dist` measures distances in
    input space, this function quantifies uncertainty based on the learned
    correlation structure of the output. Use this when validation outputs are
    available and you want uncertainty estimates that account for how the
    response varies across the input domain.

    Parameters
    ----------
    validation_df : pandas.DataFrame
        Validation points (inputs). Each column is a dimension of ``X``.
    validation_y : numpy.ndarray or pandas.Series
        Output values at validation points. Must have length equal to
        ``len(validation_df)``. These are the observed responses (e.g.,
        simulation or measurement results) used to train the GPR.
    application_df : pandas.DataFrame
        Sample of the application domain ``X_app`` (e.g. a grid or Latin
        hypercube sample). Must have the same columns as ``validation_df``.
        Predictive uncertainty is evaluated at these points.
    scale : {"minmax", "std", None}, default "minmax"
        Per-dimension scaling applied before training the GPR so that
        dimensions with different units are commensurable. Scaling factors
        are derived from the application domain (same as :func:`cal_val_dist`).
    length_scale : float, numpy.ndarray, or "auto", default "auto"
        RBF kernel length scale(s). Controls the correlation range:

        * ``"auto"`` – optimize via maximum likelihood with bounds ``[0.1, 10.0]``
        * float – fixed isotropic length scale (same for all dimensions)
        * array – per-dimension length scales (shape ``(n_dims,)``)
    alpha : float, default 1e-10
        Nugget parameter: assumed variance of observation noise. Adds
        ``alpha * I`` to the kernel matrix for numerical stability. Use
        ``alpha > 0`` if validation data contains measurement noise. With
        ``alpha ≈ 0`` the GP interpolates exactly (``sigma ≈ 0`` at validation
        points).
    n_restarts_optimizer : int, default 5
        Number of restarts for the hyperparameter optimizer (when
        ``length_scale="auto"``). Higher values increase robustness but also
        computation time.

    Returns
    -------
    GPRUncertaintyResult
        Object holding the predictive uncertainty array ``sigma`` and summary
        metrics ``sigma_max`` (G-criterion), ``sigma_q95`` (95th percentile),
        and ``sigma_mean`` (average uncertainty / A-criterion). Also includes
        the learned ``length_scales`` from the kernel.

    Raises
    ------
    ValueError
        If the column sets of the two frames differ, a frame is empty, or
        ``validation_y`` has incorrect length.

    Notes
    -----
    **Computational complexity:** Training the GPR is ``O(n_val^3)`` and
    prediction is ``O(n_val^2 * n_app)``. For ``n_val > 500`` this becomes
    expensive. Consider using a subset of validation data or a sparse GP
    approximation for large datasets.

    **Connection to optimal design:** The returned metrics correspond to
    classical optimal experimental design criteria:

    * ``sigma_max`` is the **G-optimality criterion**: minimizing this produces
      designs that minimize worst-case prediction variance.
    * ``sigma_mean`` is related to the **A-optimality criterion**: minimizing
      average variance across the domain.

    See Pronzato & Müller (2012) for connections between minimax space-filling
    designs and G-optimal designs.

    **When to use this vs cal_val_dist:**

    * Use :func:`cal_val_dist` when you only have input data (no outputs) or
      want a purely geometric, model-free coverage assessment.
    * Use :func:`cal_val_gpr_uncertainty` when validation outputs are available
      and you want model-based uncertainty that accounts for correlation
      structure and smoothness of the response surface.

    Examples
    --------
    >>> validation_df = pd.DataFrame({"T": [300, 400, 500], "P": [1.0, 1.5, 2.0]})
    >>> validation_y = np.array([10.2, 15.3, 22.1])  # measured outputs
    >>> application_df = pd.DataFrame({"T": [350, 450], "P": [1.2, 1.8]})
    >>> result = cal_val_gpr_uncertainty(validation_df, validation_y, application_df)
    >>> result.sigma_max  # worst-case uncertainty
    0.85
    >>> result.sigma_mean  # average uncertainty
    0.62

    References
    ----------
    Pronzato, L., & Müller, W. G. (2012). Design of computer experiments:
    space filling and beyond. *Statistics and Computing*, 22(3), 681-701.
    """
    # --- Input validation -------------------------------------------------------
    if not isinstance(validation_df, pd.DataFrame) or not isinstance(
            application_df, pd.DataFrame
    ):
        raise ValueError(
            "Both validation_df and application_df must be DataFrames."
        )

    if set(validation_df.columns) != set(application_df.columns):
        raise ValueError(
            "Validation and application frames must share the same columns "
            f"(dimensions of X). Got {list(validation_df.columns)} vs "
            f"{list(application_df.columns)}."
        )

    if validation_df.empty or application_df.empty:
        raise ValueError("Validation and application frames must not be empty.")

    # Convert validation_y to numpy array
    if isinstance(validation_y, pd.Series):
        validation_y = validation_y.to_numpy()
    validation_y = np.asarray(validation_y, dtype=float)

    if len(validation_y) != len(validation_df):
        raise ValueError(
            f"validation_y must have length {len(validation_df)} to match "
            f"validation_df, got {len(validation_y)}."
        )

    columns = list(validation_df.columns)
    # Align column order so dimensions match position-wise
    application_df = application_df[columns]

    # --- Scaling ----------------------------------------------------------------
    factors = _scaling_factors(application_df, scale)
    val_scaled = validation_df.to_numpy(dtype=float) / factors
    app_scaled = application_df.to_numpy(dtype=float) / factors

    # --- GPR kernel configuration -----------------------------------------------
    n_dims = val_scaled.shape[1]

    if isinstance(length_scale, str) and length_scale == "auto":
        # Optimize length scales with reasonable bounds
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
            length_scale=np.ones(n_dims),
            length_scale_bounds=(0.01, 100.0),
        )
    elif isinstance(length_scale, (int, float)):
        # Fixed isotropic length scale
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            length_scale=float(length_scale),
            length_scale_bounds="fixed",
        )
    else:
        # Per-dimension length scales (array-like)
        length_scale = np.asarray(length_scale, dtype=float)
        if length_scale.shape != (n_dims,):
            raise ValueError(
                f"length_scale array must have shape ({n_dims},), got "
                f"{length_scale.shape}."
            )
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            length_scale=length_scale,
            length_scale_bounds="fixed",
        )

    # --- Fit GPR ----------------------------------------------------------------
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=0,
    )
    gpr.fit(val_scaled, validation_y)

    # Extract learned length scales from the fitted kernel
    # kernel_.k2 is the RBF part after ConstantKernel (k1) * RBF (k2)
    learned_length_scales = gpr.kernel_.k2.length_scale
    if np.isscalar(learned_length_scales):
        learned_length_scales = np.full(n_dims, learned_length_scales)
    else:
        learned_length_scales = np.asarray(learned_length_scales)

    # --- Predict uncertainty on application domain ------------------------------
    _, sigma = gpr.predict(app_scaled, return_std=True)

    return GPRUncertaintyResult(
        sigma=sigma,
        sigma_max=float(sigma.max()),
        sigma_q95=float(np.quantile(sigma, 0.95)),
        sigma_mean=float(sigma.mean()),
        length_scales=learned_length_scales,
        columns=columns,
        scale=scale,
    )


def propose_new_val_experiments(
    application_df: pd.DataFrame,
    n: int = 1,
    scale: str | None = "minmax",
    validation_df: pd.DataFrame | None = None,
    bounds: dict[str, tuple[float, float] | np.ndarray] | None = None,
) -> pd.DataFrame:
    """Propose new validation experiments that efficiently cover the application domain.

    Implements a greedy minimax (farthest-first / k-center) heuristic: at each
    step the application point that is currently farthest from any already
    selected (or existing) validation point is chosen next. This drives down the
    worst-case coverage distance ``d_max`` with every added experiment.

    Two modes of operation
    ----------------------
    **With existing validation data** (``validation_df`` supplied):
        Proposals are points in the application domain that are worst-covered by
        the existing validation set.  The first proposal is the application point
        farthest from *any* existing validation point; subsequent proposals fill
        the next largest gaps, always accounting for all previously proposed
        points.

    **From scratch** (``validation_df=None``):
        Proposes ``n`` entirely new experiments that efficiently cover the whole
        application domain.  The first point is seeded at the application point
        nearest to the centroid of ``application_df`` (a central, representative
        starting point); subsequent points are added by the same farthest-first
        greedy rule. This produces a space-filling design that approximates the
        optimal k-center solution.

    Parameters
    ----------
    application_df : pandas.DataFrame
        Candidate points sampling the application domain ``X_app``.  Each column
        is one dimension of the input space ``X``.
    n : int, default 1
        Number of new experiments to propose.
    scale : {"minmax", "std", None}, default "minmax"
        Per-dimension scaling applied before computing distances so that
        dimensions with different units are commensurable.  Scaling factors are
        derived from ``application_df`` (see :func:`cal_val_dist`).
    validation_df : pandas.DataFrame | None, default None
        Existing validation points.  Must share the same columns as
        ``application_df``.  When *None* the function proposes ``n`` points from
        scratch without reference to any existing validation data.
    bounds : dict[str, tuple | numpy.ndarray] | None, default None
        Optional bounds specification (see :func:`cal_val_dist`).  When
        ``scale="minmax"``, the overall bound range of each dimension is used
        instead of the range observed in ``application_df``, so that selection
        and coverage evaluation share the same normalization.

    Returns
    -------
    pandas.DataFrame
        Proposed experiments taken from ``application_df`` (original, unscaled
        values), ordered by the sequence in which they were selected (most
        important / worst-covered first).

    Raises
    ------
    ValueError
        If ``application_df`` is empty, ``n`` is not positive, or (when
        ``validation_df`` is supplied) the column sets of the two frames differ
        or ``validation_df`` is empty.
    """
    if application_df.empty:
        raise ValueError("application_df must not be empty.")
    if n < 1:
        raise ValueError("n must be a positive integer.")

    columns = list(application_df.columns)
    application_df = application_df[columns]

    if validation_df is not None:
        if set(validation_df.columns) != set(columns):
            raise ValueError(
                "validation_df and application_df must share the same columns "
                "(dimensions of X)."
            )
        if validation_df.empty:
            raise ValueError("validation_df must not be empty.")

    if scale == "minmax" and bounds is not None:
        factors = _bound_scaling_factors(bounds, columns, columns)
    else:
        factors = _scaling_factors(application_df, scale)
    app = application_df.to_numpy(dtype=float) / factors

    n = min(n, app.shape[0])

    if validation_df is not None:
        val = validation_df[columns].to_numpy(dtype=float) / factors
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(val)
        d_eff = nn.kneighbors(app)[0].ravel()
    else:
        d_eff = np.full(app.shape[0], np.inf)

    chosen = _greedy_minimax(app, n, d_eff)

    return application_df.iloc[chosen].reset_index(drop=True)


def propose_val_experiments(
    application_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    n: int = 1,
    scale: str | None = "minmax",
    add_df: pd.DataFrame | None = None,
    bounds: dict[str, tuple[float, float] | np.ndarray] | None = None,
) -> pd.DataFrame:
    """Propose new validation experiments from the application domain that best complement existing ones.

    Selects *n* rows **from** ``application_df`` that, together with the
    already-existing experiments in ``validation_df``, minimise the worst-case
    coverage distance

    .. math::

        d_{\\max} = \\max_{x \\in X_{app}} \\min_{v \\in S_{val} \\cup S_{new}} \\|x - v\\|

    over the application domain (minimax / k-center criterion).  The existing
    experiments in ``validation_df`` are treated as fixed coverage; only new
    points from ``application_df`` are candidates.

    The algorithm is the greedy farthest-first heuristic shared with
    :func:`propose_new_val_experiments`: at each step the application point
    currently farthest from any existing coverage (``validation_df`` plus
    already-selected new points) is added to the selection and the running
    coverage distances are updated incrementally.

    Parameters
    ----------
    application_df : pandas.DataFrame
        Sample of the application domain ``X_app`` and pool of candidate new
        experiments.  Each column is one dimension of the input space ``X``.
    validation_df : pandas.DataFrame
        Already-completed validation experiments that contribute fixed baseline
        coverage.  Must share the same columns as ``application_df``.
    n : int, default 1
        Number of new experiments to propose.
    scale : {"minmax", "std", None}, default "minmax"
        Per-dimension scaling applied before computing distances
        (see :func:`cal_val_dist`).  Scaling factors are derived from
        ``application_df``.
    add_df : pandas.DataFrame | None, default None
        Optional pool of candidate experiments (same columns). When given,
        proposals are selected from ``add_df`` instead of ``application_df``;
        ``application_df`` remains the reference whose coverage ``d_max`` is
        minimised and which defines the scaling. Candidates are chosen
        greedily by the smallest resulting ``d_max`` over ``application_df``
        (ties broken by the smallest summed coverage distance).
    bounds : dict[str, tuple | numpy.ndarray] | None, default None
        Optional bounds specification (see :func:`cal_val_dist`).  When
        ``scale="minmax"``, the overall bound range of each dimension is used
        instead of the range observed in ``application_df``.

    Returns
    -------
    pandas.DataFrame
        Proposed experiments taken from ``add_df`` if given, else from
        ``application_df`` (original, unscaled values), in selection order.

    Raises
    ------
    ValueError
        If the column sets differ, a frame is empty, or ``n`` is not positive.

    See Also
    --------
    propose_new_val_experiments : Same algorithm; ``validation_df`` is optional
        and the function also supports a from-scratch (no existing data) mode.
    """
    if set(validation_df.columns) != set(application_df.columns):
        raise ValueError(
            "Validation and application frames must share the same columns "
            "(dimensions of X)."
        )
    if validation_df.empty or application_df.empty:
        raise ValueError("Validation and application frames must not be empty.")
    if n < 1:
        raise ValueError("n must be a positive integer.")

    columns = list(application_df.columns)
    application_df = application_df[columns]
    validation_df = validation_df[columns]

    if scale == "minmax" and bounds is not None:
        factors = _bound_scaling_factors(bounds, columns, columns)
    else:
        factors = _scaling_factors(application_df, scale)
    app = application_df.to_numpy(dtype=float) / factors
    val = validation_df.to_numpy(dtype=float) / factors

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(val)
    d_eff = nn.kneighbors(app)[0].ravel()

    if add_df is not None:
        if set(add_df.columns) != set(columns):
            raise ValueError(
                "add_df and application_df must share the same columns "
                "(dimensions of X)."
            )
        if add_df.empty:
            raise ValueError("add_df must not be empty.")
        add_df = add_df[columns]
        cand = add_df.to_numpy(dtype=float) / factors
        chosen = _greedy_minimax_candidates(app, cand, min(n, cand.shape[0]), d_eff)
        return add_df.iloc[chosen].reset_index(drop=True)

    chosen = _greedy_minimax(app, min(n, app.shape[0]), d_eff)

    return application_df.iloc[chosen].reset_index(drop=True)


def _greedy_minimax(
    app: np.ndarray,
    n: int,
    d_eff: np.ndarray,
) -> list[int]:
    """Greedy farthest-first (k-center) selection from a pre-scaled point array.

    At each step the point in *app* currently farthest from any existing
    coverage is added to the selection and the running coverage distances are
    updated incrementally.  This is the standard greedy 2-approximation for
    the k-center / minimax-covering problem.

    Parameters
    ----------
    app : numpy.ndarray, shape ``(n_app, d)``
        Pre-scaled candidate points.
    n : int
        Number of points to select.
    d_eff : numpy.ndarray, shape ``(n_app,)``
        Initial per-point coverage distance: the distance from ``app[i]`` to
        the nearest already-existing coverage point.  Pass
        ``np.full(n_app, np.inf)`` when no prior coverage exists (from-scratch
        mode); in that case the first point is seeded at the centroid-nearest
        application point.

    Returns
    -------
    list[int]
        Indices into *app* of the selected points, in selection order (most
        important / worst-covered first).
    """
    d_eff = d_eff.copy()
    chosen: list[int] = []

    # From-scratch mode: seed with the app point nearest to the centroid so
    # the first proposal is broadly representative of the domain.
    if np.all(np.isinf(d_eff)):
        centroid = app.mean(axis=0)
        seed = int(np.argmin(np.linalg.norm(app - centroid, axis=1)))
        chosen.append(seed)
        d_eff = np.linalg.norm(app - app[seed], axis=1)
        n -= 1

    for _ in range(n):
        candidates = d_eff.copy()
        candidates[chosen] = -np.inf          # prevent re-selection
        idx = int(np.argmax(candidates))      # point farthest from coverage
        chosen.append(idx)
        d_eff = np.minimum(d_eff, np.linalg.norm(app - app[idx], axis=1))

    return chosen


def _greedy_minimax_candidates(
    app: np.ndarray,
    cand: np.ndarray,
    n: int,
    d_eff: np.ndarray,
    chunk_size: int = 256,
) -> list[int]:
    """Greedy minimax selection of candidates that best cover a reference set.

    At each step the candidate minimising ``max(min(d_eff, ||app - c||))`` is
    selected; ties (e.g. several equally distant worst points) are broken by
    the smallest sum of resulting coverage distances.

    Parameters
    ----------
    app : numpy.ndarray, shape ``(n_app, d)``
        Pre-scaled reference points whose coverage is evaluated.
    cand : numpy.ndarray, shape ``(n_cand, d)``
        Pre-scaled candidate points.
    n : int
        Number of candidates to select (``<= n_cand``).
    d_eff : numpy.ndarray, shape ``(n_app,)``
        Current coverage distance of each reference point.
    chunk_size : int, default 256
        Candidates evaluated per block to bound memory use.

    Returns
    -------
    list[int]
        Indices into *cand* in selection order.
    """
    d_eff = d_eff.copy()
    chosen: list[int] = []
    for _ in range(n):
        best_idx, best_key = -1, (np.inf, np.inf)
        for start in range(0, cand.shape[0], chunk_size):
            block = cand[start:start + chunk_size]
            dist = np.linalg.norm(app[None, :, :] - block[:, None, :], axis=2)
            d_new = np.minimum(d_eff[None, :], dist)
            d_max = d_new.max(axis=1)
            d_sum = d_new.sum(axis=1)
            for j in np.lexsort((d_sum, d_max)):
                idx = start + int(j)
                if idx not in chosen:
                    key = (d_max[j], d_sum[j])
                    if key < best_key:
                        best_idx, best_key = idx, key
                    break
        chosen.append(best_idx)
        d_eff = np.minimum(d_eff, np.linalg.norm(app - cand[best_idx], axis=1))
    return chosen


def _farthest_first_indices(x: np.ndarray, n: int, start_idx: int) -> list[int]:
    """Greedy farthest-first (k-center) selection from a point array.

    Starts with *start_idx* and repeatedly picks the point currently farthest
    from any already-chosen point, updating the running minimum distance
    incrementally.  This avoids rebuilding a neighbor index at each step and
    runs in ``O(n * pool_size * d)`` time.

    Parameters
    ----------
    x : numpy.ndarray, shape ``(pool_size, d)``
        Pre-scaled point array (dimensions should already be commensurable).
    n : int
        Number of points to select (including the seed).
    start_idx : int
        Row index of the seed point (selected first).

    Returns
    -------
    list[int]
        Indices into *x* of the selected points in selection order.
    """
    chosen = [start_idx]
    d_min = np.linalg.norm(x - x[start_idx], axis=1)
    for _ in range(n - 1):
        d_min[chosen] = -np.inf
        idx = int(np.argmax(d_min))
        chosen.append(idx)
        d_min = np.minimum(d_min, np.linalg.norm(x - x[idx], axis=1))
    return chosen


def _empirical_opt_radius(app: np.ndarray, n_val: int) -> float:
    """Best achievable minimax covering radius for *n_val* points in domain F.

    The domain ``F`` is represented by the (already scaled) application points
    ``app`` – the grid ``X_N``.  A near-optimal ``n_val``-point k-center design
    is selected from ``app`` via greedy farthest-first selection
    (Pronzato & Müller 2012, Sect. 7, Eq. 32) and its covering radius

    .. math::

        d_{\\max}^{opt} = \\max_{x \\in X_N}\\ \\min_{c \\in C}\\ \\|x - c\\|

    is returned.  This serves as the empirical optimal reference for the
    coverage-efficiency ratio ``eta`` and needs no volume of ``F``, so it is
    valid for arbitrary non-box domains.

    Parameters
    ----------
    app : numpy.ndarray, shape ``(n_app, d)``
        Pre-scaled application points (the discretized domain ``F``).
    n_val : int
        Number of validation points; size of the reference design selected
        from ``app``.

    Returns
    -------
    float
        The empirical optimal covering radius.  Returns ``0.0`` when
        ``n_val >= n_app`` (coverage is then trivially exact and the metric is
        not meaningful – ``application_df`` should be much denser than the
        validation set).
    """
    n_app = app.shape[0]
    if n_val >= n_app:
        return 0.0
    centroid = app.mean(axis=0)
    seed = int(np.argmin(np.linalg.norm(app - centroid, axis=1)))
    opt_idx = _farthest_first_indices(app, n_val, seed)
    nn_opt = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(app[opt_idx])
    return float(nn_opt.kneighbors(app)[0].max())


def _is_box_spec(spec: object) -> bool:
    """Return True if ``spec`` is a ``(lo, hi)`` box bound."""
    return isinstance(spec, (tuple, list)) and len(spec) == 2


def _interp_limits(
    table: np.ndarray, indep_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear lower/upper limits for a dependent variable.

    ``table`` has columns ``[independent_value, lower, upper]`` describing the
    operating envelope; limits between the given support points are linearly
    interpolated and clamped outside the support range.
    """
    ref = table[:, 0]
    order = np.argsort(ref)
    ref = ref[order]
    lower = table[order, 1]
    upper = table[order, 2]
    return (
        np.interp(indep_values, ref, lower),
        np.interp(indep_values, ref, upper),
    )


def _resolve_bounds_and_rules(
    bounds: object,
    rules: "list[callable] | None",
    caller: str = "generate_lhs",
) -> "tuple[dict, list[callable] | None]":
    """Unwrap ValConfig / ValidationConfig objects into a plain bounds dict.

    Supports:

    * ``ValidationConfig`` (has a ``val_config`` attribute): its ``val_config``
      field is used; raises ``ValueError`` when ``val_config`` is ``None``.
    * ``ValConfig`` (has a ``get_bounds`` method): used directly.
    * Plain ``dict``: returned unchanged.

    Rules defined in a config object are used as a base; any explicitly supplied
    *rules* are appended to them.

    Parameters
    ----------
    bounds : dict | ValConfig | ValidationConfig
        Bounds specification or config object.
    rules : list[callable] | None
        Additional relational rules to merge with any config-defined rules.
    caller : str
        Name of the calling function, used in error messages.

    Returns
    -------
    tuple[dict, list[callable] | None]
        Resolved ``(bounds_dict, rules)`` pair.
    """
    if hasattr(bounds, "val_config"):
        if bounds.val_config is None:
            raise ValueError(
                f"The ValidationConfig passed to {caller} has no val_config "
                "section defined. Add a val_config block to the YAML config."
            )
        cfg_rules = bounds.val_config.get_rules()
        bounds = bounds.val_config.get_bounds()
        rules = (cfg_rules + list(rules)) if rules else (cfg_rules or None)
    elif hasattr(bounds, "get_bounds"):
        cfg_rules = bounds.get_rules()
        bounds = bounds.get_bounds()
        rules = (cfg_rules + list(rules)) if rules else (cfg_rules or None)
    return bounds, rules


def _prepare_bound_arrays(
    bounds: dict,
) -> "tuple[list, list, list, np.ndarray, np.ndarray, dict]":
    """Parse a resolved bounds dict into arrays needed for QMC sampling.

    Parameters
    ----------
    bounds : dict[str, tuple | numpy.ndarray]
        Resolved bounds dict mapping column names to ``(lo, hi)`` box specs
        or ``numpy.ndarray`` of shape ``(m, 3)`` interpolated-limit tables.

    Returns
    -------
    tuple
        ``(columns, box_names, constraint_names, l_arr, u_arr, tables)``

        * ``columns`` — ordered list of all column names.
        * ``box_names`` — column names with plain box bounds.
        * ``constraint_names`` — column names with interpolated-limit tables.
        * ``l_arr`` / ``u_arr`` — lower/upper bound arrays (one entry per column).
        * ``tables`` — mapping from constraint column name to its ``(m, 3)`` array.

    Raises
    ------
    ValueError
        For unsupported bound specs or ill-formed interpolation tables.
    """
    columns = list(bounds.keys())
    box_names = [c for c in columns if _is_box_spec(bounds[c])]
    constraint_names = [c for c in columns if isinstance(bounds[c], np.ndarray)]

    unknown = set(columns) - set(box_names) - set(constraint_names)
    if unknown:
        raise ValueError(
            f"Unsupported bound spec for {sorted(unknown)}. Use a (lo, hi) tuple "
            "for box bounds or a numpy array of shape (m, 3) for interpolated "
            "limits."
        )

    l_bounds: list[float] = []
    u_bounds: list[float] = []
    tables: dict[str, np.ndarray] = {}
    for c in columns:
        spec = bounds[c]
        if _is_box_spec(spec):
            lo, hi = float(spec[0]), float(spec[1])
            if hi < lo:
                raise ValueError(f"Box bound for '{c}' has hi < lo.")
            l_bounds.append(lo)
            u_bounds.append(hi)
        else:
            table = np.asarray(spec, dtype=float)
            if table.ndim != 2 or table.shape[1] != 3:
                raise ValueError(
                    f"Interpolated limit for '{c}' must have shape (m, 3) with "
                    "columns [independent_value, lower, upper]."
                )
            tables[c] = table
            l_bounds.append(float(table[:, 1].min()))
            u_bounds.append(float(table[:, 2].max()))

    l_arr = np.asarray(l_bounds)
    u_arr = np.asarray(u_bounds)
    return columns, box_names, constraint_names, l_arr, u_arr, tables


def _make_unit_mapper(
    columns: list,
    l_arr: np.ndarray,
    u_arr: np.ndarray,
    indep_idx: "int | None",
    constraint_names: list,
    tables: "dict[str, np.ndarray]",
) -> "callable":
    """Return a function that maps unit-hypercube coordinates to physical space.

    Box dimensions are scaled linearly to their ``[lo, hi]`` range.  Each
    interpolated-limit dimension is placed uniformly within its row-local
    feasible band ``[lower(t), upper(t)]`` via the conditional inverse-transform,
    keeping the independent axis marginal intact regardless of the underlying
    QMC sampler.

    Parameters
    ----------
    columns : list[str]
        Ordered list of all column names.
    l_arr, u_arr : numpy.ndarray
        Lower/upper bound arrays, one entry per column.
    indep_idx : int | None
        Column index of the independent (box) variable driving envelope limits,
        or ``None`` when no interpolated-limit columns are present.
    constraint_names : list[str]
        Names of interpolated-limit columns.
    tables : dict[str, numpy.ndarray]
        Mapping from constraint column name to its ``(m, 3)`` table.

    Returns
    -------
    callable
        ``map(unit) -> points`` where *unit* is an ``(n, d)`` array of unit
        coordinates and *points* is the corresponding ``(n, d)`` physical array.
    """
    def _map(unit: np.ndarray) -> np.ndarray:
        points = qmc.scale(unit, l_arr, u_arr)
        if indep_idx is not None:
            indep_values = points[:, indep_idx]
            for c in constraint_names:
                j = columns.index(c)
                lower_i, upper_i = _interp_limits(tables[c], indep_values)
                points[:, j] = lower_i + unit[:, j] * (upper_i - lower_i)
        return points

    return _map


def extract_bounds(
    df: pd.DataFrame,
    indep_col: str,
    dep_cols: list[str] | None = None,
    n_bins: int = 20,
) -> dict[str, object]:
    """Extract a ``bounds`` dict from a measured dataset for use with :func:`generate_lhs`.

    The independent variable ``indep_col`` becomes a box bound ``(lo, hi)``.
    Every dependent column in ``dep_cols`` (default: all other numeric columns)
    becomes an interpolated-limit array of shape ``(m, 3)`` with columns
    ``[indep_value, lower, upper]``, derived by binning the independent variable
    and computing the min/max of the dependent variable inside each bin.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataset (e.g. ``input_source_df`` from a validation config).
    indep_col : str
        Name of the independent (box) variable, e.g. ``"I.S [A]"``.
    dep_cols : list[str] | None, default None
        Dependent variable columns to include.  If *None*, all numeric columns
        except ``indep_col`` are used.
    n_bins : int, default 20
        Number of equal-width bins used to discretise the independent variable
        when computing per-bin min/max limits.

    Returns
    -------
    dict[str, tuple | numpy.ndarray]
        Ready-to-use ``bounds`` dict for :func:`generate_lhs`:

        * ``indep_col`` → ``(lo, hi)`` tuple (box bound)
        * each dependent column → ``numpy.ndarray`` of shape ``(m, 3)``
          with columns ``[indep_mid, lower, upper]``

    Raises
    ------
    ValueError
        If ``indep_col`` is not in ``df``, any ``dep_cols`` entry is missing,
        or the DataFrame is empty.

    Examples
    --------
    >>> bounds = extract_bounds(
    ...     input_source_df,
    ...     indep_col="I.S [A]",
    ...     dep_cols=["p.Si.A [barg]", "p.So.A [barg]"],
    ... )
    >>> app_df = generate_lhs(bounds, n=500, seed=42)
    """
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    if indep_col not in df.columns:
        raise ValueError(
            f"Independent column '{indep_col}' not found in DataFrame."
        )

    if dep_cols is None:
        dep_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if c != indep_col
        ]
    else:
        missing = [c for c in dep_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Dependent columns not found in DataFrame: {missing}"
            )

    indep = df[indep_col].to_numpy(dtype=float)
    lo_indep, hi_indep = float(np.nanmin(indep)), float(np.nanmax(indep))

    bounds: dict[str, object] = {indep_col: (lo_indep, hi_indep)}

    bin_edges = np.linspace(lo_indep, hi_indep, n_bins + 1)
    bin_idx = np.digitize(indep, bin_edges, right=False)
    bin_idx = np.clip(bin_idx, 1, n_bins)  # map edge case at hi to last bin

    for col in dep_cols:
        values = df[col].to_numpy(dtype=float)
        rows: list[list[float]] = []
        for b in range(1, n_bins + 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            mid = 0.5 * (bin_edges[b - 1] + bin_edges[b])
            lower = float(np.nanmin(values[mask]))
            upper = float(np.nanmax(values[mask]))
            rows.append([mid, lower, upper])
        if not rows:
            raise ValueError(
                f"No data found for dependent column '{col}' after binning."
            )
        bounds[col] = np.array(rows, dtype=float)

    return bounds


def fltr_bounds(
    df: pd.DataFrame,
    bounds: dict[str, object],
    tol: float = 1e-3,
) -> pd.DataFrame:
    """Filter rows of *df* that fall outside the operating envelope defined by *bounds*.

    Each entry in ``bounds`` is evaluated against the corresponding column of
    ``df``:

    * **Box bound** ``(lo, hi)`` – rows where the column value is outside
      ``[lo, hi]`` (inclusive) are removed.
    * **Interpolated limit** (a ``numpy.ndarray`` of shape ``(m, 3)`` with
      columns ``[independent_value, lower, upper]``) – the feasible range of
      the dependent column is a piecewise-linear function of the (single)
      independent (box) variable in ``bounds``.  Rows where the dependent
      column lies outside the interpolated ``[lower, upper]`` interval are
      removed.

    The function mirrors the constraint logic used in :func:`generate_lhs` and
    is the *inverse* of the rejection-sampling filter applied there: it accepts
    only rows that would be considered feasible.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to filter (e.g. ``input_source_df``).  Must contain all
        column names present as keys in ``bounds``.
    bounds : dict[str, tuple | numpy.ndarray]
        Operating-envelope specification as returned by :func:`extract_bounds`
        or constructed manually.  Keys must be column names of ``df``.
    tol : float, default 1e-3
        Relative tolerance for floating-point comparison. Bounds are expanded
        by this relative factor to account for precision errors: the effective
        bounds become ``[lo * (1 - tol), hi * (1 + tol)]``.

    Returns
    -------
    numpy.ndarray, dtype bool
        Boolean mask where ``True`` indicates rows that satisfy all bounds
        (accounting for the *tol* factor) and ``False`` indicates rows outside
        the bounds. Can be used to filter the input DataFrame via boolean indexing:

    Raises
    ------
    ValueError
        If a key in ``bounds`` is not found in ``df``, if an interpolated-limit
        bound is given but no (or more than one) independent variable can be
        identified, or if the bound specification is unsupported.

    Examples
    --------
    >>> bounds = extract_bounds(input_source_df, indep_col=I_A, dep_cols=[mf_An_kg_per_s])
    >>> mask = fltr_bounds(input_source_df, bounds)
    >>> filtered_df = input_source_df[mask]
    """
    columns = list(bounds.keys())
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in DataFrame: {missing}"
        )

    box_names = [c for c in columns if _is_box_spec(bounds[c])]
    constraint_names = [c for c in columns if isinstance(bounds[c], np.ndarray)]

    unknown = set(columns) - set(box_names) - set(constraint_names)
    if unknown:
        raise ValueError(
            f"Unsupported bound spec for {sorted(unknown)}. Use a (lo, hi) tuple "
            "for box bounds or a numpy array of shape (m, 3) for interpolated limits."
        )

    # Identify the independent variable required for interpolated limits.
    if constraint_names:
        if len(box_names) != 1:
            raise ValueError(
                "Interpolated-limit bounds require exactly one box (independent) "
                f"variable as the interpolation axis, got {box_names}."
            )
        indep_col = box_names[0]
        indep_values = df[indep_col].to_numpy(dtype=float)

    mask = np.ones(len(df), dtype=bool)

    for col in columns:
        spec = bounds[col]
        if _is_box_spec(spec):
            lo, hi = float(spec[0]), float(spec[1])
            tol_lo = abs(lo) * tol
            tol_hi = abs(hi) * tol
            values = df[col].to_numpy(dtype=float)
            mask &= (values >= lo - tol_lo) & (values <= hi + tol_hi)
        else:
            table = np.asarray(spec, dtype=float)
            if table.ndim != 2 or table.shape[1] != 3:
                raise ValueError(
                    f"Interpolated limit for '{col}' must have shape (m, 3) with "
                    "columns [independent_value, lower, upper]."
                )
            lower_i, upper_i = _interp_limits(table, indep_values)
            tol_lower = np.abs(lower_i) * tol
            tol_upper = np.abs(upper_i) * tol
            values = df[col].to_numpy(dtype=float)
            mask &= (values >= lower_i - tol_lower) & (values <= upper_i + tol_upper)

    return mask

def fltr_config(
        df_inp: pd.DataFrame,
        df_outp: pd.DataFrame,
        config: "ValidationConfig",
        verbose=False
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if config.val_config is None:
        raise ValueError(
            "The ValidationConfig has no val_config section defined. Add a val_config block to the YAML config."
        )
    bounds = config.val_config.get_bounds()

    if not config.val_config.rules:
        mask = fltr_bounds(df_inp, bounds)
    else:
        rules = config.val_config.get_rules()
        # Filter source data to the defined operation bounds
        mask = fltr_bounds(df_inp, bounds) & fltr_rules(df_inp, rules)

    n_kept = mask.sum()
    n_total = len(mask)
    pct = mask.mean() * 100
    if verbose:
        print(f"Filtered data: {n_kept} / {n_total} rows kept ({pct:.1f}%)")
    input_df = df_inp[mask].reset_index(drop=True)
    output_df = df_outp[mask].reset_index(drop=True)

    return input_df, output_df


def fltr_rules(
    df: pd.DataFrame,
    rules: list[callable],
) -> np.ndarray:
    """Filter rows of *df* that violate one or more relational rules.

    Each rule is a callable that receives *df* and must return a boolean
    ``pandas.Series`` or ``numpy.ndarray`` mask of the same length.  Only rows
    where **all** rules evaluate to ``True`` are kept.

    This mirrors the ``rules`` parameter of :func:`generate_lhs` and can be
    combined with :func:`fltr_bounds` to apply both envelope and relational
    constraints to a measured dataset::

        mask = fltr_bounds(input_source_df, bounds) & fltr_rules(input_source_df, rules)
        input_source_df = input_source_df[mask]

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to filter (e.g. ``input_source_df``).
    rules : list[callable]
        List of relational constraints.  Each callable receives *df* and must
        return a boolean mask of the same length.  Examples::

            rules=[
                lambda df: df["p.Si.A [bara]"] > df["p.So.A [bara]"],
                lambda df: df["T.Si.A [°C]"] < df["T.Si.C [°C]"] + 10,
            ]

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``(len(df),)``; ``True`` where all rules pass.

    Raises
    ------
    ValueError
        If *rules* is empty or a rule returns a mask of unexpected length.
    """
    if not rules:
        raise ValueError("rules must be a non-empty list of callables.")

    mask = np.ones(len(df), dtype=bool)
    for rule in rules:
        result = np.asarray(rule(df), dtype=bool)
        if result.shape != (len(df),):
            raise ValueError(
                f"A rule returned a mask of shape {result.shape}, "
                f"expected ({len(df)},)."
            )
        mask &= result
    return mask


def generate_lhs(
    bounds: "dict[str, object] | ValConfig | ValidationConfig",
    n: int,
    seed: int | None = None,
    max_oversample: int = 256,
    rules: list[callable] | None = None,
    space_filling: bool = False,
    pool_factor: int = 20,
    scale: str | None = "minmax",
) -> pd.DataFrame:
    """Generate a Latin-hypercube sample of the application domain ``X_app``.

    *bounds* may be:

    * A plain ``dict`` mapping column names to bound specs (original interface).
    * A ``ValConfig`` object (the ``val_config`` field of a :class:`ValidationConfig`):
      bounds and rules are extracted automatically via ``get_bounds()`` /
      ``get_rules()``.  Any explicitly supplied *rules* argument takes precedence
      and is merged with the config rules.
    * A ``ValidationConfig`` object: its ``val_config`` attribute is used as
      above; a ``ValueError`` is raised if ``val_config`` is ``None``.

    The type of each entry in the resolved ``bounds`` dict selects the behaviour:

    * **Box bound** ``(lo, hi)`` (a tuple/list) - an independent operating
      variable sampled uniformly between ``lo`` and ``hi``.
    * **Interpolated limit** (a ``numpy.ndarray`` of shape ``(m, 3)`` with
      columns ``[independent_value, lower, upper]``) - a dependent variable
      whose feasible range is a piecewise-linear function of the (single)
      independent variable. This yields a *constraint* LHS restricted to the
      coupled operating envelope.

    If every entry is a box bound, a plain Box-LHS is returned. As soon as one
    interpolated limit is present, a Constraint-LHS is built by a conditional
    inverse-transform: a single Latin hypercube is drawn and each dependent
    variable's unit coordinate is mapped into its row-local feasible band
    ``[lower(t), upper(t)]``. This keeps the independent variable's uniform LHS
    marginal intact (constant sampling density along the independent axis)
    instead of clustering points where the band is wide, as plain rejection
    sampling would. Exactly one box variable is then required as the
    independent axis.

    Additional relational constraints between variables can be passed via
    ``rules``. Each rule is a callable that receives the current candidate
    ``pandas.DataFrame`` (columns matching ``bounds`` keys) and must return a
    boolean ``pandas.Series`` or ``numpy.ndarray`` mask. Only rows where **all**
    rules evaluate to ``True`` are kept. Rules trigger rejection sampling even
    when all bounds are plain box bounds.

    The resulting frame can be passed directly as ``application_df`` to
    :func:`cal_val_dist` and :func:`propose_val_experiments`.

    Parameters
    ----------
    bounds : dict[str, tuple | numpy.ndarray] | ValConfig | ValidationConfig
        Specification per dimension of ``X`` (see above), **or** a
        ``ValConfig`` / ``ValidationConfig`` object from which bounds and rules
        are extracted automatically.  Column order of the result follows the
        dict (or config bounds) order.
    n : int
        Number of sample points to return.
    seed : int | None, default None
        Seed for reproducible sampling.
    max_oversample : int, default 256
        Maximum oversampling factor used when rejection sampling the constraint
        region before giving up.
    rules : list[callable] | None, default None
        Optional list of relational constraints applied on top of ``bounds``.
        When *bounds* is a config object, any rules defined there are used as
        a base; rules passed here are **appended** to them.  Each callable
        receives a ``pandas.DataFrame`` with the same columns as ``bounds``
        and must return a boolean mask of the same length.  Examples::

            rules=[
                lambda df: df["p.Si.A [barg]"] > df["p.So.A [barg]"],
                lambda df: df["T.Si.A [°C]"] < df["T.Si.C [°C]"] + 10,
            ]

    space_filling : bool, default False
        When *True* and ``rules`` are present (Path C), collect a large
        feasible candidate pool of ``pool_factor * n`` points and then thin it
        to ``n`` via a greedy farthest-first (minimax k-center) selection.
        This minimises the worst-case coverage gap in the constrained feasible
        region.  Rules are guaranteed because selection only picks rows from the
        already rule-filtered pool.  Has no effect when no ``rules`` are given
        (the envelope-only path already returns a proper LHS without rejection).
    pool_factor : int, default 20
        Target pool size multiplier used when ``space_filling=True``: the
        rejection loop tries to collect at least ``pool_factor * n`` feasible
        points before thinning.  Higher values give the farthest-first selector
        more candidates to choose from, improving coverage quality at the cost
        of more rejection draws.  Ignored when ``space_filling=False``.
    scale : {"minmax", "std", None}, default "minmax"
        Per-dimension scaling applied before computing farthest-first distances
        when ``space_filling=True``.  Same convention as :func:`cal_val_dist`.
        Ignored when ``space_filling=False``.

    Returns
    -------
    pandas.DataFrame
        ``n`` rows with one column per entry in ``bounds``.

    Raises
    ------
    ValueError
        For invalid specs, an unsupported number of independent variables, or
        if not enough feasible points can be drawn within the envelope.
    """
    bounds, rules = _resolve_bounds_and_rules(bounds, rules, caller="generate_lhs")

    if n < 1:
        raise ValueError("n must be a positive integer.")

    columns, box_names, constraint_names, l_arr, u_arr, tables = _prepare_bound_arrays(
        bounds
    )

    # Fast path: plain Box-LHS when there are no coupled limits and no rules.
    if not constraint_names and not rules:
        sampler = qmc.LatinHypercube(d=len(columns), seed=seed)
        unit = sampler.random(n)
        scaled = qmc.scale(unit, l_arr, u_arr)
        return pd.DataFrame(scaled, columns=columns)

    if constraint_names:
        if len(box_names) != 1:
            raise ValueError(
                "Constraint-LHS requires exactly one box (independent) variable as "
                f"the interpolation axis, got {box_names}."
            )
        indep_idx = columns.index(box_names[0])
    else:
        # Only rules present – no interpolated limits, no single-indep restriction.
        indep_idx = None

    _map_unit_to_points = _make_unit_mapper(
        columns, l_arr, u_arr, indep_idx, constraint_names, tables
    )

    # Envelope-only: a single LHS is mapped analytically into the feasible
    # band.  Every drawn point is feasible by construction, so no rejection (and
    # no oversampling) is needed and the result is a proper Latin hypercube
    # whose independent-axis marginal is preserved.
    if constraint_names and not rules:
        sampler = qmc.LatinHypercube(d=len(columns), seed=seed)
        unit = sampler.random(n)
        points = _map_unit_to_points(unit)
        return pd.DataFrame(points, columns=columns)

    # Remaining cases involve relational rules that cannot be inverted in
    # general, so they are honored by rejection sampling, growing the draw until
    # enough feasible points are collected.  Interpolated-envelope columns (if
    # any) are still placed exactly inside their band by the conditional
    # inverse-transform above, so only rule violations cause rejection.
    #
    # When space_filling=True we attempt to collect pool_factor * n feasible
    # points and then thin to n via farthest-first selection so that the
    # result minimises the worst-case coverage gap instead of keeping the first
    # n survivors (which is a random-uniform but unstructured subset).
    target = pool_factor * n if space_filling else n
    collected: list[np.ndarray] = []
    n_have = 0
    factor = 4
    base_seed = seed
    while n_have < target and factor <= max_oversample:
        sampler = qmc.LatinHypercube(d=len(columns), seed=base_seed)
        unit = sampler.random(n * factor)
        points = _map_unit_to_points(unit)

        tmp_df = pd.DataFrame(points, columns=columns)
        mask = np.ones(points.shape[0], dtype=bool)
        for rule in rules:
            mask &= np.asarray(rule(tmp_df), dtype=bool)

        feasible = points[mask]
        if feasible.size:
            collected.append(feasible)
            n_have += feasible.shape[0]

        factor *= 2
        if base_seed is not None:
            base_seed += 1

    if n_have < n:
        raise ValueError(
            f"Could only draw {n_have} feasible points of the requested {n} "
            "satisfying the relational rules. The feasible region may be very "
            "small relative to the bounding box; increase max_oversample or "
            "loosen the constraints."
        )

    pool = np.vstack(collected)
    if space_filling and pool.shape[0] > n:
        pool_df = pd.DataFrame(pool, columns=columns)
        factors = _scaling_factors(pool_df, scale)
        scaled_pool = pool / factors
        centroid = scaled_pool.mean(axis=0)
        start_idx = int(np.argmin(np.linalg.norm(scaled_pool - centroid, axis=1)))
        chosen = _farthest_first_indices(scaled_pool, n, start_idx)
        return pool_df.iloc[chosen].reset_index(drop=True)

    return pd.DataFrame(pool[:n], columns=columns)


def generate_sobol(
    bounds: "dict[str, object] | ValConfig | ValidationConfig",
    n: int,
    seed: int | None = None,
    scramble: bool = True,
    optimization: str | None = None,
    max_oversample: int = 256,
    rejection: bool = False,
    rules: list[callable] | None = None,
    space_filling: bool = False,
    pool_factor: int = 20,
    scale: str | None = "minmax",
) -> pd.DataFrame:
    """Generate a Sobol' low-discrepancy sequence over the application domain.

    Uses ``scipy.stats.qmc.Sobol`` as the underlying sampler.  The *bounds*
    specification and the constraint/rules machinery are identical to
    :func:`generate_lhs`; refer to its docstring for full details on box bounds,
    interpolated-envelope constraints, and relational ``rules``.

    **Key difference from LHS:** ``n`` must be a positive power of 2
    (``1, 2, 4, 8, …``).  Sobol' sequences achieve their low-discrepancy
    guarantee only at ``n = 2**m`` sample sizes; other sizes are rejected with
    a ``ValueError`` that suggests the nearest valid values.

    **Note on rejection sampling:** When ``rules`` are supplied, rejection
    sampling is applied on top of the Sobol' draws (identical to the LHS path).
    This perturbs the pure low-discrepancy structure of the sequence — a more
    significant degradation than for LHS because low discrepancy is the main
    selling point of Sobol'.  Using ``space_filling=True`` partially restores
    good spatial coverage by drawing a large candidate pool and thinning it via
    farthest-first selection.

    *bounds* may be:

    * A plain ``dict`` mapping column names to bound specs.
    * A ``ValConfig`` object: bounds and rules are extracted automatically.
    * A ``ValidationConfig`` object: its ``val_config`` attribute is used.

    Parameters
    ----------
    bounds : dict[str, tuple | numpy.ndarray] | ValConfig | ValidationConfig
        Specification per dimension (see :func:`generate_lhs` for full details).
    n : int
        Number of sample points to return.  **Must be a power of 2.**
    seed : int | None, default None
        Seed for reproducible sampling (requires ``scramble=True`` to take
        effect; Owen-scrambled Sobol' is a randomised QMC method).
    scramble : bool, default True
        Apply Owen-scrambling to the Sobol' sequence.  Scrambling enables
        seed-based reproducibility, avoids the deterministic origin point
        ``(0, 0, …)`` in the first sample, and is generally recommended.
        When ``False`` the classical deterministic sequence is returned and
        *seed* has no effect.
    optimization : {"random-cd", "lloyd", None}, default None
        Post-hoc optimisation passed to ``scipy.stats.qmc.Sobol``.  ``None``
        disables it (fastest).  ``"random-cd"`` minimises the centered
        discrepancy; ``"lloyd"`` applies Lloyd's algorithm to space the points
        more evenly.  Requires scipy >= 1.10.
    max_oversample : int, default 256
        Maximum oversampling factor used when rejection-sampling the constraint
        region.  Ignored when no ``rules`` are present or when
        ``rejection=False``.
    rejection : bool, default False
        Controls whether infeasible samples (those that fail ``rules``) are
        replaced by drawing additional Sobol' points.

        * ``True`` (default) – rejection sampling with replacement: draw in
          batches, discarding infeasible points, until ``n`` (or
          ``pool_factor * n`` when ``space_filling=True``) feasible points are
          collected.  Raises ``ValueError`` if the feasible region is too small.
        * ``False`` – single-pass filtering: draw exactly ``n`` points (or
          ``pool_factor * n`` when ``space_filling=True``), apply ``rules``
          once, and discard infeasible points **without replacement**.  The
          returned ``DataFrame`` may contain fewer than ``n`` rows.
          ``max_oversample`` is ignored.
    rules : list[callable] | None, default None
        Optional relational constraints; each callable receives the current
        candidate ``DataFrame`` and must return a boolean mask.
        See :func:`generate_lhs` for examples.
    space_filling : bool, default False
        When ``True`` and ``rules`` are present, collect a large feasible
        candidate pool of ``pool_factor * n`` points and thin to ``n`` via
        greedy farthest-first (minimax k-center) selection.  Recommended when
        ``rules`` reduce the effective sample size, as it partially restores
        the low-discrepancy spatial coverage lost to rejection sampling.
    pool_factor : int, default 20
        Pool size multiplier used when ``space_filling=True``.  Ignored
        otherwise.
    scale : {"minmax", "std", None}, default "minmax"
        Per-dimension scaling applied before farthest-first distances when
        ``space_filling=True``.  Same convention as :func:`cal_val_dist`.
        Ignored when ``space_filling=False``.

    Returns
    -------
    pandas.DataFrame
        ``n`` rows with one column per entry in ``bounds``.

    Raises
    ------
    ValueError
        If ``n`` is not a positive power of 2, for invalid bound specs, or if
        not enough feasible points can be drawn within the constraint envelope.
    """
    bounds, rules = _resolve_bounds_and_rules(bounds, rules, caller="generate_sobol")

    if n < 1:
        raise ValueError("n must be a positive integer.")
    if n & (n - 1) != 0:
        hi = 1
        while hi < n:
            hi <<= 1
        raise ValueError(
            f"n must be a power of 2 for Sobol' sampling (got n={n}). "
            f"Use n={hi >> 1} or n={hi}."
        )

    columns, box_names, constraint_names, l_arr, u_arr, tables = _prepare_bound_arrays(
        bounds
    )

    # Fast path: plain box bounds, no constraints, no rules.
    if not constraint_names and not rules:
        sampler = qmc.Sobol(
            d=len(columns), scramble=scramble, seed=seed, optimization=optimization
        )
        unit = sampler.random(n)
        scaled = qmc.scale(unit, l_arr, u_arr)
        return pd.DataFrame(scaled, columns=columns)

    if constraint_names:
        if len(box_names) != 1:
            raise ValueError(
                "Constraint Sobol' requires exactly one box (independent) variable "
                f"as the interpolation axis, got {box_names}."
            )
        indep_idx = columns.index(box_names[0])
    else:
        indep_idx = None

    _map_unit_to_points = _make_unit_mapper(
        columns, l_arr, u_arr, indep_idx, constraint_names, tables
    )

    # Envelope-only: a single Sobol' draw mapped analytically into the feasible
    # band via the conditional inverse-transform.  Every point is feasible by
    # construction; no rejection sampling is needed.
    if constraint_names and not rules:
        sampler = qmc.Sobol(
            d=len(columns), scramble=scramble, seed=seed, optimization=optimization
        )
        unit = sampler.random(n)
        points = _map_unit_to_points(unit)
        return pd.DataFrame(points, columns=columns)

    # Rules path: two modes depending on the `rejection` flag.
    if not rejection:
        # Single-pass: draw once, filter, return survivors (may be fewer than n).
        draw_n = pool_factor * n if space_filling else n
        sampler = qmc.Sobol(
            d=len(columns), scramble=scramble, seed=seed, optimization=optimization
        )
        unit = sampler.random(draw_n)
        points = _map_unit_to_points(unit)
        tmp_df = pd.DataFrame(points, columns=columns)
        mask = np.ones(points.shape[0], dtype=bool)
        for rule in rules:
            mask &= np.asarray(rule(tmp_df), dtype=bool)
        pool = points[mask]
        if space_filling and pool.shape[0] > n:
            pool_df = pd.DataFrame(pool, columns=columns)
            factors = _scaling_factors(pool_df, scale)
            scaled_pool = pool / factors
            centroid = scaled_pool.mean(axis=0)
            start_idx = int(np.argmin(np.linalg.norm(scaled_pool - centroid, axis=1)))
            chosen = _farthest_first_indices(scaled_pool, n, start_idx)
            return pool_df.iloc[chosen].reset_index(drop=True)
        return pd.DataFrame(pool, columns=columns)

    # Rejection sampling with replacement: draw in batches until target is met.
    # Each batch draws n * factor points.  Because n is a power of 2 and factor
    # starts at 4 (= 2**2) and doubles each round (4, 8, 16, …), every batch
    # size n * factor is also a power of 2, preserving Sobol' balance within
    # each individual batch.
    target = pool_factor * n if space_filling else n
    collected: list[np.ndarray] = []
    n_have = 0
    factor = 4
    base_seed = seed
    while n_have < target and factor <= max_oversample:
        sampler = qmc.Sobol(
            d=len(columns), scramble=scramble, seed=base_seed, optimization=optimization
        )
        unit = sampler.random(n * factor)
        points = _map_unit_to_points(unit)

        tmp_df = pd.DataFrame(points, columns=columns)
        mask = np.ones(points.shape[0], dtype=bool)
        for rule in rules:
            mask &= np.asarray(rule(tmp_df), dtype=bool)

        feasible = points[mask]
        if feasible.size:
            collected.append(feasible)
            n_have += feasible.shape[0]

        factor *= 2
        if base_seed is not None:
            base_seed += 1

    if n_have < n:
        raise ValueError(
            f"Could only draw {n_have} feasible points of the requested {n} "
            "satisfying the relational rules. The feasible region may be very "
            "small relative to the bounding box; increase max_oversample or "
            "loosen the constraints."
        )

    pool = np.vstack(collected)
    if space_filling and pool.shape[0] > n:
        pool_df = pd.DataFrame(pool, columns=columns)
        factors = _scaling_factors(pool_df, scale)
        scaled_pool = pool / factors
        centroid = scaled_pool.mean(axis=0)
        start_idx = int(np.argmin(np.linalg.norm(scaled_pool - centroid, axis=1)))
        chosen = _farthest_first_indices(scaled_pool, n, start_idx)
        return pool_df.iloc[chosen].reset_index(drop=True)

    return pd.DataFrame(pool[:n], columns=columns)


def _interp_value(table: np.ndarray, indep_values: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of a single-value dependent variable.

    ``table`` has shape ``(m, 2)`` with columns ``[independent_value, value]``.
    Values between support points are linearly interpolated and clamped outside
    the support range.
    """
    ref = table[:, 0]
    order = np.argsort(ref)
    return np.interp(indep_values, ref[order], table[order, 1])

    ValueError
def generate_doe(
    bounds: dict[str, object],
) -> pd.DataFrame:
    """Generate a deterministic Design of Experiments from explicit bound values.

    Unlike :func:`generate_lhs`, this function does **not** sample at random.
    The independent variable is given as a list of specific values; every
    dependent variable is evaluated deterministically by piecewise-linear
    interpolation at those values.

    The type of each entry in ``bounds`` selects the behaviour:

    * **DOE box bound** (a ``list`` of floats, any length ≥ 1) — the
      independent variable.  Each element becomes exactly one row in the
      result.  Exactly **one** entry may be a DOE box bound.
    * **Interpolated value** (a ``numpy.ndarray`` of shape ``(m, 2)`` with
      columns ``[independent_value, value]``) — a dependent variable whose
      value is a piecewise-linear function of the independent variable.  One
      scalar is returned per row, obtained by interpolating (and clamping
      outside the support range) the supplied table.

    Parameters
    ----------
    bounds : dict[str, list | numpy.ndarray]
        Specification per dimension:

        * independent variable → ``list`` of specific float values, e.g.
          ``[0.0, 0.5, 1.0]``
        * dependent variable → ``numpy.ndarray`` of shape ``(m, 2)`` with
          columns ``[independent_value, value]``

    Returns
    -------
    pandas.DataFrame
        One row per element in the independent-variable list; one column per
        entry in ``bounds``.

    Raises
    ------
    ValueError
        If the number of DOE box bounds is not exactly one, if a dependent
        bound array has an unexpected shape, or if ``bounds`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> bounds = {
    ...     "I [A]": [0.0, 50.0, 100.0],
    ...     "T [°C]": np.array([[0.0, 20.0], [50.0, 45.0], [100.0, 70.0]]),
    ... }
    >>> generate_doe(bounds)
       I [A]  T [°C]
    0    0.0    20.0
    1   50.0    45.0
    2  100.0    70.0
    """
    if not bounds:
        raise ValueError("bounds must be a non-empty dict.")

    columns = list(bounds.keys())
    indep_name = columns[0]
    dep_names = columns[1:]

    indep_values = np.asarray(bounds[indep_name], dtype=float)

    tables: dict[str, np.ndarray] = {}
    for c in dep_names:
        table = np.asarray(bounds[c], dtype=float)
        if table.ndim != 2 or table.shape[1] != 2:
            raise ValueError(
                f"Interpolated value for '{c}' must have shape (m, 2) with "
                "columns [independent_value, value]."
            )
        tables[c] = table

    data: dict[str, np.ndarray] = {}
    for c in columns:
        if c == indep_name:
            data[c] = indep_values
        else:
            data[c] = _interp_value(tables[c], indep_values)

    return pd.DataFrame(data, columns=columns)


def export_profile(
    template_path: str | Path,
    df: pd.DataFrame,
    output_path: str | Path,
    hold_time: float | Sequence[float] | np.ndarray | pd.Series,
    dt_ramp: float | None = None,
) -> None:
    """Write a test-bench profile .txt file from a template and a DataFrame.

    Reads *template_path* (a tab-separated profile file with German comma
    decimal separator) and produces a new profile at *output_path* with
    exactly ``len(df)`` steps.  The first row of the template supplies default
    values for all profile columns absent from *df*; columns whose names match
    *df* columns are replaced with the corresponding row values from *df*.
    ``Step``, ``TotalTime [s]``, ``Reading_Point []``, and ``I_Delay [s]`` are
    recomputed automatically.

    **Automatic I_Delay computation:** When both ``I_Delay [s]`` and
    ``I_S_set [A]`` are present in the profile, ``I_Delay [s]`` is
    automatically recomputed for current-decrease safety:

    * Row 0 → ``1.0``
    * Row j → ``0.0`` if ``I_S_set [A][j-1] > I_S_set [A][j]`` (current
      decreased), else ``1.0``

    Parameters
    ----------
    template_path : str or Path
        Path to the template profile .txt file (tab-separated, comma decimal
        separator as used by the test-bench software).
    df : pandas.DataFrame
        One row per profile step.  Column names must match the profile column
        headers for the variables to be overridden.  Extra columns present in
        *df* but absent from the profile are silently ignored.
    output_path : str or Path
        Destination path for the generated .txt file.  Parent directories are
        created automatically if they do not exist.
    hold_time : float or sequence of float
        Duration in seconds assigned to each step (``Time [s]`` column).  A
        scalar assigns the same hold time to every step.  A sequence (e.g.
        ``list``, ``numpy.ndarray``, ``pandas.Series``) of length ``len(df)``
        assigns an individual, per-step hold time (dwell time), as needed for
        e.g. load-cycle profiles with a varying step duration.  ``TotalTime
        [s]`` is recomputed as the cumulative sum of the per-step hold times.
    dt_ramp : float, optional
        Current ramp rate in A/s used to move the stack current setpoint from
        one step to the next.  If given, every row's ``I_Ramp [A/s]`` column
        is overwritten with *dt_ramp*, and ``LinearizeTime [s]`` is (re)computed
        as ``abs(I_S_set [A][step] - I_S_set [A][step - 1]) / dt_ramp``, i.e.
        the time required to linearly ramp between the two steps' current
        setpoints at the given rate.  The first step has no predecessor within
        the profile, so its ``LinearizeTime [s]`` is 0.  Requires the profile
        (template and/or *df*) to contain the ``I_Ramp [A/s]``,
        ``LinearizeTime [s]``, and ``I_S_set [A]`` columns.  If not given
        (default), ``I_Ramp [A/s]`` and ``LinearizeTime [s]`` are left
        unchanged (inherited from the template / *df*, as for any other
        column).

    Raises
    ------
    ValueError
        If the template file cannot be decoded, if ``df`` is empty, if
        ``hold_time`` is a sequence whose length does not match ``len(df)``,
        if ``dt_ramp`` is not a positive number, or if ``dt_ramp`` is given
        but the profile is missing one of the required columns.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)

    if df.empty:
        raise ValueError("df must not be empty.")

    if isinstance(hold_time, (int, float)):
        hold_times = np.full(len(df), float(hold_time))
    else:
        hold_times = np.asarray(hold_time, dtype=float)
        if hold_times.shape != (len(df),):
            raise ValueError(
                "hold_time must be a scalar or a sequence of length "
                f"len(df)={len(df)}, got shape {hold_times.shape}."
            )

    if dt_ramp is not None and dt_ramp <= 0:
        raise ValueError("dt_ramp must be a positive number of A/s.")

    enc_used: str = ""
    for enc in ("utf-8", "latin-1"):
        try:
            tmpl = pd.read_csv(template_path, sep="\t", decimal=",", encoding=enc)
            enc_used = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            f"Could not decode '{template_path}' as utf-8 or latin-1."
        )

    # Determine which columns hold floating-point values in the template.
    float_cols: set[str] = set(tmpl.select_dtypes(include="float").columns)

    n_steps = len(df)
    ref = tmpl.iloc[0].to_dict()

    # Start from the reference row repeated n_steps times.
    new_profile = pd.DataFrame([ref] * n_steps, columns=tmpl.columns)

    # Override with df values wherever column names match the profile header.
    shared = [c for c in df.columns if c in new_profile.columns]
    for col in shared:
        new_profile[col] = df[col].to_numpy()

    # Ensure float columns keep their numeric dtype after override.
    for col in float_cols:
        new_profile[col] = new_profile[col].astype(float)

    # Recompute structural / derived columns.
    new_profile["Step"] = np.arange(1, n_steps + 1)
    new_profile["Time [s]"] = hold_times
    new_profile["TotalTime [s]"] = np.cumsum(hold_times)

    if "Reading_Point []" in new_profile.columns:
        base_rp = int(ref["Reading_Point []"]) - 1
        new_profile["Reading_Point []"] = np.arange(1, n_steps + 1) + base_rp

    if dt_ramp is not None:
        required_cols = {"I_Ramp [A/s]", "LinearizeTime [s]", "I_S_set [A]"}
        missing_cols = required_cols - set(new_profile.columns)
        if missing_cols:
            raise ValueError(
                "dt_ramp requires the profile to contain the columns "
                f"{sorted(required_cols)}; missing: {sorted(missing_cols)}."
            )

        current = new_profile["I_S_set [A]"].to_numpy(dtype=float)
        delta_current = np.abs(np.diff(current, prepend=current[0]))
        new_profile["I_Ramp [A/s]"] = float(dt_ramp)
        new_profile["LinearizeTime [s]"] = delta_current / float(dt_ramp)

    # Recompute I_Delay [s] based on current setpoint changes.
    if "I_Delay [s]" in new_profile.columns and "I_S_set [A]" in new_profile.columns:
        current = new_profile["I_S_set [A]"].to_numpy(dtype=float)
        delay = np.ones(n_steps, dtype=float)
        delay[1:] = np.where(current[:-1] > current[1:], 0.0, 1.0)
        new_profile["I_Delay [s]"] = delay

    # Write in tab-separated format with German comma decimal separator.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding=enc_used, newline="\n") as fh:
        fh.write("\t".join(tmpl.columns) + "\n")
        for _, row in new_profile.iterrows():
            parts = []
            for col in tmpl.columns:
                val = row[col]
                if col in float_cols:
                    parts.append(f"{float(val):.2f}".replace(".", ","))
                else:
                    parts.append(str(int(round(float(val)))))
            fh.write("\t".join(parts) + "\n")


def remove_twins(df: pd.DataFrame, dist: float = 1e-3, remove_single: bool = True) -> pd.DataFrame:
    """Remove duplicate/close points by merging points within a distance threshold.

    Identifies points in the DataFrame that are closer than *dist* to each other
    and merges them by averaging their numeric values. Uses union-find clustering
    to handle transitive relationships: if point A is within *dist* of B, and B is
    within *dist* of C, all three are merged into a single point even if A and C
    are farther apart.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with rows as points and columns as dimensions. Typically
        contains only numeric columns, but non-numeric columns are preserved
        from the first point in each cluster.
    dist : float, default 1e-3
        Distance threshold. Points with pairwise Euclidean distance below this
        value are considered twins and merged.
    remove_single : bool, default True
        If True, points that have no twin (clusters of size 1) are dropped from
        the result, so only merged (actual duplicate) points remain. If False,
        single points are kept unmerged in the output.

    Returns
    -------
    pandas.DataFrame
        DataFrame with twin points merged. Numeric columns are averaged within
        each cluster; non-numeric columns retain the value from the first point
        in the cluster. Index is reset to [0, 1, ..., n_clusters - 1].

    Raises
    ------
    ValueError
        If the DataFrame is empty.

    Notes
    -----
    Distance is computed as Euclidean distance across all numeric columns.
    No scaling is applied; use raw column values directly.
    """
    if df.empty:
        raise ValueError("DataFrame must not be empty.")

    n = len(df)
    if n == 1:
        return df.iloc[0:0].reset_index(drop=True) if remove_single else df.reset_index(drop=True)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # No numeric columns: nothing to cluster on, so every point is single.
        return df.iloc[0:0].reset_index(drop=True) if remove_single else df.reset_index(drop=True)

    # Compute pairwise distances on numeric columns only
    numeric_data = df[numeric_cols].to_numpy(dtype=float)
    # Shape: (n, n) where [i, j] is Euclidean distance between rows i and j
    distances = np.linalg.norm(
        numeric_data[:, np.newaxis, :] - numeric_data[np.newaxis, :, :],
        axis=2,
    )

    # Union-find data structure for transitive clustering
    parent = list(range(n))

    def find(x: int) -> int:
        """Find the root representative of x's cluster."""
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x: int, y: int) -> None:
        """Merge the clusters containing x and y."""
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build clusters: union all pairs within distance threshold
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < dist:
                union(i, j)

    # Group original row indices by their cluster
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Merge each cluster into a single representative row
    result_rows = []
    for cluster_indices in clusters.values():
        if len(cluster_indices) == 1:
            # Single point, no merging needed; drop it if remove_single is set.
            if remove_single:
                continue
            result_rows.append(df.iloc[cluster_indices[0]])
        else:
            # Multiple points: average numeric columns, keep first for non-numeric
            merged_row = {}
            for col in df.columns:
                if col in numeric_cols:
                    merged_row[col] = df.iloc[cluster_indices][col].mean()
                else:
                    merged_row[col] = df.iloc[cluster_indices[0]][col]
            result_rows.append(pd.Series(merged_row))

    if not result_rows:
        return df.iloc[0:0].reset_index(drop=True)

    return pd.DataFrame(result_rows).reset_index(drop=True)


def import_profile(
    profile_path: str | Path,
    columns: list[str] | None = None,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """Read a test-bench profile .txt file exported by :func:`export_profile`.

    Parses the tab-separated, comma-decimal file and returns its contents as a
    :class:`pandas.DataFrame`.  Optionally restricts the returned columns and
    removes duplicate rows.

    Parameters
    ----------
    profile_path : str or Path
        Path to the profile .txt file written by :func:`export_profile`.
    columns : list of str, optional
        If provided, only the named columns are included in the returned
        DataFrame.  A :exc:`ValueError` is raised if any requested column is
        not present in the file.
    remove_duplicates : bool, default True
        When ``True``, duplicate rows (assessed on the returned columns after
        applying *columns* filter) are dropped before returning.

    Returns
    -------
    pandas.DataFrame
        The profile data with one row per step and the column dtypes inferred
        by :func:`pandas.read_csv`.

    Raises
    ------
    ValueError
        If the file cannot be decoded as utf-8 or latin-1, or if any column
        listed in *columns* is not found in the profile file.
    """
    profile_path = Path(profile_path)

    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(profile_path, sep="\t", decimal=",", encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            f"Could not decode '{profile_path}' as utf-8 or latin-1."
        )

    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"The following requested columns are not present in "
                f"'{profile_path}': {missing}."
            )
        df = df[columns]

    if remove_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    return df