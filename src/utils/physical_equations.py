from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.utils import constants as _model_constants

R_H2 = _model_constants.R_H2
R_N2 = _model_constants.R_N2
R_steam = _model_constants.R_steam
R = _model_constants.R
T_0 = _model_constants.T_0

p_ref = 101325.0
p_amb = 95000
# VirtualFCS.ComponentTesting.TestFuelCell inner system.p_ambient
p_ambient_test_fc = 95000.0
m_flow_small = 1e-3

# Modelica.Media.IdealGases.SingleGases.H2 / MoistAir T_default
T_default_H2 = T_0
T_default_moist_air = T_0
M_w = _model_constants.MM_water
M_H2 = _model_constants.MM_hydrogen
M_N2 = _model_constants.MM_nitrogen


def from_Celsius(T: Any) -> Any:
    """Convert Celsius to Kelvin."""
    return T + 273.15

def to_Celsius(T: Any) -> Any:
    """Convert Kelvin to Celsius."""
    return T - 273.15

def from_bara(p: Any) -> Any:
    """Convert bara to Pa."""
    return p * 1e5

def from_barg(p: Any, p_ambient: float | None = None) -> Any:
    """Convert barg to absolute pressure [Pa]."""
    amb = p_ref if p_ambient is None else p_ambient
    return from_bara(p) + amb



def from_nl_per_min(v_nl_min: Any) -> Any:
    """Convert nl/min to m³/s (TestFuelCell: OpCons.y/60000)."""
    return np.asarray(v_nl_min, dtype=float) / 60000.0

def from_mbar(p: Any) -> Any:
    """Convert mbar to Pa."""
    return p * 1e2

def _restore_like(result: Any, *templates: Any) -> Any:
    """Return scalars as float and preserve pandas Series when possible."""
    for template in templates:
        if isinstance(template, pd.Series):
            arr = np.asarray(result)
            return pd.Series(arr, index=template.index, name=template.name)

    arr = np.asarray(result)
    if arr.ndim == 0:
        return float(arr)
    return arr


def splice_function(pos: Any, neg: Any, x: Any, deltax: float = 1.0) -> Any:
    """
    Port of Modelica.Media.Air.MoistAir.Utilities.spliceFunction.

    Works with scalars, NumPy arrays and pandas Series.
    """
    scaled_x1 = np.asarray(x) / deltax
    scaled_x = scaled_x1 * (np.pi / 2.0)
    y_mid = (np.tanh(np.tan(scaled_x)) + 1.0) / 2.0
    y = np.where(
        scaled_x1 <= -0.999999999,
        0.0,
        np.where(scaled_x1 >= 0.999999999, 1.0, y_mid),
    )
    result = np.asarray(pos) * y + (1.0 - y) * np.asarray(neg)
    return _restore_like(result, x, pos, neg)

def saturation_pressure_liquid(Tsat: Any) -> Any:
    """
    Modelica.Media.Air.MoistAir.saturationPressureLiquid.

    Validity: ~273.16 K to 647.096 K.
    """
    t_critical = 647.096
    p_critical = 22.064e6
    tsat_arr = np.asarray(Tsat)
    r1 = 1.0 - tsat_arr / t_critical
    a = (-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502)
    n = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)

    poly = (
        a[0] * r1**n[0]
        + a[1] * r1**n[1]
        + a[2] * r1**n[2]
        + a[3] * r1**n[3]
        + a[4] * r1**n[4]
        + a[5] * r1**n[5]
    )
    result = np.exp((poly * t_critical) / tsat_arr) * p_critical
    return _restore_like(result, Tsat)


def sublimation_pressure_ice(Tsat: Any) -> Any:
    """
    Modelica.Media.Air.MoistAir.sublimationPressureIce.

    Validity: ~190 K to 273.16 K.
    """
    t_triple = 273.16
    p_triple = 611.657
    tsat_arr = np.asarray(Tsat)
    r1 = tsat_arr / t_triple
    a = (-13.9281690, 34.7078238)
    n = (-1.5, -1.25)

    result = np.exp(a[0] - a[0] * r1**n[0] + a[1] - a[1] * r1**n[1]) * p_triple
    return _restore_like(result, Tsat)


def saturation_pressure(Tsat: Any) -> Any:
    """
    Modelica.Media.Air.MoistAir.saturationPressure.

    Blends ice/liquid formulas with spliceFunction around 273.16 K.
    """
    return splice_function(
        pos=saturation_pressure_liquid(Tsat),
        neg=sublimation_pressure_ice(Tsat),
        x=np.asarray(Tsat) - 273.16,
        deltax=1.0,
    )


def mass_fraction_pTphi(p: Any, T: Any, phi: Any) -> Any:
    """
    Port of Modelica.Media.Air.MoistAir.massFraction_pTphi.

    Inputs:
      p   : Pressure [Pa]
      T   : Temperature [K]
      phi : Relative humidity (typically 0..1)

    Returns:
      Steam mass fraction (kg steam / kg moist air).
    """
    k = 0.621964713077499
    psat = saturation_pressure(T)
    result = phi * k / (k * phi + p / psat - phi)
    return _restore_like(result, phi, p, T)


def gas_constant_X(X: Any) -> Any:
    """
    Port of Modelica.Media.Air.MoistAir.gasConstant_X.

    Parameters
    ----------
    X : float | array-like | pandas.Series
        Water vapor mass fraction directly, or an array-like object whose
        first column/entry contains the water vapor mass fraction.

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        Ideal gas constant of the moist-air mixture [J/(kg.K)].
    """
    dryair_r = 287.0512249529787
    steam_r = 461.5233290850878

    template = X
    if isinstance(X, pd.DataFrame):
        template = X.iloc[:, 0]
        x_water_arr = template.to_numpy(dtype=float)
    elif isinstance(X, np.ndarray) and X.ndim > 1:
        template = X[..., 0]
        x_water_arr = np.asarray(template, dtype=float)
    else:
        x_water_arr = np.asarray(X, dtype=float)

    result = dryair_r * (1.0 - x_water_arr) + steam_r * x_water_arr
    return _restore_like(result, template)

def calc_eta(U_c: Any, U_0 = "LHV") -> Any:
    """
    Calculate the efficiency of a system.

    Parameters
    ----------
    U_c : float | array-like | pandas.Series
        Current cell voltage [V].
    U_0 : str
        Reference energy input ("LHV" for Lower Heating Value, "HHV" for Higher Heating Value).

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        Efficiency [%].
    """
    if U_0 == "LHV":
        U_ref = _model_constants.U_LHV
    elif U_0 == "HHV":
        U_ref = _model_constants.U_HHV
    else:
        raise ValueError("Invalid reference potential input. Use 'LHV' or 'HHV'.")

    eta = (U_c / U_ref) * 100.0
    return _restore_like(eta, U_c)

def calc_mf_h2(
    p_an_in: Any,
    phi_an: Any,
    T_an_in: Any,
    vf_an: Any,
    vf_h2: Any,
    vf_n2: Any,
) -> Any:
    """
    Calculate anode total mass flow [kg/s].

    p_an_in: Pa,
    phi_an: %,
    T_an_in: K,
    vf_an: nl/min,
    vf_h2: nl/min,
    vf_n2_An: nl/min,

    mf_An = mf_H2_An + mf_N2_An + mf_H2O_An with ideal-gas partial flows.
    """
    phi_an = np.asarray(phi_an, dtype=float) / 100.0

    vf_an = from_nl_per_min(vf_an)
    vf_h2 = from_nl_per_min(vf_h2)
    vf_n2 = from_nl_per_min(vf_n2)

    x_w_an = mass_fraction_pTphi(p_an_in, T_an_in, phi_an)

    mf_h2 = vf_h2 * p_an_in / T_an_in / R_H2
    mf_n2 = vf_n2 * p_an_in / T_an_in / R_N2
    mf_h2o = vf_an * x_w_an * p_an_in / T_an_in / R_steam

    result = mf_h2 + mf_n2 + mf_h2o
    return _restore_like(result, p_an_in, phi_an, T_an_in)

def from_nl_to_mf_air(vf_air: Any) -> Any:
    return calc_mf_air(p_ref, 0, T_0, vf_air)

def from_mf_to_nl_air(mf_air: Any) -> Any:
    return calc_vf_air(p_ref, 0, T_0, mf_air)

def calc_wet_mf_air(
    vf_air: Any,
    dew_point: Any,
    p: Any,
    T: Any,
) -> Any:
    """
    Feuchter Luft-Massenstrom nach dem Befeuchter [kg/s].

    Berechnet den Massenstrom der feuchten Luft hinter dem Befeuchter aus
    dem trockenen Normvolumenstrom vor dem Befeuchter und dem Zustand
    hinter dem Befeuchter.

    Der trockene Massenstrom bleibt durch den Befeuchter erhalten; Wasser
    wird addiert. Mit dem Modelica-Massenanteil X (kg Dampf / kg feuchte
    Luft) gilt: m_wet = m_dry / (1 - X).

    Parameters
    ----------
    vf_air    : trockener Normvolumenstrom vor dem Befeuchter [nl/min]
    dew_point : Taupunkt hinter dem Befeuchter [K]
    p         : Druck hinter dem Befeuchter [Pa]
    T         : Temperatur hinter dem Befeuchter [K]

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        Feuchter Massenstrom [kg/s].
    """
    mf_dry = from_nl_to_mf_air(vf_air)

    phi = np.clip(
        np.asarray(saturation_pressure(dew_point), dtype=float)
        / np.asarray(saturation_pressure(T), dtype=float),
        0.0,
        1.0,
    )
    X = mass_fraction_pTphi(p, T, phi)
    mf_wet = np.asarray(mf_dry, dtype=float) / (1.0 - np.asarray(X, dtype=float))
    return _restore_like(mf_wet, vf_air, dew_point, p, T)

def calc_dpt(
    p: Any,
    T: Any,
    phi: Any,
) -> Any:
    """
    Calculate dew point temperature from pressure, temperature, and relative humidity.

    Solves numerically for the temperature T_dew at which the saturation pressure
    equals the partial pressure of water vapor:
        p_sat(T_dew) = phi * p_sat(T)

    Parameters
    ----------
    p   : float | array-like | pandas.Series
        Absolute pressure [Pa]
    T   : float | array-like | pandas.Series
        Current air temperature [K]
    phi : float | array-like | pandas.Series
        Relative humidity (0..1) or relative humidity (0..100 if converted via /100)

    Returns
    -------
    float | numpy.ndarray | pandas.Series
        Dew point temperature [K], same type as input
    """
    # Convert inputs to arrays for consistent handling
    p_arr = np.asarray(p, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)

    # If phi is in percent (> 1), convert to fraction
    phi_arr = np.where(phi_arr > 1.0, phi_arr / 100.0, phi_arr)

    # Clip phi to valid range [0, 1]
    phi_arr = np.clip(phi_arr, 0.0, 1.0)

    # Calculate reference partial pressure of water vapor
    p_sat_T = saturation_pressure(T_arr)
    p_vapor = phi_arr * p_sat_T

    # Handle scalar case
    if p_arr.ndim == 0 and T_arr.ndim == 0 and phi_arr.ndim == 0:
        p_scalar = float(p_arr)
        T_scalar = float(T_arr)
        phi_scalar = float(phi_arr)
        p_sat_T_scalar = float(p_sat_T)
        p_vapor_scalar = float(p_vapor)

        def error_func(T_dew_val: float) -> float:
            return float(saturation_pressure(T_dew_val)) - p_vapor_scalar

        # Search bounds: from near 0K to current temperature
        T_dew = brentq(error_func, 200.0, T_scalar + 0.1)
        return _restore_like(T_dew, p, T, phi)

    # Handle array case - vectorize the root finding
    T_dew_arr = np.empty_like(T_arr, dtype=float)

    # Flatten for iteration
    p_flat = p_arr.ravel()
    T_flat = T_arr.ravel()
    phi_flat = phi_arr.ravel()
    p_vapor_flat = np.asarray(p_vapor, dtype=float).ravel()

    for i in range(len(p_flat)):
        p_vapor_i = p_vapor_flat[i]
        T_i = T_flat[i]

        def error_func(T_dew_val: float) -> float:
            return float(saturation_pressure(T_dew_val)) - p_vapor_i

        try:
            T_dew_arr.ravel()[i] = brentq(error_func, 200.0, T_i + 0.1)
        except ValueError:
            # If brentq fails, set to current temperature (saturated air)
            T_dew_arr.ravel()[i] = T_i

    # Restore original shape and type
    result = T_dew_arr.reshape(T_arr.shape)
    return _restore_like(result, p, T, phi)

def calc_mf_air(
    p_ca_in: Any,
    phi_ca: Any,
    T_ca_in: Any,
    vf_air: Any,
) -> Any:
    """
    Cathode moist-air mass flow [kg/s].

    mf_Air = max(Vf_Air * p / T_Ca_in / R_s, m_flow_small)

    p_ca_in: Pa,
    phi_ca: %,
    T_ca_in: K,
    vf_air: nl/min,

    """
    phi_ca = np.asarray(phi_ca, dtype=float) / 100.0

    vf_air = from_nl_per_min(vf_air)
    x_w_ca = mass_fraction_pTphi(p_ca_in, T_ca_in, phi_ca)
    r_s = gas_constant_X(x_w_ca)

    mf_raw = vf_air * p_ca_in / T_ca_in / r_s
    result = np.maximum(mf_raw, m_flow_small)
    return _restore_like(result, p_ca_in, phi_ca, T_ca_in, vf_air)

def calc_vf_air(
    p_ca_in: Any,
    phi_ca: Any,
    T_ca_in: Any,
    mf_air: Any,
) -> Any:
    """
    Cathode moist-air volume flow [nl/min]. Inverse of calc_mf_air.

    Vf_Air = mf_Air * T_ca_in * R_s / p_ca_in

    p_ca_in: Pa,
    phi_ca: %,
    T_ca_in: K,
    mf_air: kg/s,

    """
    phi_ca = np.asarray(phi_ca, dtype=float) / 100.0

    x_w_ca = mass_fraction_pTphi(p_ca_in, T_ca_in, phi_ca)
    r_s = gas_constant_X(x_w_ca)

    vf_raw = mf_air * T_ca_in * r_s / p_ca_in
    result = vf_raw * 60000.0
    return _restore_like(result, p_ca_in, phi_ca, T_ca_in, mf_air)


def from_stoic(
    I: Any,
    lambda_K: Any,
    N: Any
) -> Any:
    """
    Cathode normal air flow [nl/min].

    mf_Air = max(Vf_Air * p / T_Ca_in / R_s, m_flow_small)

    lambda_K: float,
    I: A
    N: int
    """
    n_o2_con = (I * N) / (4 * _model_constants.F)
    n_o2_in = n_o2_con * lambda_K
    n_air_in = n_o2_in / _model_constants.X_O2
    vf_air = 60000 * n_air_in * (R * T_0)/p_ref
    return _restore_like(vf_air, lambda_K, I, N)


def to_stoic(
    vf_air: Any,
    I: Any,
    N: Any
) -> Any:
    """
    Inverse of from_stoic: compute stoichiometry lambda_K from cathode normal
    air flow, current, and number of cells.

    vf_air: nl/min
    I: A
    N: int

    Returns lambda_K (dimensionless stoichiometry).
    """
    n_o2_con = (I * N) / (4 * _model_constants.F)
    n_air_in = vf_air / 60000 / (R * T_0 / p_ref)
    n_o2_in = n_air_in * _model_constants.X_O2
    lambda_K = n_o2_in / n_o2_con
    return _restore_like(lambda_K, vf_air, I, N)