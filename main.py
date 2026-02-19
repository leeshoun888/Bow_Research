"""
════════════════════════════════════════════════════════════════════════════════
DIGITAL BOW PHYSICS & BALLISTICS LABORATORY v5.0
Museum-Grade Interactive Research Platform for Experimental Archaeology
════════════════════════════════════════════════════════════════════════════════

CORE FEATURES:
- Material Physics Database (Empirical Wood Properties)
- 7-Point Precision Measurement System
- Side Profile Definition (Reflex/Deflex Geometry)
- Geometric Force-Draw Curve Engine (String Angle & Stacking)
- Energy Storage & Shooting Efficiency Analysis
- Virtual Tiller: REAL-TIME EI-Dependent Limb Deformation
- Ballistics Engine: Parabolic Trajectory & Range Prediction
- Target Engagement Simulation (Effective Range Analysis)

REAL-TIME REACTIVITY (v4.3):
- Instant response to width/thickness changes in sidebar
- EI-dependent bending: thicker sections bend less (κ = M/EI)
- Side profile initial geometry fully integrated
- Brace height constraint (17cm) maintained across all configurations
- High-resolution curve interpolation (80 segments)
- FIXED v4.2: Upper and Lower limbs calculated INDEPENDENTLY
  * Each limb uses its own EI distribution
  * Asymmetric bows now properly represented
  * Lower limb changes now immediately visible in tiller graph
- FIXED v5.0: PHYSICS ENGINE REWRITE (Binary Search on Force)
  * Removed ALL post-hoc y-scaling and arc length normalization
  * Forward kinematics preserves arc length BY CONSTRUCTION (fixed ds per segment)
  * Binary search finds exact force for string length constraint (drawn states)
  * Binary search finds exact force for brace height constraint (braced state)
  * Cumulative angle clamped at 148° to prevent string vector inversion
  * String length, limb length, and angle validity all guaranteed simultaneously

THEORETICAL FOUNDATION:
- Cantilever Beam Theory with Non-Linear Geometry
- Second Moment of Area (Multiple Cross-Sections)
- EI Distribution-Based Curvature Calculation
- String Length Constraint (Geometric Invariant)
- String Angle Dynamics & Tip Velocity Penalties
- Kinetic Energy Transfer Efficiency
- Projectile Motion Physics (Parabolic Trajectories)

MUSEUM-GRADE PRESENTATION:
- Dark Mode Research Dashboard (Deep Navy/Charcoal)
- Metallic Accent Colors (Cyan/Gold)
- Vector Graphics & Geometric Line Art
- LaTeX Mathematical Notation
- Real-Time Debugging Info (EI Range & String Length Display)
- Citation-Ready Research Documentation
════════════════════════════════════════════════════════════════════════════════
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from scipy.integrate import simpson
from scipy.interpolate import interp1d, CubicSpline

# ════════════════════════════════════════════════════════════════════════════
# MATERIAL PHYSICS DATABASE
# ════════════════════════════════════════════════════════════════════════════

WOOD_PROPERTIES: Dict[str, Dict[str, float]] = {
    "Ash (물푸레나무)": {
        "elastic_modulus_gpa": 11.5,
        "density_g_cm3": 0.68,
    },
    "Yew (주목)": {
        "elastic_modulus_gpa": 10.0,
        "density_g_cm3": 0.64,
    },
    "Birch (박달나무)": {
        "elastic_modulus_gpa": 14.5,
        "density_g_cm3": 0.92,
    },
    "Juniper (노간주나무)": {
        "elastic_modulus_gpa": 7.5,
        "density_g_cm3": 0.55,
    },
    "Mountain Mulberry (산뽕나무)": {
        "elastic_modulus_gpa": 11.0,
        "density_g_cm3": 0.65,
    },
    "Black Locust (아까시나무)": {
        "elastic_modulus_gpa": 14.5,
        "density_g_cm3": 0.75,
    },
}

# ════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL SHAPE OPTIONS
# ════════════════════════════════════════════════════════════════════════════

SHAPE_OPTIONS = [
    "Elliptical",
    "Rectangular",
    "Trapezoid",
    "Inverse Trapezoid",
]

# ════════════════════════════════════════════════════════════════════════════
# SIDE PROFILE OPTIONS (Initial Bow Geometry)
# ════════════════════════════════════════════════════════════════════════════

SIDE_PROFILE_OPTIONS = {
    "Straight": {
        "description": "Linear profile, no initial curvature",
        "reflex_factor": 0.0,
        "fdc_modifier": 1.0,
        "stacking_modifier": 1.0,
    },
    "Reflex": {
        "description": "Tips curve away from archer (higher pre-tension)",
        "reflex_factor": 1.5,
        "fdc_modifier": 1.15,
        "stacking_modifier": 0.85,
    },
    "Deflex": {
        "description": "Tips curve toward archer (smooth draw)",
        "reflex_factor": -1.0,
        "fdc_modifier": 0.85,
        "stacking_modifier": 1.15,
    },
    "Recurve": {
        "description": "Tips curve away, working limb straight (classic recurve)",
        "reflex_factor": 2.0,
        "fdc_modifier": 1.25,
        "stacking_modifier": 0.75,
    },
    "Decurve": {
        "description": "Tips curve toward, working limb deflexed (low hand shock)",
        "reflex_factor": -1.5,
        "fdc_modifier": 0.75,
        "stacking_modifier": 1.25,
    },
}

# ════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

DRAW_LENGTH_INCH = 28.0
INCH_TO_M = 0.0254
INCH_TO_CM = 2.54
MM_TO_M = 0.001
CM_TO_M = 0.01
GPA_TO_PA = 1e9
MM4_TO_M4 = 1e-12
N_TO_LBS = 0.224808943
LBS_TO_N = 1.0 / N_TO_LBS
FPS_TO_MS = 0.3048
MS_TO_FPS = 1.0 / FPS_TO_MS
MS_TO_KMH = 3.6
GRAVITY = 9.81  # m/s²

# Bow Geometry Constants
TARGET_BRACE_HEIGHT_CM = 17.0  # Standard brace height for Korean traditional bow

# Integration & Simulation Parameters
INTEGRATION_STEPS = 300
FDC_DRAW_STEPS = 100  # Force-Draw Curve resolution

# String-angle / stacking model parameters
# Reference rationale (implemented as engineering approximations):
# - VirtualBow Theory Manual: non-linear FDC and efficiency losses are strongly
#   linked to string/limb kinematics and energy partition at release.
# - Kooi-style mechanics: draw force is governed by force equilibrium and
#   geometry-dependent transmission between string tension and nock force.
TIP_STRING_STACK_REF_DEG = 95.0
TIP_STRING_STACK_CRIT_DEG = 120.0
MIN_TIP_STRING_SIN = math.sin(math.radians(8.0))

# Efficiency penalties from late-draw stacking concentration.
STACK_EFF_ANGLE_WEIGHT = 0.22
STACK_EFF_LATE_WORK_WEIGHT = 0.14
STACK_EFF_MIN_FACTOR = 0.70

# Blend between geometry-transmission force and energy-derivative force.
GEOMETRIC_FORCE_BLEND = 0.80

# ════════════════════════════════════════════════════════════════════════════
# 7-POINT MEASUREMENT DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

POINT_DEFINITIONS = [
    {"id": 0, "name": "P0: Handle Center", "limb": "Handle"},
    {"id": 1, "name": "P1: Upper 20cm", "limb": "Upper"},
    {"id": 2, "name": "P2: Lower 20cm", "limb": "Lower"},
    {"id": 3, "name": "P3: Upper Mid", "limb": "Upper"},
    {"id": 4, "name": "P4: Lower Mid", "limb": "Lower"},
    {"id": 5, "name": "P5: Upper Tip-15cm", "limb": "Upper"},
    {"id": 6, "name": "P6: Lower Tip-15cm", "limb": "Lower"},
]


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class MeasurementPoint:
    """Single measurement point with geometric and material properties"""
    point_id: int
    name: str
    limb: str
    position_cm: float
    width_mm: float
    thickness_mm: float
    shape: str
    side_angle_deg: float = 0.0  # Initial reflex/deflex angle
    inertia_mm4: float = 0.0
    ei_nm2: float = 0.0
    mass_g: float = 0.0


@dataclass
class BowGeometry:
    """Complete bow geometry specification"""
    total_length_cm: float
    measurements: List[MeasurementPoint]
    side_profile: str = "Straight"
    string_length_cm: Optional[float] = None
    brace_height_cm: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive bow performance analysis"""
    draw_weight_28_lbs: float
    stored_energy_j: float
    shooting_efficiency: float
    arrow_speed_fps: float
    arrow_speed_kmh: float
    estimated_mass_g: float
    specific_energy_j_per_kg: float
    stacking_point_inch: Optional[float] = None


@dataclass
class ForceCurvePoint:
    """Single point on the force-draw curve"""
    draw_inch: float
    force_lbs: float
    # Tip-string interior angle at the limb tip nock (degrees).
    string_angle_deg: float
    tip_deflection_cm: float
    stored_energy_j: float = 0.0
    # Retained as secondary geometry diagnostic (not plotted in the main angle chart).
    tip_bow_angle_deg: float = 0.0


# ════════════════════════════════════════════════════════════════════════════
# SECOND MOMENT OF AREA CALCULATIONS
# ════════════════════════════════════════════════════════════════════════════


def calculate_inertia(width_mm: float, thickness_mm: float, shape: str) -> float:
    """
    Calculate second moment of area I [mm⁴]
    
    Theory:
    -------
    Rectangular: I = (w × t³) / 12
    Elliptical: I ≈ (π × w × t³) / 64
    Trapezoid: I = t³(b₁² + 4b₁b₂ + b₂²) / [36(b₁ + b₂)]
    
    Parameters:
    -----------
    width_mm : float
        Cross-section width [mm]
    thickness_mm : float
        Cross-section thickness [mm]
    shape : str
        Geometric profile
    
    Returns:
    --------
    float : Second moment of area [mm⁴]
    """
    if width_mm <= 0 or thickness_mm <= 0:
        return 0.0
    
    w, t = width_mm, thickness_mm
    
    if "Rectangular" in shape:
        return (w * t**3) / 12.0
    
    elif "Elliptical" in shape:
        return (math.pi * w * t**3) / 64.0
    
    elif "Trapezoid" in shape and "Inverse" not in shape:
        # Wide belly (0.8w narrow back)
        b_wide = w
        b_narrow = 0.8 * w
        numerator = t**3 * (b_wide**2 + 4*b_wide*b_narrow + b_narrow**2)
        denominator = 36.0 * (b_wide + b_narrow)
        return numerator / denominator
    
    elif "Inverse" in shape:
        # Narrow belly (w/0.8 wide back)
        b_narrow = w
        b_wide = w / 0.8
        numerator = t**3 * (b_wide**2 + 4*b_wide*b_narrow + b_narrow**2)
        denominator = 36.0 * (b_wide + b_narrow)
        return numerator / denominator
    
    return (w * t**3) / 12.0


def calculate_cross_section_area(width_mm: float, thickness_mm: float, shape: str) -> float:
    """
    Calculate cross-sectional area [mm²]
    
    Parameters:
    -----------
    width_mm : float
    thickness_mm : float
    shape : str
    
    Returns:
    --------
    float : Cross-sectional area [mm²]
    """
    if width_mm <= 0 or thickness_mm <= 0:
        return 0.0
    
    w, t = width_mm, thickness_mm
    
    if "Rectangular" in shape:
        return w * t
    
    elif "Elliptical" in shape:
        return (math.pi * w * t) / 4.0
    
    elif "Trapezoid" in shape and "Inverse" not in shape:
        b_wide = w
        b_narrow = 0.8 * w
        return t * (b_wide + b_narrow) / 2.0
    
    elif "Inverse" in shape:
        b_narrow = w
        b_wide = w / 0.8
        return t * (b_wide + b_narrow) / 2.0
    
    return w * t


# ════════════════════════════════════════════════════════════════════════════
# GEOMETRY & POSITIONING
# ════════════════════════════════════════════════════════════════════════════


def get_point_positions_cm(total_length_cm: float) -> List[float]:
    """
    Calculate 7-point measurement positions [cm] from handle
    
    Specification:
    --------------
    P0: Handle center (0cm)
    P1, P2: 20cm from handle (upper/lower)
    P3, P4: Midpoint between P1-P5 and P2-P6
    P5, P6: 15cm from tip (upper/lower)
    
    Parameters:
    -----------
    total_length_cm : float
        Bow total length [cm]
    
    Returns:
    --------
    List[float] : Position array [cm]
    """
    half_length = total_length_cm / 2.0
    tip_inner = half_length - 15.0
    mid_position = (20.0 + tip_inner) / 2.0
    
    return [
        0.0,           # P0: Handle
        20.0,          # P1: Upper 20cm
        20.0,          # P2: Lower 20cm
        mid_position,  # P3: Upper mid
        mid_position,  # P4: Lower mid
        tip_inner,     # P5: Upper tip area
        tip_inner,     # P6: Lower tip area
    ]


def compute_ei_profile(
    measurements: List[MeasurementPoint],
    youngs_modulus_gpa: float,
    density_g_cm3: float
) -> List[MeasurementPoint]:
    """
    Compute EI (flexural rigidity) and mass for each measurement point
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
    youngs_modulus_gpa : float
        Young's modulus [GPa]
    density_g_cm3 : float
        Material density [g/cm³]
    
    Returns:
    --------
    List[MeasurementPoint] : Updated measurements with EI and mass
    """
    e_pa = youngs_modulus_gpa * GPA_TO_PA
    
    for point in measurements:
        # Second moment of area
        i_mm4 = calculate_inertia(point.width_mm, point.thickness_mm, point.shape)
        point.inertia_mm4 = i_mm4
        
        # Flexural rigidity EI [N·m²]
        i_m4 = i_mm4 * MM4_TO_M4
        point.ei_nm2 = e_pa * i_m4
        
        # Mass estimation (will be refined with segment length)
        area_mm2 = calculate_cross_section_area(
            point.width_mm, point.thickness_mm, point.shape
        )
        area_cm2 = area_mm2 / 100.0
        # Placeholder: will be computed properly in mass calculation
        point.mass_g = area_cm2 * density_g_cm3
    
    return measurements


# ════════════════════════════════════════════════════════════════════════════
# MASS DISTRIBUTION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════


def calculate_bow_mass_distribution(
    measurements: List[MeasurementPoint],
    total_length_cm: float,
    density_g_cm3: float
) -> Tuple[float, List[float]]:
    """
    Calculate total bow mass and segment mass distribution
    
    Critical for efficiency: tip mass heavily penalizes performance
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
    total_length_cm : float
    density_g_cm3 : float
    
    Returns:
    --------
    Tuple[float, List[float]] : (total_mass_g, segment_masses_g)
    """
    positions = [p.position_cm for p in measurements]
    segment_masses = []
    
    for i in range(len(measurements) - 1):
        segment_length_cm = abs(positions[i+1] - positions[i])
        
        # Average cross-sectional area
        area1_cm2 = calculate_cross_section_area(
            measurements[i].width_mm,
            measurements[i].thickness_mm,
            measurements[i].shape
        ) / 100.0
        
        area2_cm2 = calculate_cross_section_area(
            measurements[i+1].width_mm,
            measurements[i+1].thickness_mm,
            measurements[i+1].shape
        ) / 100.0
        
        avg_area_cm2 = (area1_cm2 + area2_cm2) / 2.0
        
        # Segment volume and mass
        volume_cm3 = avg_area_cm2 * segment_length_cm
        mass_g = volume_cm3 * density_g_cm3
        
        segment_masses.append(mass_g)
    
    # Both limbs
    total_mass_g = 2.0 * sum(segment_masses)
    
    return total_mass_g, segment_masses


# ════════════════════════════════════════════════════════════════════════════
# GEOMETRIC FORCE-DRAW CURVE ENGINE
# ════════════════════════════════════════════════════════════════════════════


def compute_limb_deflection_profile(
    measurements: List[MeasurementPoint],
    limb_length_cm: float,
    force_n: float,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute limb deflection shape under applied force
    
    Uses Euler-Bernoulli beam theory with variable EI
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
    limb_length_cm : float
    force_n : float
        Applied force at tip [N]
    n_points : int
        Number of interpolation points
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : (positions_cm, deflections_cm)
    """
    # Extract EI profile
    positions_cm = np.array([p.position_cm for p in measurements])
    ei_values = np.array([max(p.ei_nm2, 1e-10) for p in measurements])
    
    # Sort and remove duplicates
    sort_idx = np.argsort(positions_cm)
    positions_sorted = positions_cm[sort_idx]
    ei_sorted = ei_values[sort_idx]
    
    unique_pos = np.unique(positions_sorted)
    unique_ei = np.array([ei_sorted[positions_sorted == p].mean() for p in unique_pos])
    
    # Ensure valid range
    if len(unique_pos) < 2:
        unique_pos = np.array([0.0, limb_length_cm])
        unique_ei = np.array([unique_ei[0], unique_ei[0]])
    
    # Interpolate EI along limb
    s_samples_cm = np.linspace(0, limb_length_cm, n_points)
    ei_samples = np.maximum(np.interp(s_samples_cm, unique_pos, unique_ei), 1e-10)
    
    # Convert to meters
    s_m = s_samples_cm * CM_TO_M
    L_m = limb_length_cm * CM_TO_M
    
    # Cantilever beam deflection: δ(s) = P ∫∫ M(x)/(EI(x)) dx dx
    # For tip load: M(x) = P(L-x)
    # δ(s) = P ∫_s^L (L-x)²/(EI(x)) dx
    
    deflections_m = np.zeros_like(s_m)
    
    for i, s in enumerate(s_m):
        # Integrate from s to L
        if i < len(s_m) - 1:
            integrand = ((L_m - s_m[i:]) ** 2) / ei_samples[i:]
            deflections_m[i] = force_n * np.trapezoid(integrand, s_m[i:])
    
    deflections_cm = deflections_m * 100.0
    
    return s_samples_cm, deflections_cm


def compute_string_angle(
    tip_deflection_cm: float,
    limb_length_cm: float,
    brace_height_cm: float = 15.0
) -> float:
    """
    Calculate string angle at tip relative to unbraced position
    
    Critical for stacking: higher angles → rapid force increase
    
    Parameters:
    -----------
    tip_deflection_cm : float
    limb_length_cm : float
    brace_height_cm : float
    
    Returns:
    --------
    float : String angle [degrees]
    """
    # Simplified geometry
    # Angle ≈ arctan(deflection / limb_length)
    if limb_length_cm <= 0:
        return 0.0
    
    angle_rad = math.atan2(tip_deflection_cm, limb_length_cm)
    angle_deg = math.degrees(angle_rad)
    
    return abs(angle_deg)


def _integrate_force_curve(draw_m_values: np.ndarray, force_n: np.ndarray) -> np.ndarray:
    """Cumulative work integral U(x)=∫F dx from a force curve."""
    n = len(draw_m_values)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)
    
    dx = np.diff(draw_m_values)
    trap = 0.5 * (force_n[1:] + force_n[:-1]) * dx
    cumulative = np.zeros(n, dtype=float)
    cumulative[1:] = np.cumsum(trap)
    return np.maximum(cumulative, 0.0)


def _smooth_force_curve(
    draw_m_values: np.ndarray,
    force_n: np.ndarray,
    target_end_energy_j: Optional[float] = None,
) -> np.ndarray:
    """
    Smooth force curve while preserving monotonic draw-force behavior and
    optionally matching terminal stored energy.
    """
    if len(draw_m_values) != len(force_n):
        return np.zeros_like(draw_m_values)
    
    force_clamped = np.maximum(np.asarray(force_n, dtype=float), 0.0)
    pseudo_energy = _integrate_force_curve(draw_m_values, force_clamped)
    smoothed = _estimate_force_from_energy(draw_m_values, pseudo_energy)
    
    if target_end_energy_j is None:
        return smoothed
    
    area = float(np.trapezoid(smoothed, draw_m_values))
    if area > 1e-12 and target_end_energy_j > 0:
        smoothed *= target_end_energy_j / area
    return np.maximum(smoothed, 0.0)


def _estimate_force_from_energy(
    draw_m_values: np.ndarray,
    stored_energy_j: np.ndarray,
    window: int = 13,
    poly_order: int = 3,
) -> np.ndarray:
    """
    Estimate F(draw)=dU/dx with a local polynomial derivative to suppress
    numerical jitter from finite-difference on discretely solved energy states.
    """
    n = len(draw_m_values)
    if n < 3:
        return np.zeros(n)
    
    if n < 7:
        force = np.gradient(stored_energy_j, draw_m_values)
        return np.maximum(force, 0.0)
    
    if window % 2 == 0:
        window += 1
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window < 5:
        window = 5
    if window % 2 == 0:
        window += 1
    
    half = window // 2
    force_n = np.zeros(n, dtype=float)
    
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        
        # Keep near-constant window size at boundaries.
        if right - left < window:
            if left == 0:
                right = min(n, window)
            elif right == n:
                left = max(0, n - window)
        
        xw = draw_m_values[left:right] - draw_m_values[i]
        yw = stored_energy_j[left:right]
        
        deg = min(poly_order, len(xw) - 1)
        if deg < 1:
            force_n[i] = 0.0
            continue
        
        # Gaussian weights: emphasize local neighborhood.
        span = max(float(np.max(np.abs(xw))), 1e-12)
        sigma = max(span * 0.6, 1e-12)
        w = np.exp(-0.5 * (xw / sigma) ** 2)
        
        # Weighted least-squares polynomial in shifted coordinate xw.
        # U(x) ≈ a0 + a1 x + a2 x² + ... -> dU/dx at center is a1.
        A = np.vstack([xw ** k for k in range(deg + 1)]).T
        Aw = A * w[:, None]
        bw = yw * w
        coeffs, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        force_n[i] = coeffs[1]
    
    force_n = np.maximum(force_n, 0.0)
    
    # Physical monotonicity (draw force should not decrease with draw).
    force_n = np.maximum.accumulate(force_n)
    
    # Energy consistency: rescale so ∫F dx matches terminal stored energy.
    end_energy = float(stored_energy_j[-1])
    force_area = float(np.trapezoid(force_n, draw_m_values))
    if force_area > 1e-12 and end_energy > 0:
        force_n *= end_energy / force_area
    
    return force_n


def compute_geometric_fdc(
    measurements: List[MeasurementPoint],
    limb_length_cm: float,
    side_profile: str = "Straight",
    max_draw_inch: float = DRAW_LENGTH_INCH,
    n_steps: int = FDC_DRAW_STEPS
) -> List[ForceCurvePoint]:
    """
    Compute force-draw curve from limb deformation energy states.
    
    Method:
    - Solve upper/lower limb shapes independently at each draw step.
    - Enforce fixed string length determined from braced state.
    - Include side profile geometry + 7-point EI/side-angle distribution.
    - Compute stored energy U(draw) from strain energy.
    - Convert to draw force via F(draw) = dU/dx.
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
    limb_length_cm : float
    max_draw_inch : float
    n_steps : int
    
    Returns:
    --------
    List[ForceCurvePoint] : Force-draw curve data
    """
    draw_inches = np.linspace(0, max_draw_inch, n_steps)
    draw_cm_values = draw_inches * INCH_TO_CM
    draw_m_values = draw_inches * INCH_TO_M
    
    # Braced state defines fixed string half-length for subsequent draw states.
    upper_braced = _solve_limb_deformation_state(
        measurements=measurements,
        limb_length_cm=limb_length_cm,
        draw_cm=0.0,
        side_profile=side_profile,
        limb_type="Upper",
        string_half_length=None,
        n_segments=100,
        draw_from_brace=True,
    )
    lower_braced = _solve_limb_deformation_state(
        measurements=measurements,
        limb_length_cm=limb_length_cm,
        draw_cm=0.0,
        side_profile=side_profile,
        limb_type="Lower",
        string_half_length=None,
        n_segments=100,
        draw_from_brace=True,
    )
    
    string_half_upper = calculate_string_length(
        upper_braced["tip_x_cm"], upper_braced["tip_y_cm"], TARGET_BRACE_HEIGHT_CM, 0.0
    )
    string_half_lower = calculate_string_length(
        lower_braced["tip_x_cm"], lower_braced["tip_y_cm"], TARGET_BRACE_HEIGHT_CM, 0.0
    )
    
    total_energy_j: List[float] = []
    geometric_draw_force_n: List[float] = []
    mean_tip_string_inner_angle_deg: List[float] = []
    mean_tip_bow_angle_deg: List[float] = []
    
    for draw_cm in draw_cm_values:
        upper_state = _solve_limb_deformation_state(
            measurements=measurements,
            limb_length_cm=limb_length_cm,
            draw_cm=float(draw_cm),
            side_profile=side_profile,
            limb_type="Upper",
            string_half_length=string_half_upper,
            n_segments=100,
            draw_from_brace=True,
        )
        lower_state = _solve_limb_deformation_state(
            measurements=measurements,
            limb_length_cm=limb_length_cm,
            draw_cm=float(draw_cm),
            side_profile=side_profile,
            limb_type="Lower",
            string_half_length=string_half_lower,
            n_segments=100,
            draw_from_brace=True,
        )
        
        total_energy_j.append(
            float(upper_state["strain_energy_j"] + lower_state["strain_energy_j"])
        )
        geometric_draw_force_n.append(
            float(
                upper_state["applied_force_n"] * upper_state["draw_force_gain"]
                + lower_state["applied_force_n"] * lower_state["draw_force_gain"]
            )
        )
        mean_tip_string_inner_angle_deg.append(
            0.5 * (upper_state["tip_string_angle_deg"] + lower_state["tip_string_angle_deg"])
        )
        mean_tip_bow_angle_deg.append(
            0.5 * (upper_state["tip_bow_angle_deg"] + lower_state["tip_bow_angle_deg"])
        )
    
    # Stored energy baseline is the braced state (draw = 0).
    energy_array = np.array(total_energy_j, dtype=float)
    stored_energy_from_strain = np.maximum(energy_array - energy_array[0], 0.0)
    # Numerical safety: enforce non-decreasing stored energy with draw.
    stored_energy_from_strain = np.maximum.accumulate(stored_energy_from_strain)
    end_energy = float(stored_energy_from_strain[-1]) if len(stored_energy_from_strain) else 0.0
    
    # Baseline force from energy derivative (stable fallback).
    force_from_energy = _estimate_force_from_energy(draw_m_values, stored_energy_from_strain)
    
    # Geometry-transmission force: directly couples draw force to string/tip angles.
    geometric_force = np.maximum(np.array(geometric_draw_force_n, dtype=float), 0.0)
    force_n = force_from_energy.copy()
    
    if len(geometric_force) == len(draw_m_values) and np.any(geometric_force > 0):
        area_geo = float(np.trapezoid(geometric_force, draw_m_values))
        if area_geo > 1e-12 and end_energy > 0:
            geometric_force *= end_energy / area_geo
        
        geometric_force = _smooth_force_curve(
            draw_m_values,
            geometric_force,
            target_end_energy_j=end_energy if end_energy > 0 else None,
        )
        
        # Blend to keep robustness near degenerate geometries while preserving
        # strong angle-driven stacking behavior.
        force_n = (
            GEOMETRIC_FORCE_BLEND * geometric_force
            + (1.0 - GEOMETRIC_FORCE_BLEND) * force_from_energy
        )
        
        # Final consistency scaling to terminal strain energy.
        area_final = float(np.trapezoid(force_n, draw_m_values))
        if area_final > 1e-12 and end_energy > 0:
            force_n *= end_energy / area_final
    
    force_n = np.maximum(force_n, 0.0)
    force_n = np.maximum.accumulate(force_n)
    stored_energy = _integrate_force_curve(draw_m_values, force_n)
    if len(stored_energy) and end_energy > 0 and stored_energy[-1] > 1e-12:
        stored_energy *= end_energy / stored_energy[-1]
    
    fdc_points: List[ForceCurvePoint] = []
    for i, draw_inch in enumerate(draw_inches):
        fdc_points.append(ForceCurvePoint(
            draw_inch=float(draw_inch),
            force_lbs=float(force_n[i] * N_TO_LBS),
            string_angle_deg=float(mean_tip_string_inner_angle_deg[i]),
            tip_deflection_cm=float(draw_cm_values[i]),
            stored_energy_j=float(stored_energy[i]),
            tip_bow_angle_deg=float(mean_tip_bow_angle_deg[i]),
        ))
    
    return fdc_points


def find_stacking_point(fdc: List[ForceCurvePoint], threshold_ratio: float = 1.5) -> Optional[float]:
    """
    Identify stacking point from force-rate acceleration + string-angle criterion.
    
    Parameters:
    -----------
    fdc : List[ForceCurvePoint]
    threshold_ratio : float
        Relative threshold multiplier on force-rate
    
    Returns:
    --------
    Optional[float] : Stacking point draw length [inch], or None
    """
    if len(fdc) < 3:
        return None
    
    draws = np.array([p.draw_inch for p in fdc], dtype=float)
    forces = np.array([p.force_lbs for p in fdc], dtype=float)
    angles = np.array([p.string_angle_deg for p in fdc], dtype=float)
    
    dF_dx = np.gradient(forces, draws)
    d2F_dx2 = np.gradient(dF_dx, draws)
    
    draw_max = float(np.max(draws))
    if draw_max <= 0:
        return None
    
    # Baseline force-rate in early/mid draw.
    baseline_mask = draws <= draw_max * 0.55
    baseline_rate = float(np.median(dF_dx[baseline_mask])) if np.any(baseline_mask) else float(np.median(dF_dx))
    baseline_rate = max(baseline_rate, 1e-9)
    
    # Angle-aware criterion: stacking should emerge in late draw where
    # tip-string interior angle has grown beyond reference region.
    for i in range(len(draws)):
        if draws[i] < draw_max * 0.55:
            continue
        if angles[i] < TIP_STRING_STACK_REF_DEG:
            continue
        if dF_dx[i] > threshold_ratio * baseline_rate and d2F_dx2[i] > 0:
            return float(draws[i])
    
    # Fallback for edge cases: pure curvature criterion in latter half.
    mean_d2F = float(np.mean(np.abs(d2F_dx2[1:-1]))) if len(d2F_dx2) > 2 else 0.0
    if mean_d2F <= 0:
        return None
    for i, val in enumerate(d2F_dx2):
        if draws[i] >= draw_max * 0.55 and abs(val) > threshold_ratio * mean_d2F:
            return float(draws[i])
    
    return None


# ════════════════════════════════════════════════════════════════════════════
# ENERGY & EFFICIENCY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════


def calculate_stored_energy(fdc: List[ForceCurvePoint]) -> float:
    """
    Calculate total stored energy from FDC
    
    E_stored = ∫ F(x) dx [area under curve]
    
    Parameters:
    -----------
    fdc : List[ForceCurvePoint]
    
    Returns:
    --------
    float : Stored energy [Joules]
    """
    if not fdc:
        return 0.0
    
    solved_energy = max((p.stored_energy_j for p in fdc), default=0.0)
    if solved_energy > 0:
        return float(solved_energy)
    
    draws_m = np.array([p.draw_inch * INCH_TO_M for p in fdc])
    forces_n = np.array([p.force_lbs * LBS_TO_N for p in fdc])
    
    # Fallback integration for legacy curves without per-point energy.
    return float(simpson(forces_n, x=draws_m))


def calculate_shooting_efficiency(
    measurements: List[MeasurementPoint],
    segment_masses_g: List[float],
    total_mass_g: float,
    fdc: Optional[List[ForceCurvePoint]] = None,
) -> float:
    """
    Calculate shooting efficiency (energy transfer ratio)
    
    Theory:
    -------
    Limb mass, especially at tips, reduces efficiency due to:
    1. Kinetic energy wasted on limb motion
    2. Tip mass has highest velocity → highest energy penalty
    
    Virtual Mass Model:
    M_virtual = M_arrow + k × M_limb
    
    Where k depends on mass distribution:
    - Tip mass: k ≈ 0.5 (moves with arrow velocity)
    - Mid limb: k ≈ 0.3
    - Handle: k ≈ 0.1 (minimal motion)
    
    Efficiency = 1 / (1 + M_virtual/M_arrow)
    
    Additional dynamic correction:
    - High tip-string interior angle in late draw increases string/limb
      kinetic-energy share and practical stack severity.
    - We model this with a bounded penalty based on:
      (a) late-draw angle excess over a reference region
      (b) late-draw work concentration (stack-heavy FDC shape)
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
    segment_masses_g : List[float]
    total_mass_g : float
    
    Returns:
    --------
    float : Efficiency ratio [0-1]
    """
    if len(segment_masses_g) == 0:
        return 0.5  # Default estimate
    
    # Position-dependent efficiency factors
    positions = [p.position_cm for p in measurements[:-1]]
    max_pos = max(positions) if positions else 1.0
    
    virtual_mass_g = 0.0
    
    for i, mass_g in enumerate(segment_masses_g):
        position_cm = measurements[i].position_cm
        
        # Normalized position [0=handle, 1=tip]
        norm_pos = position_cm / max_pos if max_pos > 0 else 0.0
        
        # Velocity factor: v(x) ≈ x/L (linear approximation)
        # Energy factor: E ∝ v² ∝ (x/L)²
        velocity_factor = norm_pos
        energy_factor = velocity_factor ** 2
        
        # Virtual mass contribution
        virtual_mass_g += mass_g * energy_factor
    
    # Both limbs
    virtual_mass_g *= 2.0
    
    # Standard arrow mass
    arrow_mass_g = 25.0
    
    # Base mass-driven efficiency
    efficiency = arrow_mass_g / (arrow_mass_g + virtual_mass_g)
    
    if fdc and len(fdc) > 4:
        draws = np.array([p.draw_inch for p in fdc], dtype=float)
        forces_n = np.array([p.force_lbs * LBS_TO_N for p in fdc], dtype=float)
        angles = np.array([p.string_angle_deg for p in fdc], dtype=float)
        
        draw_max = float(np.max(draws))
        if draw_max > 0 and np.any(forces_n > 0):
            # Late-draw weighted angle excess (stacking-sensitive region).
            draw_norm = draws / draw_max
            late_weights = np.clip((draw_norm - 0.5) / 0.5, 0.0, 1.0) + 0.15
            angle_excess = np.maximum(angles - TIP_STRING_STACK_REF_DEG, 0.0)
            angle_excess /= max(TIP_STRING_STACK_CRIT_DEG - TIP_STRING_STACK_REF_DEG, 1e-9)
            angle_excess = np.clip(angle_excess, 0.0, 1.0)
            weighted_angle_excess = float(np.average(angle_excess, weights=late_weights))
            
            # Work concentration in last quarter of draw:
            # linear-force baseline gives ~43.75% in last quarter.
            total_work = float(np.trapezoid(forces_n, draws * INCH_TO_M))
            late_start = 0.75 * draw_max
            late_mask = draws >= late_start
            late_work = float(np.trapezoid(forces_n[late_mask], draws[late_mask] * INCH_TO_M)) if np.any(late_mask) else 0.0
            late_ratio = (late_work / total_work) if total_work > 1e-9 else 0.0
            late_ratio_excess = max(0.0, (late_ratio - 0.4375) / (1.0 - 0.4375))
            
            stack_penalty = (
                STACK_EFF_ANGLE_WEIGHT * weighted_angle_excess
                + STACK_EFF_LATE_WORK_WEIGHT * late_ratio_excess
            )
            dynamic_factor = max(STACK_EFF_MIN_FACTOR, 1.0 - stack_penalty)
            efficiency *= dynamic_factor
    
    # Clamp to reasonable range
    efficiency = max(0.2, min(0.9, efficiency))
    
    return efficiency


def calculate_arrow_speed(
    stored_energy_j: float,
    efficiency: float,
    arrow_mass_g: float = 25.0
) -> Tuple[float, float]:
    """
    Calculate arrow speed from stored energy and efficiency
    
    Theory:
    -------
    E_kinetic = efficiency × E_stored
    v = √(2 × E_kinetic / m_arrow)
    
    Parameters:
    -----------
    stored_energy_j : float
    efficiency : float
    arrow_mass_g : float
    
    Returns:
    --------
    Tuple[float, float] : (speed_fps, speed_kmh)
    """
    arrow_mass_kg = arrow_mass_g / 1000.0
    
    kinetic_energy_j = efficiency * stored_energy_j
    
    if arrow_mass_kg <= 0:
        return 0.0, 0.0
    
    speed_ms = math.sqrt(2.0 * kinetic_energy_j / arrow_mass_kg)
    speed_fps = speed_ms * MS_TO_FPS
    speed_kmh = speed_ms * MS_TO_KMH
    
    return speed_fps, speed_kmh


# ════════════════════════════════════════════════════════════════════════════
# BALLISTICS ENGINE (TRAJECTORY & RANGE PREDICTION)
# ════════════════════════════════════════════════════════════════════════════


def calculate_maximum_range(
    arrow_speed_ms: float,
    launch_angle_deg: float = 45.0,
    launch_height_m: float = 1.5
) -> float:
    """
    Calculate maximum range for projectile motion
    
    Theory:
    -------
    R = (v² sin(2θ)) / g + horizontal distance from height
    
    Parameters:
    -----------
    arrow_speed_ms : float
        Initial arrow velocity [m/s]
    launch_angle_deg : float
        Launch angle [degrees]
    launch_height_m : float
        Launch height above ground [m]
    
    Returns:
    --------
    float : Maximum range [m]
    """
    if arrow_speed_ms <= 0:
        return 0.0
    
    theta_rad = math.radians(launch_angle_deg)
    v = arrow_speed_ms
    
    # Velocity components
    v_x = v * math.cos(theta_rad)
    v_y = v * math.sin(theta_rad)
    
    # Time to reach ground from y(t)=h0+v_y*t-0.5*g*t², choose positive root.
    t_flight = _solve_projectile_flight_time(v_y, launch_height_m)
    if t_flight <= 0:
        return 0.0
    
    # Horizontal range
    range_m = v_x * t_flight
    
    return range_m


def _solve_projectile_flight_time(v_y: float, launch_height_m: float) -> float:
    """Solve positive flight time for projectile from elevated launch."""
    a = -0.5 * GRAVITY
    b = v_y
    c = launch_height_m
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return 0.0
    
    sqrt_disc = math.sqrt(discriminant)
    roots = [
        (-b + sqrt_disc) / (2*a),
        (-b - sqrt_disc) / (2*a),
    ]
    positive_roots = [t for t in roots if t > 0]
    return max(positive_roots) if positive_roots else 0.0


def calculate_effective_range(
    arrow_speed_ms: float,
    arrow_mass_g: float = 25.0,
    min_kinetic_energy_j: float = 40.0
) -> float:
    """
    Calculate effective range based on minimum kinetic energy
    
    Effective range: distance where arrow retains sufficient energy
    for hunting/combat (typically 40-50 J for deer)
    
    Parameters:
    -----------
    arrow_speed_ms : float
    arrow_mass_g : float
    min_kinetic_energy_j : float
        Minimum energy for effective hit
    
    Returns:
    --------
    float : Effective range [m]
    """
    if arrow_speed_ms <= 0:
        return 0.0
    
    arrow_mass_kg = arrow_mass_g / 1000.0
    initial_ke = 0.5 * arrow_mass_kg * arrow_speed_ms**2
    
    if initial_ke < min_kinetic_energy_j:
        return 0.0
    
    # Simplified model: energy decreases with distance due to drag
    # Assume exponential decay: E(x) = E0 * exp(-k*x)
    # Solve for x when E(x) = E_min
    
    # Drag coefficient (simplified for arrow)
    # Typical arrow loses ~50% energy at 50m
    k = -math.log(0.5) / 50.0  # decay constant
    
    if k <= 0:
        return 0.0
    
    effective_range = -math.log(min_kinetic_energy_j / initial_ke) / k
    
    # Clamp to reasonable range
    max_range = calculate_maximum_range(arrow_speed_ms)
    effective_range = min(effective_range, max_range * 0.7)
    
    return max(0.0, effective_range)


def calculate_effective_range_adaptive(
    arrow_speed_ms: float,
    arrow_mass_g: float = 25.0,
    absolute_threshold_j: float = 40.0,
    fallback_fraction_of_initial: float = 0.35,
) -> Tuple[float, float, float, bool]:
    """
    Compute effective range with adaptive threshold when absolute criterion is unattainable.

    Returns:
    --------
    Tuple[float, float, float, bool]
        (effective_range_m, initial_ke_j, used_threshold_j, used_adaptive_threshold)
    """
    if arrow_speed_ms <= 0:
        return 0.0, 0.0, absolute_threshold_j, False
    
    arrow_mass_kg = arrow_mass_g / 1000.0
    initial_ke_j = 0.5 * arrow_mass_kg * arrow_speed_ms**2
    
    if initial_ke_j <= 0:
        return 0.0, 0.0, absolute_threshold_j, False
    
    used_threshold_j = absolute_threshold_j
    used_adaptive = False
    
    if initial_ke_j <= absolute_threshold_j:
        # Adaptive criterion preserves usefulness for low-energy bows while staying
        # physically bounded below the launch energy.
        used_adaptive = True
        adaptive_candidate = initial_ke_j * fallback_fraction_of_initial
        used_threshold_j = min(max(adaptive_candidate, 0.5), initial_ke_j * 0.95)
    
    effective_range_m = calculate_effective_range(
        arrow_speed_ms=arrow_speed_ms,
        arrow_mass_g=arrow_mass_g,
        min_kinetic_energy_j=used_threshold_j,
    )
    return effective_range_m, initial_ke_j, used_threshold_j, used_adaptive


def generate_trajectory_points(
    arrow_speed_ms: float,
    launch_angle_deg: float = 45.0,
    launch_height_m: float = 1.5,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate parabolic trajectory points
    
    Parameters:
    -----------
    arrow_speed_ms : float
    launch_angle_deg : float
    launch_height_m : float
    n_points : int
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : (x_positions_m, y_positions_m)
    """
    if arrow_speed_ms <= 0:
        return np.array([0.0]), np.array([0.0])
    
    theta_rad = math.radians(launch_angle_deg)
    v = arrow_speed_ms
    
    v_x = v * math.cos(theta_rad)
    v_y = v * math.sin(theta_rad)
    
    # Flight time (must use positive root).
    t_flight = _solve_projectile_flight_time(v_y, launch_height_m)
    if t_flight <= 0:
        return np.array([0.0]), np.array([launch_height_m])
    
    # Time array
    t_array = np.linspace(0, t_flight, n_points)
    
    # Position arrays
    x_array = v_x * t_array
    y_array = launch_height_m + v_y * t_array - 0.5 * GRAVITY * t_array**2
    
    # Ensure no negative heights
    y_array = np.maximum(y_array, 0.0)
    
    return x_array, y_array


# ════════════════════════════════════════════════════════════════════════════
# TILLERING PROFILE SIMULATION
# ════════════════════════════════════════════════════════════════════════════


def simulate_tillering_at_draw(
    measurements: List[MeasurementPoint],
    limb_length_cm: float,
    draw_inch: float,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate limb curvature at specific draw length
    
    Returns limb shape including initial reflex/deflex
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
    limb_length_cm : float
    draw_inch : float
    n_points : int
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : (positions_cm, curvatures_cm)
    """
    # Convert draw to force
    draw_m = draw_inch * INCH_TO_M
    
    # Estimate force at this draw
    positions = [p.position_cm for p in measurements]
    ei_values = [max(p.ei_nm2, 1e-10) for p in measurements]
    
    s_cm = np.array(positions, dtype=float)
    ei = np.array(ei_values, dtype=float)
    
    sort_idx = np.argsort(s_cm)
    s_sorted = s_cm[sort_idx]
    ei_sorted = ei[sort_idx]
    
    s_unique = np.unique(s_sorted)
    ei_unique = np.array([ei_sorted[s_sorted == s].mean() for s in s_unique])
    
    if len(s_unique) < 2:
        s_unique = np.array([0.0, limb_length_cm])
        ei_unique = np.array([ei_unique[0], ei_unique[0]])
    
    s_samples_cm = np.linspace(0, limb_length_cm, n_points)
    ei_samples = np.maximum(np.interp(s_samples_cm, s_unique, ei_unique), 1e-10)
    
    s_m = s_samples_cm * CM_TO_M
    L_m = limb_length_cm * CM_TO_M
    
    # Compliance
    integrand = ((L_m - s_m) ** 2) / ei_samples
    compliance = np.trapezoid(integrand, s_m)
    
    force_n = draw_m / compliance if compliance > 0 else 0.0
    
    # Deflection profile
    _, deflections_cm = compute_limb_deflection_profile(
        measurements, limb_length_cm, force_n, n_points
    )
    
    # Add initial side profile (reflex/deflex)
    initial_angles = np.interp(
        s_samples_cm,
        [p.position_cm for p in measurements],
        [p.side_angle_deg for p in measurements]
    )
    
    # Convert angles to offsets (simplified)
    initial_offsets_cm = np.zeros_like(s_samples_cm)
    for i in range(1, len(s_samples_cm)):
        ds = s_samples_cm[i] - s_samples_cm[i-1]
        angle_rad = math.radians(initial_angles[i])
        initial_offsets_cm[i] = initial_offsets_cm[i-1] + ds * math.tan(angle_rad)
    
    # Total curvature
    total_curvature_cm = deflections_cm + initial_offsets_cm
    
    return s_samples_cm, total_curvature_cm


# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI - ACADEMIC THEME
# ════════════════════════════════════════════════════════════════════════════


def apply_museum_dark_theme() -> None:
    """Apply refined museum-style dark interface inspired by experimental archaeology."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {
            --bg-deep: #111317;
            --bg-panel: #1a1d23;
            --bg-elev: #20242b;
            --line: #3a3d42;
            --text-main: #e7e1d5;
            --text-muted: #b9b3a8;
            --accent-brass: #c8a86b;
            --accent-oxide: #b8644a;
            --accent-patina: #6ea7a5;
        }

        .stApp {
            background:
                radial-gradient(1200px 500px at 12% -15%, rgba(200, 168, 107, 0.14), transparent 58%),
                radial-gradient(900px 460px at 88% 110%, rgba(110, 167, 165, 0.12), transparent 55%),
                linear-gradient(175deg, #0f1115 0%, var(--bg-deep) 45%, #14171c 100%);
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--text-main);
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1520px;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(190deg, #171a20 0%, #13161b 55%, #111317 100%);
            border-right: 1px solid var(--line);
            box-shadow: 8px 0 28px rgba(0, 0, 0, 0.45);
        }

        [data-testid="stSidebar"] .stMarkdown {
            color: var(--text-main);
        }

        h1 {
            color: var(--text-main) !important;
            font-family: 'Cormorant Garamond', serif !important;
            font-weight: 700 !important;
            font-size: clamp(2rem, 2.4vw, 3rem) !important;
            letter-spacing: 0.045em !important;
            text-transform: none !important;
            border-bottom: 1px solid var(--line) !important;
            padding-bottom: 0.9rem !important;
            margin-bottom: 1.35rem !important;
            position: relative;
        }

        h1::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: -1px;
            width: 11rem;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-brass), var(--accent-patina));
        }

        h2 {
            color: var(--accent-brass) !important;
            font-family: 'Cormorant Garamond', serif !important;
            font-weight: 600 !important;
            font-size: clamp(1.3rem, 1.55vw, 1.9rem) !important;
            letter-spacing: 0.04em !important;
            text-transform: none !important;
            margin-top: 2.4rem !important;
            margin-bottom: 1.15rem !important;
            border-left: 3px solid var(--accent-patina) !important;
            padding-left: 0.85rem !important;
        }

        h3 {
            color: var(--text-muted) !important;
            font-weight: 600 !important;
            font-size: 1.04rem !important;
            letter-spacing: 0.03em !important;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(150deg, rgba(41, 46, 55, 0.78), rgba(22, 25, 31, 0.9));
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1.2rem 1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
        }

        [data-testid="stMetricValue"] {
            color: var(--accent-patina) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 2.2rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            font-size: 0.76rem !important;
            letter-spacing: 0.13em !important;
            text-transform: uppercase !important;
            font-weight: 500 !important;
        }

        .stNumberInput input, .stSelectbox select {
            background-color: var(--bg-panel) !important;
            border: 1px solid var(--line) !important;
            color: var(--text-main) !important;
            border-radius: 8px !important;
            font-family: 'IBM Plex Mono', monospace !important;
        }

        .stNumberInput input:focus, .stSelectbox select:focus {
            border-color: var(--accent-patina) !important;
            box-shadow: 0 0 0 2px rgba(110, 167, 165, 0.18) !important;
        }

        .stNumberInput label, .stSelectbox label, .stSlider label {
            color: var(--text-muted) !important;
            font-weight: 500 !important;
            font-size: 0.84rem !important;
            letter-spacing: 0.04em !important;
        }

        .streamlit-expanderHeader {
            background: linear-gradient(90deg, #20252d 0%, #181c23 100%) !important;
            border: 1px solid var(--line) !important;
            border-radius: 8px !important;
            color: var(--text-main) !important;
            font-weight: 600 !important;
            letter-spacing: 0.03em !important;
        }

        .streamlit-expanderHeader:hover {
            border-color: var(--accent-patina) !important;
        }

        .streamlit-expanderContent {
            background-color: #14181f !important;
            border: 1px solid var(--line) !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
        }

        .dataframe {
            font-size: 0.84rem !important;
            background-color: #12161d !important;
        }

        .dataframe th {
            background: linear-gradient(180deg, #232933 0%, #191d25 100%) !important;
            color: var(--accent-brass) !important;
            border-bottom: 1px solid var(--line) !important;
            font-weight: 600 !important;
            letter-spacing: 0.04em !important;
        }

        .dataframe td {
            border-bottom: 1px solid #2f3339 !important;
            color: var(--text-main) !important;
        }

        .caption {
            color: #8e877b;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
        }

        .cyan-accent {
            color: var(--accent-patina);
            font-weight: 600;
        }

        .gold-accent {
            color: var(--accent-brass);
            font-weight: 600;
        }

        .critical-value {
            color: var(--accent-oxide);
            font-weight: 700;
        }

        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--line), transparent);
            margin: 2.6rem 0;
        }

        .section-card {
            background: linear-gradient(150deg, #1f242c 0%, #151920 100%);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1.8rem;
            margin: 1.8rem 0;
        }

        .geometric-line {
            width: 100%;
            height: 1px;
            background: linear-gradient(
                90deg,
                transparent 0%,
                rgba(200, 168, 107, 0.75) 22%,
                rgba(110, 167, 165, 0.75) 50%,
                rgba(200, 168, 107, 0.75) 78%,
                transparent 100%
            );
            margin: 1.3rem 0;
        }

        .hero-shell {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.3rem 1.25rem 1.15rem 1.25rem;
            background:
                radial-gradient(440px 140px at 0% 0%, rgba(200, 168, 107, 0.16), transparent 72%),
                linear-gradient(155deg, rgba(41, 47, 56, 0.86), rgba(20, 24, 31, 0.92));
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.30);
            margin-bottom: 1.1rem;
        }

        .hero-kicker {
            color: var(--accent-brass);
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            color: var(--text-main);
            font-family: 'Cormorant Garamond', serif;
            font-weight: 700;
            font-size: clamp(2rem, 3vw, 3.2rem);
            line-height: 1.03;
            letter-spacing: 0.02em;
            margin: 0;
        }

        .hero-sub {
            color: var(--text-muted);
            margin-top: 0.45rem;
            margin-bottom: 0;
            font-size: 0.94rem;
            letter-spacing: 0.025em;
        }

        .section-banner {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            border-bottom: 1px solid var(--line);
            margin-top: 1.45rem;
            margin-bottom: 0.9rem;
            padding-bottom: 0.45rem;
        }

        .section-banner-title {
            color: var(--accent-brass);
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.6rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            margin: 0;
        }

        .section-banner-sub {
            color: #9a9487;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            white-space: nowrap;
            margin-bottom: 0.08rem;
        }

        [data-testid="stPlotlyChart"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.2rem 0.2rem 0 0.2rem;
            background: linear-gradient(160deg, rgba(38, 43, 52, 0.78), rgba(20, 24, 30, 0.9));
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.26);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
            background: #131820;
        }

        @keyframes riseIn {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        [data-testid="stMetric"],
        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"] {
            animation: riseIn 0.45s ease both;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 1.1rem;
            }
            h1 {
                letter-spacing: 0.03em !important;
            }
            .section-banner {
                display: block;
            }
            .section-banner-sub {
                margin-top: 0.3rem;
                white-space: normal;
            }
        }
        </style>
    """, unsafe_allow_html=True)


def render_hero_header() -> None:
    """Render refined museum-style hero header."""
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Experimental Archaeology · Bow Laboratory</div>
            <h1 class="hero-title">디지털 활 물리 & 탄도 연구소</h1>
            <p class="hero-sub">Museum-grade interactive research platform for traditional bow mechanics and ballistics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_banner(title: str, subtitle: str) -> None:
    """Render section banner with title and contextual subtitle."""
    st.markdown(
        f"""
        <div class="section-banner">
            <div class="section-banner-title">{title}</div>
            <div class="section-banner-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_inputs() -> Tuple[str, float, str, List[MeasurementPoint]]:
    """
    Render sidebar input interface - Museum Archive Style
    
    Returns:
    --------
    Tuple[str, float, str, List[MeasurementPoint]]
        (wood_species, total_length_cm, side_profile, measurements)
    """
    with st.sidebar:
        st.markdown("## 활 제원 설정")
        st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
        
        # Wood selection
        species = st.selectbox(
            "목재 수종",
            options=list(WOOD_PROPERTIES.keys()),
            index=0,
        )
        
        props = WOOD_PROPERTIES[species]
        st.caption(f"탄성계수 E = {props['elastic_modulus_gpa']} GPa, 밀도 ρ = {props['density_g_cm3']} g/cm³")
        
        # Total length
        total_length_cm = st.slider(
            "활 전체 길이 (cm)",
            min_value=100,
            max_value=200,
            value=160,
            step=5,
        )
        
        st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
        
        # Side Profile Selection
        st.markdown("### 사이드 프로파일")
        
        side_profile = st.selectbox(
            "프로파일 유형",
            options=list(SIDE_PROFILE_OPTIONS.keys()),
            index=0,
            help="초기 활 곡률이 장력과 스택킹에 영향을 미칩니다"
        )
        
        # Show profile description
        profile_info = SIDE_PROFILE_OPTIONS[side_profile]
        st.caption(f"설명: {profile_info['description']}")
        st.caption(f"장력 배율: {profile_info['fdc_modifier']:.2f}x")
        st.caption(f"스택킹 배율: {profile_info['stacking_modifier']:.2f}x")
        
        st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
        st.markdown("## 7개 지점 측정값")
        st.caption("정밀 기하학적 프로파일 및 측면 각도 입력")
        
        # Calculate positions
        positions = get_point_positions_cm(total_length_cm)
        
        # Default values
        default_widths = [35.0, 28.0, 28.0, 24.0, 24.0, 20.0, 20.0]
        default_thicknesses = [12.0, 10.0, 10.0, 8.0, 8.0, 6.0, 6.0]
        default_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        measurements: List[MeasurementPoint] = []
        
        for i, point_def in enumerate(POINT_DEFINITIONS):
            with st.expander(f"{point_def['name']}", expanded=(i == 0)):
                st.caption(f"부위: {point_def['limb']} | 위치: {positions[i]:.1f} cm")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    width = st.number_input(
                        "너비 (mm)",
                        min_value=1.0,
                        max_value=300.0,  # EXPANDED RANGE
                        value=default_widths[i],
                        step=1.0,
                        key=f"w_{i}",
                    )
                
                with col2:
                    thickness = st.number_input(
                        "두께 (mm)",
                        min_value=1.0,
                        max_value=300.0,  # EXPANDED RANGE
                        value=default_thicknesses[i],
                        step=0.5,
                        key=f"t_{i}",
                    )
                
                # Shape selection
                if i == 0:
                    shape = "Rectangular"
                    st.caption("핸들: 직사각형 (고정)")
                else:
                    shape = st.selectbox(
                        "단면 형상",
                        options=SHAPE_OPTIONS,
                        index=0,
                        key=f"shape_{i}",
                    )
                
                # Side profile angle
                side_angle = st.number_input(
                    "측면 각도 (도)",
                    min_value=-30.0,
                    max_value=30.0,
                    value=default_angles[i],
                    step=1.0,
                    key=f"angle_{i}",
                    help="리플렉스 (+) / 디플렉스 (-)"
                )
                
                # Calculate and display I (moment of inertia) for debugging
                i_value = calculate_inertia(width, thickness, shape)
                st.caption(f"💡 I = {i_value:.2f} mm⁴ (단면 2차 모멘트)")
                
                point = MeasurementPoint(
                    point_id=point_def['id'],
                    name=point_def['name'],
                    limb=point_def['limb'],
                    position_cm=positions[i],
                    width_mm=width,
                    thickness_mm=thickness,
                    shape=shape,
                    side_angle_deg=side_angle,
                )
                
                measurements.append(point)
        
        return species, total_length_cm, side_profile, measurements


def render_performance_metrics(metrics: PerformanceMetrics) -> None:
    """Render key performance metrics - Museum Display Style"""
    
    render_section_banner("성능 분석", "Material response · Energy transfer · Shooting efficiency")
    
    # Primary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="28인치 장력",
            value=f"{metrics.draw_weight_28_lbs:.1f}",
            help="풀 드로우 시 필요한 힘"
        )
        st.caption("파운드 (lbs)")
    
    with col2:
        st.metric(
            label="저장 에너지",
            value=f"{metrics.stored_energy_j:.1f}",
            help="총 탄성 위치 에너지"
        )
        st.caption("줄 (Joules)")
    
    with col3:
        efficiency_pct = metrics.shooting_efficiency * 100
        st.metric(
            label="발시 효율",
            value=f"{efficiency_pct:.1f}",
            help="화살로 전달되는 에너지 (팁 질량 페널티 반영)"
        )
        st.caption("퍼센트 (%)")
    
    with col4:
        st.metric(
            label="활 질량",
            value=f"{metrics.estimated_mass_g:.0f}",
            help="총 추정 질량 (양쪽 림)"
        )
        st.caption("그램 (g)")
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Arrow velocity metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<p class="cyan-accent" style="font-size: 0.85rem; margin-bottom: 0.3rem;">화살 속도</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 2rem; font-weight: 700; color: #6ea7a5; margin: 0;">{metrics.arrow_speed_fps:.1f}</p>', unsafe_allow_html=True)
        st.caption("피트/초 (FPS)")
    
    with col2:
        st.markdown('<p class="gold-accent" style="font-size: 0.85rem; margin-bottom: 0.3rem;">화살 속도</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 2rem; font-weight: 700; color: #c8a86b; margin: 0;">{metrics.arrow_speed_kmh:.1f}</p>', unsafe_allow_html=True)
        st.caption("킬로미터/시 (km/h)")
    
    with col3:
        st.metric(
            label="비에너지",
            value=f"{metrics.specific_energy_j_per_kg:.1f}",
            help="활 질량 대비 에너지"
        )
        st.caption("J/kg")
    
    # Stacking warning
    if metrics.stacking_point_inch is not None:
        st.markdown(f"""
            <div style='background: linear-gradient(90deg, rgba(184, 100, 74, 0.2), transparent); 
                        border-left: 4px solid #b8644a; padding: 1rem; margin-top: 1rem; border-radius: 4px;'>
                <span style='color: #b8644a; font-weight: 700;'>⚠ 스택킹 감지</span><br>
                <span style='color: #e7e1d5;'>드로우 {metrics.stacking_point_inch:.1f}인치부터 급격한 힘 증가 시작</span>
            </div>
        """, unsafe_allow_html=True)


def create_fdc_chart(fdc: List[ForceCurvePoint]) -> go.Figure:
    """Create Force-Draw Curve with stacking visualization - Museum Dark Mode"""
    
    draws = [p.draw_inch for p in fdc]
    forces = [p.force_lbs for p in fdc]
    
    fig = go.Figure()
    
    # Main curve with glow effect
    fig.add_trace(go.Scatter(
        x=draws,
        y=forces,
        mode='lines',
        name='장력-드로우 곡선',
        line=dict(color='#6ea7a5', width=3),
        fill='tozeroy',
        fillcolor='rgba(110, 167, 165, 0.15)',
    ))
    
    # Highlight 28" point
    idx_28 = min(range(len(draws)), key=lambda i: abs(draws[i] - 28.0))
    fig.add_trace(go.Scatter(
        x=[draws[idx_28]],
        y=[forces[idx_28]],
        mode='markers',
        name='28인치 드로우 지점',
        marker=dict(color='#c8a86b', size=15, symbol='diamond', line=dict(color='#fff', width=2)),
    ))
    
    # Add vertical line at 28"
    fig.add_shape(
        type="line",
        x0=28, y0=0,
        x1=28, y1=forces[idx_28],
        line=dict(color='#c8a86b', width=2, dash='dash'),
    )
    
    fig.update_layout(
        title="장력-드로우 곡선 (FDC)",
        xaxis_title="드로우 길이 (인치)",
        yaxis_title="장력 (파운드)",
        template="plotly_dark",
        paper_bgcolor='#111317',
        plot_bgcolor='#1a1d23',
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#e7e1d5"),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(24, 28, 34, 0.82)',
            bordercolor='#3a3d42',
            borderwidth=1
        ),
        height=500,
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
    )
    
    return fig


def create_energy_storage_chart(fdc: List[ForceCurvePoint]) -> go.Figure:
    """Create stored energy vs draw chart (separate axis from FDC)."""
    draws = [p.draw_inch for p in fdc]
    energies = [p.stored_energy_j for p in fdc]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=draws,
        y=energies,
        mode='lines',
        name='저장 에너지 U(d)',
        line=dict(color='#c8a86b', width=3),
        fill='tozeroy',
        fillcolor='rgba(200, 168, 107, 0.12)',
    ))
    
    fig.update_layout(
        title="저장 에너지 곡선 U(d)",
        xaxis_title="드로우 길이 (인치)",
        yaxis_title="저장 에너지 (J)",
        template="plotly_dark",
        paper_bgcolor='#111317',
        plot_bgcolor='#1a1d23',
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#e7e1d5"),
        height=300,
        showlegend=False,
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
    )
    
    return fig


def create_string_angle_chart(fdc: List[ForceCurvePoint]) -> go.Figure:
    """Visualize tip-string interior angle progression."""
    
    draws = [p.draw_inch for p in fdc]
    tip_string_inner_angles = [p.string_angle_deg for p in fdc]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=draws,
        y=tip_string_inner_angles,
        mode='lines',
        name='Tip-String 내각',
        line=dict(color='#b8644a', width=3),
        fill='tozeroy',
        fillcolor='rgba(184, 100, 74, 0.15)',
    ))
    
    fig.update_layout(
        title="각도 변화 (Tip-String 내각)",
        xaxis_title="드로우 길이 (인치)",
        yaxis_title="각도 (도)",
        template="plotly_dark",
        paper_bgcolor='#111317',
        plot_bgcolor='#1a1d23',
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#e7e1d5"),
        height=400,
        showlegend=False,
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
    )
    
    return fig


def compute_initial_side_profile(
    s_positions: np.ndarray,
    limb_length_cm: float,
    profile_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute initial bow shape based on side profile type
    
    Coordinate system: Handle at (0,0), limb extends along +Y axis
    
    Parameters:
    -----------
    s_positions : np.ndarray
        Position array along limb [cm]
    limb_length_cm : float
    profile_type : str
        Side profile type
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] : (x_initial, y_initial)
    """
    profile_config = SIDE_PROFILE_OPTIONS.get(profile_type, SIDE_PROFILE_OPTIONS["Straight"])
    reflex_factor = profile_config["reflex_factor"]
    
    # Normalized position [0-1]
    s_norm = s_positions / limb_length_cm
    
    x_initial = np.zeros_like(s_positions)
    y_initial = s_positions.copy()
    
    # Scale curvature amplitude by limb size so side-profile effects
    # remain physically meaningful across bow lengths.
    base_amp = 0.08 * limb_length_cm
    
    if profile_type == "Straight":
        # No curvature
        pass
    
    elif profile_type == "Reflex":
        # Whole limb curves away from archer (preload).
        x_initial = -reflex_factor * base_amp * (s_norm ** 1.8)
    
    elif profile_type == "Deflex":
        # Whole limb curves toward archer (smooth draw).
        x_initial = -reflex_factor * 0.85 * base_amp * (s_norm ** 1.7)
    
    elif profile_type == "Recurve":
        # Working limb mostly straight, last 30% recurved strongly.
        recurve_start = 0.7
        for i, s in enumerate(s_norm):
            if s < recurve_start:
                x_initial[i] = 0.0
            else:
                local_norm = (s - recurve_start) / (1.0 - recurve_start)
                x_initial[i] = -reflex_factor * 1.25 * base_amp * (local_norm ** 1.8)
    
    elif profile_type == "Decurve":
        # Deflexed working limb with stronger near-tip forward set.
        x_initial = -reflex_factor * 0.95 * base_amp * (s_norm ** 1.5)
    
    return x_initial, y_initial


def calculate_string_length(
    tip_x: float,
    tip_y: float,
    nock_x: float,
    nock_y: float = 0.0
) -> float:
    """
    Calculate string length from tip to nock point
    
    Parameters:
    -----------
    tip_x, tip_y : float
        Tip coordinates [cm]
    nock_x, nock_y : float
        Nock point coordinates [cm]
    
    Returns:
    --------
    float : String length [cm]
    """
    return np.sqrt((tip_x - nock_x)**2 + (tip_y - nock_y)**2)


def compute_braced_geometry(
    x_unbraced_tip: float,
    y_unbraced_tip: float,
    brace_height_cm: float = TARGET_BRACE_HEIGHT_CM
) -> Tuple[float, float, float]:
    """
    Calculate braced tip position based on string length constraint
    
    Physical model:
    ---------------
    When bow is braced (string attached but not drawn):
    1. String is shorter than unbraced tip-to-tip distance
    2. This pulls tips inward and backward
    3. String must pass through (brace_height, 0)
    4. Tip position constrained by: x_tip ≈ brace_height
    
    Geometric constraint:
    ---------------------
    String length (tip to center) ≈ limb length - brace_height
    When braced: sqrt((x_tip - brace_height)² + y_tip²) = string_length
    
    Parameters:
    -----------
    x_unbraced_tip : float
        Tip x-coordinate in unbraced state [cm]
    y_unbraced_tip : float
        Tip y-coordinate in unbraced state [cm]
    brace_height_cm : float
        Target brace height [cm]
    
    Returns:
    --------
    Tuple[float, float, float] : (x_braced_tip, y_braced_tip, string_half_length)
    """
    # Limb length (approximated from unbraced tip position)
    limb_length_approx = np.sqrt(x_unbraced_tip**2 + y_unbraced_tip**2)
    
    # String length (half): assumes string is slightly shorter than twice limb length
    # Typical string is 95-98% of tip-to-tip distance
    string_half_length = limb_length_approx * 0.96
    
    # Braced tip position
    x_braced_tip = brace_height_cm
    
    # Calculate y from string constraint
    # String length from tip to nock: sqrt((x_tip - brace)² + y_tip²) = string_length
    # For x_tip ≈ brace: y_tip ≈ string_length
    
    # However, string also curves the limb, so actual y is slightly less
    # Use geometric constraint: tip must satisfy both string length and bending
    y_braced_tip = string_half_length - (brace_height_cm * 0.1)  # Correction factor
    
    # Ensure reasonable bounds
    y_braced_tip = max(y_unbraced_tip * 0.75, min(y_unbraced_tip * 0.95, y_braced_tip))
    
    return x_braced_tip, y_braced_tip, string_half_length


def _solve_limb_deformation_state(
    measurements: List[MeasurementPoint],
    limb_length_cm: float,
    draw_cm: float,
    side_profile: str,
    limb_type: str = "Upper",
    string_half_length: Optional[float] = None,
    n_segments: int = 100,
    draw_from_brace: bool = True,
) -> Dict[str, object]:
    """
    Solve one limb deformation state and return geometry + physics diagnostics.
    
    Includes:
    - EI-distributed bending response
    - side profile baseline geometry
    - 7-point side-angle interpolation
    - tip/string/bow angle diagnostics
    - local bending at measurement points
    """
    target_limb = [p for p in measurements if p.limb in ["Handle", limb_type]]
    target_limb.sort(key=lambda p: p.position_cm)
    
    s_array = np.linspace(0, limb_length_cm, n_segments)
    if len(target_limb) < 2:
        zeros = np.zeros_like(s_array)
        return {
            "x_coords": zeros,
            "y_coords": s_array.copy(),
            "thickness_profile": np.ones_like(s_array),
            "applied_force_n": 0.0,
            "strain_energy_j": 0.0,
            "tip_x_cm": 0.0,
            "tip_y_cm": float(limb_length_cm),
            "tip_bow_angle_deg": 0.0,
            "tip_string_angle_deg": 0.0,
            "string_to_draw_angle_deg": 90.0,
            "draw_force_gain": 0.0,
            "point_bending_deg": {},
            "mean_abs_bending_deg": 0.0,
        }
    
    positions = np.array([p.position_cm for p in target_limb], dtype=float)
    ei_values = np.array([max(p.ei_nm2, 1e-10) for p in target_limb], dtype=float)
    thickness_values = np.array([p.thickness_mm / 10.0 for p in target_limb], dtype=float)
    side_angle_deg_values = np.array([p.side_angle_deg for p in target_limb], dtype=float)
    
    unique_pos = np.unique(positions)
    unique_ei = np.array([ei_values[positions == pos].mean() for pos in unique_pos], dtype=float)
    unique_thickness = np.array([thickness_values[positions == pos].mean() for pos in unique_pos], dtype=float)
    unique_side_angle_deg = np.array(
        [side_angle_deg_values[positions == pos].mean() for pos in unique_pos], dtype=float
    )
    
    if len(unique_pos) < 2:
        unique_pos = np.array([0.0, limb_length_cm], dtype=float)
        unique_ei = np.array([unique_ei[0], unique_ei[0]], dtype=float)
        unique_thickness = np.array([unique_thickness[0], unique_thickness[0]], dtype=float)
        unique_side_angle_deg = np.array([unique_side_angle_deg[0], unique_side_angle_deg[0]], dtype=float)
    
    ei_interp = np.maximum(np.interp(s_array, unique_pos, unique_ei), 1e-10)
    thickness_interp = np.interp(s_array, unique_pos, unique_thickness)
    side_angle_interp_rad = np.radians(np.interp(s_array, unique_pos, unique_side_angle_deg))
    
    x_initial, y_initial = compute_initial_side_profile(s_array, limb_length_cm, side_profile)
    
    # Baseline tangent from side profile + user side-angle input.
    dx0 = np.gradient(x_initial, s_array)
    dy0 = np.gradient(y_initial, s_array)
    theta_initial = np.arctan2(dy0, dx0)
    theta_rest = theta_initial + side_angle_interp_rad
    
    s_m = s_array * CM_TO_M
    limb_m = limb_length_cm * CM_TO_M
    
    compliance_integrand = ((limb_m - s_m) ** 2) / ei_interp
    compliance = float(np.trapezoid(compliance_integrand, s_m))
    compliance = max(compliance, 1e-12)
    
    max_delta_angle_rad = np.pi * 0.82
    
    def _forward_from_force(force_n: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        moments = force_n * (limb_m - s_m)
        curvatures = moments / ei_interp
        
        delta_theta = np.zeros(n_segments)
        for i in range(1, n_segments):
            delta_theta[i] = delta_theta[i - 1] + curvatures[i] * (s_m[i] - s_m[i - 1])
            if delta_theta[i] > max_delta_angle_rad:
                delta_theta[i] = max_delta_angle_rad
        
        # Draw force bends limb toward +X direction.
        theta_total = theta_rest - delta_theta
        
        x_coords = np.zeros(n_segments)
        y_coords = np.zeros(n_segments)
        x_coords[0] = x_initial[0]
        y_coords[0] = 0.0
        for i in range(1, n_segments):
            ds_cm = s_array[i] - s_array[i - 1]
            x_coords[i] = x_coords[i - 1] + ds_cm * np.cos(theta_total[i - 1])
            y_coords[i] = y_coords[i - 1] + ds_cm * np.sin(theta_total[i - 1])
        
        return x_coords, y_coords, theta_total, delta_theta, moments
    
    force_n = 0.0
    x_state, y_state, theta_state, delta_state, moments_state = _forward_from_force(0.0)
    
    if draw_cm >= 0:
        is_braced = draw_cm == 0
        
        def _error_for_force(test_force: float) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x_t, y_t, theta_t, delta_t, moments_t = _forward_from_force(test_force)
            if is_braced:
                err = x_t[-1] - TARGET_BRACE_HEIGHT_CM
            else:
                nock_x = TARGET_BRACE_HEIGHT_CM + draw_cm if draw_from_brace else draw_cm
                target_len = string_half_length if (string_half_length is not None and string_half_length > 0) else None
                if target_len is None:
                    # fallback when no string constraint
                    err = x_t[-1] - (TARGET_BRACE_HEIGHT_CM + draw_cm)
                else:
                    err = calculate_string_length(x_t[-1], y_t[-1], nock_x, 0.0) - target_len
            return err, x_t, y_t, theta_t, delta_t, moments_t
        
        f_low = 0.0
        f_guess = (max(draw_cm, TARGET_BRACE_HEIGHT_CM) * CM_TO_M) / compliance
        f_high = max(10.0, f_guess * 8.0)
        
        err_low, x_l, y_l, theta_l, delta_l, moments_l = _error_for_force(f_low)
        err_high, x_h, y_h, theta_h, delta_h, moments_h = _error_for_force(f_high)
        
        # Expand upper bound until sign change or safety limit.
        tries = 0
        while err_low * err_high > 0 and tries < 24 and f_high < 1e8:
            f_high *= 2.0
            err_high, x_h, y_h, theta_h, delta_h, moments_h = _error_for_force(f_high)
            tries += 1
        
        best_abs_err = float("inf")
        if err_low * err_high <= 0:
            for _ in range(90):
                f_mid = 0.5 * (f_low + f_high)
                err_mid, x_m, y_m, theta_m, delta_m, moments_m = _error_for_force(f_mid)
                
                if y_m[-1] < 1.5:
                    f_high = f_mid
                    continue
                
                if abs(err_mid) < best_abs_err:
                    best_abs_err = abs(err_mid)
                    force_n = f_mid
                    x_state, y_state = x_m, y_m
                    theta_state, delta_state, moments_state = theta_m, delta_m, moments_m
                
                if abs(err_mid) < 0.01:
                    break
                
                if err_low * err_mid <= 0:
                    f_high = f_mid
                    err_high = err_mid
                else:
                    f_low = f_mid
                    err_low = err_mid
        else:
            # No clear bracket: choose better endpoint.
            candidates = [
                (abs(err_low), f_low, x_l, y_l, theta_l, delta_l, moments_l),
                (abs(err_high), f_high, x_h, y_h, theta_h, delta_h, moments_h),
            ]
            best = min(candidates, key=lambda t: t[0])
            force_n = best[1]
            x_state, y_state, theta_state, delta_state, moments_state = best[2], best[3], best[4], best[5], best[6]
    
    tip_x = float(x_state[-1])
    tip_y = float(y_state[-1])
    
    # U = ∫ M^2 / (2EI) ds
    strain_energy = float(np.trapezoid((moments_state ** 2) / (2.0 * ei_interp), s_m))
    
    tip_tangent_rad = float(theta_state[-1])
    tip_tangent_deg = math.degrees(tip_tangent_rad)
    # Deviation from local vertical bow axis.
    tip_bow_angle_deg = abs(((tip_tangent_deg - 90.0 + 180.0) % 360.0) - 180.0)
    
    nock_x_eval = TARGET_BRACE_HEIGHT_CM + max(draw_cm, 0.0) if draw_from_brace else max(draw_cm, 0.0)
    string_dir_rad = math.atan2(-tip_y, nock_x_eval - tip_x)
    # Interior angle at tip between:
    # 1) limb tangent directed toward handle (inward ray from tip)
    # 2) string segment directed from tip to nock
    tip_limb_inward_rad = tip_tangent_rad + math.pi
    tip_string_angle_deg = abs(math.degrees(string_dir_rad - tip_limb_inward_rad))
    tip_string_angle_deg = abs(((tip_string_angle_deg + 180.0) % 360.0) - 180.0)
    
    # Geometric transmission from string tension to nock draw force.
    # F_draw,limb = T * cos(alpha)
    # N_tip (solver load) approximates perpendicular component: N_tip = T * sin(beta)
    # => F_draw,limb = N_tip * cos(alpha) / sin(beta)
    string_to_draw_angle_deg = abs(math.degrees(string_dir_rad))
    cos_alpha = abs(math.cos(string_dir_rad))
    sin_beta = abs(math.sin(math.radians(tip_string_angle_deg)))
    draw_force_gain = cos_alpha / max(sin_beta, MIN_TIP_STRING_SIN)
    
    point_bending_deg: Dict[int, float] = {}
    for p in target_limb:
        point_bending_deg[p.point_id] = math.degrees(float(np.interp(p.position_cm, s_array, delta_state)))
    
    bending_vals = [abs(v) for pid, v in point_bending_deg.items() if pid != 0]
    mean_abs_bending_deg = float(np.mean(bending_vals)) if bending_vals else 0.0
    
    return {
        "x_coords": x_state,
        "y_coords": y_state,
        "thickness_profile": thickness_interp,
        "applied_force_n": float(force_n),
        "strain_energy_j": strain_energy,
        "tip_x_cm": tip_x,
        "tip_y_cm": tip_y,
        "tip_bow_angle_deg": float(tip_bow_angle_deg),
        "tip_string_angle_deg": float(tip_string_angle_deg),
        "string_to_draw_angle_deg": float(string_to_draw_angle_deg),
        "draw_force_gain": float(draw_force_gain),
        "point_bending_deg": point_bending_deg,
        "mean_abs_bending_deg": mean_abs_bending_deg,
    }


def compute_bow_deformation_realistic(
    measurements: List[MeasurementPoint],
    limb_length_cm: float,
    draw_cm: float,
    side_profile: str,
    limb_type: str = "Upper",
    string_half_length: Optional[float] = None,
    n_segments: int = 80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return limb geometry for visualization using the shared physics solver."""
    state = _solve_limb_deformation_state(
        measurements=measurements,
        limb_length_cm=limb_length_cm,
        draw_cm=draw_cm,
        side_profile=side_profile,
        limb_type=limb_type,
        string_half_length=string_half_length,
        n_segments=n_segments,
        draw_from_brace=True,
    )
    return (
        state["x_coords"],        # type: ignore[return-value]
        state["y_coords"],        # type: ignore[return-value]
        state["thickness_profile"]  # type: ignore[return-value]
    )


def create_virtual_tiller_realistic(
    measurements: List[MeasurementPoint],
    limb_length_cm: float,
    side_profile: str,
    draw_inches: List[float] = [-1, 0, 20, 28]
) -> go.Figure:
    """
    Create REALISTIC Virtual Tiller with kinematic bow deformation
    
    Shows actual bow shape as physical object with:
    - Symmetric limbs (upper/lower)
    - Thickness-based geometry
    - Bowstring visualization (braced and drawn states)
    - Side profile initial shape
    - CORRECT physics: tips move INWARD and BACKWARD when drawn
    - BRACE HEIGHT constraint: braced state maintains fixed brace height
    """
    
    fig = go.Figure()
    
    colors = ['#66615a', '#8e877b', '#7f9f9d', '#6ea7a5']
    alphas = [0.15, 0.25, 0.45, 0.7]
    labels = ['미휨(시위 없음)', '휨(시위 걸림)', '20인치 드로우', '28인치 드로우']
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Calculate STRING LENGTH from Braced state (draw = 0)
    # This establishes the fixed string length for all subsequent draws
    # ═══════════════════════════════════════════════════════════════════════
    string_half_length_upper = None
    string_half_length_lower = None
    
    # Find braced state index
    braced_idx = None
    for i, d in enumerate(draw_inches):
        if d == 0:
            braced_idx = i
            break
    
    if braced_idx is not None:
        draw_cm_braced = 0.0
        
        # Calculate braced geometry to get string length
        x_upper_braced, y_upper_braced, _ = compute_bow_deformation_realistic(
            measurements, limb_length_cm, draw_cm_braced, side_profile, 
            limb_type="Upper", string_half_length=None, n_segments=80
        )
        
        x_lower_braced, y_lower_braced, _ = compute_bow_deformation_realistic(
            measurements, limb_length_cm, draw_cm_braced, side_profile, 
            limb_type="Lower", string_half_length=None, n_segments=80
        )
        
        # Calculate string length from braced tip to nock
        tip_x_upper = x_upper_braced[-1]
        tip_y_upper = y_upper_braced[-1]
        string_half_length_upper = calculate_string_length(
            tip_x_upper, tip_y_upper, TARGET_BRACE_HEIGHT_CM, 0.0
        )
        
        tip_x_lower = x_lower_braced[-1]
        tip_y_lower = y_lower_braced[-1]
        string_half_length_lower = calculate_string_length(
            tip_x_lower, tip_y_lower, TARGET_BRACE_HEIGHT_CM, 0.0
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Draw all states with fixed string length constraint
    # ═══════════════════════════════════════════════════════════════════════
    
    for idx, draw_inch in enumerate(draw_inches):
        draw_cm = draw_inch * 2.54
        
        # ═══════════════════════════════════════════════════════════════════
        # REAL-TIME REACTIVITY: Uses current session measurements
        # CRITICAL FIX: Calculate Upper and Lower INDEPENDENTLY
        # Each limb uses its own EI distribution from measurements
        # ═══════════════════════════════════════════════════════════════════
        
        # Calculate UPPER limb with Upper measurements
        # Pass string length for drawn states to maintain constant string length
        x_upper, y_upper, thickness_upper = compute_bow_deformation_realistic(
            measurements,      # Current measurements with up-to-date EI
            limb_length_cm,
            draw_cm,
            side_profile,      # Current side profile selection
            limb_type="Upper", # UPPER limb
            string_half_length=string_half_length_upper if draw_cm > 0 else None,  # FIX: String length constraint
            n_segments=80      # High resolution for smooth curves
        )
        
        # Calculate LOWER limb with Lower measurements (INDEPENDENT calculation!)
        x_lower_raw, y_lower_raw, thickness_lower = compute_bow_deformation_realistic(
            measurements,      # Current measurements with up-to-date EI
            limb_length_cm,
            draw_cm,
            side_profile,      # Current side profile selection
            limb_type="Lower", # LOWER limb (uses different EI values!)
            string_half_length=string_half_length_lower if draw_cm > 0 else None,  # FIX: String length constraint
            n_segments=80      # High resolution for smooth curves
        )
        
        # Mirror lower limb coordinates (flip Y-axis for visualization)
        x_lower = x_lower_raw.copy()
        y_lower = -y_lower_raw  # Negative Y for lower limb positioning
        
        # Create back and belly curves for upper limb
        # Perpendicular offset based on thickness
        angles_upper = np.zeros(len(x_upper))
        for i in range(1, len(x_upper)):
            dx = x_upper[i] - x_upper[i-1]
            dy = y_upper[i] - y_upper[i-1]
            angles_upper[i] = np.arctan2(dy, dx)
        angles_upper[0] = angles_upper[1]
        
        # Back (positive perpendicular)
        x_upper_back = x_upper + (thickness_upper / 2.0) * np.sin(angles_upper)
        y_upper_back = y_upper - (thickness_upper / 2.0) * np.cos(angles_upper)
        
        # Belly (negative perpendicular)
        x_upper_belly = x_upper - (thickness_upper / 2.0) * np.sin(angles_upper)
        y_upper_belly = y_upper + (thickness_upper / 2.0) * np.cos(angles_upper)
        
        # Same for lower limb
        angles_lower = np.zeros(len(x_lower))
        for i in range(1, len(x_lower)):
            dx = x_lower[i] - x_lower[i-1]
            dy = y_lower[i] - y_lower[i-1]
            angles_lower[i] = np.arctan2(dy, dx)
        angles_lower[0] = angles_lower[1]
        
        x_lower_back = x_lower + (thickness_lower / 2.0) * np.sin(angles_lower)
        y_lower_back = y_lower - (thickness_lower / 2.0) * np.cos(angles_lower)
        
        x_lower_belly = x_lower - (thickness_lower / 2.0) * np.sin(angles_lower)
        y_lower_belly = y_lower + (thickness_lower / 2.0) * np.cos(angles_lower)
        
        # Fill upper limb body
        x_fill_upper = np.concatenate([x_upper_back, x_upper_belly[::-1]])
        y_fill_upper = np.concatenate([y_upper_back, y_upper_belly[::-1]])
        
        fig.add_trace(go.Scatter(
            x=x_fill_upper,
            y=y_fill_upper,
            fill='toself',
            fillcolor=f'rgba(110, 167, 165, {alphas[idx]})' if idx == len(draw_inches)-1 else f'rgba(128, 128, 128, {alphas[idx]})',
            line=dict(width=0),
            name=f'{labels[idx]} - 상부',
            showlegend=False,
            hoverinfo='skip',
        ))
        
        # Fill lower limb body
        x_fill_lower = np.concatenate([x_lower_back, x_lower_belly[::-1]])
        y_fill_lower = np.concatenate([y_lower_back, y_lower_belly[::-1]])
        
        fig.add_trace(go.Scatter(
            x=x_fill_lower,
            y=y_fill_lower,
            fill='toself',
            fillcolor=f'rgba(110, 167, 165, {alphas[idx]})' if idx == len(draw_inches)-1 else f'rgba(128, 128, 128, {alphas[idx]})',
            line=dict(width=0),
            name=f'{labels[idx]} - 하부',
            showlegend=False,
            hoverinfo='skip',
        ))
        
        # Add centerlines
        fig.add_trace(go.Scatter(
            x=x_upper,
            y=y_upper,
            mode='lines',
            name=f'{labels[idx]} - 상부',
            line=dict(color=colors[idx], width=1.5, dash='dot' if idx < len(draw_inches)-1 else 'solid'),
            showlegend=False,
        ))
        
        fig.add_trace(go.Scatter(
            x=x_lower,
            y=y_lower,
            mode='lines',
            name=f'{labels[idx]}',
            line=dict(color=colors[idx], width=1.5, dash='dot' if idx < len(draw_inches)-1 else 'solid'),
        ))
        
        # Add bowstring for braced and drawn positions
        if draw_inch >= 0:
            # String connects tip to nock point
            tip_upper_x = x_upper[-1]
            tip_upper_y = y_upper[-1]
            tip_lower_x = x_lower[-1]
            tip_lower_y = y_lower[-1]
            
            if draw_inch == 0:
                # Braced state: string at brace height, no nock point
                nock_x = TARGET_BRACE_HEIGHT_CM
                nock_y = 0.0
                
                # String passes through brace height vertically
                string_x = [tip_upper_x, nock_x, tip_lower_x]
                string_y = [tip_upper_y, nock_y, tip_lower_y]
                
                line_style = dict(color='#6ea7a5', width=1.5, dash='dot')
                
            else:
                # Drawn state: nock point is where arrow rests (pulled back by draw length)
                nock_x = TARGET_BRACE_HEIGHT_CM + draw_cm
                nock_y = 0.0
                
                # String: upper tip -> nock -> lower tip
                string_x = [tip_upper_x, nock_x, tip_lower_x]
                string_y = [tip_upper_y, nock_y, tip_lower_y]
                
                line_style = dict(
                    color='#c8a86b' if idx == len(draw_inches)-1 else '#999999', 
                    width=2, 
                    dash='solid'
                )
            
            fig.add_trace(go.Scatter(
                x=string_x,
                y=string_y,
                mode='lines',
                name=f'시위 ({labels[idx]})',
                line=line_style,
                showlegend=False,
            ))
            
            # Nock point marker (only for drawn states)
            if draw_inch > 0 and idx == len(draw_inches) - 1:
                fig.add_trace(go.Scatter(
                    x=[nock_x],
                    y=[nock_y],
                    mode='markers',
                    name='노크 지점',
                    marker=dict(color='#c8a86b', size=10, symbol='circle'),
                    showlegend=False,
                ))
            
            # Brace height indicator (only for braced state)
            if draw_inch == 0:
                fig.add_shape(
                    type="line",
                    x0=TARGET_BRACE_HEIGHT_CM, y0=-limb_length_cm*0.3,
                    x1=TARGET_BRACE_HEIGHT_CM, y1=limb_length_cm*0.3,
                    line=dict(color='#6ea7a5', width=1, dash='dash'),
                )
                fig.add_annotation(
                    x=TARGET_BRACE_HEIGHT_CM,
                    y=limb_length_cm*0.35,
                    text=f"Brace Height<br>{TARGET_BRACE_HEIGHT_CM} cm",
                    showarrow=False,
                    font=dict(color='#6ea7a5', size=9),
                    bgcolor='rgba(24, 28, 34, 0.82)',
                    bordercolor='#6ea7a5',
                    borderwidth=1,
                )
    
    # Add handle marker
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        name='핸들',
        marker=dict(color='#b8644a', size=12, symbol='square'),
    ))
    
    fig.update_layout(
        title="가상 틸러링: 활 변형 시뮬레이션",
        xaxis_title="수평 위치 (cm)",
        yaxis_title="수직 위치 (cm)",
        template="plotly_dark",
        paper_bgcolor='#111317',
        plot_bgcolor='#1a1d23',
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#e7e1d5"),
        height=700,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(24, 28, 34, 0.82)',
            bordercolor='#3a3d42',
            borderwidth=1
        ),
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='#c8a86b',
        zerolinewidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='#c8a86b',
        zerolinewidth=1,
    )
    
    return fig


def create_ballistics_trajectory_chart(
    arrow_speed_ms: float,
    max_range_m: float,
    effective_range_m: float
) -> go.Figure:
    """
    Create parabolic trajectory visualization with target silhouette
    
    Museum-style vector graphics
    """
    
    fig = go.Figure()
    
    # Maximum range trajectory (45 degrees)
    x_max, y_max = generate_trajectory_points(arrow_speed_ms, 45.0, 1.5, 150)
    
    fig.add_trace(go.Scatter(
        x=x_max,
        y=y_max,
        mode='lines',
        name='최대 사거리 (45°)',
        line=dict(color='#6ea7a5', width=2, dash='dot'),
        fill='tozeroy',
        fillcolor='rgba(110, 167, 165, 0.1)',
    ))
    
    # Effective range trajectory reference (35° flatter trajectory)
    x_eff, y_eff = generate_trajectory_points(arrow_speed_ms, 35.0, 1.5, 100)
    
    # Truncate to effective range; if not available (0), keep a visible reference segment.
    if effective_range_m > 0:
        mask = x_eff <= effective_range_m
    else:
        fallback_limit = max(1.0, min(max_range_m * 0.35, float(np.max(x_eff)) if len(x_eff) else 1.0))
        mask = x_eff <= fallback_limit
    
    x_eff_truncated = x_eff[mask]
    y_eff_truncated = y_eff[mask]
    
    if len(x_eff_truncated) == 0:
        x_eff_truncated = np.array([0.0])
        y_eff_truncated = np.array([1.5])
    
    fig.add_trace(go.Scatter(
        x=x_eff_truncated,
        y=y_eff_truncated,
        mode='lines',
        name='유효 사거리 기준 궤적 (35°)' if effective_range_m > 0 else '참고 궤적 (35°)',
        line=dict(
            color='#c8a86b' if effective_range_m > 0 else '#999999',
            width=3 if effective_range_m > 0 else 2,
            dash='solid' if effective_range_m > 0 else 'dot',
        ),
    ))
    
    # Launch point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[1.5],
        mode='markers',
        name='발사 지점',
        marker=dict(color='#b8644a', size=12, symbol='triangle-right'),
    ))
    
    # Impact point (max range)
    fig.add_trace(go.Scatter(
        x=[max_range_m],
        y=[0],
        mode='markers',
        name='최대 사거리 착탄',
        marker=dict(color='#6ea7a5', size=10, symbol='x'),
    ))
    
    # Effective range marker
    if effective_range_m > 0:
        # Find y-coordinate at effective range
        y_at_effective = np.interp(effective_range_m, x_eff_truncated, y_eff_truncated)
        
        fig.add_trace(go.Scatter(
            x=[effective_range_m],
            y=[y_at_effective],
            mode='markers',
            name='유효 사거리 한계',
            marker=dict(color='#c8a86b', size=12, symbol='diamond'),
        ))
        
        # Vertical line to target
        fig.add_trace(go.Scatter(
            x=[effective_range_m, effective_range_m],
            y=[0, y_at_effective],
            mode='lines',
            name='목표물 구역',
            line=dict(color='#c8a86b', width=2, dash='dash'),
            showlegend=False,
        ))
        
        # Target silhouette (simplified deer shape)
        target_x = effective_range_m
        target_height = 1.5  # meters (deer height)
        target_width = 0.8   # meters
        
        # Rectangle for target body
        fig.add_shape(
            type="rect",
            x0=target_x - target_width/2,
            y0=0,
            x1=target_x + target_width/2,
            y1=target_height,
            line=dict(color='#c8a86b', width=2),
            fillcolor='rgba(200, 168, 107, 0.2)',
        )
        
        # Add annotation
        fig.add_annotation(
            x=target_x,
            y=target_height + 0.5,
            text="목표물",
            showarrow=False,
            font=dict(color='#c8a86b', size=10, family='monospace'),
            bgcolor='rgba(24, 28, 34, 0.82)',
            bordercolor='#c8a86b',
            borderwidth=1,
        )
    else:
        fig.add_annotation(
            x=max_range_m * 0.45,
            y=max(y_max) * 0.65 if len(y_max) > 0 else 1.0,
            text="유효 사거리 기준 미충족<br>(현재 조건에서 0 m)",
            showarrow=False,
            font=dict(color='#999999', size=10),
            bgcolor='rgba(24, 28, 34, 0.82)',
            bordercolor='#666666',
            borderwidth=1,
        )
    
    fig.update_layout(
        title="탄도학: 포물선 궤적 분석",
        xaxis_title="수평 거리 (m)",
        yaxis_title="높이 (m)",
        template="plotly_dark",
        paper_bgcolor='#111317',
        plot_bgcolor='#1a1d23',
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#e7e1d5"),
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(24, 28, 34, 0.82)',
            bordercolor='#3a3d42',
            borderwidth=1
        ),
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
        range=[0, max(1.0, max_range_m * 1.1)],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
        range=[0, max(y_max) * 1.2] if len(y_max) > 0 else [0, 10],
    )
    
    return fig


def create_geometry_profile_chart(measurements: List[MeasurementPoint]) -> go.Figure:
    """Visualize width and thickness profiles - Museum Dark Mode"""
    
    upper = [p for p in measurements if p.limb == "Upper" or p.limb == "Handle"]
    lower = [p for p in measurements if p.limb == "Lower" or p.limb == "Handle"]
    
    fig = go.Figure()
    
    # Width - Upper
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in upper],
        y=[p.width_mm for p in upper],
        mode='lines+markers',
        name='상부 림 - 너비',
        line=dict(color='#6ea7a5', width=2),
        marker=dict(size=8, symbol='circle'),
    ))
    
    # Width - Lower
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in lower],
        y=[p.width_mm for p in lower],
        mode='lines+markers',
        name='하부 림 - 너비',
        line=dict(color='#6ea7a5', width=2, dash='dot'),
        marker=dict(size=8, symbol='circle-open'),
    ))
    
    # Thickness - Upper
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in upper],
        y=[p.thickness_mm for p in upper],
        mode='lines+markers',
        name='상부 림 - 두께',
        line=dict(color='#c8a86b', width=2),
        marker=dict(size=8, symbol='diamond'),
    ))
    
    # Thickness - Lower
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in lower],
        y=[p.thickness_mm for p in lower],
        mode='lines+markers',
        name='하부 림 - 두께',
        line=dict(color='#c8a86b', width=2, dash='dot'),
        marker=dict(size=8, symbol='diamond-open'),
    ))
    
    fig.update_layout(
        title="기하학적 프로파일: 너비 및 두께 분포",
        xaxis_title="핸들로부터 위치 (cm)",
        yaxis_title="치수 (mm)",
        template="plotly_dark",
        paper_bgcolor='#111317',
        plot_bgcolor='#1a1d23',
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#e7e1d5"),
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(24, 28, 34, 0.82)',
            bordercolor='#3a3d42',
            borderwidth=1
        ),
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#3a3d42',
        gridwidth=1,
    )
    
    return fig


def render_research_foundation() -> None:
    """Render comprehensive mathematical foundation - Museum Archive Style"""
    
    with st.expander("연구 기반: 수학적 및 물리적 원리", expanded=False):
        st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
        
        st.markdown("### I. 구조 역학")
        
        st.markdown("**오일러-베르누이 보 이론 (캔틸레버 구성)**")
        st.latex(r"\delta(s) = P \int_s^L \frac{(L-x)^2}{EI(x)} \, dx")
        st.caption("가변 굽힘 강성을 고려한 위치 함수로서의 림 처짐")
        
        st.markdown("---")
        
        st.markdown("**단면 2차 모멘트 (단면 기하학)**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**직사각형:**")
            st.latex(r"I_{\text{rect}} = \frac{w \cdot t^3}{12}")
            
            st.markdown("**타원형:**")
            st.latex(r"I_{\text{ellipse}} = \frac{\pi w t^3}{64}")
        
        with col2:
            st.markdown("**사다리꼴:**")
            st.latex(r"I_{\text{trap}} = \frac{t^3(b_1^2 + 4b_1b_2 + b_2^2)}{36(b_1 + b_2)}")
            
            st.markdown("**굽힘 강성:**")
            st.latex(r"EI = E \cdot I")
        
        st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
        
        st.markdown("### II. 에너지 분석")
        
        st.markdown("**저장 탄성 에너지 (FDC 적분)**")
        st.latex(r"E_{\text{stored}} = \int_0^{d_{\text{max}}} F(d) \, dd")
        st.caption("장력-드로우 곡선 하단 면적은 총 에너지 저장 용량을 나타냄")
        
        st.markdown("**발시 효율 (가상 질량 모델)**")
        st.latex(r"\eta = \frac{m_{\text{arrow}}}{m_{\text{arrow}} + m_{\text{virtual}}}")
        
        st.latex(r"m_{\text{virtual}} = 2 \int_0^L \rho(s) A(s) \left(\frac{s}{L}\right)^2 \, ds")
        st.caption("가상 질량은 위치 의존 속도로 가중된 림 운동 에너지를 고려함")
        
        st.markdown("**화살 운동 에너지 및 속도**")
        st.latex(r"E_{\text{kinetic}} = \eta \cdot E_{\text{stored}}")
        
        st.latex(r"v_{\text{arrow}} = \sqrt{\frac{2 E_{\text{kinetic}}}{m_{\text{arrow}}}}")
        
        st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
        
        st.markdown("### III. 탄도학 및 궤적")
        
        st.markdown("**발사체 운동 (포물선 궤적)**")
        st.latex(r"x(t) = v_0 \cos(\theta) \cdot t")
        st.latex(r"y(t) = h_0 + v_0 \sin(\theta) \cdot t - \frac{1}{2}gt^2")
        st.caption("발사 각도와 초기 높이를 고려한 시간의 함수로서의 위치")
        
        st.markdown("**최대 사거리 (최적 발사 각도)**")
        st.latex(r"R_{\text{max}} = \frac{v_0^2 \sin(2\theta)}{g} + \Delta x_{\text{height}}")
        st.caption("θ = 45°일 때 최대 수평 거리 달성")
        
        st.markdown("**유효 사거리 (에너지 기반 기준)**")
        st.latex(r"E(x) = E_0 \cdot e^{-kx}")
        st.latex(r"R_{\text{eff}} = -\frac{1}{k} \ln\left(\frac{E_{\text{min}}}{E_0}\right)")
        st.caption("화살이 목표물 관통을 위한 최소 운동 에너지(≥40 J)를 유지하는 거리")
        
        st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
        
        st.markdown("### IV. 상수 및 매개변수")
        
        params_df = pd.DataFrame([
            {"기호": "g", "값": "9.81 m/s²", "설명": "중력 가속도"},
            {"기호": "m_arrow", "값": "25 g", "설명": "표준 화살 질량"},
            {"기호": "E_min", "값": "40 J", "설명": "최소 관통 에너지"},
            {"기호": "h_0", "값": "1.5 m", "설명": "발사 높이"},
            {"기호": "θ_max", "값": "45°", "설명": "최적 발사 각도"},
        ])
        
        st.dataframe(params_df, use_container_width=True, hide_index=True)


def render_detailed_data_table(measurements: List[MeasurementPoint]) -> None:
    """Render comprehensive data table"""
    
    df = pd.DataFrame([
        {
            "지점": p.name,
            "부위": p.limb,
            "위치 (cm)": f"{p.position_cm:.1f}",
            "너비 (mm)": f"{p.width_mm:.1f}",
            "두께 (mm)": f"{p.thickness_mm:.1f}",
            "형상": p.shape,
            "측면 각도 (°)": f"{p.side_angle_deg:.1f}",
            "I (mm⁴)": f"{p.inertia_mm4:.2f}",
            "EI (N·m²)": f"{p.ei_nm2:.3f}",
        }
        for p in measurements
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="디지털 활 물리 & 탄도 연구소",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Apply museum dark theme
    apply_museum_dark_theme()
    
    # Header
    render_hero_header()
    
    # ═══════════════════════════════════════════════════════════════════════
    # REAL-TIME INPUT COLLECTION
    # Every sidebar change triggers re-render with updated values
    # ═══════════════════════════════════════════════════════════════════════
    species, total_length_cm, side_profile, measurements = render_sidebar_inputs()
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTE MATERIAL PROPERTIES (EI calculation)
    # This updates each measurement point with current EI based on:
    # - Young's modulus (E) from selected wood species
    # - Moment of inertia (I) from width/thickness input
    # Result: EI = E × I for each point
    # ═══════════════════════════════════════════════════════════════════════
    wood_props = WOOD_PROPERTIES[species]
    measurements = compute_ei_profile(
        measurements,
        wood_props['elastic_modulus_gpa'],
        wood_props['density_g_cm3']
    )
    
    limb_length_cm = total_length_cm / 2.0
    
    # Mass analysis
    total_mass_g, segment_masses_g = calculate_bow_mass_distribution(
        measurements, total_length_cm, wood_props['density_g_cm3']
    )
    
    # Force-Draw Curve with side profile
    fdc = compute_geometric_fdc(measurements, limb_length_cm, side_profile)
    
    # Stacking analysis
    stacking_point = find_stacking_point(fdc)
    
    # Energy & Efficiency
    stored_energy_j = calculate_stored_energy(fdc)
    efficiency = calculate_shooting_efficiency(
        measurements,
        segment_masses_g,
        total_mass_g,
        fdc=fdc,
    )
    arrow_speed_fps, arrow_speed_kmh = calculate_arrow_speed(stored_energy_j, efficiency)
    
    # Draw weight at 28"
    idx_28 = min(range(len(fdc)), key=lambda i: abs(fdc[i].draw_inch - 28.0))
    draw_weight_28 = fdc[idx_28].force_lbs
    
    # Specific energy
    specific_energy = stored_energy_j / (total_mass_g / 1000.0) if total_mass_g > 0 else 0.0
    
    # Package metrics
    metrics = PerformanceMetrics(
        draw_weight_28_lbs=draw_weight_28,
        stored_energy_j=stored_energy_j,
        shooting_efficiency=efficiency,
        arrow_speed_fps=arrow_speed_fps,
        arrow_speed_kmh=arrow_speed_kmh,
        estimated_mass_g=total_mass_g,
        specific_energy_j_per_kg=specific_energy,
        stacking_point_inch=stacking_point,
    )
    
    # Ballistics calculations
    arrow_speed_ms = arrow_speed_fps * FPS_TO_MS
    max_range_m = calculate_maximum_range(arrow_speed_ms, 45.0, 1.5)
    effective_range_m, initial_ke_j, effective_ke_threshold_j, used_adaptive_threshold = (
        calculate_effective_range_adaptive(
            arrow_speed_ms=arrow_speed_ms,
            arrow_mass_g=25.0,
            absolute_threshold_j=40.0,
            fallback_fraction_of_initial=0.35,
        )
    )
    
    # Render results
    render_performance_metrics(metrics)
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Main charts
    render_section_banner("장력-드로우 분석", "Force-draw relation · Tip-string angle · Stored energy")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_fdc = create_fdc_chart(fdc)
        st.plotly_chart(fig_fdc, use_container_width=True)
    
    with col2:
        fig_angle = create_string_angle_chart(fdc)
        st.plotly_chart(fig_angle, use_container_width=True)
        fig_energy = create_energy_storage_chart(fdc)
        st.plotly_chart(fig_energy, use_container_width=True)
    
    st.caption("참고: FDC(장력)와 저장 에너지(U)는 물리량 단위가 다르므로 별도 그래프로 분리 표시됩니다.")
    st.caption("참고: 발시 효율은 질량 분포 + late-draw 시위각/스태킹 손실 모델을 함께 반영합니다.")
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Ballistics Section
    render_section_banner("탄도학: 궤적 및 사거리 분석", "Trajectory envelope · Effective range modeling")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="최대 사거리",
            value=f"{max_range_m:.1f} m",
            help="45도 발사 각도 기준"
        )
    
    with col2:
        st.metric(
            label="유효 사거리",
            value=f"{effective_range_m:.1f} m",
            help=f"유효 기준 에너지 ≥ {effective_ke_threshold_j:.1f} J 유지 거리"
        )
    
    with col3:
        range_ratio = (effective_range_m / max_range_m * 100) if max_range_m > 0 else 0
        st.metric(
            label="유효/최대 비율",
            value=f"{range_ratio:.0f}%",
            help="전투 효용성 지표"
        )
    
    if used_adaptive_threshold:
        st.caption(
            f"ℹ️ 초기 운동에너지 {initial_ke_j:.1f} J가 절대 기준 40 J 미만이어서, "
            f"상대 기준(초기 에너지의 35% = {effective_ke_threshold_j:.1f} J)으로 유효 사거리를 계산했습니다."
        )
    else:
        st.caption(f"ℹ️ 유효 사거리 절대 기준: 최소 운동에너지 40 J (초기 {initial_ke_j:.1f} J)")
    
    fig_ballistics = create_ballistics_trajectory_chart(
        arrow_speed_ms, max_range_m, effective_range_m
    )
    st.plotly_chart(fig_ballistics, use_container_width=True)
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Virtual Tiller simulation (Realistic Kinematic)
    render_section_banner("가상 틸러링: 활 변형 시뮬레이션", "Real-time limb deformation · Brace-constrained kinematics")
    
    # Calculate EI range for debugging info
    ei_values_upper = [p.ei_nm2 for p in measurements if p.limb in ["Upper", "Handle"]]
    ei_values_lower = [p.ei_nm2 for p in measurements if p.limb in ["Lower", "Handle"]]
    
    ei_min_upper = min(ei_values_upper) if ei_values_upper else 0
    ei_max_upper = max(ei_values_upper) if ei_values_upper else 0
    ei_ratio_upper = ei_max_upper / ei_min_upper if ei_min_upper > 0 else 1.0
    
    ei_min_lower = min(ei_values_lower) if ei_values_lower else 0
    ei_max_lower = max(ei_values_lower) if ei_values_lower else 0
    ei_ratio_lower = ei_max_lower / ei_min_lower if ei_min_lower > 0 else 1.0
    
    st.caption(f"사실적 2D 물리 시뮬레이션 • 사이드 프로파일: {side_profile} • Brace Height: {TARGET_BRACE_HEIGHT_CM} cm (고정)")
    st.caption(f"⚙️ 상부 강성 (EI): {ei_min_upper:.2f} ~ {ei_max_upper:.2f} N·m² (비율: {ei_ratio_upper:.2f}x)")
    st.caption(f"⚙️ 하부 강성 (EI): {ei_min_lower:.2f} ~ {ei_max_lower:.2f} N·m² (비율: {ei_ratio_lower:.2f}x)")
    st.caption(f"🔧 물리 엔진 v5.0: 이진 탐색(Binary Search) 기반 — ① Forward Kinematics로 림 길이 자동 보존 ② 힘(Force) 최적화로 시위 길이 정밀 고정 ③ 누적 각도 제한(< 148°)으로 시위 벡터 역전 방지")
    
    fig_tiller = create_virtual_tiller_realistic(measurements, limb_length_cm, side_profile, [-1, 0, 20, 28])
    st.plotly_chart(fig_tiller, use_container_width=True)
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Geometry profile
    render_section_banner("기하학적 프로파일", "Seven-point survey of width and thickness")
    fig_geometry = create_geometry_profile_chart(measurements)
    st.plotly_chart(fig_geometry, use_container_width=True)
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Detailed data
    with st.expander("상세 측정 데이터", expanded=False):
        render_detailed_data_table(measurements)
    
    # Research foundation
    render_research_foundation()
    
    # Footer
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #9a9487; font-size: 0.82rem; padding: 2.1rem 0 1.5rem 0; border-top: 1px solid #3a3d42;'>
            <p style='letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem;'>Digital Experimental Archaeology Laboratory</p>
            <p style='margin: 0.15rem 0;'>Museum-grade interactive research platform © 2026</p>
            <p style='font-style: italic; margin-top: 0.55rem; letter-spacing: 0.03em;'>
                "고대 장인정신과 현대 계산 물리학의 융합"
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
