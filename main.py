"""
════════════════════════════════════════════════════════════════════════════════
DIGITAL BOW PHYSICS & BALLISTICS LABORATORY v4.1
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

REAL-TIME REACTIVITY (v4.1):
- Instant response to width/thickness changes in sidebar
- EI-dependent bending: thicker sections bend less (κ = M/EI)
- Side profile initial geometry fully integrated
- Brace height constraint (17cm) maintained across all configurations
- High-resolution curve interpolation (80 segments)

THEORETICAL FOUNDATION:
- Cantilever Beam Theory with Non-Linear Geometry
- Second Moment of Area (Multiple Cross-Sections)
- EI Distribution-Based Curvature Calculation
- String Angle Dynamics & Tip Velocity Penalties
- Kinetic Energy Transfer Efficiency
- Projectile Motion Physics (Parabolic Trajectories)

MUSEUM-GRADE PRESENTATION:
- Dark Mode Research Dashboard (Deep Navy/Charcoal)
- Metallic Accent Colors (Cyan/Gold)
- Vector Graphics & Geometric Line Art
- LaTeX Mathematical Notation
- Real-Time Debugging Info (EI Range Display)
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
    string_angle_deg: float
    tip_deflection_cm: float


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


def compute_geometric_fdc(
    measurements: List[MeasurementPoint],
    limb_length_cm: float,
    side_profile: str = "Straight",
    max_draw_inch: float = DRAW_LENGTH_INCH,
    n_steps: int = FDC_DRAW_STEPS
) -> List[ForceCurvePoint]:
    """
    Compute Force-Draw Curve with geometric string angle effects
    
    STACKING PHENOMENON:
    - As draw increases, tip bends more → string angle increases
    - Higher angle → more force required for same additional draw
    - Taper (thin tips) → early stacking
    - Parallel (thick tips) → delayed stacking
    
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
    fdc_points = []
    
    # Get side profile modifiers
    profile_config = SIDE_PROFILE_OPTIONS.get(side_profile, SIDE_PROFILE_OPTIONS["Straight"])
    fdc_modifier = profile_config["fdc_modifier"]
    stacking_modifier = profile_config["stacking_modifier"]
    
    for draw_inch in draw_inches:
        draw_m = draw_inch * INCH_TO_M
        
        # Initial force estimate (linear approximation)
        # Refine iteratively considering string angle
        
        # Extract EI profile
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
        
        # Interpolate and integrate compliance
        s_samples_cm = np.linspace(0, limb_length_cm, INTEGRATION_STEPS)
        ei_samples = np.maximum(np.interp(s_samples_cm, s_unique, ei_unique), 1e-10)
        
        s_m = s_samples_cm * CM_TO_M
        L_m = limb_length_cm * CM_TO_M
        
        # Compliance integral: C = ∫ (L-s)²/EI ds
        integrand = ((L_m - s_m) ** 2) / ei_samples
        compliance = np.trapezoid(integrand, s_m)
        
        if compliance <= 0:
            force_n = 0.0
        else:
            force_n = draw_m / compliance
        
        # String angle effect (stacking multiplier)
        # Compute tip deflection
        tip_deflection_cm = draw_inch * INCH_TO_CM
        string_angle_deg = compute_string_angle(tip_deflection_cm, limb_length_cm)
        
        # Stacking factor: increases exponentially with angle
        # Physical basis: F_effective = F_linear / cos(θ)
        # Additional taper effect: thinner tips stack earlier
        
        # Taper ratio: tip EI / handle EI
        handle_ei = measurements[0].ei_nm2
        tip_ei = measurements[-1].ei_nm2
        taper_ratio = tip_ei / max(handle_ei, 1e-10)
        
        # Stacking multiplier
        angle_rad = math.radians(string_angle_deg)
        cos_angle = math.cos(angle_rad)
        
        if cos_angle < 0.1:
            cos_angle = 0.1  # Prevent singularity
        
        stacking_factor = 1.0 / cos_angle
        
        # Taper penalty: thin tips stack earlier
        taper_penalty = 1.0 + (1.0 - taper_ratio) * (angle_rad ** 2)
        
        # Apply side profile modifiers
        force_n_effective = force_n * stacking_factor * taper_penalty * fdc_modifier
        
        # Stacking modifier affects how early stacking occurs
        stacking_factor_adjusted = stacking_factor ** stacking_modifier
        force_n_effective = force_n * stacking_factor_adjusted * taper_penalty * fdc_modifier
        
        force_lbs = force_n_effective * N_TO_LBS
        
        fdc_point = ForceCurvePoint(
            draw_inch=draw_inch,
            force_lbs=force_lbs,
            string_angle_deg=string_angle_deg,
            tip_deflection_cm=tip_deflection_cm
        )
        
        fdc_points.append(fdc_point)
    
    return fdc_points


def find_stacking_point(fdc: List[ForceCurvePoint], threshold_ratio: float = 1.5) -> Optional[float]:
    """
    Identify stacking point: where force increase rate exceeds threshold
    
    Stacking = d²F/dx² > threshold
    
    Parameters:
    -----------
    fdc : List[ForceCurvePoint]
    threshold_ratio : float
        Second derivative threshold
    
    Returns:
    --------
    Optional[float] : Stacking point draw length [inch], or None
    """
    if len(fdc) < 3:
        return None
    
    draws = np.array([p.draw_inch for p in fdc])
    forces = np.array([p.force_lbs for p in fdc])
    
    # First derivative (force rate)
    dF_dx = np.gradient(forces, draws)
    
    # Second derivative (acceleration)
    d2F_dx2 = np.gradient(dF_dx, draws)
    
    # Find where second derivative exceeds threshold
    mean_d2F = np.mean(np.abs(d2F_dx2[1:-1]))
    
    for i, val in enumerate(d2F_dx2):
        if i > len(d2F_dx2) // 2:  # Only check latter half
            if abs(val) > threshold_ratio * mean_d2F:
                return draws[i]
    
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
    draws_m = np.array([p.draw_inch * INCH_TO_M for p in fdc])
    forces_n = np.array([p.force_lbs * LBS_TO_N for p in fdc])
    
    # Integrate using Simpson's rule
    energy_j = simpson(forces_n, x=draws_m)
    
    return energy_j


def calculate_shooting_efficiency(
    measurements: List[MeasurementPoint],
    segment_masses_g: List[float],
    total_mass_g: float
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
    
    # Efficiency
    efficiency = arrow_mass_g / (arrow_mass_g + virtual_mass_g)
    
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
    
    # Time to reach ground (quadratic formula)
    # h = h0 + v_y*t - 0.5*g*t²
    # 0 = launch_height_m + v_y*t - 0.5*g*t²
    
    a = -0.5 * GRAVITY
    b = v_y
    c = launch_height_m
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return 0.0
    
    t1 = (-b + math.sqrt(discriminant)) / (2*a)
    t2 = (-b - math.sqrt(discriminant)) / (2*a)
    
    t_flight = max(t1, t2)
    
    # Horizontal range
    range_m = v_x * t_flight
    
    return range_m


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
    
    # Flight time
    a = -0.5 * GRAVITY
    b = v_y
    c = launch_height_m
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return np.array([0.0]), np.array([launch_height_m])
    
    t_flight = (-b + math.sqrt(discriminant)) / (2*a)
    
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
    """Apply museum-grade dark mode research dashboard theme"""
    st.markdown("""
        <style>
        /* Museum Dark Mode: Deep Navy & Charcoal Foundation */
        .stApp {
            background: linear-gradient(180deg, #0a0e27 0%, #121212 50%, #0a0e27 100%);
            font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif;
            color: #E0E0E0;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1600px;
        }
        
        /* Sidebar: Archive Drawer Style */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1f3a 0%, #0d111f 100%);
            border-right: 1px solid #2a3f5f;
            box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5);
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #E0E0E0;
        }
        
        /* Typography: Museum Exhibition Labels */
        h1 {
            color: #00d4ff !important;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
            border-bottom: 2px solid #ffd700 !important;
            padding-bottom: 1rem !important;
            margin-bottom: 2rem !important;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        
        h2 {
            color: #ffd700 !important;
            font-weight: 600 !important;
            font-size: 1.5rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            margin-top: 3rem !important;
            margin-bottom: 1.5rem !important;
            border-left: 4px solid #00d4ff !important;
            padding-left: 1rem !important;
        }
        
        h3 {
            color: #B0B0B0 !important;
            font-weight: 500 !important;
            font-size: 1.1rem !important;
            letter-spacing: 0.05em !important;
            margin-top: 1.5rem !important;
        }
        
        /* Metric Cards: Illuminated Display Panels */
        [data-testid="stMetricValue"] {
            color: #00d4ff !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            font-family: 'JetBrains Mono', 'Courier New', monospace !important;
            text-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        }
        
        [data-testid="stMetricLabel"] {
            color: #B0B0B0 !important;
            font-size: 0.75rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.15em !important;
            font-weight: 500 !important;
        }
        
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #1a1f3a 0%, #0d111f 100%);
            border: 1px solid #2a3f5f;
            border-radius: 8px;
            padding: 1.5rem 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }
        
        /* Input Fields: Technical Instrument Style */
        .stNumberInput input, .stSelectbox select {
            background-color: #1a1f3a !important;
            border: 1px solid #2a3f5f !important;
            color: #E0E0E0 !important;
            border-radius: 6px !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        .stNumberInput input:focus, .stSelectbox select:focus {
            border-color: #00d4ff !important;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.3) !important;
        }
        
        .stNumberInput label, .stSelectbox label {
            color: #B0B0B0 !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
        }
        
        /* Slider: Precision Control */
        .stSlider label {
            color: #B0B0B0 !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em !important;
        }
        
        /* Expander: Archive Drawer */
        .streamlit-expanderHeader {
            background: linear-gradient(90deg, #1a1f3a 0%, #0d111f 100%) !important;
            border: 1px solid #2a3f5f !important;
            border-radius: 6px !important;
            color: #E0E0E0 !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: linear-gradient(90deg, #2a3f5f 0%, #1a1f3a 100%) !important;
            border-color: #00d4ff !important;
        }
        
        .streamlit-expanderContent {
            background-color: #0d111f !important;
            border: 1px solid #2a3f5f !important;
            border-top: none !important;
            border-radius: 0 0 6px 6px !important;
        }
        
        /* Dataframe: Digital Archive Table */
        .dataframe {
            font-size: 0.85rem !important;
            border-collapse: collapse !important;
            background-color: #0d111f !important;
        }
        
        .dataframe th {
            background: linear-gradient(180deg, #1a1f3a 0%, #0d111f 100%) !important;
            color: #ffd700 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #2a3f5f !important;
            padding: 0.75rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        
        .dataframe td {
            border-bottom: 1px solid #2a3f5f !important;
            padding: 0.6rem 0.75rem !important;
            color: #E0E0E0 !important;
        }
        
        /* Caption: Museum Label */
        .caption {
            color: #808080;
            font-size: 0.8rem;
            font-style: italic;
            margin-top: 0.5rem;
            letter-spacing: 0.03em;
        }
        
        /* Strategic Metallic Accents */
        .cyan-accent {
            color: #00d4ff;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.4);
        }
        
        .gold-accent {
            color: #ffd700;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(255, 215, 0, 0.4);
        }
        
        .critical-value {
            color: #ff6b6b;
            font-weight: 700;
        }
        
        /* Dividers: Light Beam */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #2a3f5f, transparent);
            margin: 3rem 0;
            box-shadow: 0 0 10px rgba(42, 63, 95, 0.5);
        }
        
        /* Section Cards */
        .section-card {
            background: linear-gradient(135deg, #1a1f3a 0%, #0d111f 100%);
            border: 1px solid #2a3f5f;
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        }
        
        /* Geometric Line Decorations */
        .geometric-line {
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, 
                transparent 0%, 
                #00d4ff 20%, 
                #ffd700 50%, 
                #00d4ff 80%, 
                transparent 100%);
            margin: 1.5rem 0;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar_inputs() -> Tuple[str, float, str, List[MeasurementPoint]]:
    """
    Render sidebar input interface - Museum Archive Style
    
    Returns:
    --------
    Tuple[str, float, str, List[MeasurementPoint]]
        (wood_species, total_length_cm, side_profile, measurements)
    """
    with st.sidebar:
        st.markdown("## ═══ 활 제원 설정 ═══")
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
        st.markdown("### 사이드 프로파일 (초기 형상)")
        
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
        st.markdown("## ═══ 7개 지점 측정값 ═══")
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
    
    st.markdown("## ═══ 성능 분석 ═══")
    
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
    
    st.markdown("")
    
    # Arrow velocity metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<p class="cyan-accent" style="font-size: 0.85rem; margin-bottom: 0.3rem;">화살 속도</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 2rem; font-weight: 700; color: #00d4ff; margin: 0;">{metrics.arrow_speed_fps:.1f}</p>', unsafe_allow_html=True)
        st.caption("피트/초 (FPS)")
    
    with col2:
        st.markdown('<p class="gold-accent" style="font-size: 0.85rem; margin-bottom: 0.3rem;">화살 속도</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 2rem; font-weight: 700; color: #ffd700; margin: 0;">{metrics.arrow_speed_kmh:.1f}</p>', unsafe_allow_html=True)
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
            <div style='background: linear-gradient(90deg, rgba(255, 107, 107, 0.2), transparent); 
                        border-left: 4px solid #ff6b6b; padding: 1rem; margin-top: 1rem; border-radius: 4px;'>
                <span style='color: #ff6b6b; font-weight: 700;'>⚠ 스택킹 감지</span><br>
                <span style='color: #E0E0E0;'>드로우 {metrics.stacking_point_inch:.1f}인치부터 급격한 힘 증가 시작</span>
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
        line=dict(color='#00d4ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.15)',
    ))
    
    # Highlight 28" point
    idx_28 = min(range(len(draws)), key=lambda i: abs(draws[i] - 28.0))
    fig.add_trace(go.Scatter(
        x=[draws[idx_28]],
        y=[forces[idx_28]],
        mode='markers',
        name='28인치 드로우 지점',
        marker=dict(color='#ffd700', size=15, symbol='diamond', line=dict(color='#fff', width=2)),
    ))
    
    # Add vertical line at 28"
    fig.add_shape(
        type="line",
        x0=28, y0=0,
        x1=28, y1=forces[idx_28],
        line=dict(color='#ffd700', width=2, dash='dash'),
    )
    
    fig.update_layout(
        title="장력-드로우 곡선 (FDC)",
        xaxis_title="드로우 길이 (인치)",
        yaxis_title="장력 (파운드)",
        template="plotly_dark",
        paper_bgcolor='#0d111f',
        plot_bgcolor='#1a1f3a',
        font=dict(family="Inter, sans-serif", size=12, color="#E0E0E0"),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(26, 31, 58, 0.8)',
            bordercolor='#2a3f5f',
            borderwidth=1
        ),
        height=500,
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
        gridwidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
        gridwidth=1,
    )
    
    return fig


def create_string_angle_chart(fdc: List[ForceCurvePoint]) -> go.Figure:
    """Visualize string angle progression (stacking indicator) - Museum Dark Mode"""
    
    draws = [p.draw_inch for p in fdc]
    angles = [p.string_angle_deg for p in fdc]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=draws,
        y=angles,
        mode='lines',
        name='시위 각도',
        line=dict(color='#ff6b6b', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.15)',
    ))
    
    # Critical angle indicator (around 45 degrees - high stacking)
    fig.add_shape(
        type="line",
        x0=min(draws), y0=45,
        x1=max(draws), y1=45,
        line=dict(color='#ffd700', width=1, dash='dash'),
    )
    
    fig.add_annotation(
        x=max(draws) * 0.95,
        y=45,
        text="임계 각도 (45°)",
        showarrow=False,
        font=dict(color='#ffd700', size=9),
        xanchor='right',
        yanchor='bottom',
    )
    
    fig.update_layout(
        title="시위 각도 변화 (스택킹 지표)",
        xaxis_title="드로우 길이 (인치)",
        yaxis_title="시위 각도 (도)",
        template="plotly_dark",
        paper_bgcolor='#0d111f',
        plot_bgcolor='#1a1f3a',
        font=dict(family="Inter, sans-serif", size=12, color="#E0E0E0"),
        height=400,
        showlegend=False,
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
        gridwidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
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
    
    if profile_type == "Straight":
        # No curvature
        pass
    
    elif profile_type == "Reflex":
        # Tips curve away from archer (negative X direction initially)
        # Parabolic curve: x = -k * s²
        x_initial = -reflex_factor * 2.0 * (s_norm ** 2)
    
    elif profile_type == "Deflex":
        # Tips curve toward archer (positive X direction initially)
        x_initial = -reflex_factor * 2.0 * (s_norm ** 2)  # reflex_factor is negative
    
    elif profile_type == "Recurve":
        # Working limb straight, tips recurve strongly
        # Only last 30% curves
        recurve_start = 0.7
        for i, s in enumerate(s_norm):
            if s < recurve_start:
                x_initial[i] = 0.0
            else:
                # Strong curve at tips
                local_norm = (s - recurve_start) / (1.0 - recurve_start)
                x_initial[i] = -reflex_factor * 3.0 * (local_norm ** 2)
    
    elif profile_type == "Decurve":
        # Smooth deflex throughout
        x_initial = -reflex_factor * 1.5 * (s_norm ** 1.5)
    
    return x_initial, y_initial


def compute_braced_geometry(
    x_unbraced_tip: float,
    y_unbraced_tip: float,
    brace_height_cm: float = TARGET_BRACE_HEIGHT_CM
) -> Tuple[float, float]:
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
    
    Simplification for x_tip ≈ brace_height:
    y_tip ≈ sqrt(string_length²) ≈ limb_length - brace_height
    
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
    Tuple[float, float] : (x_braced_tip, y_braced_tip)
    """
    # Limb length (approximated from unbraced tip position)
    limb_length_approx = np.sqrt(x_unbraced_tip**2 + y_unbraced_tip**2)
    
    # String length (half): assumes string is slightly shorter than twice limb length
    # Typical string is 95-98% of tip-to-tip distance
    string_half_length = limb_length_approx * 0.96
    
    # Braced tip position
    x_braced_tip = brace_height_cm
    
    # Calculate y from string constraint
    # String from (brace_height, y_tip) to (brace_height, 0)
    # But string also pulls tip backward slightly
    # Distance from tip to nock: sqrt((brace_height - brace_height)^2 + y_tip^2)
    # This simplifies to: y_tip = string_half_length
    
    # However, string also curves the limb, so actual y is slightly less
    # Use geometric constraint: tip must satisfy both string length and bending
    y_braced_tip = string_half_length - (brace_height_cm * 0.1)  # Correction factor
    
    # Ensure reasonable bounds
    y_braced_tip = max(y_unbraced_tip * 0.75, min(y_unbraced_tip * 0.95, y_braced_tip))
    
    return x_braced_tip, y_braced_tip


def compute_bow_deformation_realistic(
    measurements: List[MeasurementPoint],
    limb_length_cm: float,
    draw_cm: float,
    side_profile: str,
    n_segments: int = 80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PHYSICALLY CORRECT bow deformation under draw with EI-dependent bending
    
    Key physics:
    - Bow stands vertically along Y-axis
    - String pulls tips inward (toward Y=0) and backward (toward +X)
    - Draw force creates bending moment that curves limbs
    - CRITICAL: Bending inversely proportional to EI (stiffer sections bend less)
    
    Real-time reactivity:
    - Uses current measurements EI values directly
    - Thickness changes immediately affect local stiffness
    - Side profile provides initial geometry
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
        Current measurement data with up-to-date EI values
    limb_length_cm : float
    draw_cm : float
    side_profile : str
    n_segments : int
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (x_coords, y_coords, thickness_profile)
    """
    # Extract upper limb data - MUST be current session data
    upper_limb = [p for p in measurements if p.limb in ["Handle", "Upper"]]
    upper_limb.sort(key=lambda p: p.position_cm)
    
    positions = np.array([p.position_cm for p in upper_limb])
    ei_values = np.array([max(p.ei_nm2, 1e-10) for p in upper_limb])
    thicknesses = np.array([p.thickness_mm / 10.0 for p in upper_limb])
    
    # High-resolution interpolation for smooth curves
    s_array = np.linspace(0, limb_length_cm, n_segments)
    ei_interp = np.interp(s_array, positions, ei_values)
    thickness_interp = np.interp(s_array, positions, thicknesses)
    
    # Get initial profile shape from Side Profile selection
    x_initial, y_initial = compute_initial_side_profile(s_array, limb_length_cm, side_profile)
    
    # Handle special case: Braced state (draw_cm == 0)
    is_braced = (draw_cm == 0)
    
    if draw_cm < 0:
        # Truly unbraced (negative draw = no string attached)
        return x_initial, y_initial, thickness_interp
    
    # Calculate string tension and EI-dependent bending
    nock_x = draw_cm if draw_cm > 0 else TARGET_BRACE_HEIGHT_CM
    nock_y = 0.0
    
    # Estimate string force based on draw
    draw_m = max(draw_cm, 0) / 100.0
    s_m = s_array / 100.0
    limb_m = limb_length_cm / 100.0
    
    # Compliance calculation (critical for EI sensitivity)
    integrand = ((limb_m - s_m) ** 2) / ei_interp
    compliance = np.trapezoid(integrand, s_m)
    
    if is_braced:
        # Braced state: smaller pre-tension force
        string_force_n = (TARGET_BRACE_HEIGHT_CM / 100.0) / compliance if compliance > 0 else 0.0
    else:
        # Drawn state: full draw force
        string_force_n = draw_m / compliance if compliance > 0 else 0.0
    
    # Apply side profile force modifier
    profile_config = SIDE_PROFILE_OPTIONS.get(side_profile, SIDE_PROFILE_OPTIONS["Straight"])
    string_force_n *= profile_config["fdc_modifier"]
    
    # ═══════════════════════════════════════════════════════════════════════
    # CRITICAL: EI-DEPENDENT BENDING CALCULATION
    # Higher EI → Less bending (κ = M/EI)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Calculate curvature at each point: κ(s) = M(s) / EI(s)
    curvatures = np.zeros(n_segments)
    
    for i in range(n_segments):
        s_m_local = s_array[i] / 100.0
        
        # Bending moment: M(s) = F × (L - s)
        moment_arm = limb_m - s_m_local
        bending_moment = string_force_n * moment_arm
        
        # Curvature: inversely proportional to EI
        # This is where thickness changes take effect!
        curvatures[i] = bending_moment / ei_interp[i]
    
    # Integrate curvature to get angles
    angles = np.zeros(n_segments)
    for i in range(1, n_segments):
        ds = (s_array[i] - s_array[i-1]) / 100.0
        # Cumulative rotation
        angles[i] = angles[i-1] + curvatures[i] * ds
    
    # Apply side profile initial angle offset
    profile_angle_offset = 0.0
    if side_profile == "Reflex":
        profile_angle_offset = -0.2
    elif side_profile == "Deflex":
        profile_angle_offset = 0.15
    elif side_profile == "Recurve":
        # Progressive offset at tips
        for i in range(n_segments):
            s_norm = s_array[i] / limb_length_cm
            if s_norm > 0.7:
                angles[i] -= 0.4 * ((s_norm - 0.7) / 0.3) ** 2
    
    angles += profile_angle_offset
    
    # ═══════════════════════════════════════════════════════════════════════
    # FORWARD KINEMATICS: Integrate angles to cartesian coordinates
    # ═══════════════════════════════════════════════════════════════════════
    
    x_deformed = np.zeros(n_segments)
    y_deformed = np.zeros(n_segments)
    
    # Start at handle (origin)
    x_deformed[0] = x_initial[0]
    y_deformed[0] = 0.0
    
    for i in range(1, n_segments):
        ds = (s_array[i] - s_array[i-1]) / 100.0  # meters
        
        # Current direction: start vertical, rotate by accumulated angle
        # θ = 0 means pointing right (+X), θ = π/2 means pointing up (+Y)
        # Limb starts at π/2 (vertical), bends toward 0 (horizontal/backward)
        direction_angle = np.pi/2 - angles[i-1]
        
        # Displacement components
        dx = ds * 100.0 * np.cos(direction_angle)  # back to cm
        dy = ds * 100.0 * np.sin(direction_angle)
        
        x_deformed[i] = x_deformed[i-1] + dx
        y_deformed[i] = y_deformed[i-1] + dy
    
    # ═══════════════════════════════════════════════════════════════════════
    # POST-PROCESSING: Apply geometric constraints
    # ═══════════════════════════════════════════════════════════════════════
    
    if is_braced:
        # Braced state: Enforce brace height constraint while respecting EI distribution
        # The deformation computed above is EI-dependent
        # Now scale to match target brace height
        
        tip_idx = -1
        current_tip_x = x_deformed[tip_idx]
        current_tip_y = y_deformed[tip_idx]
        
        # Target tip position from geometric constraint
        x_unbraced_tip = x_initial[tip_idx]
        y_unbraced_tip = y_initial[tip_idx]
        x_target_tip, y_target_tip = compute_braced_geometry(
            x_unbraced_tip, y_unbraced_tip, TARGET_BRACE_HEIGHT_CM
        )
        
        # Scale deformation to match target tip position
        # This preserves the EI-dependent curve shape
        for i in range(n_segments):
            s_norm = s_array[i] / limb_length_cm
            
            # Blend between initial and deformed, scaled to target
            if current_tip_x > 0:
                x_scale = (x_target_tip - x_initial[0]) / (current_tip_x - x_initial[0])
            else:
                x_scale = 1.0
                
            if current_tip_y > 0:
                y_scale = (y_target_tip - y_initial[0]) / (current_tip_y - y_initial[0])
            else:
                y_scale = 1.0
            
            # Apply progressive scaling (more at tip, less at handle)
            x_deformed[i] = x_initial[i] + (x_deformed[i] - x_initial[i]) * x_scale
            y_deformed[i] = y_initial[i] + (y_deformed[i] - y_initial[i]) * y_scale
            
    else:
        # Drawn state: Apply inward compression (tips move toward centerline)
        # This simulates string pulling tips together
        draw_ratio = draw_cm / (limb_length_cm * 2.0)
        compression_factor = max(0.5, 1.0 - draw_ratio * 0.5)
        
        for i in range(n_segments):
            s_norm = s_array[i] / limb_length_cm
            # Progressive compression: more at tips, less at handle
            local_compression = 1.0 - s_norm * (1.0 - compression_factor)
            y_deformed[i] = y_deformed[i] * local_compression
    
    return x_deformed, y_deformed, thickness_interp


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
    
    colors = ['#606060', '#808080', '#00a0d0', '#00d4ff']
    alphas = [0.15, 0.25, 0.45, 0.7]
    labels = ['미휨(시위 없음)', '휨(시위 걸림)', '20인치 드로우', '28인치 드로우']
    
    for idx, draw_inch in enumerate(draw_inches):
        draw_cm = draw_inch * 2.54
        
        # ═══════════════════════════════════════════════════════════════════
        # REAL-TIME REACTIVITY: Uses current session measurements
        # When user changes width/thickness in sidebar, measurements are updated
        # with new EI values, which immediately affect the deformation calculation
        # ═══════════════════════════════════════════════════════════════════
        x_upper, y_upper, thickness_upper = compute_bow_deformation_realistic(
            measurements,      # Current measurements with up-to-date EI
            limb_length_cm,
            draw_cm,
            side_profile,      # Current side profile selection
            n_segments=80      # High resolution for smooth curves
        )
        
        # Mirror for lower limb (reflect across x-axis at handle)
        x_lower = x_upper.copy()
        y_lower = -y_upper
        thickness_lower = thickness_upper.copy()
        
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
            fillcolor=f'rgba(0, 212, 255, {alphas[idx]})' if idx == len(draw_inches)-1 else f'rgba(128, 128, 128, {alphas[idx]})',
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
            fillcolor=f'rgba(0, 212, 255, {alphas[idx]})' if idx == len(draw_inches)-1 else f'rgba(128, 128, 128, {alphas[idx]})',
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
                
                line_style = dict(color='#00d4ff', width=1.5, dash='dot')
                
            else:
                # Drawn state: nock point is where arrow rests (pulled back by draw length)
                nock_x = draw_cm
                nock_y = 0.0
                
                # String: upper tip -> nock -> lower tip
                string_x = [tip_upper_x, nock_x, tip_lower_x]
                string_y = [tip_upper_y, nock_y, tip_lower_y]
                
                line_style = dict(
                    color='#ffd700' if idx == len(draw_inches)-1 else '#999999', 
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
                    marker=dict(color='#ffd700', size=10, symbol='circle'),
                    showlegend=False,
                ))
            
            # Brace height indicator (only for braced state)
            if draw_inch == 0:
                fig.add_shape(
                    type="line",
                    x0=TARGET_BRACE_HEIGHT_CM, y0=-limb_length_cm*0.3,
                    x1=TARGET_BRACE_HEIGHT_CM, y1=limb_length_cm*0.3,
                    line=dict(color='#00d4ff', width=1, dash='dash'),
                )
                fig.add_annotation(
                    x=TARGET_BRACE_HEIGHT_CM,
                    y=limb_length_cm*0.35,
                    text=f"Brace Height<br>{TARGET_BRACE_HEIGHT_CM} cm",
                    showarrow=False,
                    font=dict(color='#00d4ff', size=9),
                    bgcolor='rgba(26, 31, 58, 0.8)',
                    bordercolor='#00d4ff',
                    borderwidth=1,
                )
    
    # Add handle marker
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        name='핸들',
        marker=dict(color='#ff6b6b', size=12, symbol='square'),
    ))
    
    fig.update_layout(
        title="가상 틸러링: 활 변형 시뮬레이션",
        xaxis_title="수평 위치 (cm)",
        yaxis_title="수직 위치 (cm)",
        template="plotly_dark",
        paper_bgcolor='#0d111f',
        plot_bgcolor='#1a1f3a',
        font=dict(family="Inter, sans-serif", size=12, color="#E0E0E0"),
        height=700,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(26, 31, 58, 0.8)',
            bordercolor='#2a3f5f',
            borderwidth=1
        ),
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='#ffd700',
        zerolinewidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='#ffd700',
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
        line=dict(color='#00d4ff', width=2, dash='dot'),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)',
    ))
    
    # Effective range trajectory (slightly lower angle for flatter trajectory)
    x_eff, y_eff = generate_trajectory_points(arrow_speed_ms, 35.0, 1.5, 100)
    
    # Truncate to effective range
    mask = x_eff <= effective_range_m
    x_eff_truncated = x_eff[mask]
    y_eff_truncated = y_eff[mask]
    
    fig.add_trace(go.Scatter(
        x=x_eff_truncated,
        y=y_eff_truncated,
        mode='lines',
        name='유효 사거리 (35°)',
        line=dict(color='#ffd700', width=3),
    ))
    
    # Launch point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[1.5],
        mode='markers',
        name='발사 지점',
        marker=dict(color='#ff6b6b', size=12, symbol='triangle-right'),
    ))
    
    # Impact point (max range)
    fig.add_trace(go.Scatter(
        x=[max_range_m],
        y=[0],
        mode='markers',
        name='최대 사거리 착탄',
        marker=dict(color='#00d4ff', size=10, symbol='x'),
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
            marker=dict(color='#ffd700', size=12, symbol='diamond'),
        ))
        
        # Vertical line to target
        fig.add_trace(go.Scatter(
            x=[effective_range_m, effective_range_m],
            y=[0, y_at_effective],
            mode='lines',
            name='목표물 구역',
            line=dict(color='#ffd700', width=2, dash='dash'),
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
            line=dict(color='#ffd700', width=2),
            fillcolor='rgba(255, 215, 0, 0.2)',
        )
        
        # Add annotation
        fig.add_annotation(
            x=target_x,
            y=target_height + 0.5,
            text="목표물",
            showarrow=False,
            font=dict(color='#ffd700', size=10, family='monospace'),
            bgcolor='rgba(26, 31, 58, 0.8)',
            bordercolor='#ffd700',
            borderwidth=1,
        )
    
    fig.update_layout(
        title="탄도학: 포물선 궤적 분석",
        xaxis_title="수평 거리 (m)",
        yaxis_title="높이 (m)",
        template="plotly_dark",
        paper_bgcolor='#0d111f',
        plot_bgcolor='#1a1f3a',
        font=dict(family="Inter, sans-serif", size=12, color="#E0E0E0"),
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(26, 31, 58, 0.8)',
            bordercolor='#2a3f5f',
            borderwidth=1
        ),
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
        gridwidth=1,
        range=[0, max_range_m * 1.1],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
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
        line=dict(color='#00d4ff', width=2),
        marker=dict(size=8, symbol='circle'),
    ))
    
    # Width - Lower
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in lower],
        y=[p.width_mm for p in lower],
        mode='lines+markers',
        name='하부 림 - 너비',
        line=dict(color='#00d4ff', width=2, dash='dot'),
        marker=dict(size=8, symbol='circle-open'),
    ))
    
    # Thickness - Upper
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in upper],
        y=[p.thickness_mm for p in upper],
        mode='lines+markers',
        name='상부 림 - 두께',
        line=dict(color='#ffd700', width=2),
        marker=dict(size=8, symbol='diamond'),
    ))
    
    # Thickness - Lower
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in lower],
        y=[p.thickness_mm for p in lower],
        mode='lines+markers',
        name='하부 림 - 두께',
        line=dict(color='#ffd700', width=2, dash='dot'),
        marker=dict(size=8, symbol='diamond-open'),
    ))
    
    fig.update_layout(
        title="기하학적 프로파일: 너비 및 두께 분포",
        xaxis_title="핸들로부터 위치 (cm)",
        yaxis_title="치수 (mm)",
        template="plotly_dark",
        paper_bgcolor='#0d111f',
        plot_bgcolor='#1a1f3a',
        font=dict(family="Inter, sans-serif", size=12, color="#E0E0E0"),
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(26, 31, 58, 0.8)',
            bordercolor='#2a3f5f',
            borderwidth=1
        ),
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
        gridwidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#2a3f5f',
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
    
    # Header with geometric decoration
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    st.title("디지털 활 물리 & 탄도 연구소")
    st.caption("박물관급 인터랙티브 연구 플랫폼 · 실험 고고학 · v4.0")
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
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
    efficiency = calculate_shooting_efficiency(measurements, segment_masses_g, total_mass_g)
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
    effective_range_m = calculate_effective_range(arrow_speed_ms, 25.0, 40.0)
    
    # Render results
    render_performance_metrics(metrics)
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Main charts
    st.markdown("## 장력-드로우 분석")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_fdc = create_fdc_chart(fdc)
        st.plotly_chart(fig_fdc, use_container_width=True)
    
    with col2:
        fig_angle = create_string_angle_chart(fdc)
        st.plotly_chart(fig_angle, use_container_width=True)
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Ballistics Section
    st.markdown("## 탄도학: 궤적 및 사거리 분석")
    
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
            help="≥40 J 운동 에너지 유지 거리"
        )
    
    with col3:
        range_ratio = (effective_range_m / max_range_m * 100) if max_range_m > 0 else 0
        st.metric(
            label="유효/최대 비율",
            value=f"{range_ratio:.0f}%",
            help="전투 효용성 지표"
        )
    
    fig_ballistics = create_ballistics_trajectory_chart(
        arrow_speed_ms, max_range_m, effective_range_m
    )
    st.plotly_chart(fig_ballistics, use_container_width=True)
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Virtual Tiller simulation (Realistic Kinematic)
    st.markdown("## 가상 틸러링: 활 변형 시뮬레이션")
    
    # Calculate EI range for debugging info
    ei_values = [p.ei_nm2 for p in measurements if p.limb in ["Upper", "Handle"]]
    ei_min = min(ei_values) if ei_values else 0
    ei_max = max(ei_values) if ei_values else 0
    ei_ratio = ei_max / ei_min if ei_min > 0 else 1.0
    
    st.caption(f"사실적 2D 물리 시뮬레이션 • 사이드 프로파일: {side_profile} • Brace Height: {TARGET_BRACE_HEIGHT_CM} cm")
    st.caption(f"⚙️ 강성 분포 (EI): {ei_min:.2f} ~ {ei_max:.2f} N·m² (비율: {ei_ratio:.2f}x) • 강성이 높은 부위는 덜 휘어짐")
    
    fig_tiller = create_virtual_tiller_realistic(measurements, limb_length_cm, side_profile, [-1, 0, 20, 28])
    st.plotly_chart(fig_tiller, use_container_width=True)
    
    st.markdown('<div class="geometric-line"></div>', unsafe_allow_html=True)
    
    # Geometry profile
    st.markdown("## 기하학적 프로파일")
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
        <div style='text-align: center; color: #808080; font-size: 0.85rem; padding: 2rem 0;'>
            <p>디지털 실험 고고학 연구소</p>
            <p>박물관급 인터랙티브 연구 플랫폼 © 2026</p>
            <p style='font-style: italic; margin-top: 0.5rem;'>
                "고대 장인정신과 현대 계산 물리학의 융합"
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
