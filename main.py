"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  고대 활 장력 시뮬레이터 — 논문 수준 정밀 버전 v2.0                           ║
║  Ancient Bow Draw Weight Simulator · Digital Experimental Archaeology      ║
╚═══════════════════════════════════════════════════════════════════════════╝

🎯 핵심 기능:
- 실측 기반 목재 물성 데이터베이스 (Material Physics)
- 7-Point 정밀 측정 시스템
- 단면 형상별 2차 모멘트 계산 (타원/직사각형/사다리꼴/역사다리꼴)
- EI 보간 적분 기반 굽힘 강성 해석
- 28인치 드로우 장력 예측
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Dict
from dataclasses import dataclass

# =============================================================================
# 실측 기반 목재 물성 데이터 (Material Physics Database)
# =============================================================================

WOOD_PROPERTIES: Dict[str, Dict[str, float]] = {
    "물푸레나무 (Ash)": {
        "elastic_modulus_gpa": 11.5,
        "density_g_cm3": 0.68,
    },
    "주목 (Yew)": {
        "elastic_modulus_gpa": 10.0,
        "density_g_cm3": 0.64,
    },
    "박달나무 (Birch, Bakdal)": {
        "elastic_modulus_gpa": 14.5,
        "density_g_cm3": 0.92,
    },
    "노간주나무 (Juniper)": {
        "elastic_modulus_gpa": 7.5,
        "density_g_cm3": 0.55,
    },
    "산뽕나무 (Mountain Mulberry)": {
        "elastic_modulus_gpa": 11.0,
        "density_g_cm3": 0.65,
    },
    "아까시나무 (Black Locust)": {
        "elastic_modulus_gpa": 14.5,
        "density_g_cm3": 0.75,
    },
}

# =============================================================================
# 단면 형상 옵션 및 상수
# =============================================================================

SHAPE_OPTIONS = [
    "타원형 (Elliptical)",
    "직사각형 (Rectangular)",
    "사다리꼴 (Trapezoid)",
    "역사다리꼴 (Inv. Trapezoid)",
]

# 물리 상수
DRAW_LENGTH_INCH = 28.0
INCH_TO_M = 0.0254
MM_TO_M = 0.001
GPA_TO_PA = 1e9
MM4_TO_M4 = 1e-12
N_TO_LBS = 0.224808943
INTEGRATION_STEPS = 200

# 7-Point 측정 지점 정의
POINT_DEFINITIONS = [
    {"id": 0, "name": "Point 0: 핸들 정중앙", "desc": "Handle Center", "limb": "핸들"},
    {"id": 1, "name": "Point 1: 핸들에서 20cm", "desc": "Upper Limb", "limb": "상부"},
    {"id": 2, "name": "Point 2: 핸들에서 20cm", "desc": "Lower Limb", "limb": "하부"},
    {"id": 3, "name": "Point 3: 상부 중간", "desc": "Upper Mid", "limb": "상부"},
    {"id": 4, "name": "Point 4: 하부 중간", "desc": "Lower Mid", "limb": "하부"},
    {"id": 5, "name": "Point 5: 팁에서 15cm 안쪽", "desc": "Upper Tip area", "limb": "상부"},
    {"id": 6, "name": "Point 6: 팁에서 15cm 안쪽", "desc": "Lower Tip area", "limb": "하부"},
]


@dataclass
class MeasurementPoint:
    """7개 측정 지점의 데이터 구조"""
    point_id: int
    name: str
    limb: str
    position_cm: float
    width_mm: float
    thickness_mm: float
    shape: str
    inertia_mm4: float = 0.0
    ei_nm2: float = 0.0


# =============================================================================
# 핵심 물리 엔진 (Physics Engine)
# =============================================================================


def calculate_inertia(width_mm: float, thickness_mm: float, shape: str) -> float:
    """
    단면 2차 모멘트 I [mm⁴] 계산
    
    Parameters:
    -----------
    width_mm : float
        단면 너비 [mm]
    thickness_mm : float
        단면 두께 [mm]
    shape : str
        단면 형상 (타원형/직사각형/사다리꼴/역사다리꼴)
    
    Returns:
    --------
    float : 단면 2차 모멘트 [mm⁴]
    
    공식:
    - 직사각형: I = (w × t³) / 12
    - 타원형: I ≈ (π × w × t³) / 64
    - 사다리꼴: 넓은 면 = w, 좁은 면 = 0.8w
    - 역사다리꼴: 좁은 면 = w, 넓은 면 = w/0.8
    """
    if width_mm <= 0 or thickness_mm <= 0:
        return 0.0
    
    w, t = width_mm, thickness_mm
    
    # 직사각형 단면
    if "직사각형" in shape:
        return (w * t**3) / 12.0
    
    # 타원형 단면
    elif "타원" in shape:
        return (math.pi * w * t**3) / 64.0
    
    # 사다리꼴 단면 (Belly 넓음, Back 좁음)
    elif "사다리꼴" in shape and "역" not in shape:
        b_wide = w
        b_narrow = 0.8 * w
        numerator = t**3 * (b_wide**2 + 4*b_wide*b_narrow + b_narrow**2)
        denominator = 36.0 * (b_wide + b_narrow)
        return numerator / denominator
    
    # 역사다리꼴 단면 (Belly 좁음, Back 넓음)
    elif "역사다리꼴" in shape or "Inv" in shape:
        b_narrow = w
        b_wide = w / 0.8
        numerator = t**3 * (b_wide**2 + 4*b_wide*b_narrow + b_narrow**2)
        denominator = 36.0 * (b_wide + b_narrow)
        return numerator / denominator
    
    # 기본값 (직사각형)
    return (w * t**3) / 12.0


def get_point_positions_cm(total_length_cm: float) -> List[float]:
    """
    7개 측정 지점의 위치 계산 [cm]
    
    Parameters:
    -----------
    total_length_cm : float
        활의 총 길이 [cm]
    
    Returns:
    --------
    List[float] : 각 지점의 핸들로부터의 거리 [cm]
        [P0(핸들), P1(20cm), P2(20cm), P3(중간), P4(중간), P5(팁-15cm), P6(팁-15cm)]
    """
    half_length = total_length_cm / 2.0
    tip_inner = half_length - 15.0  # 팁에서 15cm 안쪽
    mid_position = (20.0 + tip_inner) / 2.0  # P1과 P5의 중간
    
    return [
        0.0,           # P0: 핸들 정중앙
        20.0,          # P1: 핸들에서 20cm (상부)
        20.0,          # P2: 핸들에서 20cm (하부)
        mid_position,  # P3: 상부 중간
        mid_position,  # P4: 하부 중간
        tip_inner,     # P5: 팁에서 15cm 안쪽 (상부)
        tip_inner,     # P6: 팁에서 15cm 안쪽 (하부)
    ]


def compute_ei_profile(
    measurements: List[MeasurementPoint],
    youngs_modulus_gpa: float
) -> List[MeasurementPoint]:
    """
    각 측정 지점에서 EI (굽힘 강성) 계산
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
        7개 측정 지점 데이터
    youngs_modulus_gpa : float
        영계수 [GPa]
    
    Returns:
    --------
    List[MeasurementPoint] : EI 값이 업데이트된 측정 지점 리스트
    """
    e_pa = youngs_modulus_gpa * GPA_TO_PA
    
    for point in measurements:
        # 단면 2차 모멘트 계산
        i_mm4 = calculate_inertia(point.width_mm, point.thickness_mm, point.shape)
        point.inertia_mm4 = i_mm4
        
        # EI 계산 [N·m²]
        i_m4 = i_mm4 * MM4_TO_M4
        point.ei_nm2 = e_pa * i_m4
    
    return measurements


def calculate_draw_weight(
    measurements: List[MeasurementPoint],
    limb_length_cm: float,
    draw_length_inch: float = DRAW_LENGTH_INCH
) -> float:
    """
    캔틸레버 빔 모델을 사용한 드로우 장력 계산
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
        EI 프로파일이 계산된 측정 지점 리스트
    limb_length_cm : float
        한쪽 림의 길이 [cm]
    draw_length_inch : float
        드로우 길이 [inch]
    
    Returns:
    --------
    float : 예상 장력 [lbs]
    
    이론:
    -----
    캔틸레버 빔의 처짐: δ = P ∫₀^L (L-s)²/EI(s) ds
    따라서: P = δ / ∫₀^L (L-s)²/EI(s) ds
    """
    if limb_length_cm <= 0:
        return 0.0
    
    # 측정 지점들을 위치순으로 정렬
    positions_cm = [p.position_cm for p in measurements]
    ei_values = [max(p.ei_nm2, 1e-10) for p in measurements]  # 0 방지
    
    if all(e <= 1e-10 for e in ei_values):
        return 0.0
    
    # 위치 정렬 및 중복 제거
    s_cm = np.array(positions_cm, dtype=float)
    ei = np.array(ei_values, dtype=float)
    
    order = np.argsort(s_cm)
    s_sorted = s_cm[order]
    ei_sorted = ei[order]
    
    # 동일 위치의 EI는 평균 사용
    s_unique = np.unique(s_sorted)
    ei_unique = np.array([ei_sorted[s_sorted == s].mean() for s in s_unique])
    
    # 단일 지점인 경우 처리
    if len(s_unique) == 1:
        s_unique = np.array([0.0, limb_length_cm])
        ei_unique = np.array([ei_unique[0], ei_unique[0]])
    
    # EI 선형 보간 후 적분 계산
    s_samples_cm = np.linspace(0, limb_length_cm, INTEGRATION_STEPS)
    ei_samples = np.maximum(np.interp(s_samples_cm, s_unique, ei_unique), 1e-10)
    
    # 단위 변환 및 적분 계산
    s_m = s_samples_cm * 0.01
    L_m = limb_length_cm * 0.01
    delta_m = draw_length_inch * INCH_TO_M
    
    # ∫ (L-s)²/EI(s) ds 계산 (사다리꼴 적분)
    integrand = ((L_m - s_m)**2) / ei_samples
    integral = float(np.sum((integrand[:-1] + integrand[1:]) * np.diff(s_m) / 2.0))
    
    if integral <= 0:
        return 0.0
    
    # 장력 계산 및 단위 변환 [N] → [lbs]
    force_newton = delta_m / integral
    return force_newton * N_TO_LBS


def estimate_bow_mass(
    measurements: List[MeasurementPoint],
    total_length_cm: float,
    density_g_cm3: float
) -> float:
    """
    활의 대략적인 질량 추정 [g]
    
    Parameters:
    -----------
    measurements : List[MeasurementPoint]
        7개 측정 지점
    total_length_cm : float
        활의 총 길이 [cm]
    density_g_cm3 : float
        목재 밀도 [g/cm³]
    
    Returns:
    --------
    float : 추정 질량 [g]
    """
    # 각 구간별 평균 단면적 × 길이 × 밀도
    limb_length_cm = total_length_cm / 2.0
    total_volume_cm3 = 0.0
    
    positions = [p.position_cm for p in measurements]
    
    for i in range(len(measurements) - 1):
        # 구간 길이
        segment_length = abs(positions[i+1] - positions[i])
        
        # 평균 단면적 [cm²]
        area1 = (measurements[i].width_mm * measurements[i].thickness_mm) / 100.0
        area2 = (measurements[i+1].width_mm * measurements[i+1].thickness_mm) / 100.0
        avg_area = (area1 + area2) / 2.0
        
        # 부피 누적
        total_volume_cm3 += avg_area * segment_length
    
    # 상/하 림 모두 고려 (×2)
    total_volume_cm3 *= 2.0
    
    return total_volume_cm3 * density_g_cm3


def calculate_stored_energy(draw_weight_lbs: float, draw_length_inch: float = DRAW_LENGTH_INCH) -> float:
    """
    활에 저장되는 에너지 추정 [Joules]
    
    E = (1/2) × F × d (선형 근사)
    """
    force_n = draw_weight_lbs / N_TO_LBS
    distance_m = draw_length_inch * INCH_TO_M
    return 0.5 * force_n * distance_m


# =============================================================================
# Streamlit UI - 고급 테마
# =============================================================================


def apply_premium_theme() -> None:
    """프리미엄 다크 테마 적용"""
    st.markdown("""
        <style>
        /* 전체 배경 */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 50%, #0a0e27 100%);
        }
        
        /* 메인 컨테이너 */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }
        
        /* 사이드바 */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1d3a 0%, #252850 100%);
            border-right: 2px solid #3d4166;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #e1e4f0;
        }
        
        /* 헤더 스타일 */
        h1 {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 2.8rem !important;
            letter-spacing: -0.02em;
            margin-bottom: 0.3rem !important;
            background: linear-gradient(90deg, #58a6ff 0%, #8e7dff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        h2 {
            color: #c9d1d9 !important;
            font-weight: 600 !important;
            font-size: 1.6rem !important;
            margin-top: 2rem !important;
            border-bottom: 2px solid #3d4166;
            padding-bottom: 0.5rem;
        }
        
        h3 {
            color: #b1bac4 !important;
            font-weight: 600 !important;
            font-size: 1.2rem !important;
        }
        
        /* 장력 결과 카드 */
        .tension-card {
            background: linear-gradient(135deg, #1e2139 0%, #2d3250 100%);
            border: 2px solid #58a6ff;
            border-radius: 16px;
            padding: 2.5rem 2rem;
            margin: 2rem 0;
            box-shadow: 0 8px 32px rgba(88, 166, 255, 0.25);
            text-align: center;
        }
        
        .tension-value {
            font-size: 5rem;
            font-weight: 800;
            color: #58a6ff;
            font-family: 'SF Mono', 'Monaco', 'Cascadia Code', monospace;
            letter-spacing: -0.03em;
            line-height: 1.1;
            text-shadow: 0 0 40px rgba(88, 166, 255, 0.5);
        }
        
        .tension-label {
            font-size: 1.1rem;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .tension-unit {
            font-size: 2.2rem;
            color: #8b949e;
            margin-left: 0.5rem;
        }
        
        /* 측정 지점 카드 */
        .measurement-card {
            background: rgba(45, 50, 80, 0.5);
            border: 1px solid #3d4166;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .measurement-card:hover {
            background: rgba(45, 50, 80, 0.8);
            border-color: #58a6ff;
            box-shadow: 0 4px 16px rgba(88, 166, 255, 0.2);
        }
        
        .point-header {
            color: #58a6ff;
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .limb-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .limb-upper {
            background: rgba(88, 166, 255, 0.2);
            color: #58a6ff;
        }
        
        .limb-lower {
            background: rgba(142, 125, 255, 0.2);
            color: #8e7dff;
        }
        
        .limb-handle {
            background: rgba(163, 182, 138, 0.2);
            color: #a3b68a;
        }
        
        /* 입력 필드 */
        .stNumberInput label, .stSelectbox label {
            color: #c9d1d9 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }
        
        .stNumberInput input, .stSelectbox select {
            background-color: rgba(22, 27, 34, 0.6) !important;
            border: 1px solid #3d4166 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }
        
        .stNumberInput input:focus, .stSelectbox select:focus {
            border-color: #58a6ff !important;
            box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.1) !important;
        }
        
        /* 슬라이더 */
        .stSlider label {
            color: #c9d1d9 !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        
        /* 메트릭 카드 */
        [data-testid="stMetricValue"] {
            color: #58a6ff !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #8b949e !important;
            font-size: 0.9rem !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: rgba(45, 50, 80, 0.5) !important;
            border: 1px solid #3d4166 !important;
            border-radius: 8px !important;
            color: #c9d1d9 !important;
            font-weight: 600 !important;
        }
        
        .streamlit-expanderContent {
            background-color: rgba(22, 27, 34, 0.4) !important;
            border: 1px solid #3d4166 !important;
            border-radius: 0 0 8px 8px !important;
        }
        
        /* 캡션 */
        .caption-text {
            color: #8b949e;
            font-size: 0.85rem;
            font-style: italic;
        }
        
        /* 구분선 */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #3d4166, transparent);
            margin: 2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar_inputs() -> Tuple[str, float, List[MeasurementPoint]]:
    """
    사이드바: 활 제원 입력 UI
    
    Returns:
    --------
    Tuple[str, float, List[MeasurementPoint]]
        (수종, 총 길이[cm], 측정 지점 리스트)
    """
    with st.sidebar:
        st.markdown("### 🏹 활 기본 제원")
        st.markdown("---")
        
        # 목재 선택
        species = st.selectbox(
            "**목재 수종**",
            options=list(WOOD_PROPERTIES.keys()),
            index=0,
            help="각 수종의 탄성계수(E)와 밀도(ρ)가 자동 적용됩니다.",
        )
        
        wood_props = WOOD_PROPERTIES[species]
        st.caption(f"📊 E = {wood_props['elastic_modulus_gpa']} GPa, ρ = {wood_props['density_g_cm3']} g/cm³")
        
        # 활 총 길이
        total_length_cm = st.slider(
            "**활의 총 길이 (cm)**",
            min_value=100,
            max_value=200,
            value=160,
            step=5,
            help="팁부터 팁까지의 전체 길이"
        )
        
        st.markdown("---")
        st.markdown("### 📐 7-Point 실측 데이터")
        st.caption("각 측정 지점의 너비, 두께, 단면 형상을 입력하세요.")
        
        # 7개 측정 지점 위치 계산
        positions = get_point_positions_cm(total_length_cm)
        
        # 기본값 설정
        default_widths = [35.0, 28.0, 28.0, 24.0, 24.0, 20.0, 20.0]
        default_thicknesses = [12.0, 10.0, 10.0, 8.0, 8.0, 6.0, 6.0]
        
        measurements: List[MeasurementPoint] = []
        
        # 각 측정 지점 입력
        for i, point_def in enumerate(POINT_DEFINITIONS):
            with st.expander(f"**{point_def['name']}** ({point_def['desc']})", expanded=(i == 0)):
                st.markdown(f"""
                    <div class="limb-badge limb-{point_def['limb'].lower() if point_def['limb'] != '핸들' else 'handle'}">
                        {point_def['limb']}
                    </div>
                    <p class="caption-text">핸들로부터 거리: {positions[i]:.1f} cm</p>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    width = st.number_input(
                        "너비 (mm)",
                        min_value=5.0,
                        max_value=80.0,
                        value=default_widths[i],
                        step=1.0,
                        key=f"width_{i}",
                        help="활의 좌우 폭"
                    )
                
                with col2:
                    thickness = st.number_input(
                        "두께 (mm)",
                        min_value=2.0,
                        max_value=25.0,
                        value=default_thicknesses[i],
                        step=0.5,
                        key=f"thickness_{i}",
                        help="Belly부터 Back까지의 높이"
                    )
                
                # 단면 형상 선택 (핸들은 직사각형 고정)
                if i == 0:
                    shape = "직사각형 (Rectangular)"
                    st.caption("⚠️ 핸들 부위는 직사각형으로 고정됩니다.")
                else:
                    shape = st.selectbox(
                        "단면 형상",
                        options=SHAPE_OPTIONS,
                        index=0,
                        key=f"shape_{i}",
                        help="단면의 기하학적 형태"
                    )
                
                # MeasurementPoint 객체 생성
                point = MeasurementPoint(
                    point_id=point_def['id'],
                    name=point_def['name'],
                    limb=point_def['limb'],
                    position_cm=positions[i],
                    width_mm=width,
                    thickness_mm=thickness,
                    shape=shape,
                )
                measurements.append(point)
        
        return species, total_length_cm, measurements


def render_tension_result(draw_weight_lbs: float) -> None:
    """장력 결과 카드"""
    st.markdown(f"""
        <div class="tension-card">
            <div class="tension-label">예상 장력 (28" 풀 드로우 기준)</div>
            <div class="tension-value">
                {draw_weight_lbs:.1f}<span class="tension-unit">lbs</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_metrics_panel(
    species: str,
    total_length_cm: float,
    draw_weight_lbs: float,
    measurements: List[MeasurementPoint]
) -> None:
    """주요 지표 패널"""
    wood_props = WOOD_PROPERTIES[species]
    estimated_mass = estimate_bow_mass(measurements, total_length_cm, wood_props['density_g_cm3'])
    stored_energy = calculate_stored_energy(draw_weight_lbs)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="탄성계수 (E)",
            value=f"{wood_props['elastic_modulus_gpa']} GPa",
            help="목재의 강성(stiffness)"
        )
    
    with col2:
        st.metric(
            label="추정 질량",
            value=f"{estimated_mass:.0f} g",
            help="7개 지점 측정값 기반 추정"
        )
    
    with col3:
        st.metric(
            label="저장 에너지",
            value=f"{stored_energy:.1f} J",
            help="풀 드로우 시 활에 저장되는 에너지"
        )
    
    with col4:
        st.metric(
            label="비장력",
            value=f"{draw_weight_lbs/estimated_mass*1000:.2f}",
            help="장력/질량 비율 (lbs/kg)"
        )


def create_profile_chart(measurements: List[MeasurementPoint], total_length_cm: float):
    """Plotly 기반 프로파일 차트"""
    
    # 상부/하부 림 데이터 분리
    upper_points = [p for p in measurements if p.limb == "상부" or p.limb == "핸들"]
    lower_points = [p for p in measurements if p.limb == "하부" or p.limb == "핸들"]
    
    fig = go.Figure()
    
    # 상부 림 - 너비
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in upper_points],
        y=[p.width_mm for p in upper_points],
        mode='lines+markers',
        name='상부 림 - 너비',
        line=dict(color='#58a6ff', width=3),
        marker=dict(size=10, symbol='circle')
    ))
    
    # 상부 림 - 두께
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in upper_points],
        y=[p.thickness_mm for p in upper_points],
        mode='lines+markers',
        name='상부 림 - 두께',
        line=dict(color='#8e7dff', width=3, dash='dot'),
        marker=dict(size=10, symbol='diamond')
    ))
    
    # 하부 림 - 너비
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in lower_points],
        y=[p.width_mm for p in lower_points],
        mode='lines+markers',
        name='하부 림 - 너비',
        line=dict(color='#56d364', width=3),
        marker=dict(size=10, symbol='circle')
    ))
    
    # 하부 림 - 두께
    fig.add_trace(go.Scatter(
        x=[p.position_cm for p in lower_points],
        y=[p.thickness_mm for p in lower_points],
        mode='lines+markers',
        name='하부 림 - 두께',
        line=dict(color='#f0883e', width=3, dash='dot'),
        marker=dict(size=10, symbol='diamond')
    ))
    
    fig.update_layout(
        title="활 형상 프로파일 (7-Point Measurement)",
        xaxis_title="핸들로부터의 거리 (cm)",
        yaxis_title="치수 (mm)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,29,58,0.5)',
        font=dict(color='#c9d1d9', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_ei_chart(measurements: List[MeasurementPoint]):
    """EI 분포 차트"""
    
    upper_points = [p for p in measurements if p.limb == "상부" or p.limb == "핸들"]
    lower_points = [p for p in measurements if p.limb == "하부" or p.limb == "핸들"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[p.name.split(":")[0] for p in upper_points],
        y=[p.ei_nm2 for p in upper_points],
        name='상부 림',
        marker_color='#58a6ff'
    ))
    
    fig.add_trace(go.Bar(
        x=[p.name.split(":")[0] for p in lower_points],
        y=[p.ei_nm2 for p in lower_points],
        name='하부 림',
        marker_color='#56d364'
    ))
    
    fig.update_layout(
        title="굽힘 강성 (EI) 분포",
        xaxis_title="측정 지점",
        yaxis_title="EI (N·m²)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,29,58,0.5)',
        font=dict(color='#c9d1d9', size=12),
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_detailed_table(measurements: List[MeasurementPoint]) -> None:
    """상세 측정 데이터 테이블"""
    
    df = pd.DataFrame([
        {
            "지점": p.name.split(":")[0],
            "부위": p.limb,
            "위치 (cm)": f"{p.position_cm:.1f}",
            "너비 (mm)": f"{p.width_mm:.1f}",
            "두께 (mm)": f"{p.thickness_mm:.1f}",
            "형상": p.shape.split(" ")[0],
            "I (mm⁴)": f"{p.inertia_mm4:.1f}",
            "EI (N·m²)": f"{p.ei_nm2:.2f}",
        }
        for p in measurements
    ])
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def render_theory_section() -> None:
    """이론 및 수식 섹션"""
    
    with st.expander("📚 이론적 배경 및 계산 공식"):
        st.markdown("""
        ### 캔틸레버 빔 모델 (Cantilever Beam Model)
        
        활의 림을 고정단-자유단 빔으로 모델링하여 처짐과 장력의 관계를 도출합니다.
        """)
        
        st.latex(r"\delta = P \int_0^L \frac{(L-s)^2}{EI(s)} \, ds")
        
        st.latex(r"\Rightarrow P = \frac{\delta}{\int_0^L \frac{(L-s)^2}{EI(s)} \, ds}")
        
        st.markdown("""
        ### 단면 2차 모멘트 (Second Moment of Area)
        
        **직사각형 단면:**
        """)
        
        st.latex(r"I_{\text{rect}} = \frac{w \cdot t^3}{12}")
        
        st.markdown("**타원형 단면:**")
        
        st.latex(r"I_{\text{ellipse}} \approx \frac{\pi w t^3}{64}")
        
        st.markdown("""
        **사다리꼴 단면:**
        """)
        
        st.latex(r"I_{\text{trap}} = \frac{t^3 (b_1^2 + 4b_1 b_2 + b_2^2)}{36(b_1 + b_2)}")
        
        st.markdown("""
        여기서:
        - **w**: 너비 (width)
        - **t**: 두께 (thickness)  
        - **b₁, b₂**: 사다리꼴의 평행한 두 변의 길이
        - **E**: 영계수 (Young's Modulus)
        - **I**: 단면 2차 모멘트
        - **L**: 림 길이
        - **s**: 핸들로부터의 거리
        - **δ**: 드로우 길이
        - **P**: 장력 (Draw Weight)
        """)


# =============================================================================
# 메인 애플리케이션
# =============================================================================


def main() -> None:
    """메인 애플리케이션"""
    
    # 페이지 설정
    st.set_page_config(
        page_title="고대 활 장력 시뮬레이터",
        page_icon="🏹",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # 테마 적용
    apply_premium_theme()
    
    # 헤더
    st.title("🏹 고대 활 장력 시뮬레이터")
    st.caption("디지털 실험고고학 · 7-Point 정밀 측정 시스템 · Material Physics Based Simulator")
    st.markdown("---")
    
    # 사이드바 입력
    species, total_length_cm, measurements = render_sidebar_inputs()
    
    # 물리 계산
    wood_props = WOOD_PROPERTIES[species]
    measurements = compute_ei_profile(measurements, wood_props['elastic_modulus_gpa'])
    limb_length_cm = total_length_cm / 2.0
    draw_weight_lbs = calculate_draw_weight(measurements, limb_length_cm)
    
    # 결과 렌더링
    render_tension_result(draw_weight_lbs)
    
    st.markdown("## 📊 주요 지표")
    render_metrics_panel(species, total_length_cm, draw_weight_lbs, measurements)
    
    st.markdown("---")
    
    # 그래프
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📐 형상 프로파일")
        create_profile_chart(measurements, total_length_cm)
    
    with col2:
        st.markdown("## 🔬 EI 분포")
        create_ei_chart(measurements)
    
    st.markdown("---")
    
    # 상세 데이터 테이블
    with st.expander("📋 상세 측정 데이터", expanded=False):
        render_detailed_table(measurements)
    
    # 이론 섹션
    render_theory_section()
    
    # 푸터
    st.markdown("---")
    st.caption("© 2026 디지털 실험고고학 연구소 · Powered by Streamlit & Python")


if __name__ == "__main__":
    main()
