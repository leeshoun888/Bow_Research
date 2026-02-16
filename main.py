"""
고대 활 장력 시뮬레이터 — 논문 수준 정밀 버전
Ancient Bow Tension Simulator · 7-Point Measurement · Material Physics
- 실측 기반 목재 물성 데이터 · 단면 형상 보정 · EI 보간 적분
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple

# =============================================================================
# 실측 기반 목재 물성 데이터 (Material Physics Database)
# =============================================================================

WOOD_PROPERTIES: dict[str, dict[str, float]] = {
    "물푸레나무 (Ash)": {"elastic_modulus_gpa": 11.5, "density_g_cm3": 0.68},
    "주목 (Yew)": {"elastic_modulus_gpa": 10.0, "density_g_cm3": 0.64},
    "박달나무 (Birch, Bakdal)": {"elastic_modulus_gpa": 14.5, "density_g_cm3": 0.92},
    "노간주나무 (Juniper)": {"elastic_modulus_gpa": 7.5, "density_g_cm3": 0.55},
    "산뽕나무 (Mountain Mulberry)": {"elastic_modulus_gpa": 11.0, "density_g_cm3": 0.65},
    "아까시나무 (Black Locust)": {"elastic_modulus_gpa": 14.5, "density_g_cm3": 0.75},
}

# 단면 형상 옵션 (Point 0 제외)
SHAPE_OPTIONS = ["타원형 (Elliptical)", "직사각형 (Rectangular)", "사다리꼴 (Trapezoid)", "역사다리꼴 (Inv. Trapezoid)"]

# =============================================================================
# 상수 (물리·단위)
# =============================================================================

DRAW_LENGTH_INCH = 28.0
INCH_TO_M = 0.0254
MM_TO_M = 0.001
GPA_TO_PA = 1e9
MM4_TO_M4 = 1e-12
N_TO_LBS = 0.224808943
INTEGRATION_STEPS = 200  # EI 보간 적분용

# =============================================================================
# 7-Point 위치 정의 (총길이 L cm 기준, 한쪽 팔 기준: 0 = 핸들, L/2 = 팁)
# =============================================================================


def get_point_positions_cm(total_length_cm: float) -> List[float]:
    """
    한쪽 림을 따라의 위치 [cm]: 0 = 핸들 정중앙, 끝 = 팁에서 15cm 안쪽.
    순서: P0(핸들), P1(20cm), P2(20cm 하부), P3(Upper Mid), P4(Lower Mid), P5(Upper Tip), P6(Lower Tip).
    한쪽 limb 길이 = L/2; P1/P5는 상부 림, P2/P6는 하부 림. 대칭이므로 한쪽만 거리로 표현.
    반환: [0, 20, 20, mid_upper, mid_lower, tip_upper, tip_lower] → 실제로는 상/하 동일 거리 사용.
    """
    half = total_length_cm / 2.0
    tip_inner = half - 15.0  # 팁에서 15cm 안쪽
    mid_upper = (20.0 + tip_inner) / 2.0
    mid_lower = mid_upper
    return [0.0, 20.0, 20.0, mid_upper, mid_lower, tip_inner, tip_inner]


# =============================================================================
# 단면 2차 모멘트 (Shape Factor)
# =============================================================================


def calculate_inertia(width_mm: float, thickness_mm: float, shape: str) -> float:
    """
    단면 2차 모멘트 I [mm⁴].
    - 직사각형: I = (1/12) * w * t³
    - 타원형: I ≈ (π/64) * w * t³
    - 사다리꼴: 넓은 면 = width, 좁은 면 = 0.8*width, 높이 = thickness
    - 역사다리꼴: Belly 좁음 → 좁은 면 = width, 넓은 면 = width/0.8
    """
    if width_mm <= 0 or thickness_mm <= 0:
        return 0.0
    w, t = width_mm, thickness_mm

    if "직사각형" in shape or shape == "rectangular":
        return (w * (t ** 3)) / 12.0

    if "타원" in shape or "Elliptical" in shape:
        return (math.pi * w * (t ** 3)) / 64.0

    if "사다리꼴" in shape and "역" not in shape:
        # Trapezoid: 넓은 면 = w, 좁은 면 = 0.8*w, 높이 = t
        b_wide = w
        b_narrow = 0.8 * w
        # I = t³ * (b_wide² + 4*b_wide*b_narrow + b_narrow²) / (36 * (b_wide + b_narrow))
        return (t ** 3) * (b_wide ** 2 + 4 * b_wide * b_narrow + b_narrow ** 2) / (36.0 * (b_wide + b_narrow))

    if "역사다리꼴" in shape or "Inv" in shape:
        # Inv. Trapezoid: Belly 좁고 Back 넓음 → 입력 width를 좁은 면으로 해석, 넓은 면 = w/0.8
        b_narrow = w
        b_wide = w / 0.8
        return (t ** 3) * (b_wide ** 2 + 4 * b_wide * b_narrow + b_narrow ** 2) / (36.0 * (b_wide + b_narrow))

    return (w * (t ** 3)) / 12.0


# =============================================================================
# EI 보간 및 장력 계산 (Physics Engine)
# =============================================================================


def compute_ei_at_points(
    widths_mm: List[float],
    thicknesses_mm: List[float],
    shapes: List[str],
    youngs_modulus_gpa: float,
) -> List[float]:
    """7개 지점에서 EI [N·m²] 계산."""
    e_pa = youngs_modulus_gpa * GPA_TO_PA
    ei_list = []
    for w, t, shape in zip(widths_mm, thicknesses_mm, shapes):
        i_mm4 = calculate_inertia(w, t, shape)
        i_m4 = i_mm4 * MM4_TO_M4
        ei_list.append(e_pa * i_m4)
    return ei_list


def draw_weight_from_ei_interpolation(
    positions_cm: List[float],
    ei_nm2_list: List[float],
    limb_length_cm: float,
    draw_length_inch: float,
) -> float:
    """
    7개 지점 EI를 보간하여 림 전체의 굽힘 강성을 적분하고,
    28인치 드로우 시 장력(lbs) 산출.
    캔틸레버: δ = P ∫₀^L (L-s)²/(EI(s)) ds  =>  P = δ / ∫₀^L (L-s)²/(EI(s)) ds
    """
    if limb_length_cm <= 0 or not ei_nm2_list or all(e <= 0 for e in ei_nm2_list):
        return 0.0
    limb_m = limb_length_cm * 0.01
    delta_m = draw_length_inch * INCH_TO_M

    # 한쪽 림만: s는 0(핸들) ~ limb_length_cm(팁). 동일 위치는 EI 평균으로 유일화 (np.interp는 xp 증가 필요).
    s_cm = np.array(positions_cm, dtype=float)
    ei = np.maximum(np.array(ei_nm2_list, dtype=float), 1e-10)
    order = np.argsort(s_cm)
    s_sorted = s_cm[order]
    ei_sorted = ei[order]
    s_uniq = np.unique(s_sorted)
    ei_uniq = np.array([ei_sorted[s_sorted == s].mean() for s in s_uniq])
    if len(s_uniq) == 1:
        ei_uniq = np.array([ei_uniq[0], ei_uniq[0]])
        s_uniq = np.array([0.0, limb_length_cm])

    # EI 1차 보간 후 ∫ (L-s)²/(EI) ds [m/N] 계산
    s_samples_cm = np.linspace(0, limb_length_cm, INTEGRATION_STEPS)
    ei_samples = np.maximum(np.interp(s_samples_cm, s_uniq, ei_uniq), 1e-10)

    s_m = s_samples_cm * 0.01
    l_m = limb_length_cm * 0.01
    integrand_si = ((l_m - s_m) ** 2) / ei_samples
    integral_si = float(np.sum((integrand_si[:-1] + integrand_si[1:]) * np.diff(s_m) / 2.0))
    p_newton = delta_m / integral_si if integral_si > 0 else 0.0
    return p_newton * N_TO_LBS


def compute_draw_weight(
    youngs_modulus_gpa: float,
    widths_mm: List[float],
    thicknesses_mm: List[float],
    shapes: List[str],
    total_length_cm: float,
    draw_length_inch: float = DRAW_LENGTH_INCH,
) -> Tuple[float, List[float], List[float]]:
    """예상 장력(lbs), 7개 I [mm⁴], 7개 EI [N·m²] 반환."""
    positions = get_point_positions_cm(total_length_cm)
    ei_list = compute_ei_at_points(widths_mm, thicknesses_mm, shapes, youngs_modulus_gpa)
    i_list = [
        calculate_inertia(w, t, sh)
        for w, t, sh in zip(widths_mm, thicknesses_mm, shapes)
    ]
    limb_length_cm = total_length_cm / 2.0
    draw_lbs = draw_weight_from_ei_interpolation(
        positions, ei_list, limb_length_cm, draw_length_inch
    )
    return draw_lbs, i_list, ei_list


# =============================================================================
# Streamlit UI
# =============================================================================


def apply_industrial_dark_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 100%; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #161b22 0%, #21262d 100%);
            border-right: 1px solid #30363d;
        }
        [data-testid="stSidebar"] .stMarkdown { color: #c9d1d9; }
        .tension-result {
            font-size: 4.5rem; font-weight: 700; color: #58a6ff; text-align: center;
            padding: 2rem 1.5rem; margin: 2rem 0;
            background: rgba(22, 27, 34, 0.8); border: 2px solid #30363d;
            border-radius: 12px; box-shadow: 0 0 20px rgba(88, 166, 255, 0.15);
            letter-spacing: 0.02em; font-family: 'JetBrains Mono', 'Consolas', monospace;
        }
        .tension-label { font-size: 1.1rem; color: #8b949e; text-align: center; margin-bottom: 0.5rem; letter-spacing: 0.05em; }
        .stSlider label, .stSelectbox label, .stNumberInput label { color: #c9d1d9 !important; }
        h1, h2, h3 { color: #f0f6fc !important; font-weight: 600 !important; }
        h1 { border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


POINT_LABELS = [
    "Point 0: 핸들 정중앙 (Handle Center)",
    "Point 1: 핸들에서 20cm (Upper Limb)",
    "Point 2: 핸들에서 20cm (Lower Limb)",
    "Point 3: Upper Mid",
    "Point 4: Lower Mid",
    "Point 5: 팁에서 15cm 안쪽 (Upper Tip)",
    "Point 6: 팁에서 15cm 안쪽 (Lower Tip)",
]


def render_sidebar_inputs() -> Tuple[str, float, List[float], List[float], List[str]]:
    """사이드바: 수종, 총 길이, 7개 지점 Width/Thickness/Shape. 반환: (수종, 길이_cm, widths, thicknesses, shapes)."""
    with st.sidebar:
        st.markdown("### 🏹 활 제원 입력 (7-Point Measurement)")
        st.markdown("---")

        species = st.selectbox(
            "**목재 선택**",
            options=list(WOOD_PROPERTIES.keys()),
            index=0,
            help="실측 기반 탄성계수(E)와 밀도가 적용됩니다.",
        )

        total_length_cm = st.slider(
            "**활의 총 길이 (cm)**",
            min_value=100,
            max_value=200,
            value=160,
            step=5,
        )

        st.markdown("---")
        st.markdown("#### 7개 지점 실측값 (Width mm / Thickness mm / 단면 형상)")

        widths: List[float] = []
        thicknesses: List[float] = []
        shapes: List[str] = []
        default_w = [35.0, 28.0, 28.0, 24.0, 24.0, 20.0, 20.0]
        default_t = [12.0, 10.0, 10.0, 8.0, 8.0, 6.0, 6.0]

        cols = st.columns(7)
        for i in range(7):
            with cols[i]:
                st.caption(POINT_LABELS[i])
                w = st.number_input(
                    "Width (mm)",
                    min_value=5.0,
                    max_value=80.0,
                    value=default_w[i],
                    step=1.0,
                    key=f"w_{i}",
                )
                t = st.number_input(
                    "Thickness (mm)",
                    min_value=2.0,
                    max_value=25.0,
                    value=default_t[i],
                    step=0.5,
                    key=f"t_{i}",
                )
                if i == 0:
                    shape = "직사각형 (Rectangular)"
                    st.caption("직사각형 고정")
                else:
                    shape = st.selectbox(
                        "Shape",
                        options=SHAPE_OPTIONS,
                        index=0,
                        key=f"shape_{i}",
                    )
                widths.append(w)
                thicknesses.append(t)
                shapes.append(shape)

        return species, total_length_cm, widths, thicknesses, shapes


def render_main_panel(
    species: str,
    total_length_cm: float,
    draw_lbs: float,
    widths_mm: List[float],
    thicknesses_mm: List[float],
    positions_cm: List[float],
) -> None:
    """결과 패널: 예상 장력, 형상 Line Chart (X: 활 내 위치, Y: 치수)."""
    st.markdown('<p class="tension-label">예상 장력 (28" 풀 드로우 기준)</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="tension-result">예상 장력: {draw_lbs:.1f} <span style="font-size: 2rem; color: #8b949e;">lbs</span></p>',
        unsafe_allow_html=True,
    )

    st.subheader("📐 형상 그래프 (7개 지점: 너비·두께 변화)")
    # X축: 활 내 위치 (cm), Y축: 치수 (mm)
    df_chart = pd.DataFrame({
        "위치 (cm)": positions_cm,
        "너비 (mm)": widths_mm,
        "두께 (mm)": thicknesses_mm,
    }).set_index("위치 (cm)")
    st.line_chart(df_chart, height=320)

    with st.expander("📐 계산식 및 입력 요약"):
        E = WOOD_PROPERTIES[species]["elastic_modulus_gpa"]
        st.latex(r"I_{\mathrm{rect}} = \frac{w \cdot t^3}{12}, \quad I_{\mathrm{ell}} \approx \frac{\pi w t^3}{64}")
        st.latex(r"EI = E \cdot I, \quad \delta = P \int_0^L \frac{(L-s)^2}{EI(s)}\,ds \Rightarrow P = \frac{\delta}{\int \ldots}")
        st.markdown(
            f"- **수종**: {species} (E = {E} GPa)  \n"
            f"- **총 길이**: {total_length_cm} cm  \n"
            f"- **예상 장력**: **{draw_lbs:.1f} lbs**"
        )


def main() -> None:
    st.set_page_config(
        page_title="고대 활 장력 시뮬레이터",
        page_icon="🏹",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_industrial_dark_theme()

    st.title("고대 활 장력 시뮬레이터")
    st.caption("디지털 실험고고학 · 7-Point 실측 기반 정밀 시뮬레이터")

    species, total_length_cm, widths_mm, thicknesses_mm, shapes = render_sidebar_inputs()
    E = WOOD_PROPERTIES[species]["elastic_modulus_gpa"]
    positions_cm = get_point_positions_cm(total_length_cm)

    draw_lbs, i_list, ei_list = compute_draw_weight(E, widths_mm, thicknesses_mm, shapes, total_length_cm)
    render_main_panel(species, total_length_cm, draw_lbs, widths_mm, thicknesses_mm, positions_cm)


if __name__ == "__main__":
    main()
