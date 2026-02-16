"""
수종별 탄성계수(Young's Modulus) 및 밀도 데이터
- 아래 값은 참고용 임의값입니다. 실제 논문/실측 데이터로 교체하여 사용하세요.
- 키: 수종명(한글), 값: youngs_modulus_gpa (GPa), density_g_cm3 (g/cm³)
"""

from typing import Dict

# 수종별 탄성계수(E, GPa) 및 밀도(g/cm³) — 논문·실측 데이터로 수정 가능
WOOD_SPECIES: Dict[str, Dict[str, float]] = {
    "가시나무": {"youngs_modulus_gpa": 12.0, "density_g_cm3": 0.72},
    "느티나무": {"youngs_modulus_gpa": 11.5, "density_g_cm3": 0.64},
    "물태기나무": {"youngs_modulus_gpa": 10.8, "density_g_cm3": 0.68},
    "주목": {"youngs_modulus_gpa": 10.2, "density_g_cm3": 0.62},
    "백라리": {"youngs_modulus_gpa": 9.5, "density_g_cm3": 0.58},
}
