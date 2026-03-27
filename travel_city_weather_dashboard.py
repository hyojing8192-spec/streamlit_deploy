"""
도시 기반 여행지 추천 대시보드 (Streamlit)

PRD 핵심:
- 도시 -> 월 추천
- 월 -> 도시 Top N 추천
- 추천점수(score) 정렬: 온도 편안함 + 강수일수(precip_days) 기반
- 도시 상세(선택 월 기준):
  - 휴양(4) / 관광(4) / 유명 음식(8, MVP는 국가 기반 대표 예시)
  - 이번 달 액티비티(휴양/관광 각 4)
  - 당일 코스 예시(오전/오후/저녁)

주의:
- "투어 상품"이 아니라 일정 아이디어/추천 텍스트입니다.
- 기후 데이터는 Open-Meteo Climate API(모델 기반) 사용.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote_plus
import hashlib
import re

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import streamlit.components.v1 as components


PLOT_TEMPLATE = "plotly_white"

# Open-Meteo endpoints
OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_CLIMATE = "https://climate-api.open-meteo.com/v1/climate"

# Public dataset (worldcities.csv) with columns:
# city, city_ascii, lat, lng, country, iso2, iso3, admin_name, capital, population, id
# Source: ofou gist (Kaggle-derived) - raw CSV URL
WORLD_CITIES_RAW = (
    "https://gist.githubusercontent.com/ofou/9c33f377d16e033924f74fa3cce49296/"
    "raw/worldcities.csv"
)


ActivityPref = Literal["야외 우선", "혼합", "실내 우선"]

@dataclass(frozen=True)
class ScoreParams:
    # 관광: 23도 기준으로 기온이 '비슷하면' 좋은 시기
    t_target_c: float = 23.0
    delta_c: float = 6.0

    # 휴양(물놀이): 23도 근처가 아니라, 보통 "따뜻 + 건기(비 적음)"를 우선
    relax_t_target_c: float = 28.0
    relax_delta_c: float = 10.0
    precip_threshold_mm: float = 1.0
    cloud_threshold_pct: float = 60.0  # 참고 지표용(추천점수에는 직접 반영하지 않음)
    p_min: float = 0.10
    p_max: float = 0.35
    relax_p_min: float = 0.08
    relax_p_max: float = 0.22

    # 실내 우선: 비가 잦아도 실내로 대체 가능한 만큼 강수 페널티를 완화
    indoor_t_target_c: float = 21.0
    indoor_delta_c: float = 10.0
    indoor_p_min: float = 0.15
    indoor_p_max: float = 0.45

    # 활동 선호별 가중치(추천점수(score) 반영)
    # - outdoor: 휴양(따뜻 + 건기) 우선 -> 비 페널티를 더 크게
    # - mixed: 관광(23도 기준) -> 비 페널티는 중간
    # - indoor: 실내 우선 -> 비 페널티는 작게
    outdoor_precip_weight: float = 1.25
    mixed_precip_weight: float = 0.7
    indoor_precip_weight: float = 0.35

    # 점수 로직 버전(캐시 무효화용)
    logic_version: int = 2


PARAMS = ScoreParams()


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _temp_score(avg_temp_c: float, t_target: float, delta_c: float) -> float:
    if avg_temp_c is None or (isinstance(avg_temp_c, float) and math.isnan(avg_temp_c)):
        return float("nan")
    return max(0.0, 1.0 - (abs(avg_temp_c - t_target) / max(1e-9, delta_c)))


def _precip_penalty(precip_ratio: float, p_min: float, p_max: float) -> float:
    if precip_ratio is None or (isinstance(precip_ratio, float) and math.isnan(precip_ratio)):
        return float("nan")
    denom = max(1e-9, (p_max - p_min))
    return _clamp01((precip_ratio - p_min) / denom)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resolve_cache_dir() -> Path:
    """
    Streamlit Cloud/로컬 모두에서 안전하게 쓸 캐시 경로를 결정합니다.
    우선순위:
    1) 환경변수 TRAVEL_CACHE_DIR
    2) 앱 파일 기준 .cache_travel
    3) 시스템 임시폴더
    """
    env_dir = os.getenv("TRAVEL_CACHE_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir)
        try:
            _ensure_dir(candidate)
            return candidate
        except Exception:
            pass

    local_dir = Path(__file__).resolve().parent / ".cache_travel"
    try:
        _ensure_dir(local_dir)
        return local_dir
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "travel_city_weather_cache"
        _ensure_dir(fallback)
        return fallback


def _cache_paths(cache_dir: Path, top_n: int) -> dict[str, Path]:
    return {
        "cities_csv": cache_dir / f"cities_top_{top_n}.csv",
        "downloaded_worldcities": cache_dir / "worldcities.csv",
    }


def _weather_cache_paths(
    cache_dir: Path,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    model: str,
    score_params: ScoreParams,
) -> dict[str, Path]:
    # 파라미터가 바뀌면 캐시도 달라져야 합니다.
    start_s = str(start_date.date())
    end_s = str(end_date.date())
    key = (
        f"{start_s}_to_{end_s}_model_{model}"
        f"_v{score_params.logic_version}"
        f"_precip{score_params.precip_threshold_mm:.1f}"
        f"_cloud{int(score_params.cloud_threshold_pct)}"
        f"_t{score_params.t_target_c:.1f}_{score_params.delta_c:.1f}"
        f"_rt{score_params.relax_t_target_c:.1f}_{score_params.relax_delta_c:.1f}"
        f"_it{score_params.indoor_t_target_c:.1f}_{score_params.indoor_delta_c:.1f}"
        f"_p{score_params.p_min:.2f}_{score_params.p_max:.2f}"
        f"_rp{score_params.relax_p_min:.2f}_{score_params.relax_p_max:.2f}"
        f"_ip{score_params.indoor_p_min:.2f}_{score_params.indoor_p_max:.2f}"
        f"_w{score_params.outdoor_precip_weight:.2f}_{score_params.mixed_precip_weight:.2f}_{score_params.indoor_precip_weight:.2f}"
    )
    key = key.replace(":", "-").replace("/", "-").replace("\\", "-")
    return {
        "weather_monthly": cache_dir / f"weather_monthly_cache_{key}.parquet",
        "weather_monthly_tmp": cache_dir / f"weather_monthly_cache_tmp_{key}.parquet",
    }


def _download_worldcities_if_missing(download_path: Path, timeout_s: int = 60) -> None:
    if download_path.exists():
        return

    st.info("도시 좌표 데이터를 다운로드 중입니다. (최초 1회)")
    resp = requests.get(WORLD_CITIES_RAW, timeout=timeout_s)
    resp.raise_for_status()
    download_path.write_bytes(resp.content)


def _build_cities_top_200(cache_dir: Path, top_n: int = 100) -> pd.DataFrame:
    paths = _cache_paths(cache_dir, top_n=top_n)
    cities_csv = paths["cities_csv"]
    if cities_csv.exists():
        return pd.read_csv(cities_csv)

    _download_worldcities_if_missing(paths["downloaded_worldcities"])
    df = pd.read_csv(paths["downloaded_worldcities"])

    # MVP: 인덱스/일관성 위해 city_ascii 기반으로 정리
    # columns: city, city_ascii, lat, lng, country, iso2, population ...
    required = ["city_ascii", "lat", "lng", "country", "iso2", "population"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"worldcities.csv 스키마가 예상과 다릅니다. missing={missing}")

    df = df.copy()
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["city_ascii", "lat", "lng", "country", "iso2", "population"])
    df = df[df["population"] > 0]

    # 동일 이름이 많은 편이어서 (도시명, iso2)로 중복 제거 후 top-N
    df = df.drop_duplicates(subset=["city_ascii", "country", "iso2"])
    df = df.sort_values("population", ascending=False).head(top_n).copy()
    df = df.reset_index(drop=True)
    df["city_id"] = df.index.astype(int)
    df["city_name"] = df["city_ascii"].astype(str)

    keep = ["city_id", "city_name", "country", "iso2", "lat", "lng", "population"]
    df = df[keep]

    cities_csv.write_text(df.to_csv(index=False), encoding="utf-8")
    return df


def _format_cloud_display(cloudy_days: int, cloud_ratio: float) -> str:
    pct = int(round(cloud_ratio * 100))
    return f"{cloudy_days} ({pct}%)"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(min(1.0, a)))
    return 6371.0 * c


@st.cache_data(show_spinner=False)
def _geocode_open_meteo_cached(name: str, *, language: str) -> list[dict[str, Any]]:
    q = (name or "").strip()
    if len(q) < 2:
        return []
    resp = requests.get(
        OPEN_METEO_GEOCODING,
        params={"name": q, "count": 12, "language": language},
        headers={"User-Agent": "travel-city-weather-dashboard/1.0"},
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    return list(data.get("results") or [])


def _nearest_city_row(
    cities_df: pd.DataFrame, lat: float, lon: float, country_code: str | None
) -> pd.Series | None:
    df = cities_df.copy()
    if country_code:
        cc = df["iso2"].astype(str).str.upper() == country_code.upper()
        if bool(cc.any()):
            df = df.loc[cc].copy()
    if df.empty:
        return None
    dist_km = df.apply(
        lambda r: _haversine_km(float(lat), float(lon), float(r["lat"]), float(r["lng"])),
        axis=1,
    )
    imin = int(dist_km.idxmin())
    if float(dist_km.loc[imin]) > 250.0:
        return None
    return df.loc[imin]


def _format_best_period_label(months: list[int]) -> str:
    if not months:
        return "—"
    months = sorted(set(int(m) for m in months))
    ranges: list[tuple[int, int]] = []
    start = prev = months[0]
    for m in months[1:]:
        if m == prev + 1:
            prev = m
        else:
            ranges.append((start, prev))
            start = prev = m
    ranges.append((start, prev))
    parts: list[str] = []
    for a, b in ranges:
        parts.append(f"{a}월" if a == b else f"{a}~{b}월")
    return ", ".join(parts)


def _country_name_ko(name: str) -> str:
    """자주 쓰는 국가명 한글 매핑(없으면 원문 유지)."""
    country_map = {
        "South Korea": "대한민국",
        "Korea, South": "대한민국",
        "North Korea": "북한",
        "Japan": "일본",
        "China": "중국",
        "Taiwan": "대만",
        "Hong Kong": "홍콩",
        "Singapore": "싱가포르",
        "Thailand": "태국",
        "Vietnam": "베트남",
        "Malaysia": "말레이시아",
        "Indonesia": "인도네시아",
        "Philippines": "필리핀",
        "India": "인도",
        "Nepal": "네팔",
        "Mongolia": "몽골",
        "Australia": "호주",
        "New Zealand": "뉴질랜드",
        "United States": "미국",
        "USA": "미국",
        "Canada": "캐나다",
        "Mexico": "멕시코",
        "Brazil": "브라질",
        "Argentina": "아르헨티나",
        "Chile": "칠레",
        "Peru": "페루",
        "United Kingdom": "영국",
        "England": "영국",
        "Ireland": "아일랜드",
        "France": "프랑스",
        "Germany": "독일",
        "Spain": "스페인",
        "Italy": "이탈리아",
        "Portugal": "포르투갈",
        "Netherlands": "네덜란드",
        "Belgium": "벨기에",
        "Switzerland": "스위스",
        "Austria": "오스트리아",
        "Czechia": "체코",
        "Poland": "폴란드",
        "Hungary": "헝가리",
        "Greece": "그리스",
        "Turkey": "튀르키예",
        "Russia": "러시아",
        "Ukraine": "우크라이나",
        "United Arab Emirates": "아랍에미리트",
        "Saudi Arabia": "사우디아라비아",
        "Qatar": "카타르",
        "Egypt": "이집트",
        "South Africa": "남아프리카공화국",
    }
    return country_map.get(str(name), str(name))


def _safety_score(city_name: str, country_name: str) -> float:
    """치안 점수(0~100). 월→도시 추천 종합점수에 높은 비중으로 반영."""
    city_key = _city_key(city_name)
    city_overrides: dict[str, float] = {
        "서울": 84.0,
        "부산": 82.0,
        "도쿄": 88.0,
        "오사카": 84.0,
        "싱가포르": 95.0,
        "파리": 69.0,
        "런던": 71.0,
        "뉴욕": 67.0,
        "시드니": 79.0,
        "방콕": 66.0,
        "발리": 72.0,
        "바르셀로나": 68.0,
        "로마": 70.0,
        "홍콩": 80.0,
    }
    if city_key in city_overrides:
        return city_overrides[city_key]

    country_scores: dict[str, float] = {
        "Singapore": 95.0,
        "Japan": 88.0,
        "South Korea": 84.0,
        "Taiwan": 82.0,
        "Hong Kong": 80.0,
        "Australia": 79.0,
        "New Zealand": 83.0,
        "Canada": 82.0,
        "United Kingdom": 71.0,
        "France": 69.0,
        "Germany": 76.0,
        "Spain": 68.0,
        "Italy": 70.0,
        "Portugal": 78.0,
        "Netherlands": 77.0,
        "Belgium": 72.0,
        "Switzerland": 86.0,
        "Austria": 84.0,
        "Czechia": 79.0,
        "Poland": 77.0,
        "Hungary": 74.0,
        "Greece": 72.0,
        "Turkey": 62.0,
        "United States": 67.0,
        "Mexico": 55.0,
        "Brazil": 54.0,
        "Argentina": 62.0,
        "Chile": 71.0,
        "Peru": 61.0,
        "Thailand": 66.0,
        "Vietnam": 72.0,
        "Malaysia": 73.0,
        "Indonesia": 64.0,
        "Philippines": 58.0,
        "India": 57.0,
        "China": 70.0,
        "Mongolia": 73.0,
        "Russia": 56.0,
        "Ukraine": 30.0,
        "United Arab Emirates": 83.0,
        "Saudi Arabia": 74.0,
        "Qatar": 86.0,
        "Egypt": 58.0,
        "South Africa": 50.0,
    }
    return country_scores.get(str(country_name), 68.0)


@st.cache_data(show_spinner=False)
def _city_name_ko(name: str, country: str = "") -> str:
    """도시명을 가능한 한 한글로 변환(미지원 시 원문 유지)."""
    base_map = {
        "Seoul": "서울", "Busan": "부산", "Incheon": "인천", "Daegu": "대구", "Daejeon": "대전", "Gwangju": "광주",
        "Tokyo": "도쿄", "Osaka": "오사카", "Kyoto": "교토", "Nagoya": "나고야", "Fukuoka": "후쿠오카",
        "Beijing": "베이징", "Shanghai": "상하이", "Shenzhen": "선전", "Guangzhou": "광저우", "Hong Kong": "홍콩",
        "Taipei": "타이베이", "Kaohsiung": "가오슝", "Singapore": "싱가포르", "Bangkok": "방콕",
        "Hanoi": "하노이", "Ho Chi Minh City": "호치민", "Jakarta": "자카르타", "Bali": "발리", "Manila": "마닐라",
        "Delhi": "델리", "Mumbai": "뭄바이", "Sydney": "시드니", "Melbourne": "멜버른",
        "Auckland": "오클랜드", "New York": "뉴욕", "Los Angeles": "로스앤젤레스", "San Francisco": "샌프란시스코",
        "Las Vegas": "라스베이거스", "Chicago": "시카고", "Seattle": "시애틀", "Boston": "보스턴",
        "Toronto": "토론토", "Vancouver": "밴쿠버", "Montreal": "몬트리올", "Mexico City": "멕시코시티",
        "London": "런던", "Paris": "파리", "Berlin": "베를린", "Munich": "뮌헨", "Rome": "로마",
        "Milan": "밀라노", "Madrid": "마드리드", "Barcelona": "바르셀로나", "Lisbon": "리스본",
        "Amsterdam": "암스테르담", "Brussels": "브뤼셀", "Zurich": "취리히", "Vienna": "빈",
        "Prague": "프라하", "Budapest": "부다페스트", "Warsaw": "바르샤바", "Athens": "아테네",
        "Istanbul": "이스탄불", "Moscow": "모스크바", "Dubai": "두바이", "Doha": "도하", "Abu Dhabi": "아부다비",
        "Cairo": "카이로", "Cape Town": "케이프타운", "Johannesburg": "요하네스버그",
    }
    city = str(name)
    if city in base_map:
        return base_map[city]
    try:
        cands = _geocode_open_meteo_cached(city, language="ko")
    except (requests.HTTPError, requests.RequestException):
        return city
    ctry_ko = _country_name_ko(country)
    for c in cands:
        nm = str(c.get("name", "")).strip()
        ctry = str(c.get("country", "")).strip()
        if not nm:
            continue
        if ctry and (ctry == country or ctry == ctry_ko):
            return nm
    if cands:
        nm = str(cands[0].get("name", "")).strip()
        if nm:
            return nm
    return city


def _best_travel_window_and_reason(df_city: pd.DataFrame, score_col: str) -> tuple[str, str]:
    """도시 월별 데이터에서 추천 시기·한 줄 요약 문구 생성."""
    d = df_city.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    if d.empty:
        return "—", "월별 추천 점수가 없어 요약을 만들 수 없습니다."

    m1 = int(d.iloc[0]["month"])
    months_pick: list[int] = [m1]
    if len(d) > 1:
        m2 = int(d.iloc[1]["month"])
        s1 = float(d.iloc[0][score_col])
        s2 = float(d.iloc[1][score_col])
        if s2 >= s1 - 0.08:
            months_pick.append(m2)

    period = _format_best_period_label(months_pick)
    sub = df_city[df_city["month"].isin(months_pick)].copy()
    avg_t = float(sub["avg_temp_c"].mean())
    prs: list[float] = []
    for _, r in sub.iterrows():
        dim = int(r["days_in_month_observed"]) if not pd.isna(r["days_in_month_observed"]) else 0
        if dim <= 0:
            continue
        prs.append(float(r["precip_days"]) / float(dim))
    avg_pr = sum(prs) / len(prs) if prs else 0.0

    parts: list[str] = []
    if avg_t >= 22:
        parts.append(
            f"**따뜻한 기온**(이 시기 평균 약 **{avg_t:.0f}℃**)이라 야외에서 가볍게 걷기 좋은 편이에요."
        )
    elif avg_t >= 18:
        parts.append(
            f"평균 기온이 약 **{avg_t:.0f}℃**로 **산책·관광에 쾌적한** 때예요."
        )
    elif avg_t >= 12:
        parts.append(
            f"평균 기온은 약 **{avg_t:.0f}℃**예요. 아침·저녁에는 얇은 겉옷이 있으면 좋아요."
        )
    else:
        parts.append(f"평균 기온은 약 **{avg_t:.0f}℃**로 보온을 챙기면 더 편하게 돌아다닐 수 있어요.")

    if avg_pr <= 0.16:
        parts.append("**비가 자주 오지 않는 시기**라 야외에서 여유 있게 돌아다니기 정말 좋은 때예요!")
    elif avg_pr <= 0.26:
        parts.append("비는 있을 수 있지만 **상대적으로 야외 일정 짜기 무난한** 편이에요.")
    else:
        parts.append("이 구간은 **강수일이 잦을 수 있어** 우산과 실내 대체 코스를 함께 준비하는 걸 추천해요.")

    return period, " ".join(parts)


def _render_hero_title() -> None:
    """앱 전체 우주 배경 + 중앙 지구/비행기 궤도 + 중앙 타이틀."""
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at 12% 18%, rgba(255,255,255,0.25) 1px, transparent 1.5px) 0 0/46px 46px,
              radial-gradient(circle at 76% 24%, rgba(255,255,255,0.22) 1px, transparent 1.5px) 0 0/56px 56px,
              radial-gradient(circle at 26% 74%, rgba(255,255,255,0.18) 1px, transparent 1.5px) 0 0/66px 66px,
              linear-gradient(180deg, #040816 0%, #0c173b 40%, #111c45 100%) !important;
            min-height: 100vh;
          }
          .stApp [data-testid="stAppViewContainer"] .main .block-container {
            position: relative;
            z-index: 3;
          }
          .space-scene {
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
          }
          .earth-center {
            position: fixed;
            left: 50%;
            top: 50%;
            width: clamp(195px, 21vw, 300px);
            height: clamp(195px, 21vw, 300px);
            transform: translate(-50%, -50%);
            border-radius: 50%;
            overflow: hidden;
            background:
              radial-gradient(circle at 68% 30%, rgba(186, 230, 253, 0.85) 0%, rgba(186, 230, 253, 0.2) 18%, transparent 30%),
              radial-gradient(circle at 40% 70%, rgba(56, 189, 248, 0.3) 0%, rgba(56, 189, 248, 0.05) 30%, transparent 52%),
              radial-gradient(circle at 60% 35%, #9ad6ff 0%, #4a8fdc 44%, #173a8f 100%);
            box-shadow: inset -26px -20px 42px rgba(6, 24, 77, 0.62), 0 0 48px rgba(96, 165, 250, 0.34);
            animation: earthRotate 58s linear infinite;
          }
          .earth-center:before, .earth-center:after {
            content: "";
            position: absolute;
            border-radius: 48% 52% 45% 55% / 52% 48% 56% 44%;
          }
          .earth-center:before {
            width: 42%;
            height: 24%;
            left: 12%;
            top: 40%;
            transform: rotate(-15deg);
            background: linear-gradient(145deg, rgba(74, 222, 128, 0.8), rgba(22, 163, 74, 0.82));
          }
          .earth-center:after {
            width: 30%;
            height: 19%;
            left: 55%;
            top: 57%;
            transform: rotate(22deg);
            background: linear-gradient(160deg, rgba(74, 222, 128, 0.76), rgba(21, 128, 61, 0.82));
          }
          .earth-map {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            opacity: 0.85;
          }
          .earth-map .land {
            fill: rgba(47, 129, 74, 0.82);
            stroke: rgba(30, 90, 56, 0.42);
            stroke-width: 0.6;
          }
          .earth-map .land-2 {
            fill: rgba(73, 160, 96, 0.72);
            stroke: rgba(36, 104, 62, 0.4);
            stroke-width: 0.6;
          }
          .earth-shade {
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 24% 30%, rgba(255,255,255,0.0) 0%, rgba(255,255,255,0.0) 45%, rgba(0,0,0,0.24) 100%);
          }
          .earth-atmosphere {
            position: absolute;
            inset: -2px;
            border-radius: 50%;
            border: 2px solid rgba(186, 230, 253, 0.35);
          }
          .cloud-band {
            position: absolute;
            left: 8%;
            right: 8%;
            top: 30%;
            height: 8%;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(255,255,255,0.0), rgba(255,255,255,0.34), rgba(255,255,255,0.0));
          }
          .flight-arc {
            position: fixed;
            left: 50%;
            top: 50%;
            width: clamp(370px, 47vw, 590px);
            height: clamp(155px, 22vw, 265px);
            transform: translate(-50%, -50%);
            border-top: 1.5px dashed rgba(191, 219, 254, 0.35);
            border-radius: 300px 300px 0 0;
          }
          .plane-north {
            position: absolute;
            left: calc(100% - 4px);
            top: calc(100% - 10px);
            width: clamp(26px, 2.8vw, 40px);
            height: clamp(26px, 2.8vw, 40px);
            filter: drop-shadow(0 2px 2px rgba(0,0,0,0.35));
            animation: northArcFlight 11s ease-in-out infinite;
          }
          .plane-north svg {
            width: 100%;
            height: 100%;
            display: block;
          }
          .hero-title-center {
            margin: 20vh auto 5vh auto;
            text-align: center;
            font-family: 'BMJUA', 'Malgun Gothic', sans-serif;
            font-size: clamp(2rem, 4.8vw, 3rem);
            color: #eff6ff;
            text-shadow: 0 3px 14px rgba(15, 23, 42, 0.5);
            letter-spacing: -0.02em;
            animation: titleBlinkOnce 0.6s ease-in-out 1 2.2s;
            position: relative;
            z-index: 5;
          }
          @keyframes titleBlinkOnce {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.2; }
          }
          @keyframes earthRotate {
            from { transform: translate(-50%, -50%) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg); }
          }
          @keyframes northArcFlight {
            0% {
              left: calc(100% - 4px);
              top: calc(100% - 10px);
              transform: rotate(-22deg);
            }
            20% {
              left: 75%;
              top: 28%;
              transform: rotate(-10deg);
            }
            40% {
              left: 50%;
              top: -12px;
              transform: rotate(0deg);
            }
            60% {
              left: 25%;
              top: 28%;
              transform: rotate(10deg);
            }
            72.7% {
              left: 0%;
              top: calc(100% - 10px);
              transform: rotate(18deg);
            }
            100% {
              left: 0%;
              top: calc(100% - 10px);
              transform: rotate(18deg);
            }
          }
        </style>
        <div class="space-scene">
          <div class="earth-center">
            <svg class="earth-map" viewBox="0 0 200 200" aria-hidden="true">
              <path class="land" d="M46 60c10-9 28-13 42-8 10 3 13 12 9 19-4 8-16 13-27 15-8 2-11 9-8 16 4 8 4 16-4 22-10 8-23 6-30-3-6-8-9-20-5-30 4-13 10-23 23-31z"/>
              <path class="land-2" d="M84 121c9-5 20-2 22 7 4 14-2 30-13 39-7 6-16 3-18-7-3-15 1-32 9-39z"/>
              <path class="land" d="M108 62c12-8 31-8 40 2 7 8 6 18-3 25-8 6-19 8-23 16-3 6 1 13 4 18 6 10 3 22-7 29-11 8-25 7-32-4-7-11-7-25 1-35 5-6 11-11 13-18 2-7-2-14 0-21 1-5 4-9 7-12z"/>
              <path class="land-2" d="M126 128c10-5 21 0 24 11 4 13-2 28-12 36-8 6-18 3-22-7-6-15-1-34 10-40z"/>
              <path class="land" d="M150 74c10-6 24-6 31 1 6 6 6 14 0 20-7 7-18 9-26 7-8-2-12-8-11-16 1-5 2-9 6-12z"/>
            </svg>
            <div class="earth-shade"></div>
            <div class="cloud-band"></div>
            <div class="earth-atmosphere"></div>
          </div>
          <div class="flight-arc">
            <div class="plane-north" aria-hidden="true">
              <svg viewBox="0 0 64 64">
                <!-- 왼쪽을 보는 만화풍 비행기 -->
                <path d="M6 34 C6 24 14 18 24 18 L44 18 C51 18 57 22 60 28 L58 44 C56 48 52 50 46 50 L26 50 C16 50 6 44 6 34 Z" fill="#7dd3fc" stroke="#0f172a" stroke-width="2.8" />
                <path d="M18 23 L10 30 L14 37 L24 37 L24 23 Z" fill="#dbeafe" stroke="#0f172a" stroke-width="2.2"/>
                <path d="M42 18 L52 7 L61 9 L58 22 Z" fill="#93c5fd" stroke="#0f172a" stroke-width="2.2"/>
                <path d="M34 50 L51 50 L63 58 L54 60 L30 53 Z" fill="#dbeafe" stroke="#0f172a" stroke-width="2.2"/>
                <rect x="28" y="28" width="2.3" height="6.8" rx="1" fill="#0f172a"/>
                <rect x="33" y="28" width="2.3" height="6.8" rx="1" fill="#0f172a"/>
                <rect x="38" y="28" width="2.3" height="6.8" rx="1" fill="#0f172a"/>
              </svg>
            </div>
          </div>
        </div>
        <div class="hero-title-center">당신의 여행계획을 세워드립니다!</div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_weather_monthly_cache(cache_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache_path)
    # 과거 캐시에 컬럼이 없을 수 있어 호환 처리
    if "precip_mm_avg" not in df.columns:
        if "precip_days" in df.columns:
            df["precip_mm_avg"] = pd.to_numeric(df["precip_days"], errors="coerce")
        else:
            df["precip_mm_avg"] = float("nan")
    return df


def _fetch_climate_daily_batch_once(
    lats: list[float],
    lons: list[float],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    model: str,
    session: requests.Session,
    *,
    timeout_s: int = 120,
) -> requests.Response:
    params = {
        "latitude": ",".join([str(x) for x in lats]),
        "longitude": ",".join([str(x) for x in lons]),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "models": model,
        "daily": "temperature_2m_mean,precipitation_sum,cloud_cover_mean",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }
    headers = {"User-Agent": "travel-city-weather-dashboard/1.0"}
    return session.get(OPEN_METEO_CLIMATE, params=params, headers=headers, timeout=timeout_s)


def _sleep_for_rate_limit(resp: requests.Response, attempt: int) -> None:
    """429 시 대기: Retry-After 헤더 우선, 없으면 지수 백오프."""
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            wait = float(ra)
        except ValueError:
            wait = min(120.0, 3.0 * (2**attempt))
    else:
        wait = min(120.0, 3.0 * (2**attempt) + 0.5 * attempt)
    time.sleep(wait)


def _parse_climate_list_response(resp: requests.Response) -> list[dict]:
    data = resp.json()
    if not isinstance(data, list):
        data = [data]
    return data


def _fetch_climate_daily_batch(
    lats: list[float],
    lons: list[float],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    model: str,
    session: requests.Session,
    *,
    max_retries_per_attempt: int = 8,
) -> list[dict]:
    """
    Open-Meteo는 무료 한도에서 429가 자주 납니다.
    - 동일 배치에 대해 재시도(백오프)
    - 끝까지 429면 배치를 둘로 나눠 재귀 호출(1개까지)
    """
    if len(lats) != len(lons) or not lats:
        raise ValueError("latitude/longitude 길이가 맞지 않거나 비어 있습니다.")

    last_resp: requests.Response | None = None
    for attempt in range(max_retries_per_attempt):
        resp = _fetch_climate_daily_batch_once(
            lats, lons, start_date, end_date, model, session
        )
        last_resp = resp
        if resp.status_code == 429:
            _sleep_for_rate_limit(resp, attempt)
            continue
        if resp.status_code >= 500:
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
            continue
        resp.raise_for_status()
        return _parse_climate_list_response(resp)

    # 여전히 실패(대부분 429): 배치 축소
    if len(lats) > 1:
        mid = len(lats) // 2
        left = _fetch_climate_daily_batch(
            lats[:mid],
            lons[:mid],
            start_date,
            end_date,
            model,
            session,
            max_retries_per_attempt=max_retries_per_attempt,
        )
        right = _fetch_climate_daily_batch(
            lats[mid:],
            lons[mid:],
            start_date,
            end_date,
            model,
            session,
            max_retries_per_attempt=max_retries_per_attempt,
        )
        return left + right

    # 단일 좌표인데도 실패(429 등) — 추가 대기 후 재시도
    for extra in range(12):
        time.sleep(min(30.0, 5.0 * (extra + 1)))
        resp = _fetch_climate_daily_batch_once(
            lats, lons, start_date, end_date, model, session, timeout_s=180
        )
        last_resp = resp
        if resp.status_code == 429:
            _sleep_for_rate_limit(resp, max_retries_per_attempt + extra)
            continue
        if resp.status_code >= 500:
            continue
        resp.raise_for_status()
        return _parse_climate_list_response(resp)

    raise RuntimeError(
        "Open-Meteo Climate API가 일시적으로 요청을 거부했습니다(429). "
        "몇 분 뒤에 다시 시도하거나, 기간을 짧게(예: 1년) 줄이거나, 배치 크기를 1~2로 낮춰 주세요."
    )


def _daily_to_monthly_stats(
    daily_json: dict,
    precip_threshold_mm: float,
    cloud_threshold_pct: float,
) -> pd.DataFrame:
    daily = daily_json.get("daily", {}) or {}
    times = daily.get("time", []) or []
    temps = daily.get("temperature_2m_mean", []) or []
    precips = daily.get("precipitation_sum", []) or []
    clouds = daily.get("cloud_cover_mean", []) or []

    if not times:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(times, errors="coerce"),
            "avg_temp_c": pd.to_numeric(temps, errors="coerce"),
            "precip_mm": pd.to_numeric(precips, errors="coerce"),
            "cloud_pct": pd.to_numeric(clouds, errors="coerce"),
        }
    ).dropna(subset=["date"])

    # month 1~12 기준 집계
    df["month"] = df["date"].dt.month.astype(int)
    grouped = df.groupby("month", as_index=False).agg(
        avg_temp_c=("avg_temp_c", "mean"),
        precip_mm_avg=("precip_mm", "mean"),
        precip_days=("precip_mm", lambda s: int((s.fillna(0.0) >= precip_threshold_mm).sum())),
        cloudy_days=("cloud_pct", lambda s: int((s.fillna(0.0) >= cloud_threshold_pct).sum())),
        days_in_month_observed=("date", "count"),
    )
    grouped = grouped.sort_values("month")
    return grouped


def _compute_scores_for_monthly(
    monthly_df: pd.DataFrame,
    score_params: ScoreParams,
) -> pd.DataFrame:
    if monthly_df.empty:
        return monthly_df

    days = monthly_df["days_in_month_observed"].replace(0, pd.NA)
    monthly_df = monthly_df.copy()
    monthly_df["precip_ratio"] = (monthly_df["precip_days"] / days).astype(float)

    # 활동 선호별 추천점수
    # - outdoor: 휴양(따뜻 + 건기) 우선 -> "기온이 더 높아도 + 비가 적으면" 점수 상승
    # - mixed: 관광(23도 기준) -> 23도 근접이 중심, 비는 중간 패널티
    # - indoor: 실내 우선 -> 비 페널티를 작게, 극단 온도만 살짝 완화

    monthly_df["temp_score_outdoor"] = monthly_df["avg_temp_c"].apply(
        lambda t: _temp_score(t, score_params.relax_t_target_c, score_params.relax_delta_c)
    )
    monthly_df["precip_penalty_outdoor"] = monthly_df["precip_ratio"].apply(
        lambda r: _precip_penalty(r, score_params.relax_p_min, score_params.relax_p_max)
    )
    monthly_df["recommendation_score_outdoor"] = (
        monthly_df["temp_score_outdoor"] - score_params.outdoor_precip_weight * monthly_df["precip_penalty_outdoor"]
    )

    monthly_df["temp_score_mixed"] = monthly_df["avg_temp_c"].apply(
        lambda t: _temp_score(t, score_params.t_target_c, score_params.delta_c)
    )
    monthly_df["precip_penalty_mixed"] = monthly_df["precip_ratio"].apply(
        lambda r: _precip_penalty(r, score_params.p_min, score_params.p_max)
    )
    monthly_df["recommendation_score_mixed"] = (
        monthly_df["temp_score_mixed"] - score_params.mixed_precip_weight * monthly_df["precip_penalty_mixed"]
    )

    monthly_df["temp_score_indoor"] = monthly_df["avg_temp_c"].apply(
        lambda t: _temp_score(t, score_params.indoor_t_target_c, score_params.indoor_delta_c)
    )
    monthly_df["precip_penalty_indoor"] = monthly_df["precip_ratio"].apply(
        lambda r: _precip_penalty(r, score_params.indoor_p_min, score_params.indoor_p_max)
    )
    monthly_df["recommendation_score_indoor"] = (
        monthly_df["temp_score_indoor"] - score_params.indoor_precip_weight * monthly_df["precip_penalty_indoor"]
    )
    return monthly_df


def _build_weather_monthly_cache(
    cache_dir: Path,
    cities_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    model: str,
    batch_size: int = 5,
    pause_between_batches_s: float = 1.5,
    max_cities: int | None = None,
    score_params: ScoreParams = PARAMS,
) -> pd.DataFrame:
    wpaths = _weather_cache_paths(
        cache_dir=cache_dir,
        start_date=start_date,
        end_date=end_date,
        model=model,
        score_params=score_params,
    )
    cache_path = wpaths["weather_monthly"]
    tmp_path = wpaths["weather_monthly_tmp"]

    # 이미 있으면 로드(스트림릿 캐시 + disk cache)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    _ensure_dir(cache_dir)
    cities = cities_df.copy()
    if max_cities is not None:
        cities = cities.head(max_cities).copy()

    n = len(cities)
    st.info(f"기후 월별 캐시 생성 시작: cities={n}, range={start_date.date()}~{end_date.date()}, model={model}")

    all_rows: list[dict] = []
    with requests.Session() as session:
        for start_i in range(0, n, batch_size):
            end_i = min(n, start_i + batch_size)
            batch = cities.iloc[start_i:end_i]
            lats = batch["lat"].astype(float).tolist()
            lons = batch["lng"].astype(float).tolist()

            st.write(f"  - batch {start_i} ~ {end_i-1} / {n-1}")
            daily_list = _fetch_climate_daily_batch(
                lats=lats,
                lons=lons,
                start_date=start_date,
                end_date=end_date,
                model=model,
                session=session,
            )

            # daily_list는 배치의 순서대로 각 위치 구조가 들어오는 형태를 기대
            for i_loc, daily_json in enumerate(daily_list):
                city_row = batch.iloc[i_loc]
                monthly = _daily_to_monthly_stats(
                    daily_json=daily_json,
                    precip_threshold_mm=score_params.precip_threshold_mm,
                    cloud_threshold_pct=score_params.cloud_threshold_pct,
                )
                monthly = _compute_scores_for_monthly(monthly, score_params=score_params)
                if monthly.empty:
                    continue

                for _, r in monthly.iterrows():
                    month = int(r["month"])
                    days_in_month = int(r["days_in_month_observed"]) if not pd.isna(r["days_in_month_observed"]) else 0
                    cloudy_days = int(r["cloudy_days"]) if not pd.isna(r["cloudy_days"]) else 0
                    cloud_ratio = (cloudy_days / days_in_month) if days_in_month else 0.0

                    all_rows.append(
                        {
                            "city_id": int(city_row["city_id"]),
                            "city_name": str(city_row["city_name"]),
                            "country": str(city_row["country"]),
                            "iso2": str(city_row["iso2"]),
                            "month": month,
                            "avg_temp_c": float(r["avg_temp_c"]) if not pd.isna(r["avg_temp_c"]) else float("nan"),
                            "precip_mm_avg": float(r["precip_mm_avg"]) if "precip_mm_avg" in r and not pd.isna(r["precip_mm_avg"]) else float("nan"),
                            "precip_days": int(r["precip_days"]) if not pd.isna(r["precip_days"]) else 0,
                            "cloudy_days": cloudy_days,
                            "days_in_month_observed": days_in_month,
                            "precip_ratio": float(r["precip_ratio"]) if "precip_ratio" in r else float("nan"),
                            "recommendation_score_outdoor": float(r["recommendation_score_outdoor"])
                            if not pd.isna(r["recommendation_score_outdoor"])
                            else float("nan"),
                            "recommendation_score_mixed": float(r["recommendation_score_mixed"])
                            if not pd.isna(r["recommendation_score_mixed"])
                            else float("nan"),
                            "recommendation_score_indoor": float(r["recommendation_score_indoor"])
                            if not pd.isna(r["recommendation_score_indoor"])
                            else float("nan"),
                        }
                    )

            # 다음 배치 전 짧은 휴지(무료 API 429 방지)
            if end_i < n and pause_between_batches_s > 0:
                time.sleep(pause_between_batches_s)

    df_out = pd.DataFrame(all_rows)
    if df_out.empty:
        raise RuntimeError("weather_monthly_cache 생성 결과가 비어 있습니다. (API 응답/파라미터 확인)")

    _ensure_dir(cache_dir)
    # 원자성(임시 후 교체)
    df_out.to_parquet(tmp_path, index=False)
    if cache_path.exists():
        cache_path.unlink()
    os.replace(tmp_path, cache_path)
    return df_out


def _select_score_column(activity_pref: ActivityPref) -> str:
    if activity_pref == "야외 우선":
        return "recommendation_score_outdoor"
    if activity_pref == "혼합":
        return "recommendation_score_mixed"
    return "recommendation_score_indoor"


def _best_month_for_city(df_city: pd.DataFrame, score_col: str) -> int:
    if df_city.empty or score_col not in df_city.columns:
        return 1
    s = df_city.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    if s.empty:
        return 1
    return int(s.iloc[0]["month"])


def _outdoor_grade(precip_ratio: float) -> Literal["야외 적합", "혼합 추천", "실내 우선"]:
    if math.isnan(precip_ratio):
        return "혼합 추천"
    if precip_ratio <= 0.15:
        return "야외 적합"
    if precip_ratio <= 0.30:
        return "혼합 추천"
    return "실내 우선"


def _activity_reason(
    precip_ratio: float,
    avg_temp_c: float,
    activity_kind: Literal["휴양", "관광"],
) -> str:
    grade = _outdoor_grade(precip_ratio)
    pr_pct = int(round(precip_ratio * 100)) if not math.isnan(precip_ratio) else 0

    # 휴양(물놀이)은 "23도 근처"보다 "따뜻 + 건기"가 더 중요
    if activity_kind == "휴양":
        warm_enough = not math.isnan(avg_temp_c) and avg_temp_c >= 24
        if grade == "야외 적합":
            if warm_enough:
                return f"강수일수 비율이 낮아(비가 적은 편) 물놀이/해변 산책에 특히 좋아요. 평균기온은 약 {avg_temp_c:.0f}℃예요."
            return f"비가 적은 편이라 야외 중심으로 일정을 짜기 좋아요. 평균기온은 약 {avg_temp_c:.0f}℃로, 가벼운 물놀이 위주가 잘 맞아요."
        if grade == "혼합 추천":
            return f"비가 가끔 섞일 수 있지만(강수일수 비율 {pr_pct}%) 평균기온이 비교적 무난해요. 물놀이 시간은 오전/오후로 분산해 보세요."
        return f"강수일이 잦은 편이라(강수일수 비율 {pr_pct}%) 야외 물놀이 비중을 낮추고 실내 대체 코스를 함께 추천해요."

    # 관광은 기본적으로 23도 전후의 쾌적함에 더 비중
    if grade == "야외 적합":
        t_diff = abs(avg_temp_c - PARAMS.t_target_c) if not math.isnan(avg_temp_c) else float("nan")
        return f"강수일수 비율이 낮아(야외 적합) 야외 산책·관광에 유리해요. 평균기온이 목표(23C)와 {t_diff:.1f}C 정도 차이입니다."
    if grade == "혼합 추천":
        return f"기후 변수가 있는 편이라(혼합) 오전/오후는 유연하게, 실내 대체도 함께 추천해요."
    return f"강수일수 비율이 높은 편이라(실내 우선) 실내 활동 비중을 늘리는 구성이 좋아요."


def _pick_from_templates(templates: list[str], used: set[str], k: int) -> list[str]:
    out: list[str] = []
    for t in templates:
        if t not in used:
            out.append(t)
            used.add(t)
        if len(out) >= k:
            break
    return out


def _city_key(city_name: str) -> str:
    return str(city_name).strip().lower()


def _place(
    name: str,
    link: str | None = None,
    *,
    desc: str | None = None,
    image: str | None = None,
    menu: str | None = None,
    activity: str | None = None,
) -> dict[str, str | None]:
    return {
        "name": name,
        "link": link,
        "desc": desc,
        "image": image,
        "menu": menu,
        "activity": activity,
    }


def _gmaps_link(query: str) -> str:
    q = str(query).strip().replace(" ", "+")
    return f"https://www.google.com/maps/search/?api=1&query={q}"


def _img_query(query: str) -> str:
    q = quote_plus(str(query).strip())
    # source.unsplash는 간헐적으로 404/차단이 있어 picsum seed로 안정화
    return f"https://picsum.photos/seed/{q}/640/420"


def _link_image(link: str | None, query: str) -> str:
    """링크와 연관된 썸네일 우선(페이지 스냅샷), 실패 대비 쿼리 이미지."""
    if isinstance(link, str) and link.strip():
        safe = link.strip()
        return f"https://image.thum.io/get/width/640/noanimate/{safe}"
    return _img_query(query)


def _generic_city_profile(city_name: str) -> dict[str, Any]:
    """프로필이 없는 도시용: 추상 문구 없이 구체 장소명/링크 생성."""
    city = str(city_name).strip()
    return {
        "dominant": "혼합",
        "sightseeing": [
            _place(f"{city} 올드타운", _gmaps_link(f"{city} old town")),
            _place(f"{city} 국립박물관", _gmaps_link(f"{city} national museum")),
            _place(f"{city} 중앙시장", _gmaps_link(f"{city} central market")),
            _place(f"{city} 전망대", _gmaps_link(f"{city} observation deck")),
        ],
        "relax": [
            _place(f"{city} 보타닉 가든", _gmaps_link(f"{city} botanical garden")),
            _place(f"{city} 리버사이드 산책로", _gmaps_link(f"{city} riverside walk")),
            _place(f"{city} 시티파크", _gmaps_link(f"{city} city park")),
            _place(f"{city} 브런치 카페 거리", _gmaps_link(f"{city} brunch cafe")),
        ],
        "activities": [
            _place(f"{city} 도보 투어", _gmaps_link(f"{city} walking tour")),
            _place(f"{city} 선셋 크루즈", _gmaps_link(f"{city} sunset cruise")),
            _place(f"{city} 자전거 투어", _gmaps_link(f"{city} bike tour")),
            _place(f"{city} 쿠킹 클래스", _gmaps_link(f"{city} cooking class")),
        ],
        "restaurants": [
            _place(f"{city} 미슐랭 가이드 레스토랑", _gmaps_link(f"{city} michelin restaurant")),
            _place(f"{city} 시푸드 레스토랑", _gmaps_link(f"{city} seafood restaurant")),
            _place(f"{city} 스테이크 하우스", _gmaps_link(f"{city} steak house")),
            _place(f"{city} 로컬 브런치 카페", _gmaps_link(f"{city} brunch cafe")),
        ],
    }


def _pick_place_templates(templates: list[dict[str, str | None]], used: set[str], k: int) -> list[dict[str, str | None]]:
    out: list[dict[str, str | None]] = []
    for t in templates:
        nm = str(t.get("name", "")).strip()
        if not nm:
            continue
        if nm not in used:
            out.append(t)
            used.add(nm)
        if len(out) >= k:
            break
    return out


def _enrich_place_list(city_name: str, kind: Literal["관광", "휴양", "액티비티", "식당", "음식"], places: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    out: list[dict[str, str | None]] = []
    for p in places:
        name = str(p.get("name", "")).strip()
        if not name:
            continue
        link = p.get("link") or _gmaps_link(f"{city_name} {name}")
        desc = p.get("desc")
        image = p.get("image") or _link_image(str(link), f"{city_name} {name}")
        menu = p.get("menu")
        activity = p.get("activity")
        if not desc:
            if kind == "관광":
                desc = f"{city_name}에서 대표적으로 방문하는 명소예요."
            elif kind == "휴양":
                desc = f"{city_name}에서 여유롭게 쉬기 좋은 장소예요."
            elif kind == "식당":
                desc = f"{city_name}에서 평이 좋은 인기 식당이에요."
            elif kind == "액티비티":
                desc = f"{city_name}에서 체험하기 좋은 활동 코스예요."
            else:
                desc = f"{city_name}에서 자주 찾는 음식/미식 포인트예요."
        if kind == "식당" and not menu:
            menu = f"주메뉴: {name}의 시그니처 메뉴"
        if kind == "액티비티" and not activity:
            activity = f"활동 내용: {name} 중심으로 현장 체험/가이드 진행"
        out.append(_place(name, str(link), desc=desc, image=str(image), menu=menu, activity=activity))
    return out


def _city_profile(city_name: str) -> dict[str, Any]:
    """도시별 추천 성향/액티비티/식당 추천 템플릿."""
    k = _city_key(city_name)
    profiles: dict[str, dict[str, Any]] = {
        "시드니": {
            "dominant": "관광",
            "sightseeing": [
                _place("시드니 하버 브리지(BridgeClimb)", "https://www.bridgeclimb.com/"),
                _place("시드니 오페라하우스", "https://www.sydneyoperahouse.com/"),
                _place("더 록스(The Rocks)", "https://www.therocks.com/"),
                _place("본다이-쿠지 코스탈 워크", "https://www.bondivisitorinformation.com.au/bondi-to-coogee-coastal-walk"),
            ],
            "relax": [
                _place("본다이 비치", "https://www.bondivisitorinformation.com.au/"),
                _place("로열 보타닉 가든 시드니", "https://www.rbgsyd.nsw.gov.au/"),
                _place("달링 하버", "https://www.darlingharbour.com/"),
                _place("맨리 비치", "https://www.sydney.com/destinations/sydney/sydney-north/manly"),
            ],
            "activities": [
                _place("시드니 하버 브리지 워킹 투어", "https://www.bridgeclimb.com/"),
                _place("시드니 하버 크루즈", "https://www.captaincook.com.au/sydney-harbour-cruises/"),
                _place("서큘러 키-맨리 페리", "https://transportnsw.info/routes/ferry"),
                _place("본다이 서핑 클래스", "https://letsgosurfing.com.au/"),
            ],
            "restaurants": [
                _place("Bennelong", "https://www.bennelong.com.au/"),
                _place("Quay", "https://www.quay.com.au/"),
                _place("The Grounds of Alexandria", "https://thegrounds.com.au/"),
                _place("Single O Surry Hills(브런치)", "https://singleo.com.au/"),
            ],
        },
        "도쿄": {
            "dominant": "관광",
            "sightseeing": [
                _place("아사쿠사 센소지", "https://www.senso-ji.jp/"),
                _place("도쿄 스카이트리", "https://www.tokyo-skytree.jp/"),
                _place("메이지 신궁", "https://www.meijijingu.or.jp/"),
                _place("우에노 공원/도쿄국립박물관", "https://www.tnm.jp/"),
            ],
            "relax": [
                _place("요요기 공원", "https://www.tokyo-park.or.jp/park/yoyogi/"),
                _place("가구라자카 카페 거리", "https://www.gotokyo.org/en/destinations/central-tokyo/kagurazaka/index.html"),
                _place("오다이바 해변공원", "https://www.tokyo-odaiba.net/en/"),
                _place("스미다강 크루즈", "https://www.suijobus.co.jp/en/"),
            ],
            "activities": [
                _place("teamLab Planets TOKYO", "https://planets.teamlab.art/tokyo/"),
                _place("쓰키지/도요스 스시 체험", "https://www.toyosu-market.or.jp/en/"),
                _place("도쿄 베이 야경 크루즈", "https://www.symphony-cruise.co.jp/en/"),
                _place("도쿄 자전거 투어", "https://www.tokyobike.tours/"),
            ],
            "restaurants": [
                _place("스시다이(도요스)", "https://www.toyosu-market.or.jp/en/"),
                _place("긴자 이치란 라멘", "https://en.ichiran.com/"),
                _place("규카츠 모토무라", "https://www.gyukatsu-motomura.com/en/"),
                _place("로컬 브런치: Path(요요기하치만)", "https://www.instagram.com/path_restaurant/"),
            ],
        },
        "발리": {
            "dominant": "휴양",
            "sightseeing": [
                _place("우붓 왕궁", "https://www.gotravelaindonesia.com/ubud-palace"),
                _place("우붓 전통시장", "https://www.gotravelaindonesia.com/ubud-art-market"),
                _place("울루와뚜 사원", "https://www.balitourismboard.org/uluwatu-temple/"),
                _place("뜨그눙안 폭포", "https://www.indonesia.travel/"),
            ],
            "relax": [
                _place("스미냑 비치", "https://www.balitourismboard.org/"),
                _place("우붓 라이스 테라스(뜨갈랄랑)", "https://www.indonesia.travel/"),
                _place("스파/웰니스 데이", "https://www.tripadvisor.com/Attractions-g294226-Activities-c40-Bali.html"),
                _place("짱구 비치 선셋", "https://www.balitourismboard.org/"),
            ],
            "activities": [
                _place("발리 서핑 입문 클래스", "https://www.getyourguide.com/bali-l347/"),
                _place("누사 페니다 스노클링 투어", "https://www.getyourguide.com/bali-l347/"),
                _place("요가 반야드 클래스", "https://www.theyogabarn.com/"),
                _place("선셋 비치클럽(핀스/포테이토헤드)", "https://finnsbali.com/"),
            ],
            "restaurants": [
                _place("Locavore NXT", "https://www.locavorenxt.com/"),
                _place("Merah Putih", "https://merahputihbali.com/"),
                _place("Sisterfields", "https://www.sisterfieldsbali.com/"),
                _place("브런치: Milk & Madu", "https://milknmadu.com/"),
            ],
        },
        "서울": {
            "dominant": "관광",
            "sightseeing": [
                _place("경복궁", "https://www.royalpalace.go.kr/"),
                _place("북촌한옥마을", "https://bukchon.seoul.go.kr/"),
                _place("N서울타워", "https://www.nseoultower.co.kr/"),
                _place("국립중앙박물관", "https://www.museum.go.kr/"),
            ],
            "relax": [
                _place("서울숲", "https://seoulforest.or.kr/"),
                _place("한강공원(여의도)", "https://hangang.seoul.go.kr/"),
                _place("정독도서관/삼청동 산책", _gmaps_link("삼청동 산책")),
                _place("익선동 카페거리", _gmaps_link("익선동 카페거리")),
            ],
            "activities": [
                _place("한강 유람선", "https://www.elandcruise.com/"),
                _place("DMZ 당일 투어", _gmaps_link("DMZ tour seoul")),
                _place("광장시장 푸드투어", _gmaps_link("광장시장")),
                _place("잠실 롯데월드타워 전망대", "https://seoulsky.lotteworld.com/"),
            ],
            "restaurants": [
                _place("광장시장 부촌육회", _gmaps_link("부촌육회")),
                _place("몽탄", _gmaps_link("몽탄 용산")),
                _place("진미식당(게장)", _gmaps_link("진미식당")),
                _place("브런치: 오아시스 한남", _gmaps_link("오아시스 한남")),
            ],
        },
        "오사카": {
            "dominant": "관광",
            "sightseeing": [
                _place("오사카성", "https://www.osakacastle.net/"),
                _place("도톤보리", _gmaps_link("Dotonbori")),
                _place("우메다 스카이빌딩", "https://www.skybldg.co.jp/en/"),
                _place("신세카이/츠텐카쿠", "https://www.tsutenkaku.co.jp/"),
            ],
            "relax": [
                _place("나카노시마 공원", _gmaps_link("Nakanoshima Park")),
                _place("우츠보 공원", _gmaps_link("Utsubo Park")),
                _place("호리에 카페거리", _gmaps_link("Horie Osaka cafe")),
                _place("유니버설시티 워터프론트", _gmaps_link("Universal Citywalk Osaka")),
            ],
            "activities": [
                _place("우메다 야경 투어", _gmaps_link("Umeda night view")),
                _place("구로몬시장 식도락", _gmaps_link("Kuromon Market")),
                _place("오사카 강 리버크루즈", _gmaps_link("Tombori River Cruise")),
                _place("나라/교토 당일치기 투어", _gmaps_link("Kyoto day trip from Osaka")),
            ],
            "restaurants": [
                _place("하리주 도톤보리 본점", _gmaps_link("Hariju Dotonbori")),
                _place("자유켄", _gmaps_link("Jiyuken Osaka")),
                _place("치보 오코노미야키", _gmaps_link("Chibo Dotonbori")),
                _place("브런치: bills 오사카", _gmaps_link("bills Osaka")),
            ],
        },
        "파리": {
            "dominant": "관광",
            "sightseeing": [
                _place("에펠탑", "https://www.toureiffel.paris/en"),
                _place("루브르 박물관", "https://www.louvre.fr/"),
                _place("오르세 미술관", "https://www.musee-orsay.fr/"),
                _place("몽마르트르/사크레쾨르", _gmaps_link("Montmartre Sacre-Coeur")),
            ],
            "relax": [
                _place("튈르리 정원", _gmaps_link("Tuileries Garden")),
                _place("뤽상부르 공원", _gmaps_link("Luxembourg Gardens")),
                _place("생마르탱 운하 산책", _gmaps_link("Canal Saint Martin")),
                _place("마레 지구 카페", _gmaps_link("Le Marais cafe")),
            ],
            "activities": [
                _place("세느강 크루즈(Bateaux Mouches)", "https://www.bateaux-mouches.fr/en"),
                _place("루브르 야간 관람", "https://www.louvre.fr/"),
                _place("마카롱 베이킹 클래스", _gmaps_link("Paris macaron class")),
                _place("베르사유 궁전 반일 투어", "https://en.chateauversailles.fr/"),
            ],
            "restaurants": [
                _place("Le Relais de l'Entrecôte", "https://www.relaisentrecote.fr/"),
                _place("Bouillon Chartier", "https://www.bouillon-chartier.com/en/"),
                _place("Septime", "https://www.septime-charonne.fr/"),
                _place("브런치: Holybelly", "https://holybellycafe.com/"),
            ],
        },
        "런던": {
            "dominant": "관광",
            "sightseeing": [
                _place("대영박물관", "https://www.britishmuseum.org/"),
                _place("타워 브리지", "https://www.towerbridge.org.uk/"),
                _place("버킹엄 궁전", "https://www.rct.uk/visit/buckingham-palace"),
                _place("코벤트 가든", "https://www.coventgarden.london/"),
            ],
            "relax": [
                _place("하이드파크", _gmaps_link("Hyde Park London")),
                _place("리젠트파크", _gmaps_link("Regent's Park")),
                _place("노팅힐 카페거리", _gmaps_link("Notting Hill cafe")),
                _place("템즈강 산책", _gmaps_link("Thames riverside walk")),
            ],
            "activities": [
                _place("웨스트엔드 뮤지컬", _gmaps_link("West End musical")),
                _place("런던아이 탑승", "https://www.londoneye.com/"),
                _place("버로우마켓 푸드투어", "https://boroughmarket.org.uk/"),
                _place("그리니치 당일 투어", _gmaps_link("Greenwich tour")),
            ],
            "restaurants": [
                _place("Dishoom Covent Garden", "https://www.dishoom.com/covent-garden/"),
                _place("Flat Iron Soho", "https://flatironsteak.co.uk/"),
                _place("Padella Borough", "https://www.padella.co/"),
                _place("브런치: Duck & Waffle", "https://duckandwaffle.com/"),
            ],
        },
        "뉴욕": {
            "dominant": "관광",
            "sightseeing": [
                _place("센트럴파크", "https://www.centralparknyc.org/"),
                _place("메트로폴리탄 미술관", "https://www.metmuseum.org/"),
                _place("타임스스퀘어", "https://www.timessquarenyc.org/"),
                _place("브루클린 브리지", _gmaps_link("Brooklyn Bridge")),
            ],
            "relax": [
                _place("브라이언트파크", _gmaps_link("Bryant Park")),
                _place("하이라인 파크", "https://www.thehighline.org/"),
                _place("소호 카페거리", _gmaps_link("Soho NYC cafe")),
                _place("허드슨 리버파크", _gmaps_link("Hudson River Park")),
            ],
            "activities": [
                _place("자유의여신상 페리", "https://www.statuecitycruises.com/"),
                _place("브로드웨이 공연", _gmaps_link("Broadway show")),
                _place("양키스타디움/메츠 경기", _gmaps_link("New York baseball game")),
                _place("로어맨해튼 워킹투어", _gmaps_link("Lower Manhattan walking tour")),
            ],
            "restaurants": [
                _place("Katz's Delicatessen", "https://katzsdelicatessen.com/"),
                _place("Joe's Pizza", "https://joespizzanyc.com/"),
                _place("Keens Steakhouse", "https://www.keens.com/"),
                _place("브런치: Balthazar", "https://balthazarny.com/"),
            ],
        },
        "싱가포르": {
            "dominant": "관광",
            "sightseeing": [
                _place("가든스 바이 더 베이", "https://www.gardensbythebay.com.sg/"),
                _place("마리나 베이 샌즈 스카이파크", "https://www.marinabaysands.com/"),
                _place("머라이언 파크", _gmaps_link("Merlion Park")),
                _place("차이나타운", _gmaps_link("Chinatown Singapore")),
            ],
            "relax": [
                _place("보타닉 가든", "https://www.nparks.gov.sg/sbg"),
                _place("이스트코스트 파크", _gmaps_link("East Coast Park Singapore")),
                _place("티옹바루 카페거리", _gmaps_link("Tiong Bahru cafe")),
                _place("센토사 비치", _gmaps_link("Sentosa beach")),
            ],
            "activities": [
                _place("나이트 사파리", "https://www.mandai.com/en/night-safari.html"),
                _place("싱가포르 리버 크루즈", _gmaps_link("Singapore river cruise")),
                _place("호커센터 미식투어", _gmaps_link("hawker food tour singapore")),
                _place("주얼 창이 실내 폭포", "https://www.jewelchangiairport.com/"),
            ],
            "restaurants": [
                _place("Jumbo Seafood", "https://www.jumboseafood.com.sg/"),
                _place("Lau Pa Sat", "https://www.laupasat.sg/"),
                _place("Din Tai Fung", "https://www.dintaifung.com.sg/"),
                _place("브런치: Common Man Coffee Roasters", "https://commonmancoffeeroasters.com/"),
            ],
        },
        "방콕": {
            "dominant": "관광",
            "sightseeing": [
                _place("왕궁/왓 프라깨우", _gmaps_link("Grand Palace Bangkok")),
                _place("왓 아룬", _gmaps_link("Wat Arun")),
                _place("왓 포", _gmaps_link("Wat Pho")),
                _place("짜뚜짝 시장", _gmaps_link("Chatuchak Market")),
            ],
            "relax": [
                _place("룸피니 공원", _gmaps_link("Lumphini Park")),
                _place("짜오프라야 강변 산책", _gmaps_link("Chao Phraya riverside")),
                _place("아리 카페거리", _gmaps_link("Ari cafe Bangkok")),
                _place("벤짜끼띠 공원", _gmaps_link("Benjakitti Park")),
            ],
            "activities": [
                _place("짜오프라야 디너 크루즈", _gmaps_link("Chao Phraya dinner cruise")),
                _place("툭툭 나이트투어", _gmaps_link("Bangkok tuk tuk tour")),
                _place("태국 쿠킹 클래스", _gmaps_link("Bangkok cooking class")),
                _place("마사지/스파 체험", _gmaps_link("Bangkok spa")),
            ],
            "restaurants": [
                _place("Thipsamai Pad Thai", _gmaps_link("Thipsamai")),
                _place("Sorn", _gmaps_link("Sorn Bangkok")),
                _place("Jay Fai", _gmaps_link("Jay Fai")),
                _place("브런치: Roast", _gmaps_link("Roast Bangkok")),
            ],
        },
        "바르셀로나": {
            "dominant": "관광",
            "sightseeing": [
                _place("사그라다 파밀리아", "https://sagradafamilia.org/en/"),
                _place("구엘 공원", "https://parkguell.barcelona/en"),
                _place("고딕 지구", _gmaps_link("Gothic Quarter Barcelona")),
                _place("카사 바트요", "https://www.casabatllo.es/en/"),
            ],
            "relax": [
                _place("바르셀로네타 해변", _gmaps_link("Barceloneta Beach")),
                _place("시우타데야 공원", _gmaps_link("Parc de la Ciutadella")),
                _place("엘 보른 카페거리", _gmaps_link("El Born cafe")),
                _place("벙커스 전망포인트", _gmaps_link("Bunkers del Carmel")),
            ],
            "activities": [
                _place("몬주익 케이블카", _gmaps_link("Montjuic cable car")),
                _place("타파스 투어", _gmaps_link("Barcelona tapas tour")),
                _place("캄프누 스타디움 투어", "https://www.fcbarcelona.com/en/tickets/camp-nou-experience"),
                _place("해변 자전거 투어", _gmaps_link("Barcelona bike tour beach")),
            ],
            "restaurants": [
                _place("Cerveceria Catalana", _gmaps_link("Cerveceria Catalana")),
                _place("Disfrutar", "https://www.disfrutarbarcelona.com/"),
                _place("Can Culleretes", _gmaps_link("Can Culleretes")),
                _place("브런치: Brunch & Cake", "https://brunchandcake.com/"),
            ],
        },
        "로마": {
            "dominant": "관광",
            "sightseeing": [
                _place("콜로세움", "https://colosseo.it/en/"),
                _place("바티칸 박물관", "https://www.museivaticani.va/"),
                _place("트레비 분수", _gmaps_link("Trevi Fountain")),
                _place("판테온", _gmaps_link("Pantheon Rome")),
            ],
            "relax": [
                _place("보르게세 공원", _gmaps_link("Villa Borghese")),
                _place("트라스테베레 산책", _gmaps_link("Trastevere")),
                _place("테라스 카페", _gmaps_link("Rome terrace cafe")),
                _place("티베르 강변 산책", _gmaps_link("Tiber riverside")),
            ],
            "activities": [
                _place("콜로세움 야간 투어", _gmaps_link("Colosseum night tour")),
                _place("바티칸 가이드 투어", _gmaps_link("Vatican guided tour")),
                _place("로마 쿠킹 클래스", _gmaps_link("Rome cooking class")),
                _place("전동카트 시티투어", _gmaps_link("Rome golf cart tour")),
            ],
            "restaurants": [
                _place("Roscioli", "https://www.salumeriaroscioli.com/"),
                _place("Armando al Pantheon", _gmaps_link("Armando al Pantheon")),
                _place("Da Enzo al 29", _gmaps_link("Da Enzo al 29")),
                _place("브런치: Marigold Roma", _gmaps_link("Marigold Roma")),
            ],
        },
    }
    return profiles.get(k, _generic_city_profile(city_name))


def _build_relax_activities(
    city_name: str,
    country: str,
    activity_pref: ActivityPref,
    avg_temp_c: float,
    precip_ratio: float,
) -> list[dict]:
    grade = _outdoor_grade(precip_ratio)
    used: set[str] = set()
    base_reason = _activity_reason(precip_ratio, avg_temp_c, activity_kind="휴양")

    profile = _city_profile(city_name)
    prof_relax = _enrich_place_list(city_name, "휴양", profile["relax"])
    outdoor_templates = prof_relax + [
        _place(f"{city_name} 시티파크", _gmaps_link(f"{city_name} city park")),
        _place(f"{city_name} 워터프론트", _gmaps_link(f"{city_name} waterfront")),
        _place(f"{city_name} 선셋 포인트", _gmaps_link(f"{city_name} sunset point")),
        _place(f"{city_name} 브런치 카페", _gmaps_link(f"{city_name} brunch cafe")),
    ]
    indoor_templates: list[dict[str, str | None]] = [
        _place(f"{city_name} 시립미술관", _gmaps_link(f"{city_name} art museum")),
        _place(f"{city_name} 박물관", _gmaps_link(f"{city_name} museum")),
        _place(f"{city_name} 실내 마켓", _gmaps_link(f"{city_name} indoor market")),
        _place(f"{city_name} 쿠킹 스튜디오", _gmaps_link(f"{city_name} cooking class")),
    ]
    mixed_templates: list[dict[str, str | None]] = [
        _place(f"{city_name} 하이라이트 산책", _gmaps_link(f"{city_name} highlights walk")),
        _place(f"{city_name} 아트센터", _gmaps_link(f"{city_name} art center")),
        _place(f"{city_name} 마켓 거리", _gmaps_link(f"{city_name} market street")),
        _place(f"{city_name} 전망 포인트", _gmaps_link(f"{city_name} viewpoint")),
    ]

    if activity_pref == "야외 우선":
        candidates = outdoor_templates if grade != "실내 우선" else mixed_templates + indoor_templates
    elif activity_pref == "실내 우선":
        candidates = indoor_templates if grade != "야외 적합" else mixed_templates + outdoor_templates
    else:
        if grade == "야외 적합":
            candidates = outdoor_templates + mixed_templates
        elif grade == "실내 우선":
            candidates = indoor_templates + mixed_templates
        else:
            candidates = mixed_templates + outdoor_templates + indoor_templates

    picks = _pick_place_templates(candidates, used=used, k=4)
    return [
        {
            "activity_type": "휴양",
            "title": str(p.get("name", "")),
            "reason": base_reason,
            "weather_fit": "야외/실내 유연(강수 대응 포함)",
            "link": p.get("link"),
        }
        for p in picks
    ]


def _build_sightseeing_activities(
    city_name: str,
    country: str,
    activity_pref: ActivityPref,
    avg_temp_c: float,
    precip_ratio: float,
) -> list[dict]:
    grade = _outdoor_grade(precip_ratio)
    used: set[str] = set()
    base_reason = _activity_reason(precip_ratio, avg_temp_c, activity_kind="관광")

    profile = _city_profile(city_name)
    prof_sight = _enrich_place_list(city_name, "관광", profile["sightseeing"])
    outdoor_templates = prof_sight + [
        _place(f"{city_name} 랜드마크", _gmaps_link(f"{city_name} landmark")),
        _place(f"{city_name} 역사 지구", _gmaps_link(f"{city_name} historic district")),
        _place(f"{city_name} 스카이 전망대", _gmaps_link(f"{city_name} sky observatory")),
        _place(f"{city_name} 야경 스팟", _gmaps_link(f"{city_name} night view point")),
    ]
    indoor_templates: list[dict[str, str | None]] = [
        _place(f"{city_name} 국립박물관", _gmaps_link(f"{city_name} national museum")),
        _place(f"{city_name} 현대미술관", _gmaps_link(f"{city_name} modern art museum")),
        _place(f"{city_name} 실내 쇼핑몰", _gmaps_link(f"{city_name} mall")),
        _place(f"{city_name} 테마 전시관", _gmaps_link(f"{city_name} exhibition hall")),
    ]
    mixed_templates: list[dict[str, str | None]] = [
        _place(f"{city_name} 랜드마크 투어", _gmaps_link(f"{city_name} landmark tour")),
        _place(f"{city_name} 전망대", _gmaps_link(f"{city_name} observation deck")),
        _place(f"{city_name} 도심 투어", _gmaps_link(f"{city_name} downtown tour")),
        _place(f"{city_name} 실내 명소", _gmaps_link(f"{city_name} indoor attractions")),
    ]

    if activity_pref == "야외 우선":
        candidates = outdoor_templates if grade != "실내 우선" else mixed_templates + indoor_templates
    elif activity_pref == "실내 우선":
        candidates = indoor_templates if grade != "야외 적합" else mixed_templates + outdoor_templates
    else:
        if grade == "야외 적합":
            candidates = outdoor_templates + mixed_templates
        elif grade == "실내 우선":
            candidates = indoor_templates + mixed_templates
        else:
            candidates = mixed_templates + outdoor_templates + indoor_templates

    picks = _pick_place_templates(candidates, used=used, k=4)
    return [
        {
            "activity_type": "관광",
            "title": str(p.get("name", "")),
            "reason": base_reason,
            "weather_fit": "우천 대체 포함",
            "link": p.get("link"),
        }
        for p in picks
    ]


def _foods_for_iso2(iso2: str, city_name: str) -> list[dict[str, str | None]]:
    # MVP: 실제 "유명 음식"을 Wikidata로 뽑지 못하므로 국가 기반 대표 예시(대표성을 우선)
    # iso2가 빈 값이면 공통 템플릿 사용.
    mapping: dict[str, list[dict[str, str | None]]] = {
        "VN": [_place("퍼(Pho)", _gmaps_link(f"{city_name} pho")), _place("반미(Banh Mi)", _gmaps_link(f"{city_name} banh mi")), _place("분짜(Bun Cha)", _gmaps_link(f"{city_name} bun cha")), _place("월남쌈(Goi Cuon)", _gmaps_link(f"{city_name} goi cuon"))],
        "TH": [_place("팟타이", _gmaps_link(f"{city_name} pad thai")), _place("똠얌꿍", _gmaps_link(f"{city_name} tom yum")), _place("쏨땀", _gmaps_link(f"{city_name} som tam")), _place("망고 스티키 라이스", _gmaps_link(f"{city_name} mango sticky rice"))],
        "ID": [_place("나시고랭", _gmaps_link(f"{city_name} nasi goreng")), _place("사테", _gmaps_link(f"{city_name} satay")), _place("렌당", _gmaps_link(f"{city_name} rendang")), _place("가도가도", _gmaps_link(f"{city_name} gado gado"))],
        "JP": [_place("스시", _gmaps_link(f"{city_name} sushi")), _place("라멘", _gmaps_link(f"{city_name} ramen")), _place("오코노미야키", _gmaps_link(f"{city_name} okonomiyaki")), _place("타코야키", _gmaps_link(f"{city_name} takoyaki"))],
        "KR": [_place("비빔밥", _gmaps_link(f"{city_name} bibimbap")), _place("불고기", _gmaps_link(f"{city_name} bulgogi")), _place("김치찌개", _gmaps_link(f"{city_name} kimchi jjigae")), _place("떡볶이", _gmaps_link(f"{city_name} tteokbokki"))],
        "CN": [_place("베이징덕", _gmaps_link(f"{city_name} peking duck")), _place("훠궈", _gmaps_link(f"{city_name} hot pot")), _place("마파두부", _gmaps_link(f"{city_name} mapo tofu")), _place("만두", _gmaps_link(f"{city_name} jiaozi"))],
        "IT": [_place("마르게리타 피자", _gmaps_link(f"{city_name} margherita pizza")), _place("카르보나라", _gmaps_link(f"{city_name} carbonara")), _place("티라미수", _gmaps_link(f"{city_name} tiramisu")), _place("젤라토", _gmaps_link(f"{city_name} gelato"))],
        "FR": [_place("크루아상", _gmaps_link(f"{city_name} croissant")), _place("크레페", _gmaps_link(f"{city_name} crepe")), _place("키시 로렌", _gmaps_link(f"{city_name} quiche lorraine")), _place("라따뚜이", _gmaps_link(f"{city_name} ratatouille"))],
        "US": [_place("버거", _gmaps_link(f"{city_name} burger")), _place("바비큐 립", _gmaps_link(f"{city_name} bbq ribs")), _place("클램 차우더", _gmaps_link(f"{city_name} clam chowder")), _place("키 라임 파이", _gmaps_link(f"{city_name} key lime pie"))],
        "GB": [_place("피시 앤 칩스", _gmaps_link(f"{city_name} fish and chips")), _place("셰퍼드 파이", _gmaps_link(f"{city_name} shepherd pie")), _place("풀 잉글리시 브렉퍼스트", _gmaps_link(f"{city_name} english breakfast")), _place("스티키 토피 푸딩", _gmaps_link(f"{city_name} sticky toffee pudding"))],
        "ES": [_place("빠에야", _gmaps_link(f"{city_name} paella")), _place("가스파초", _gmaps_link(f"{city_name} gazpacho")), _place("하몽", _gmaps_link(f"{city_name} jamon")), _place("토르티야 에스파뇰라", _gmaps_link(f"{city_name} tortilla espanola"))],
        "TR": [_place("도너 케밥", _gmaps_link(f"{city_name} doner kebab")), _place("라흐마준", _gmaps_link(f"{city_name} lahmacun")), _place("메제", _gmaps_link(f"{city_name} meze")), _place("바클라바", _gmaps_link(f"{city_name} baklava"))],
        "MX": [_place("타코스", _gmaps_link(f"{city_name} tacos")), _place("추로스", _gmaps_link(f"{city_name} churros")), _place("몰레 포블라노", _gmaps_link(f"{city_name} mole poblano")), _place("과카몰리", _gmaps_link(f"{city_name} guacamole"))],
    }

    iso2 = (iso2 or "").upper().strip()
    if iso2 in mapping:
        return _enrich_place_list(city_name, "음식", mapping[iso2])

    # fallback: 모든 국가에서 구체 메뉴명 + 링크 제공
    return _enrich_place_list(city_name, "음식", [
        _place(f"{city_name} 시그니처 누들", _gmaps_link(f"{city_name} noodle")),
        _place(f"{city_name} 대표 스튜", _gmaps_link(f"{city_name} stew")),
        _place(f"{city_name} 로컬 스트리트푸드", _gmaps_link(f"{city_name} street food")),
        _place(f"{city_name} 시푸드 메뉴", _gmaps_link(f"{city_name} seafood")),
        _place(f"{city_name} 로컬 디저트", _gmaps_link(f"{city_name} dessert")),
        _place(f"{city_name} 그릴/바비큐", _gmaps_link(f"{city_name} bbq")),
        _place(f"{city_name} 베이커리", _gmaps_link(f"{city_name} bakery")),
        _place(f"{city_name} 브런치 플레이트", _gmaps_link(f"{city_name} brunch")),
    ])


def _restaurant_recommendations(city_name: str, iso2: str) -> list[dict[str, str | None]]:
    profile = _city_profile(city_name)
    if profile["restaurants"]:
        return _enrich_place_list(city_name, "식당", profile["restaurants"][:4])
    fallback: dict[str, list[dict[str, str | None]]] = {
        "AU": [_place("Bills Sydney", "https://www.billssydney.com.au/"), _place("Single O", "https://singleo.com.au/"), _place("Mr. Wong", "https://merivale.com/venues/mrwong/"), _place("The Apollo", "https://theapollo.com.au/")],
        "JP": [_place("스시다이(도요스)", "https://www.toyosu-market.or.jp/en/"), _place("아후리 라멘", "https://afuri.com/"), _place("규카츠 모토무라", "https://www.gyukatsu-motomura.com/en/"), _place("우나기 히츠마부시 비노", "https://www.hitsumabushi-bino.com/")],
        "KR": [_place("광장시장 먹거리", "https://map.naver.com/"), _place("을지로 노포 맛집", "https://map.naver.com/"), _place("익선동 한식 레스토랑", "https://map.naver.com/"), _place("연남동 브런치 카페", "https://map.naver.com/")],
        "US": [_place("로컬 다이너 브런치", "https://www.google.com/maps"), _place("스테이크 하우스", "https://www.google.com/maps"), _place("루프탑 바", "https://www.google.com/maps"), _place("아시안 퓨전 레스토랑", "https://www.google.com/maps")],
    }
    return _enrich_place_list(
        city_name,
        "식당",
        fallback.get(
        (iso2 or "").upper().strip(),
        [
            _place(f"{city_name} 비스트로", _gmaps_link(f"{city_name} bistro")),
            _place(f"{city_name} 브런치 카페", _gmaps_link(f"{city_name} brunch cafe")),
            _place(f"{city_name} 시푸드 레스토랑", _gmaps_link(f"{city_name} seafood restaurant")),
            _place(f"{city_name} 뷰 다이닝", _gmaps_link(f"{city_name} view dining")),
        ],
        ),
    )


def _build_day_plan(
    activity_pref: ActivityPref,
    grade: Literal["야외 적합", "혼합 추천", "실내 우선"],
    avg_temp_c: float,
    precip_ratio: float,
) -> dict[str, dict]:
    # 시간대 고정(PRD)
    plan = {
        "오전": {},
        "오후": {},
        "저녁": {},
    }

    # 강수 대응 문구(짧게)
    if grade == "야외 적합":
        fallback = "비가 오면: 카페/전시 30~60분으로 전환"
    elif grade == "혼합 추천":
        fallback = "비가 오면: 실내 관람 중심으로 순서 변경"
    else:
        fallback = "비가 오면: 실내 일정으로 완전 전환"

    # 활동 선호에 따라 야외/실내 우선 템플릿 선택
    if activity_pref == "야외 우선":
        plan["오전"] = {"title": "야외 산책/전망 포인트", "desc": "날씨 좋은 시간대에 가볍게 걷고 사진 포인트를 먼저 찍어요."}
        plan["오후"] = {"title": "도보+마켓(실내 대체 포함)", "desc": "야외는 짧게, 비가 오면 실내 전시/카페로 옮겨요."}
        plan["저녁"] = {"title": "선셋/야경 느낌", "desc": "가벼운 저녁 산책 또는 실내 관람으로 마무리합니다."}
    elif activity_pref == "실내 우선":
        plan["오전"] = {"title": "박물관/전시 하이라이트", "desc": "실내에서 핵심만 빠르게 보고 이동 시간을 줄입니다."}
        plan["오후"] = {"title": "카페+로컬 마켓(실내 우선)", "desc": "실내 구역 중심으로 즐기고 기상 변화에 맞춰 조정해요."}
        plan["저녁"] = {"title": "디저트/야경(전망 포인트 대체 가능)", "desc": "비가 오면 야경 대신 실내 관람/식사로 자연스럽게 변경합니다."}
    else:
        plan["오전"] = {"title": "야외 산책(가벼운 코스)", "desc": "짧게라도 야외를 넣어 만족도를 높이는 구성이에요."}
        plan["오후"] = {"title": "야외+실내 혼합(비 대체 포함)", "desc": "기후가 변하면 실내 관람으로 전환해 동선을 유지합니다."}
        plan["저녁"] = {"title": "전망/야경 또는 실내 전시 마무리", "desc": "저녁은 걷기 난이도보다 만족도가 큰 선택지를 우선합니다."}

    for k in plan.keys():
        plan[k]["fallback"] = fallback

    return plan


def _recommended_trip_days(
    *,
    precip_ratio: float,
    avg_temp_c: float,
    dominant: str,
    activity_pref: ActivityPref,
) -> int:
    """도시 기후/성향 기준으로 추천 여행일수(3~6일)."""
    days = 4
    if dominant == "관광":
        days += 1
    elif dominant == "휴양":
        days -= 1

    if not math.isnan(precip_ratio):
        if precip_ratio <= 0.15:
            days += 1
        elif precip_ratio >= 0.35:
            days -= 1
    if not math.isnan(avg_temp_c) and (avg_temp_c < 8 or avg_temp_c > 32):
        days -= 1

    if activity_pref == "야외 우선" and not math.isnan(precip_ratio) and precip_ratio <= 0.20:
        days += 1
    if activity_pref == "실내 우선":
        days -= 1
    return max(3, min(6, days))


def _build_trip_itinerary(
    *,
    city_name: str,
    trip_days: int,
    trip_focus: Literal["관광위주", "휴양위주"],
    relax: list[dict],
    sightseeing: list[dict],
    activities: list[str],
    restaurants: list[str],
    airport_label: str,
) -> list[dict[str, Any]]:
    """추천 리스트를 활용해 일차별 코스 생성(첫날/마지막날 반나절)."""
    relax_titles = [x.get("title", "") for x in relax if x.get("title")]
    sight_titles = [x.get("title", "") for x in sightseeing if x.get("title")]
    act_titles = activities[:]
    rest_titles = restaurants[:]

    def pick(seq: list[str], idx: int, fallback: str) -> str:
        if not seq:
            return fallback
        return seq[idx % len(seq)]

    days: list[dict[str, Any]] = []
    for d in range(1, trip_days + 1):
        is_first = d == 1
        is_last = d == trip_days
        half_day = is_first or is_last

        if trip_focus == "관광위주":
            primary = pick(sight_titles, d - 1, "대표 랜드마크 탐방")
            secondary = pick(relax_titles, d - 1, "워터프론트 산책")
        else:
            primary = pick(relax_titles, d - 1, "도시 공원 여유 산책")
            secondary = pick(sight_titles, d - 1, "핵심 명소 1곳 방문")

        activity = pick(act_titles, d - 1, f"{city_name} 로컬 체험")
        restaurant = pick(rest_titles, d - 1, "현지 인기 레스토랑")

        if half_day:
            if is_first:
                slots = [f"{airport_label} IN", f"{primary}", f"{restaurant}"]
            else:
                slots = [f"{primary}", f"{restaurant}", f"{airport_label} OUT"]
        else:
            slots = [
                f"{primary}",
                f"{activity}",
                f"{secondary}",
                f"{restaurant}",
            ]
        days.append({"day": d, "half_day": half_day, "slots": slots})
    return days


def _city_airport_label(city_name: str, country_name: str, iso2: str) -> str:
    k = _city_key(city_name)
    m = {
        "시드니": "시드니 킹스포드 스미스 국제공항(SYD)",
        "도쿄": "도쿄 하네다/나리타 국제공항(HND/NRT)",
        "오사카": "오사카 간사이 국제공항(KIX)",
        "서울": "인천국제공항(ICN)",
        "부산": "김해국제공항(PUS)",
        "싱가포르": "창이 국제공항(SIN)",
        "방콕": "수완나품 국제공항(BKK)",
        "발리": "응우라라이 국제공항(DPS)",
        "파리": "파리 샤를드골 국제공항(CDG)",
        "런던": "런던 히드로 공항(LHR)",
        "뉴욕": "뉴욕 JFK 국제공항(JFK)",
        "로스앤젤레스": "LA 국제공항(LAX)",
        "두바이": "두바이 국제공항(DXB)",
    }
    if k in m:
        return m[k]
    ctry_ko = _country_name_ko(country_name)
    return f"{city_name} 대표 국제공항"


def _build_itinerary_points(
    *,
    city_name: str,
    city_lat: float,
    city_lng: float,
    itinerary: list[dict[str, Any]],
) -> pd.DataFrame:
    """코스 순서대로 지도에 찍을 이동 포인트 생성(고정 좌표 우선, 실패 시 결정적 오프셋)."""

    def _city_airport_coords(city_key: str) -> tuple[float, float] | None:
        airports: dict[str, tuple[float, float]] = {
            "시드니": (-33.9399, 151.1753),
            "도쿄": (35.5494, 139.7798),  # 하네다
            "오사카": (34.7855, 135.4384),  # 간사이
            "서울": (37.4602, 126.4407),  # 인천
            "부산": (35.1796, 129.0756),  # 김해
            "싱가포르": (1.3644, 103.9915),  # 창이
            "방콕": (13.6900, 100.7501),  # 수완나품
            "발리": (-8.7468, 115.1689),  # 응우라라이
            "파리": (49.0097, 2.5479),  # 샤를드골
            "런던": (51.4700, -0.4543),  # 히드로
            "뉴욕": (40.6413, -73.7781),  # JFK
            "로스앤젤레스": (33.9416, -118.4085),  # LAX
            "두바이": (25.2532, 55.3657),  # DXB
            "이스탄불": (41.2754, 28.7519),  # (IST) 근사
            "홍콩": (22.3080, 113.9185),  # HKG 근사
        }
        return airports.get(city_key)

    def _fixed_coords(city_key: str, slot: str) -> tuple[float, float] | None:
        s = str(slot)
        # 공항 IN/OUT은 도시 공항 고정좌표 사용
        if "국제공항" in s or s.endswith(" IN") or s.endswith(" OUT"):
            ac = _city_airport_coords(city_key)
            if ac:
                return ac

        patterns: dict[str, list[tuple[str, tuple[float, float]]]] = {
            "시드니": [
                ("시드니 하버 브리지", (-33.8523, 151.2108)),
                ("하버 브리지", (-33.8523, 151.2108)),
                ("시드니 오페라하우스", (-33.8568, 151.2153)),
                ("오페라하우스", (-33.8568, 151.2153)),
                ("더 록스", (-33.8599, 151.2100)),
                ("본다이", (-33.8909, 151.2743)),
                ("로열 보타닉 가든", (-33.8645, 151.2163)),
                ("달링 하버", (-33.8709, 151.1986)),
                ("맨리 비치", (-33.7993, 151.2872)),
                ("서큘러 키", (-33.8615, 151.2111)),
                ("서큘러", (-33.8615, 151.2111)),
                ("Bennelong", (-33.8563, 151.2157)),
                ("Quay", (-33.8749, 151.1996)),
                ("The Grounds", (-33.8930, 151.2040)),
                ("Single O", (-33.8849, 151.2070)),
            ],
            "도쿄": [
                ("센소지", (35.7148, 139.7967)),
                ("스카이트리", (35.7101, 139.8107)),
                ("메이지 신궁", (35.6764, 139.7016)),
                ("우에노", (35.7153, 139.7730)),
                ("요요기 공원", (35.6717, 139.5650)),
                ("가구라자카", (35.6942, 139.7380)),
                ("오다이바", (35.6290, 139.7753)),
                ("스미다강", (35.7100, 139.7985)),
                ("teamLab", (35.6288, 139.7907)),
                ("쓰키지", (35.6668, 139.7706)),
                ("도요스", (35.6646, 139.7909)),
                ("도쿄 베이", (35.6351, 139.7737)),
                ("이치란", (35.6707, 139.7704)),
            ],
            "발리": [
                ("우붓 왕궁", (-8.5069, 115.2625)),
                ("우붓", (-8.5069, 115.2625)),
                ("울루와뚜", (-8.8326, 115.0916)),
                ("뜨그눙안", (-8.6928, 115.2384)),
                ("스미냑", (-8.6886, 115.1660)),
                ("라이스 테라스", (-8.5110, 115.2600)),
                ("응우라라이", (-8.7468, 115.1689)),
            ],
            "서울": [
                ("경복궁", (37.5796, 126.9770)),
                ("북촌", (37.5839, 126.9827)),
                ("N서울타워", (37.5512, 126.9882)),
                ("국립중앙박물관", (37.5051, 126.9817)),
                ("서울숲", (37.5426, 127.0406)),
                ("한강공원", (37.5222, 126.9321)),
                ("광장시장", (37.5700, 126.9920)),
                ("몽탄", (37.5310, 127.0070)),
                ("DMZ", (38.0330, 126.7420)),
                ("김해", (35.1796, 129.0756)),
                ("인천", (37.4602, 126.4407)),
            ],
        }

        for pat, coord in patterns.get(city_key, []):
            if pat in s:
                return coord
        return None

    def _deterministic_offset(slot: str) -> tuple[float, float]:
        # slot 문자열 해시로 결정적 오프셋 생성(완전 랜덤 아님)
        h = hashlib.md5(str(slot).encode("utf-8")).hexdigest()
        a = int(h[:8], 16) % 360
        # 최대 약 12km 내(0.108도 내외)
        r_deg_lat = 12.0 / 111.0
        rad = math.radians(a)
        lat = city_lat + r_deg_lat * math.sin(rad)
        lng = city_lng + (r_deg_lat * math.cos(rad)) / max(0.3, math.cos(math.radians(city_lat)))
        return lat, lng

    def _geocode_slot_coords(slot: str) -> tuple[float, float] | None:
        """고정 좌표가 없을 때 slot+도시로 지오코딩 후, 도시 중심에서 가까운 후보만 채택."""
        # 지오코딩 잡음을 줄이기 위해 IN/OUT, 괄호, 기호 등을 제거
        cleaned = re.sub(r"\(.*?\)", " ", str(slot))
        cleaned = re.sub(r"\b(IN|OUT)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("·", " ").replace("→", " ").replace("->", " ")
        cleaned = re.sub(r"[^0-9A-Za-z가-힣\s/+-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        query = f"{cleaned} {city_name}".strip()
        try:
            geo = _geocode_open_meteo_cached(query, language="ko")
        except Exception:
            return None
        if not geo:
            return None
        best: tuple[float, float] | None = None
        best_score = float("-inf")
        cleaned_norm = cleaned.lower()
        cleaned_norm = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", cleaned_norm)
        tokens = [t for t in re.split(r"\s+", cleaned_norm) if t]
        for g in geo[:10]:
            try:
                glat = float(g.get("latitude"))
                glng = float(g.get("longitude"))
            except Exception:
                continue
            if math.isnan(glat) or math.isnan(glng):
                continue
            d = _haversine_km(city_lat, city_lng, glat, glng)
            # 도시 중심에서 너무 먼 후보는 오탐 가능성이 커서 제외
            if d > 80.0:
                continue
            # 후보 이름 기반 간단 토큰 유사도
            cand_name = str(g.get("name", "") or "").lower()
            cand_name = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", cand_name)
            token_hits = sum(1 for t in tokens if t and t in cand_name)
            sim = token_hits / max(1, len(tokens))

            # 거리 페널티 + 유사도 보상
            dist_norm = max(0.0, 1.0 - d / 40.0)  # 0km->1, 40km->0
            score = dist_norm * 0.7 + sim * 0.3
            if score > best_score:
                best_score = score
                best = (glat, glng)

        # 너무 멀면 오탐 방지(기본 40km 타이트)
        if best is None:
            return None
        return best
    points: list[dict[str, Any]] = []
    step = 1
    city_key = _city_key(city_name)
    for d in itinerary:
        day_step = 1
        for si, s in enumerate(d["slots"]):
            fixed = _fixed_coords(city_key, s)
            if fixed:
                lat, lng = fixed
            else:
                geo = _geocode_slot_coords(s)
                if geo:
                    lat, lng = geo
                else:
                    lat, lng = _deterministic_offset(s)
            points.append(
                {
                    "순서": step,
                    "일차순서": day_step,
                    "day": int(d["day"]),
                    "일차": f"{d['day']}일차",
                    "일정": s,
                    "lat": lat,
                    "lon": lng,
                    "size": 40 + (si * 5),
                }
            )
            step += 1
            day_step += 1
    return pd.DataFrame(points)


def _build_detail_section(
    city_row: pd.Series,
    active_month: int,
    activity_pref: ActivityPref,
    monthly_row: pd.Series,
) -> None:
    avg_temp_c = float(monthly_row.get("avg_temp_c", float("nan")))
    precip_days = int(monthly_row.get("precip_days", 0))
    days_in_month = int(monthly_row.get("days_in_month_observed", 0)) or 0
    precip_ratio = (precip_days / days_in_month) if days_in_month else float("nan")
    cloudy_days = int(monthly_row.get("cloudy_days", 0))
    cloud_ratio = (cloudy_days / days_in_month) if days_in_month else 0.0

    grade = _outdoor_grade(precip_ratio)

    city_name_ko = _city_name_ko(str(city_row["city_name"]), str(city_row["country"]))
    st.subheader(f"상세: {city_name_ko} · {active_month}월")
    st.caption(
        f"강수일수: {precip_days}/{days_in_month if days_in_month else '-'}일 "
        f"({precip_ratio*100:.1f}% 내외) · 흐림참고: {cloudy_days} ({int(round(cloud_ratio*100))}%) · 추천근거: 온도+강수일수"
    )
    st.info(f"이번 달 기후 적합도: {grade} (액티비티/코스는 이 기준을 반영합니다)")

    # 상세 요약 문구는 선택된 성향(야외 우선=휴양 중심, 그 외=관광 중심)에 맞춰 표현
    reason_kind: Literal["휴양", "관광"] = "휴양" if activity_pref == "야외 우선" else "관광"
    reason = _activity_reason(precip_ratio, avg_temp_c, activity_kind=reason_kind)
    profile = _city_profile(city_name_ko)
    city_activities = _enrich_place_list(city_name_ko, "액티비티", profile.get("activities", []) or [])

    # 휴양/관광 장소 카드(여기서는 MVP용 아이디어 생성)
    relax = _build_relax_activities(
        city_name=city_name_ko,
        country=str(city_row["country"]),
        activity_pref=activity_pref,
        avg_temp_c=avg_temp_c,
        precip_ratio=precip_ratio,
    )
    sightseeing = _build_sightseeing_activities(
        city_name=city_name_ko,
        country=str(city_row["country"]),
        activity_pref=activity_pref,
        avg_temp_c=avg_temp_c,
        precip_ratio=precip_ratio,
    )

    foods = _foods_for_iso2(str(city_row.get("iso2", "")), city_name_ko)
    restaurants = _restaurant_recommendations(city_name_ko, str(city_row.get("iso2", "")))

    # 도시 성향 우선 추천 문구
    def _render_place(title: str, link: Any, desc: Any = None, *, menu: Any = None, activity: Any = None, image: Any = None) -> None:
        t = str(title)
        lnk = str(link) if isinstance(link, str) and link.strip() else None
        if lnk:
            st.markdown(f"- [{t}]({lnk})")
        else:
            st.markdown(f"- {t}")

    if profile.get("dominant") == "관광":
        st.markdown(f"### {city_name_ko}는 **휴양보다는 관광 위주**로 즐기기 좋은 도시예요!")
        st.caption(reason)
        st.markdown("#### 관광 추천 코스")
        for a in sightseeing:
            _render_place(str(a["title"]), a.get("link"), a.get("desc"), activity=a.get("activity"), image=a.get("image"))
        st.markdown("#### 휴양을 즐기고 싶다면")
        for a in relax[:3]:
            _render_place(str(a["title"]), a.get("link"), a.get("desc"), activity=a.get("activity"), image=a.get("image"))
    elif profile.get("dominant") == "휴양":
        st.markdown(f"### {city_name_ko}는 **관광보다는 휴양 위주**로 즐기기 좋은 도시예요!")
        st.caption(reason)
        st.markdown("#### 휴양 추천 코스")
        for a in relax:
            _render_place(str(a["title"]), a.get("link"), a.get("desc"), activity=a.get("activity"), image=a.get("image"))
        st.markdown("#### 관광도 함께 하고 싶다면")
        for a in sightseeing[:3]:
            _render_place(str(a["title"]), a.get("link"), a.get("desc"), activity=a.get("activity"), image=a.get("image"))
    else:
        st.markdown(f"### {city_name_ko}는 **관광/휴양을 균형 있게** 즐기기 좋아요!")
        st.caption(reason)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 휴양 추천")
            for a in relax:
                _render_place(str(a["title"]), a.get("link"), a.get("desc"), activity=a.get("activity"), image=a.get("image"))
        with c2:
            st.markdown("#### 관광 추천")
            for a in sightseeing:
                _render_place(str(a["title"]), a.get("link"), a.get("desc"), activity=a.get("activity"), image=a.get("image"))

    st.markdown("### 유명 음식 추천 (예시)")
    fcols = st.columns(2)
    for i, food in enumerate(foods):
        col = fcols[i % 2]
        with col:
            _render_place(str(food.get("name", "")), food.get("link"), food.get("desc"), image=food.get("image"))
    st.markdown("#### 함께 가기 좋은 유명 식당 추천")
    for r in restaurants:
        _render_place(str(r.get("name", "")), r.get("link"), r.get("desc"), menu=r.get("menu"), image=r.get("image"))

    st.divider()

    # 이번 달 액티비티(휴양/관광 각 4)
    st.markdown("### 이번 달 액티비티 추천")
    if city_activities:
        for a in city_activities[:4]:
            if isinstance(a, dict):
                _render_place(str(a.get("name", "")), a.get("link"), a.get("desc"), activity=a.get("activity"), image=a.get("image"))
            else:
                st.markdown(f"- {a}")
    else:
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("#### 휴양 액티비티")
            for a in relax:
                st.write(f"- {a['title']}  _(야외/실내 유연)_")
        with a2:
            st.markdown("#### 관광 액티비티")
            for a in sightseeing:
                st.write(f"- {a['title']}  _(우천 대체 포함)_")

    st.markdown("### 여행 코스 예시")
    trip_days = _recommended_trip_days(
        precip_ratio=precip_ratio,
        avg_temp_c=avg_temp_c,
        dominant=str(profile.get("dominant", "혼합")),
        activity_pref=activity_pref,
    )
    st.markdown(f"**최적의 여행일수 추천:** 약 **{trip_days}일**")
    st.caption("1일차/마지막날은 이동 시간(반나절)을 고려해 일정을 구성합니다.")

    trip_focus = st.radio(
        "코스 성향 선택",
        ["관광위주", "휴양위주"],
        horizontal=True,
        key=f"trip_focus_{city_row['city_id']}_{active_month}",
    )
    itinerary = _build_trip_itinerary(
        city_name=city_name_ko,
        trip_days=trip_days,
        trip_focus=trip_focus,
        relax=relax,
        sightseeing=sightseeing,
        activities=[str(a.get("name")) if isinstance(a, dict) else str(a) for a in city_activities],
        restaurants=[str(r.get("name")) if isinstance(r, dict) else str(r) for r in restaurants],
        airport_label=_city_airport_label(city_name_ko, str(city_row["country"]), str(city_row.get("iso2", ""))),
    )

    st.markdown("#### 일차별 코스")
    for d in itinerary:
        label = f"{d['day']}일차"
        st.markdown(f"**{label}**")
        flow_parts: list[str] = ["<div class='itinerary-flow'>"]
        for i, s in enumerate(d["slots"]):
            safe_text = str(s).replace("<", "&lt;").replace(">", "&gt;")
            flow_parts.append(f"<div class='itinerary-node'>{safe_text}</div>")
            if i < len(d["slots"]) - 1:
                flow_parts.append("<div class='itinerary-arrow'>→</div>")
        flow_parts.append("</div>")
        st.markdown("".join(flow_parts), unsafe_allow_html=True)

    st.markdown("#### 이동 위치 지도")
    center_lat = float(city_row["lat"])
    center_lng = float(city_row["lng"])
    map_df = _build_itinerary_points(city_name=city_name_ko, city_lat=center_lat, city_lng=center_lng, itinerary=itinerary)
    day_options = [f"{d['day']}일차" for d in itinerary]
    day_pick = st.radio(
        "일자별 보기",
        day_options,
        horizontal=True,
        key=f"map_day_pick_{city_row['city_id']}_{active_month}",
    )
    day_num = int(day_pick.replace("일차", ""))
    view_df = map_df[map_df["day"] == day_num].sort_values("일차순서").copy()
    view_df["일차순서_int"] = view_df["일차순서"].astype(int)
    # 지도 라벨 가독성 개선(안 뜨는 문제 대비): ASCII 1/2/3로 고정
    view_df["표시순서"] = view_df["일차순서_int"].astype(str)

    # 1->2->3 이동선 + 번호 포인트
    path_points = view_df.copy()
    path_data = [{"path": path_points[["lon", "lat"]].values.tolist()}]
    view_state = pdk.ViewState(
        latitude=float(path_points["lat"].mean()),
        longitude=float(path_points["lon"].mean()),
        zoom=11,
        pitch=0,
    )
    deck = pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=view_state,
        layers=[
            pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_color=[37, 99, 235, 190],
                width_scale=14,
                width_min_pixels=3,
                pickable=False,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=view_df,
                get_position="[lon, lat]",
                # 하늘색 원 마커(숫자 배경)
                get_fill_color=[56, 189, 248, 235],
                get_line_color=[125, 211, 252, 255],
                get_line_width=1.2,
                get_radius=62,
                radius_units="pixels",
                radius_min_pixels=14,
                radius_max_pixels=28,
                pickable=True,
                # 마우스 오버 시 하이라이트로 반경이 크게 보이게 함
                auto_highlight=True,
                highlight_color=[37, 99, 235, 255],
                highlight_radius=120,
            ),
            pdk.Layer(
                "TextLayer",
                data=path_points,
                get_position="[lon, lat]",
                get_text="표시순서",
                # 원 안 숫자(흰색)
                get_size=20,
                get_color=[255, 255, 255, 255],
                get_alignment_baseline="'center'",
                get_text_anchor="'middle'",
                pickable=False,
                background=False,
                get_pixel_offset=[0, 0],
            ),
        ],
        tooltip={"text": "{순서}. {일차}\n{일정}"},
    )
    try:
        st.pydeck_chart(deck, use_container_width=True)
    except Exception:
        st.map(view_df[["lat", "lon"]], use_container_width=True)
    st.dataframe(
        view_df[["일차순서", "일차", "일정"]]
        .rename(columns={"일차순서": "일차 내 순서"})
        .copy(),
        use_container_width=True,
        hide_index=True,
        height=min(340, 90 + 30 * len(view_df)),
    )

    st.caption("면책: 본 대시보드는 기후 데이터 기반 아이디어 추천이며 실제 날씨/현장 운영은 별도 확인이 필요합니다.")


def main() -> None:
    st.set_page_config(
        page_title="당신의 여행계획을 세워드립니다!",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
          @font-face {
            font-family: 'BMJUA';
            src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_one@1.0/BMJUA.woff') format('woff');
            font-weight: normal;
            font-display: swap;
          }
          /* 사이드바·헤더·툴바는 기본 폰트(BMJUA 없음)로 두어 열기/닫기·아이콘 주변이 깨지지 않게 함 */
          .stApp {
            background: #ffffff;
            font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", "Malgun Gothic", sans-serif;
          }
          section.main,
          section.main * {
            font-family: 'BMJUA', 'Malgun Gothic', sans-serif !important;
          }
          section.main h1,
          section.main h2,
          section.main h3,
          section.main h4,
          section.main p,
          section.main li,
          section.main label,
          section.main .stCaption {
            color: #f8fbff !important;
          }
          /* 링크 가독성: 흰색 글로우/그림자 (다른 CSS 덮어쓰기 방지) */
          section.main a,
          section.main .stMarkdown a,
          section.main .markdown-body a,
          section.main a:visited,
          section.main a:hover,
          section.main a:active {
            color: #f8fbff !important;
            text-shadow: 0 0 8px rgba(255, 255, 255, 0.8), 0 2px 12px rgba(0, 0, 0, 0.45) !important;
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.5)) !important;
          }
          .entry-card {
            border: 1px solid rgba(191, 219, 254, 0.55);
            border-radius: 14px;
            padding: 16px 14px 12px 14px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.82) 0%, rgba(30, 41, 59, 0.78) 100%);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.28);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            min-height: 130px;
            text-align: center;
          }
          .entry-card:hover {
            transform: translateY(-3px);
            border-color: #bfdbfe;
            box-shadow: 0 12px 28px rgba(56, 189, 248, 0.22);
          }
          .entry-card-title {
            margin: 0 0 6px 0;
            font-size: 1.15rem;
            color: #e0f2fe;
          }
          .entry-card-desc {
            margin: 0;
            color: #dbeafe;
            font-size: 0.93rem;
            line-height: 1.45;
          }
          .entry-question {
            text-align: center;
            color: #f8fbff;
            font-family: 'BMJUA', 'Malgun Gothic', sans-serif;
            font-size: clamp(1.2rem, 2.6vw, 1.7rem);
            margin: 12px 0 18px 0;
            text-shadow: 0 3px 12px rgba(15, 23, 42, 0.45);
          }
          .st-key-go_entry_page button {
            font-size: 0.5rem !important;
            padding: 0.06rem 0.9rem !important;
            min-height: 1.2rem !important;
            line-height: 1.0 !important;
            white-space: nowrap !important;
            background: #fef08a !important;
            color: #1f2937 !important;
            border: 1px solid #facc15 !important;
          }
          .st-key-go_entry_page button:hover {
            background: #fde047 !important;
            border-color: #eab308 !important;
            color: #111827 !important;
          }
          .st-key-entry_city_month button,
          .st-key-entry_month_city button {
            color: #e0f2fe !important;
            background-color: #1e3a8a !important;
            opacity: 1 !important;
            border: 1px solid #1d4ed8 !important;
            box-shadow: none !important;
          }
          .st-key-entry_city_month button:hover,
          .st-key-entry_month_city button:hover {
            color: #e0f2fe !important;
            background-color: #1e40af !important;
            opacity: 1 !important;
            border: 1px solid #2563eb !important;
          }
          .stSelectbox label,
          .stRadio label,
          .stSlider label,
          .stDateInput label {
            color: #f8fbff !important;
          }
          body, p, label, h1, h2, h3, h4, h5, h6 {
            color: white !important;
          }
          .stTextInput>div>div>input {
            color: #0A2540 !important;
          }
          .stSelectbox div[data-baseweb="select"] * {
            color: #0A2540 !important;
          }
          input {
            color: #0A2540 !important;
          }
          textarea {
            color: #0A2540 !important;
          }
          div[data-baseweb="select"] > div {
            color: #0A2540 !important;
          }
          div[data-baseweb="select"] * {
            color: #0A2540 !important;
          }
          .stSelectbox [data-baseweb="select"] {
            color: #0A2540 !important;
          }
          /* TextInput/Selectbox hover 시에도 가독성을 위해 배경/투명도 고정 */
          .stTextInput:hover, .stSelectbox:hover, .stTextArea:hover {
            background-color: white !important;
            opacity: 1 !important;
          }
          /* 입력창 actual box hover 제거 */
          div[data-baseweb="input"] {
            background-color: white !important;
            opacity: 1 !important;
          }
          div[data-baseweb="input"]:hover {
            background-color: white !important;
            opacity: 1 !important;
          }
          /* Selectbox 기본 박스 */
          div[data-baseweb="select"] {
            background-color: white !important;
            opacity: 1 !important;
          }
          div[data-baseweb="select"]:hover {
            background-color: white !important;
            opacity: 1 !important;
          }
          /* 드롭다운 옵션 리스트 글자 색 */
          div[data-baseweb="menu"] div[role="option"] {
            color: #0A2540 !important;
          }
          /* 드롭다운 옵션 hover 시 색 변화 방지 */
          div[data-baseweb="menu"] div[role="option"]:hover {
            background-color: rgba(0, 0, 0, 0.05) !important;
            color: #0A2540 !important;
          }
          /* 드롭다운 전체 메뉴 레이어 */
          div[data-baseweb="menu"] * {
            color: #0A2540 !important;
            background-color: white !important;
          }

          /* 개별 옵션 */
          div[data-baseweb="menu"] div[role="option"] {
            color: #0A2540 !important;
          }

          /* li 형태 옵션 대응 */
          div[data-baseweb="menu"] li {
            color: #0A2540 !important;
          }

          /* hover 시 색 유지 */
          div[data-baseweb="menu"] div[role="option"]:hover,
          div[data-baseweb="menu"] li:hover {
            background-color: rgba(0, 0, 0, 0.08) !important;
            color: #0A2540 !important;
          }

          /* (추가) 드롭다운 색상 강제 (요청 반영) */
          div[data-baseweb="menu"] * {
            color: #0A2540 !important;
            background-color: white !important;
          }

          div[data-baseweb="menu"] div[role="option"] {
            color: #0A2540 !important;
          }

          div[data-baseweb="menu"] li {
            color: #0A2540 !important;
          }

          div[data-baseweb="menu"] div[role="option"]:hover,
          div[data-baseweb="menu"] li:hover {
            background-color: rgba(0, 0, 0, 0.08) !important;
            color: #0A2540 !important;
          }

          div[data-baseweb="select"] > div {
            color: #0A2540 !important;
          }

          div[data-baseweb="select"] svg {
            fill: #0A2540 !important;
          }

          /* 선택된 텍스트 */
          div[data-baseweb="select"] > div {
            color: #0A2540 !important;
          }

          /* 드롭다운 화살표까지 색 변경 */
          div[data-baseweb="select"] svg {
            fill: #0A2540 !important;
          }
          /* 흰색 박스 내부는 가독성을 위해 어두운 텍스트로 예외 처리 */
          [data-testid="stVerticalBlockBorderWrapper"],
          [data-testid="stVerticalBlockBorderWrapper"] * {
            color: #0f172a !important;
          }
          [data-testid="stVerticalBlockBorderWrapper"] .stTextInput>div>div>input,
          [data-testid="stVerticalBlockBorderWrapper"] .stSelectbox div[data-baseweb="select"] * {
            color: #0A2540 !important;
          }
          /* 흰 박스 안 selectbox(클래스명이 stSelectbox가 아닐 때도)까지 남색으로 강제 */
          [data-testid="stVerticalBlockBorderWrapper"] div[data-baseweb="select"] *,
          [data-testid="stVerticalBlockBorderWrapper"] div[data-baseweb="menu"] * {
            color: #0A2540 !important;
          }
          [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.96) !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 14px !important;
            padding: 0.65rem 0.8rem !important;
          }
          [data-testid="stVerticalBlockBorderWrapper"] h1,
          [data-testid="stVerticalBlockBorderWrapper"] h2,
          [data-testid="stVerticalBlockBorderWrapper"] h3,
          [data-testid="stVerticalBlockBorderWrapper"] h4,
          [data-testid="stVerticalBlockBorderWrapper"] p,
          [data-testid="stVerticalBlockBorderWrapper"] label,
          [data-testid="stVerticalBlockBorderWrapper"] li,
          [data-testid="stVerticalBlockBorderWrapper"] .stCaption {
            color: #0f172a !important;
          }
          /* 최종 강제: Selectbox 표시/옵션 텍스트를 남색으로 */
          section.main div[data-baseweb="select"] *,
          section.main div[data-baseweb="menu"] * {
            color: #0A2540 !important;
            background-color: white !important;
          }

          /* (추가) Selectbox 드롭다운이 포털 레이어로 렌더링될 때 role 기반으로 강제 */
          div[role="menu"] *,
          div[role="listbox"] *,
          div[role="option"] * {
            color: #0A2540 !important;
          }
          div[role="menu"] [role="option"],
          div[role="listbox"] [role="option"],
          div[role="option"] {
            color: #0A2540 !important;
            background-color: #ffffff !important;
          }
          div[role="menu"] [role="option"]:hover,
          div[role="listbox"] [role="option"]:hover,
          div[role="option"]:hover {
            background-color: rgba(0, 0, 0, 0.06) !important;
            color: #0A2540 !important;
          }
          div[role="menu"] [role="option"][aria-selected="true"],
          div[role="listbox"] [role="option"][aria-selected="true"] {
            background-color: rgba(14, 165, 233, 0.14) !important;
          }
          /* 최종 강제: Streamlit/BaseWeb 드롭다운 텍스트 명도 문제 해결 */
          div[data-baseweb="menu"],
          div[role="menu"],
          div[role="listbox"] {
            color: #0A2540 !important;
            background-color: #ffffff !important;
          }
          div[data-baseweb="menu"] *,
          div[role="menu"] *,
          div[role="listbox"] *,
          div[data-baseweb="menu"] div[role="option"],
          div[role="menu"] div[role="option"],
          div[role="listbox"] div[role="option"] {
            color: #0A2540 !important;
            -webkit-text-fill-color: #0A2540 !important;
            opacity: 1 !important;
            filter: none !important;
          }
          .recommend-title {
            font-family: 'BMJUA', 'Malgun Gothic', sans-serif !important;
            letter-spacing: -0.01em;
          }
          .itinerary-slot {
            background: #e0f2fe;
            border: 1px solid #7dd3fc;
            border-radius: 999px;
            color: #0f172a !important;
            padding: 0.42rem 0.9rem;
            margin: 0.28rem 0;
            font-size: 0.93rem;
            display: inline-block;
          }
          .itinerary-flow {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin: 0.35rem 0 0.8rem 0;
          }
          .itinerary-node {
            background: #e0f2fe;
            border: 1px solid #60a5fa;
            border-radius: 999px;
            color: #0f172a !important;
            padding: 0.42rem 0.88rem;
            font-size: 0.9rem;
            line-height: 1.2;
            white-space: nowrap;
          }
          .itinerary-arrow {
            color: #38bdf8 !important;
            font-weight: 700;
            font-size: 1rem;
            line-height: 1;
          }
          @media (max-width: 768px) {
            .entry-card {
              min-height: auto;
              padding: 14px 12px 10px 12px;
              margin-bottom: 8px;
            }
            .entry-card-title { font-size: 1.02rem; }
            .entry-card-desc { font-size: 0.86rem; }
            div[data-testid="column"] {
              min-width: 100% !important;
              flex: 1 1 100% !important;
            }
          }
          .small-caption { font-size: 0.85rem; color: #f8fbff !important; }
          /* Streamlit caption 계열(회색 글자) 흰색 강제 */
          .stCaption, .stCaption * {
            color: #f8fbff !important;
          }
          /* st.write/st.markdown으로 렌더되는 마크다운 영역 회색/투명도 강제 제거 */
          section.main .stMarkdown,
          section.main .stMarkdown *,
          section.main .markdown-body,
          section.main .markdown-body * {
            color: #f8fbff !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #f8fbff !important;
          }
          section.main em,
          section.main i,
          section.main strong,
          section.main span,
          section.main li,
          section.main ul,
          section.main p {
            color: #f8fbff !important;
            opacity: 1 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _render_hero_title()
    st.caption("기후(온도·강수일수) 기반 여행 아이디어입니다. 실제 투어 예약 서비스는 아닙니다.")

    if "entry_page_mode" not in st.session_state:
        st.session_state.entry_page_mode = None
    if "should_scroll_to_results" not in st.session_state:
        st.session_state.should_scroll_to_results = False

    st.markdown("<div class='entry-question'>어떤 추천이 필요하신가요?</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="entry-card">
              <h4 class="entry-card-title">최적의 여행시기가 궁금해요</h4>
              <p class="entry-card-desc">도시를 고르면, 가장 좋은 여행 월을 추천해드려요.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        with st.container():
            if st.button("도시 → 월 추천 시작", use_container_width=True, key="entry_city_month"):
                st.session_state.entry_page_mode = "city_to_month"
                st.session_state.should_scroll_to_results = True
                st.rerun()
    with c2:
        st.markdown(
            """
            <div class="entry-card">
              <h4 class="entry-card-title">추천여행지가 궁금해요</h4>
              <p class="entry-card-desc">월을 고르면, 그 시기에 좋은 도시를 추천해드려요.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        with st.container():
            if st.button("월 → 도시 추천 시작", use_container_width=True, key="entry_month_city"):
                st.session_state.entry_page_mode = "month_to_city"
                st.session_state.should_scroll_to_results = True
                st.rerun()

    if st.session_state.entry_page_mode is None:
        return

    cache_dir = _resolve_cache_dir()

    TOP_N_CITIES = 100
    with st.sidebar:
        st.caption("왼쪽 ▶ **기후 데이터 캐시** 를 열어서 생성·재생성할 수 있어요.")
        with st.expander("기후 데이터 캐시 (클릭하여 열기)", expanded=False):
            st.subheader("캐시 생성 옵션")
            default_start = pd.Timestamp("2025-01-01")
            default_end = pd.Timestamp("2025-12-31")
            start_date = st.date_input("기후 데이터 시작일", value=default_start.date(), key="cache_start")
            end_date = st.date_input("기후 데이터 종료일", value=default_end.date(), key="cache_end")
            if start_date > end_date:
                st.error("시작일이 종료일보다 큽니다.")
                st.stop()

            model = st.selectbox(
                "기후 모델",
                options=["EC_Earth3P_HR", "MPI_ESM1_2_XR", "MRI_AGCM3_2_S"],
                index=0,
                key="cache_model",
            )
            st.caption("캐시는 오픈메테오 기후 모델 데이터로 생성합니다.")

            batch_size = st.slider(
                "배치 크기(좌표)",
                min_value=1,
                max_value=15,
                value=5,
                step=1,
                key="cache_batch",
                help="429가 나면 2~3 이하로 낮추세요.",
            )
            pause_between_batches = st.slider(
                "배치 간 대기(초)",
                min_value=0.0,
                max_value=8.0,
                value=2.0,
                step=0.5,
                key="cache_pause",
            )
            st.caption("막히면: 기간 짧게 · 배치↓ · 대기↑ 후 다시 생성.")
            compute = st.button("기후 월별 캐시 생성(100개)", type="primary", key="cache_compute")

    cities_df = _build_cities_top_200(cache_dir=cache_dir, top_n=TOP_N_CITIES)

    wpaths = _weather_cache_paths(
        cache_dir=cache_dir,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        model=model,
        score_params=PARAMS,
    )
    weather_cache_path = wpaths["weather_monthly"]
    cache_exists = weather_cache_path.exists()

    if not cache_exists:
        st.warning("선택한 기간/모델에 대한 기후 캐시가 아직 없습니다. 생성 버튼을 눌러주세요.")
        if not compute:
            return

    if compute and cache_exists:
        weather_cache_path.unlink(missing_ok=True)

    if not weather_cache_path.exists():
        with st.spinner("기후 캐시 생성 중입니다..."):
            _build_weather_monthly_cache(
                cache_dir=cache_dir,
                cities_df=cities_df,
                start_date=pd.Timestamp(start_date),
                end_date=pd.Timestamp(end_date),
                model=model,
                batch_size=int(batch_size),
                pause_between_batches_s=float(pause_between_batches),
                max_cities=TOP_N_CITIES,
                score_params=PARAMS,
            )

    with st.spinner("캐시 로드 중..."):
        df_weather = _load_weather_monthly_cache(weather_cache_path)

    st.divider()
    st.markdown("<div id='results-section'></div>", unsafe_allow_html=True)
    if st.session_state.should_scroll_to_results:
        components.html(
            """
            <script>
              const target = window.parent.document.getElementById("results-section");
              if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
              }
            </script>
            """,
            height=0,
        )
        st.session_state.should_scroll_to_results = False
    if "activity_pref_main" not in st.session_state:
        st.session_state.activity_pref_main = "혼합"
    activity_pref: ActivityPref = st.session_state.activity_pref_main
    score_col = _select_score_column(activity_pref)

    top_back_col, _ = st.columns([1.35, 6.65])
    with top_back_col:
        if st.button("처음으로", key="go_entry_page"):
            st.session_state.entry_page_mode = None
            st.rerun()

    show_city_to_month = st.session_state.entry_page_mode == "city_to_month"

    # 선택 상태
    if "active_city_id" not in st.session_state:
        st.session_state.active_city_id = int(cities_df.iloc[0]["city_id"])
    if "active_month" not in st.session_state:
        init_city = int(st.session_state.active_city_id)
        df_city_init = df_weather[df_weather["city_id"] == init_city]
        st.session_state.active_month = _best_month_for_city(df_city_init, score_col=score_col)

    with st.container(border=True):
        if show_city_to_month:
            st.markdown("<h3 class='recommend-title'>도시 → 월 추천</h3>", unsafe_allow_html=True)

            country_list_raw = sorted(
                cities_df["country"].astype(str).unique().tolist(),
                key=lambda x: str(x).lower(),
            )
            country_display_map = {c: _country_name_ko(c) for c in country_list_raw}
            country_display_options = [country_display_map[c] for c in country_list_raw]
            country_sel = st.selectbox(
                "국가 선택",
                options=["전체"] + country_display_options,
                index=0,
                key="country_filter",
            )
            reverse_country_map = {v: k for k, v in country_display_map.items()}
            country_sel_raw = reverse_country_map.get(country_sel, country_sel)
            base_df = cities_df if country_sel == "전체" else cities_df[cities_df["country"] == country_sel_raw].copy()
            if base_df.empty:
                st.warning("선택한 국가에 도시 데이터가 없습니다.")
                st.stop()

            city_options = base_df.sort_values("city_name").reset_index(drop=True)
            city_options_display = [
                f"{_city_name_ko(str(row.city_name), str(row.country))} (국가: {_country_name_ko(str(row.country))})"
                for row in city_options.itertuples(index=False)
            ]
            city_options_map = {
                display: int(row.city_id)
                for display, row in zip(city_options_display, city_options.itertuples(index=False))
            }

            active_display = None
            for disp, cid in city_options_map.items():
                if cid == int(st.session_state.active_city_id):
                    active_display = disp
                    break
            if active_display is None:
                first_row = city_options.iloc[0]
                st.session_state.active_city_id = int(first_row["city_id"])
                active_display = (
                    f"{_city_name_ko(str(first_row['city_name']), str(first_row['country']))} "
                    f"(국가: {_country_name_ko(str(first_row['country']))})"
                )

            selected_city_id = st.selectbox(
                "도시 선택",
                options=city_options_display,
                index=city_options_display.index(active_display) if active_display in city_options_display else 0,
                key="left_city_select",
            )
            st.session_state.active_city_id = city_options_map[selected_city_id]

            df_city = df_weather[df_weather["city_id"] == st.session_state.active_city_id].copy()
            if df_city.empty:
                st.error("해당 도시 기후 데이터가 없습니다.")
                st.stop()

            period, reason = _best_travel_window_and_reason(df_city, score_col)
            st.markdown(f"### 최적의 여행일시 : **{period}**")
            st.markdown(reason)

            st.markdown("#### 월별 평균기온 추이 🌡️")
            df_temp_plot = df_city[["month", "avg_temp_c", "precip_mm_avg"]].copy().sort_values("month")
            # 표와 동일한 표시 규칙(반올림 후 정수)로 맞춤
            df_temp_plot["avg_temp_c_plot"] = df_temp_plot["avg_temp_c"].map(lambda x: None if pd.isna(x) else int(round(float(x))))
            df_temp_plot["precip_mm_avg_plot"] = df_temp_plot["precip_mm_avg"].map(
                lambda x: None if pd.isna(x) else int(round(float(x)))
            )
            # Vega-Lite 렌더 실패 시에도 화면이 비지 않도록 스펙을 보수적으로 단순화
            try:
                st.vega_lite_chart(
                    df_temp_plot,
                    {
                        "layer": [
                            {
                                "mark": {"type": "bar", "opacity": 0.75, "color": "#2563eb"},
                                "encoding": {
                                    "x": {"field": "month", "type": "ordinal", "title": "월", "axis": {"labelAngle": 0}},
                                    "y": {
                                        "field": "precip_mm_avg_plot",
                                        "type": "quantitative",
                                        "title": "월평균 강수량(mm)",
                                        "axis": {
                                            "orient": "left",
                                            "titleAngle": 0,
                                            "titlePadding": 16,
                                            "labelPadding": 6,
                                        },
                                    },
                                    "color": {
                                        "datum": "월평균 강수량(mm)",
                                        "type": "nominal",
                                        "legend": {"title": "범례", "orient": "bottom"},
                                    },
                                    "tooltip": [
                                        {"field": "month", "type": "ordinal", "title": "월"},
                                        {"field": "precip_mm_avg_plot", "type": "quantitative", "title": "월평균 강수량(mm)"},
                                    ],
                                },
                            },
                            {
                                "mark": {"type": "line", "point": True, "strokeWidth": 3, "color": "#ea580c"},
                                "encoding": {
                                    "x": {"field": "month", "type": "ordinal", "title": "월", "axis": {"labelAngle": 0}},
                                    "y": {
                                        "field": "avg_temp_c_plot",
                                        "type": "quantitative",
                                        "title": "평균기온(℃)",
                                        "axis": {
                                            "orient": "right",
                                            "titleAngle": 0,
                                            "titlePadding": 16,
                                            "labelPadding": 6,
                                        },
                                    },
                                    "color": {
                                        "datum": "평균기온(℃)",
                                        "type": "nominal",
                                        "legend": {"title": "범례", "orient": "bottom"},
                                    },
                                    "tooltip": [
                                        {"field": "month", "type": "ordinal", "title": "월"},
                                        {"field": "avg_temp_c_plot", "type": "quantitative", "title": "기온(℃)"},
                                    ],
                                },
                            },
                        ],
                        "height": 420,
                        "padding": {"left": 96, "right": 96, "top": 28, "bottom": 72},
                        "resolve": {"scale": {"y": "independent"}},
                        "config": {
                            "view": {"stroke": None},
                            "legend": {"orient": "bottom", "direction": "horizontal"},
                            "axis": {"titleLimit": 180},
                        },
                    },
                    use_container_width=True,
                )
            except Exception:
                # fallback: 최소한의 시각화 제공
                st.line_chart(df_temp_plot.set_index("month")["avg_temp_c_plot"], use_container_width=True)
                st.bar_chart(df_temp_plot.set_index("month")["precip_mm_avg_plot"], use_container_width=True)

            with st.expander("월별 세부 수치 보기", expanded=False):
                df_city_tbl = (
                    df_city[["month", "avg_temp_c", "precip_mm_avg", "precip_days", "cloudy_days", "days_in_month_observed", score_col]]
                    .copy()
                    .rename(columns={score_col: "recommendation_score_raw"})
                    .sort_values("month")
                )

                df_city_tbl["cloud_ratio"] = df_city_tbl.apply(
                    lambda r: (r["cloudy_days"] / r["days_in_month_observed"]) if r["days_in_month_observed"] else 0.0,
                    axis=1,
                )
                df_city_tbl["흐린날 수(%)"] = df_city_tbl.apply(
                    lambda r: _format_cloud_display(int(r["cloudy_days"]), float(r["cloud_ratio"])),
                    axis=1,
                )

                score_min = float(df_city_tbl["recommendation_score_raw"].min())
                score_max = float(df_city_tbl["recommendation_score_raw"].max())
                if pd.isna(score_min) or pd.isna(score_max):
                    df_city_tbl["추천점수(0~100)"] = None
                elif abs(score_max - score_min) < 1e-9:
                    df_city_tbl["추천점수(0~100)"] = 100.0
                else:
                    df_city_tbl["추천점수(0~100)"] = (
                        (df_city_tbl["recommendation_score_raw"] - score_min) / (score_max - score_min) * 100.0
                    )

                months = df_city_tbl["month"].astype(int).tolist()
                month_cols = [f"{m}월" for m in months]
                table_rows = [
                    ("🌡️ 평균기온(°C)", [None if pd.isna(v) else int(round(float(v))) for v in df_city_tbl["avg_temp_c"].tolist()]),
                    ("🌧️ 월평균 강수량(mm)", [None if pd.isna(v) else int(round(float(v))) for v in df_city_tbl["precip_mm_avg"].tolist()]),
                    ("☁️ 흐린날 수(%)", df_city_tbl["흐린날 수(%)"].tolist()),
                    ("⭐ 추천점수(0~100)", [None if pd.isna(v) else int(round(float(v))) for v in df_city_tbl["추천점수(0~100)"].tolist()]),
                ]

                table_data: dict[str, list[Any]] = {"항목": [r[0] for r in table_rows]}
                for i, mc in enumerate(month_cols):
                    table_data[mc] = [r[1][i] for r in table_rows]
                df_city_wide = pd.DataFrame(table_data)
                month_cols_only = [c for c in df_city_wide.columns if c.endswith("월")]
                if month_cols_only:
                    # 월 값이 전부 비어있는 행은 제거
                    df_city_wide = df_city_wide[
                        ~df_city_wide[month_cols_only].apply(
                            lambda r: all((v is None) or (isinstance(v, str) and v.strip() == "") or pd.isna(v) for v in r),
                            axis=1,
                        )
                    ].reset_index(drop=True)

                def _style_city_score_row(row: pd.Series) -> list[str]:
                    styles = [""] * len(row)
                    if row.get("항목") != "⭐ 추천점수(0~100)":
                        return styles
                    for ci, col in enumerate(df_city_wide.columns):
                        if col not in month_cols_only:
                            continue
                        val = row[col]
                        if val is None or pd.isna(val):
                            styles[ci] = ""
                            continue
                        ratio = max(0.0, min(1.0, float(val) / 100.0))
                        hue = int(120 * ratio)
                        styles[ci] = f"background-color: hsl({hue}, 75%, 86%); font-weight: 700;"
                    return styles

                styled_city_wide = df_city_wide.style.apply(_style_city_score_row, axis=1)
                table_height = 70 + max(1, len(df_city_wide)) * 42

                st.dataframe(
                    styled_city_wide,
                    use_container_width=True,
                    hide_index=True,
                    height=table_height,
                )
                st.markdown(
                    "<span style='color: white;'>* 흐림 기준: 구름량 평균 60% 이상인 날 / 해당 월 일수 기준</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<h3 class='recommend-title'>월 → 도시 상위 추천</h3>", unsafe_allow_html=True)
            top_n = st.slider(
                "상위 추천 개수",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                key="top_n_main",
            )
            month_val = st.selectbox(
                "월 선택",
                options=list(range(1, 13)),
                index=int(st.session_state.active_month) - 1,
                key="right_month_select",
            )
            st.session_state.active_month = int(month_val)

            df_month = df_weather[df_weather["month"] == st.session_state.active_month].copy()
            if df_month.empty:
                st.error("해당 월 데이터가 없습니다.")
                st.stop()

            df_month = df_month.dropna(subset=[score_col]).copy()
            if df_month.empty:
                st.error("해당 월 추천 점수 데이터가 없습니다.")
                st.stop()

            # 기후 점수(월 내 상대 0~100) + 치안 점수(큰 비중) 결합
            ws_min = float(df_month[score_col].min())
            ws_max = float(df_month[score_col].max())
            if abs(ws_max - ws_min) < 1e-9:
                df_month["기후점수(0~100)"] = 100.0
            else:
                df_month["기후점수(0~100)"] = ((df_month[score_col] - ws_min) / (ws_max - ws_min) * 100.0)

            df_month["치안점수(0~100)"] = df_month.apply(
                lambda r: _safety_score(str(r["city_name"]), str(r["country"])),
                axis=1,
            )
            safety_weight = 0.7
            weather_weight = 0.3
            df_month["⭐ 종합점수(0~100)"] = (
                df_month["치안점수(0~100)"] * safety_weight + df_month["기후점수(0~100)"] * weather_weight
            )

            df_month = df_month.sort_values("⭐ 종합점수(0~100)", ascending=False)
            df_top = df_month.head(int(top_n)).copy()
            if not df_top.empty:
                st.session_state.active_city_id = int(df_top.iloc[0]["city_id"])

            df_top_tbl = df_top[
                [
                    "city_name",
                    "country",
                    "avg_temp_c",
                    "precip_days",
                    "cloudy_days",
                    "days_in_month_observed",
                    "치안점수(0~100)",
                    "기후점수(0~100)",
                    "⭐ 종합점수(0~100)",
                ]
            ].copy()
            df_top_tbl["cloud_ratio"] = df_top_tbl.apply(
                lambda r: (r["cloudy_days"] / r["days_in_month_observed"]) if r["days_in_month_observed"] else 0.0,
                axis=1,
            )
            df_top_tbl["흐린날 수(%)"] = df_top_tbl.apply(
                lambda r: _format_cloud_display(int(r["cloudy_days"]), float(r["cloud_ratio"])),
                axis=1,
            )

            df_top_tbl["city_name"] = df_top_tbl.apply(
                lambda r: _city_name_ko(str(r["city_name"]), str(r["country"])),
                axis=1,
            )
            df_top_tbl["country"] = df_top_tbl["country"].map(lambda x: _country_name_ko(str(x)))

            df_top_tbl = df_top_tbl.rename(
                columns={
                    "city_name": "🏙️ 도시",
                    "country": "🌍 국가",
                    "avg_temp_c": "🌡️ 평균기온(°C)",
                    "precip_days": "🌧️ 강수일수(일)",
                    "흐린날 수(%)": "☁️ 흐린날 수(%)",
                    "치안점수(0~100)": "🔐 치안점수(0~100)",
                    "기후점수(0~100)": "🌦️ 기후점수(0~100)",
                    "⭐ 종합점수(0~100)": "⭐ 종합점수(0~100)",
                }
            )
            df_top_tbl = df_top_tbl[
                [
                    "🏙️ 도시",
                    "🌍 국가",
                    "🌡️ 평균기온(°C)",
                    "🌧️ 강수일수(일)",
                    "☁️ 흐린날 수(%)",
                    "🔐 치안점수(0~100)",
                    "🌦️ 기후점수(0~100)",
                    "⭐ 종합점수(0~100)",
                ]
            ]
            df_top_tbl["🌡️ 평균기온(°C)"] = df_top_tbl["🌡️ 평균기온(°C)"].map(lambda x: None if pd.isna(x) else int(round(float(x))))
            for c in ["🔐 치안점수(0~100)", "🌦️ 기후점수(0~100)", "⭐ 종합점수(0~100)"]:
                df_top_tbl[c] = df_top_tbl[c].map(lambda x: None if pd.isna(x) else int(round(float(x))))
            df_top_tbl = df_top_tbl.sort_values("⭐ 종합점수(0~100)", ascending=False)
            styled_top_tbl = df_top_tbl.style.background_gradient(
                cmap="RdYlGn",
                subset=["⭐ 종합점수(0~100)"],
                axis=0,
            )

            st.dataframe(
                styled_top_tbl,
                use_container_width=True,
                hide_index=True,
                height=380,
            )
            st.markdown(
                "<span style='color: white;'>* 종합점수 가중치: 치안 70% + 기후 30%</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<span style='color: white;'>* 흐림 기준: 구름량 평균 60% 이상인 날 / 해당 월 일수 기준</span>",
                unsafe_allow_html=True,
            )

    # 상세 영역(하단)
    st.divider()
    activity_pref = st.radio(
        "활동 선호 (휴양/관광 추천 반영)",
        ["야외 우선", "혼합", "실내 우선"],
        index=["야외 우선", "혼합", "실내 우선"].index(st.session_state.activity_pref_main),
        horizontal=True,
        key="activity_pref_main",
        help="아래 휴양/관광/액티비티 추천 문구와 점수 반영에 사용됩니다.",
    )
    city_sel_id = int(st.session_state.active_city_id)
    month_sel = int(st.session_state.active_month)

    city_row = cities_df[cities_df["city_id"] == city_sel_id].iloc[0]
    monthly_row = df_weather[(df_weather["city_id"] == city_sel_id) & (df_weather["month"] == month_sel)].copy()
    if monthly_row.empty:
        st.error("선택한 도시/월의 상세 데이터가 없습니다.")
        return
    monthly_row = monthly_row.iloc[0]

    with st.container(border=True):
        _build_detail_section(
            city_row=city_row,
            active_month=month_sel,
            activity_pref=activity_pref,
            monthly_row=monthly_row,
        )


if __name__ == "__main__":
    main()

