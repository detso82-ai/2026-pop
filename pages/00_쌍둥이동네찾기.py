import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="지역별 인구구조 비교",
    page_icon="👥",
    layout="wide",
)

DATA_FILENAME = "202606_202606_연령별인구현황_월간.csv"
DATA_PATH = Path(__file__).resolve().parent / DATA_FILENAME

AGE_LABELS = [
    f"{age}세"
    for age in range(100)
] + ["100세 이상"]

AGE_ORDERS = list(range(101))


# =========================================================
# CSV 불러오기
# =========================================================

@st.cache_data(show_spinner="인구 데이터를 불러오는 중입니다.")
def load_data(file_path: Path) -> pd.DataFrame:
    """
    app.py와 같은 폴더에 있는 CSV 파일을 읽습니다.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path.name} 파일을 찾을 수 없습니다."
        )

    encodings = [
        "cp949",
        "euc-kr",
        "utf-8-sig",
        "utf-8",
    ]

    last_error = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                file_path,
                encoding=encoding,
                dtype=str,
                low_memory=False,
            )

            dataframe.columns = [
                str(column).strip()
                for column in dataframe.columns
            ]

            return dataframe

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        "CSV 파일의 인코딩을 확인할 수 없습니다."
    ) from last_error


# =========================================================
# 문자열 및 숫자 정리
# =========================================================

def normalize_text(value) -> str:
    """연속된 공백과 앞뒤 공백을 정리합니다."""

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def remove_region_code(region_text: str) -> str:
    """
    지역명 뒤의 10자리 행정구역 코드를 제거합니다.

    예:
    서울특별시 종로구 청운효자동(1111051500)
    → 서울특별시 종로구 청운효자동
    """

    return re.sub(
        r"\s*\(\d{10}\)\s*$",
        "",
        normalize_text(region_text),
    ).strip()


def extract_region_code(region_text: str) -> str:
    """지역명 뒤의 10자리 행정구역 코드를 추출합니다."""

    match = re.search(
        r"\((\d{10})\)\s*$",
        normalize_text(region_text),
    )

    if match:
        return match.group(1)

    return ""


def convert_to_number(value) -> int:
    """쉼표가 포함된 문자열을 정수로 변환합니다."""

    if pd.isna(value):
        return 0

    text = str(value).replace(",", "").strip()

    if text in {
        "",
        "-",
        "nan",
        "None",
        "null",
    }:
        return 0

    try:
        return int(float(text))

    except (ValueError, TypeError):
        return 0


# =========================================================
# 행정구역 단계 판별
# =========================================================

def determine_admin_level(region_code: str) -> str:
    """
    행정구역 코드의 뒤쪽 0 개수를 기준으로 행정단위를 분류합니다.

    예:
    1100000000 → 시·도
    1111000000 → 시·군·구
    1111051500 → 읍·면·동
    """

    if not region_code:
        return "기타"

    if region_code.endswith("00000000"):
        return "시·도"

    if region_code.endswith("00000"):
        return "시·군·구"

    return "읍·면·동"


def get_short_region_name(region_name: str) -> str:
    """
    긴 지역명에서 마지막 지역명을 반환합니다.

    예:
    서울특별시 종로구 청운효자동
    → 청운효자동
    """

    words = normalize_text(region_name).split()

    if not words:
        return ""

    return words[-1]


# =========================================================
# 연령별 열 찾기
# =========================================================

def find_age_column(
    columns,
    sex_code: str,
    age_label: str,
) -> str | None:
    """
    연월에 상관없이 연령별 열을 찾습니다.

    예:
    2026년06월_계_30세
    """

    suffix = f"_{sex_code}_{age_label}"

    for column in columns:
        if str(column).endswith(suffix):
            return column

    return None


def find_total_column(
    columns,
    sex_code: str = "계",
) -> str | None:
    """총인구수 열을 찾습니다."""

    suffix = f"_{sex_code}_총인구수"

    for column in columns:
        if str(column).endswith(suffix):
            return column

    return None


# =========================================================
# 전체 지역 연령별 행렬 생성
# =========================================================

@st.cache_data(show_spinner="전국 연령별 인구구조를 계산하는 중입니다.")
def prepare_population_data(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    전체 지역에 대해 0세부터 100세 이상까지의
    인구수와 인구 비율 행렬을 만듭니다.

    반환값:
    1. 지역 정보 데이터프레임
    2. 연령별 비율 행렬
    3. 실제 사용된 연령 열 목록
    """

    dataframe = raw_df.copy()

    dataframe["행정구역_원본"] = (
        dataframe["행정구역"]
        .apply(normalize_text)
    )

    dataframe["지역명"] = (
        dataframe["행정구역_원본"]
        .apply(remove_region_code)
    )

    dataframe["행정구역코드"] = (
        dataframe["행정구역_원본"]
        .apply(extract_region_code)
    )

    dataframe["행정단위"] = (
        dataframe["행정구역코드"]
        .apply(determine_admin_level)
    )

    dataframe["짧은지역명"] = (
        dataframe["지역명"]
        .apply(get_short_region_name)
    )

    dataframe = dataframe[
        dataframe["지역명"] != ""
    ].reset_index(drop=True)

    age_columns = []

    for age_label in AGE_LABELS:
        column = find_age_column(
            columns=dataframe.columns,
            sex_code="계",
            age_label=age_label,
        )

        if column is None:
            raise ValueError(
                f"`계_{age_label}` 열을 찾을 수 없습니다."
            )

        age_columns.append(column)

    numeric_age_df = dataframe[
        age_columns
    ].copy()

    for column in age_columns:
        numeric_age_df[column] = (
            numeric_age_df[column]
            .fillna("0")
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        numeric_age_df[column] = pd.to_numeric(
            numeric_age_df[column],
            errors="coerce",
        ).fillna(0)

    age_counts = numeric_age_df.to_numpy(
        dtype=np.float64
    )

    age_totals = age_counts.sum(
        axis=1,
        keepdims=True,
    )

    age_totals[age_totals == 0] = 1

    age_proportions = age_counts / age_totals

    total_column = find_total_column(
        dataframe.columns,
        "계",
    )

    if total_column is not None:
        dataframe["총인구수"] = (
            dataframe[total_column]
            .apply(convert_to_number)
        )

    else:
        dataframe["총인구수"] = (
            age_counts.sum(axis=1)
            .astype(int)
        )

    dataframe["데이터행번호"] = np.arange(
        len(dataframe)
    )

    return (
        dataframe,
        age_proportions,
        age_columns,
    )


# =========================================================
# 선택 지역 연령별 데이터
# =========================================================

def make_age_profile(
    selected_index: int,
    region_df: pd.DataFrame,
    age_proportions: np.ndarray,
) -> pd.DataFrame:
    """선택한 지역의 연령별 인구수와 비율을 만듭니다."""

    source_row = region_df.iloc[selected_index]

    age_counts = []

    for age_label in AGE_LABELS:
        column = find_age_column(
            columns=region_df.columns,
            sex_code="계",
            age_label=age_label,
        )

        age_counts.append(
            convert_to_number(
                source_row[column]
            )
        )

    proportions = (
        age_proportions[selected_index]
        * 100
    )

    return pd.DataFrame(
        {
            "연령순서": AGE_ORDERS,
            "연령": AGE_LABELS,
            "인구수": age_counts,
            "비율": proportions,
        }
    )


# =========================================================
# 코사인 유사도 계산
# =========================================================

def calculate_cosine_similarities(
    target_vector: np.ndarray,
    comparison_matrix: np.ndarray,
) -> np.ndarray:
    """
    선택 지역과 모든 비교 지역의 코사인 유사도를 계산합니다.

    1에 가까울수록 연령별 인구 비율의 모양이 비슷합니다.
    """

    target_norm = np.linalg.norm(
        target_vector
    )

    matrix_norms = np.linalg.norm(
        comparison_matrix,
        axis=1,
    )

    denominator = (
        target_norm
        * matrix_norms
    )

    denominator[
        denominator == 0
    ] = 1

    similarities = (
        comparison_matrix
        @ target_vector
    ) / denominator

    similarities = np.clip(
        similarities,
        0,
        1,
    )

    return similarities


def find_similar_regions(
    selected_index: int,
    region_df: pd.DataFrame,
    age_proportions: np.ndarray,
    comparison_scope: str,
    minimum_population: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    선택 지역과 인구구조가 가장 유사한 지역을 찾습니다.
    """

    selected_row = region_df.iloc[
        selected_index
    ]

    target_vector = age_proportions[
        selected_index
    ]

    candidate_mask = (
        region_df["총인구수"]
        >= minimum_population
    ).to_numpy()

    if comparison_scope == "선택 지역과 같은 행정단위":
        same_level_mask = (
            region_df["행정단위"]
            == selected_row["행정단위"]
        ).to_numpy()

        candidate_mask = (
            candidate_mask
            & same_level_mask
        )

    elif comparison_scope != "전국 모든 행정단위":
        level_mask = (
            region_df["행정단위"]
            == comparison_scope
        ).to_numpy()

        candidate_mask = (
            candidate_mask
            & level_mask
        )

    # 자기 자신은 비교 대상에서 제외
    candidate_mask[selected_index] = False

    candidate_indices = np.where(
        candidate_mask
    )[0]

    if len(candidate_indices) == 0:
        return pd.DataFrame()

    candidate_matrix = age_proportions[
        candidate_indices
    ]

    similarities = calculate_cosine_similarities(
        target_vector=target_vector,
        comparison_matrix=candidate_matrix,
    )

    result = region_df.iloc[
        candidate_indices
    ][
        [
            "지역명",
            "짧은지역명",
            "행정구역코드",
            "행정단위",
            "총인구수",
            "데이터행번호",
        ]
    ].copy()

    result["코사인유사도"] = similarities

    result["유사도점수"] = (
        result["코사인유사도"]
        * 100
    )

    result = (
        result
        .sort_values(
            [
                "코사인유사도",
                "총인구수",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    result["순위"] = (
        np.arange(len(result))
        + 1
    )

    return result


# =========================================================
# 그래프 생성
# =========================================================

def create_selected_region_chart(
    profile_df: pd.DataFrame,
    region_name: str,
    chart_value: str,
) -> go.Figure:
    """선택 지역의 연령별 인구구조 그래프입니다."""

    if chart_value == "인구 비율 (%)":
        y_values = profile_df["비율"]
        y_title = "지역 전체 인구 대비 비율 (%)"
        hover_template = (
            "<b>%{x}</b><br>"
            "%{y:.3f}%"
            "<extra></extra>"
        )

    else:
        y_values = profile_df["인구수"]
        y_title = "인구수 (명)"
        hover_template = (
            "<b>%{x}</b><br>"
            "%{y:,.0f}명"
            "<extra></extra>"
        )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=profile_df["연령"],
            y=y_values,
            mode="lines",
            name=region_name,
            line={
                "width": 3,
            },
            hovertemplate=hover_template,
        )
    )

    figure.update_layout(
        title={
            "text": f"{region_name} 연령별 인구구조",
            "x": 0.01,
        },
        xaxis_title="연령",
        yaxis_title=y_title,
        height=550,
        hovermode="x unified",
        margin={
            "l": 30,
            "r": 20,
            "t": 70,
            "b": 30,
        },
    )

    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=AGE_LABELS,
        tickmode="array",
        tickvals=[
            f"{age}세"
            for age in range(0, 100, 5)
        ] + ["100세 이상"],
        showgrid=False,
    )

    figure.update_yaxes(
        rangemode="tozero",
        gridcolor="rgba(128,128,128,0.18)",
    )

    return figure


def create_similarity_ranking_chart(
    similar_df: pd.DataFrame,
    selected_region_name: str,
) -> go.Figure:
    """인구구조 유사도 상위 지역을 가로 막대그래프로 표시합니다."""

    chart_df = (
        similar_df
        .sort_values(
            "유사도점수",
            ascending=True,
        )
        .copy()
    )

    chart_df["표시명"] = (
        chart_df["순위"].astype(str)
        + "위 · "
        + chart_df["지역명"]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=chart_df["유사도점수"],
            y=chart_df["표시명"],
            orientation="h",
            text=chart_df["유사도점수"].map(
                lambda value: f"{value:.3f}점"
            ),
            textposition="outside",
            customdata=np.stack(
                [
                    chart_df["총인구수"],
                    chart_df["행정단위"],
                    chart_df["행정구역코드"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "유사도: %{x:.4f}점<br>"
                "총인구: %{customdata[0]:,.0f}명<br>"
                "행정단위: %{customdata[1]}<br>"
                "행정구역 코드: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    minimum_score = max(
        0,
        chart_df["유사도점수"].min() - 0.5,
    )

    figure.update_layout(
        title={
            "text": (
                f"{selected_region_name}과 "
                "인구구조가 비슷한 지역 TOP 5"
            ),
            "x": 0.01,
        },
        xaxis_title="인구구조 유사도 점수",
        yaxis_title="",
        height=430,
        margin={
            "l": 40,
            "r": 90,
            "t": 70,
            "b": 40,
        },
        showlegend=False,
    )

    figure.update_xaxes(
        range=[
            minimum_score,
            100.05,
        ],
        ticksuffix="점",
        gridcolor="rgba(128,128,128,0.18)",
    )

    return figure


def create_comparison_line_chart(
    selected_index: int,
    selected_region_name: str,
    similar_df: pd.DataFrame,
    region_df: pd.DataFrame,
    age_proportions: np.ndarray,
) -> go.Figure:
    """
    선택 지역과 유사 지역 TOP 5의
    연령별 인구 비율을 한 그래프에 표시합니다.
    """

    figure = go.Figure()

    selected_percentages = (
        age_proportions[selected_index]
        * 100
    )

    figure.add_trace(
        go.Scatter(
            x=AGE_LABELS,
            y=selected_percentages,
            mode="lines",
            name=f"선택 · {selected_region_name}",
            line={
                "width": 5,
            },
            hovertemplate=(
                "<b>%{x}</b><br>"
                "%{fullData.name}<br>"
                "%{y:.3f}%"
                "<extra></extra>"
            ),
        )
    )

    dash_styles = [
        "dash",
        "dot",
        "dashdot",
        "longdash",
        "longdashdot",
    ]

    for row_number, result_row in similar_df.iterrows():
        data_index = int(
            result_row["데이터행번호"]
        )

        percentages = (
            age_proportions[data_index]
            * 100
        )

        trace_name = (
            f"{int(result_row['순위'])}위 · "
            f"{result_row['지역명']} "
            f"({result_row['유사도점수']:.3f}점)"
        )

        figure.add_trace(
            go.Scatter(
                x=AGE_LABELS,
                y=percentages,
                mode="lines",
                name=trace_name,
                line={
                    "width": 2,
                    "dash": dash_styles[
                        row_number
                        % len(dash_styles)
                    ],
                },
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "%{fullData.name}<br>"
                    "%{y:.3f}%"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title={
            "text": (
                f"{selected_region_name}과 "
                "유사 지역의 연령별 인구 비율"
            ),
            "x": 0.01,
        },
        xaxis_title="연령",
        yaxis_title="지역 전체 인구 대비 비율 (%)",
        height=700,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 30,
            "r": 20,
            "t": 150,
            "b": 40,
        },
    )

    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=AGE_LABELS,
        tickmode="array",
        tickvals=[
            f"{age}세"
            for age in range(0, 100, 5)
        ] + ["100세 이상"],
        showgrid=False,
        rangeslider={
            "visible": True,
            "thickness": 0.05,
        },
    )

    figure.update_yaxes(
        rangemode="tozero",
        ticksuffix="%",
        gridcolor="rgba(128,128,128,0.18)",
    )

    return figure


# =========================================================
# 데이터 불러오기
# =========================================================

try:
    raw_population_df = load_data(
        DATA_PATH
    )

except FileNotFoundError:
    st.error(
        f"`{DATA_FILENAME}` 파일을 찾을 수 없습니다."
    )

    st.write(
        "CSV 파일을 `app.py`와 같은 폴더에 넣어 주세요."
    )

    st.stop()

except Exception as error:
    st.error(
        "CSV 파일을 읽는 중 오류가 발생했습니다."
    )

    st.exception(error)
    st.stop()


if "행정구역" not in raw_population_df.columns:
    st.error(
        "CSV 파일에서 `행정구역` 열을 찾지 못했습니다."
    )

    st.stop()


try:
    (
        region_df,
        age_proportion_matrix,
        age_columns,
    ) = prepare_population_data(
        raw_population_df
    )

except Exception as error:
    st.error(
        "연령별 데이터를 처리하는 중 오류가 발생했습니다."
    )

    st.exception(error)
    st.stop()


# =========================================================
# 화면 제목
# =========================================================

st.title("👥 전국 지역별 인구구조 비교")

st.write(
    "지역을 선택하면 해당 지역의 연령별 인구구조와 "
    "전국에서 인구구조가 가장 비슷한 지역 5곳을 보여줍니다."
)

st.caption(
    "유사도는 각 연령 인구가 지역 전체 인구에서 차지하는 비율을 "
    "0세부터 100세 이상까지 비교한 코사인 유사도입니다. "
    "총인구수가 달라도 연령 분포의 모양이 비슷하면 높은 점수를 받습니다."
)


# =========================================================
# 사이드바 설정
# =========================================================

with st.sidebar:
    st.header("비교 설정")

    comparison_scope = st.selectbox(
        "비교할 행정단위",
        options=[
            "선택 지역과 같은 행정단위",
            "읍·면·동",
            "시·군·구",
            "시·도",
            "전국 모든 행정단위",
        ],
        index=0,
        help=(
            "같은 행정단위를 선택하면 읍·면·동은 전국 읍·면·동끼리, "
            "시·군·구는 전국 시·군·구끼리 비교합니다."
        ),
    )

    minimum_population = st.number_input(
        "비교 지역 최소 인구수",
        min_value=0,
        max_value=1_000_000,
        value=1000,
        step=500,
        help=(
            "인구가 매우 적은 지역은 작은 수치 변화에 따라 "
            "인구구조가 크게 달라질 수 있습니다."
        ),
    )

    selected_chart_value = st.radio(
        "선택 지역 그래프 표시",
        options=[
            "인구 비율 (%)",
            "인구수 (명)",
        ],
        index=0,
    )

    st.divider()

    st.caption(
        f"데이터 파일: {DATA_FILENAME}"
    )

    st.caption(
        f"전체 지역 수: {len(region_df):,}개"
    )


# =========================================================
# 지역 검색 및 선택
# =========================================================

st.subheader("1. 기준 지역 선택")

search_keyword = st.text_input(
    "지역명 검색",
    placeholder=(
        "예: 서울 종로구 청운효자동, "
        "수원 영통구, 제주 애월읍"
    ),
    help=(
        "여러 단어를 띄어 입력하면 모든 단어가 포함된 지역만 표시합니다."
    ),
)

search_terms = [
    term.lower()
    for term in search_keyword.split()
    if term.strip()
]


def region_matches_search(row) -> bool:
    searchable_text = (
        f"{row['지역명']} "
        f"{row['행정구역코드']} "
        f"{row['행정단위']}"
    ).lower()

    return all(
        term in searchable_text
        for term in search_terms
    )


if search_terms:
    filtered_regions = region_df[
        region_df.apply(
            region_matches_search,
            axis=1,
        )
    ].copy()

else:
    filtered_regions = region_df.copy()


if filtered_regions.empty:
    st.warning(
        "검색어와 일치하는 지역이 없습니다."
    )

    st.stop()


filtered_indices = filtered_regions.index.tolist()


def format_region_index(index: int) -> str:
    row = region_df.loc[index]

    return (
        f"{row['지역명']} "
        f"· {row['행정단위']} "
        f"· 인구 {row['총인구수']:,}명"
    )


selected_index = st.selectbox(
    "지역 선택",
    options=filtered_indices,
    format_func=format_region_index,
)

selected_row = region_df.loc[
    selected_index
]

selected_region_name = selected_row[
    "지역명"
]

selected_admin_level = selected_row[
    "행정단위"
]

selected_population = int(
    selected_row["총인구수"]
)


# =========================================================
# 선택 지역 정보
# =========================================================

metric_columns = st.columns(4)

metric_columns[0].metric(
    "선택 지역",
    selected_region_name,
)

metric_columns[1].metric(
    "행정단위",
    selected_admin_level,
)

metric_columns[2].metric(
    "총인구",
    f"{selected_population:,}명",
)

metric_columns[3].metric(
    "행정구역 코드",
    selected_row["행정구역코드"],
)


# =========================================================
# 선택 지역 연령별 그래프
# =========================================================

selected_profile_df = make_age_profile(
    selected_index=selected_index,
    region_df=region_df,
    age_proportions=age_proportion_matrix,
)

selected_figure = create_selected_region_chart(
    profile_df=selected_profile_df,
    region_name=selected_region_name,
    chart_value=selected_chart_value,
)

st.plotly_chart(
    selected_figure,
    use_container_width=True,
    config={
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
    },
)


# =========================================================
# 유사 지역 계산
# =========================================================

st.subheader("2. 인구구조가 가장 비슷한 지역 TOP 5")

similar_regions_df = find_similar_regions(
    selected_index=selected_index,
    region_df=region_df,
    age_proportions=age_proportion_matrix,
    comparison_scope=comparison_scope,
    minimum_population=int(
        minimum_population
    ),
    top_n=5,
)


if similar_regions_df.empty:
    st.warning(
        "현재 조건에서 비교할 지역이 없습니다. "
        "최소 인구수를 낮추거나 비교 행정단위를 변경해 주세요."
    )

    st.stop()


ranking_figure = create_similarity_ranking_chart(
    similar_df=similar_regions_df,
    selected_region_name=selected_region_name,
)

st.plotly_chart(
    ranking_figure,
    use_container_width=True,
    config={
        "displaylogo": False,
        "responsive": True,
    },
)


# =========================================================
# 유사 지역 비교 꺾은선 그래프
# =========================================================

st.subheader("3. 선택 지역과 유사 지역의 연령별 분포 비교")

st.caption(
    "굵은 실선이 선택한 지역이며, 나머지 선은 유사도 상위 5개 지역입니다."
)

comparison_figure = create_comparison_line_chart(
    selected_index=selected_index,
    selected_region_name=selected_region_name,
    similar_df=similar_regions_df,
    region_df=region_df,
    age_proportions=age_proportion_matrix,
)

st.plotly_chart(
    comparison_figure,
    use_container_width=True,
    config={
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
        ],
        "toImageButtonOptions": {
            "format": "png",
            "filename": (
                f"{selected_region_name}_인구구조_유사지역"
            ),
            "height": 900,
            "width": 1600,
            "scale": 2,
        },
    },
)


# =========================================================
# 결과 표
# =========================================================

st.subheader("4. 유사 지역 상세 결과")

display_result_df = similar_regions_df[
    [
        "순위",
        "지역명",
        "행정단위",
        "총인구수",
        "유사도점수",
        "행정구역코드",
    ]
].copy()

display_result_df["유사도점수"] = (
    display_result_df["유사도점수"]
    .round(4)
)

st.dataframe(
    display_result_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "순위": st.column_config.NumberColumn(
            "순위",
            format="%d위",
        ),
        "지역명": st.column_config.TextColumn(
            "지역명"
        ),
        "행정단위": st.column_config.TextColumn(
            "행정단위"
        ),
        "총인구수": st.column_config.NumberColumn(
            "총인구수",
            format="%d명",
        ),
        "유사도점수": st.column_config.NumberColumn(
            "유사도 점수",
            format="%.4f점",
        ),
        "행정구역코드": st.column_config.TextColumn(
            "행정구역 코드"
        ),
    },
)


# =========================================================
# 결과 다운로드
# =========================================================

download_df = display_result_df.copy()

download_df.insert(
    0,
    "기준지역",
    selected_region_name,
)

download_df.insert(
    1,
    "기준지역행정단위",
    selected_admin_level,
)

download_df.insert(
    2,
    "비교범위",
    comparison_scope,
)

download_csv = download_df.to_csv(
    index=False,
).encode("utf-8-sig")

safe_region_name = re.sub(
    r'[\\/:*?"<>|]',
    "_",
    selected_region_name,
)

st.download_button(
    label="유사 지역 결과 CSV 다운로드",
    data=download_csv,
    file_name=(
        f"{safe_region_name}_인구구조_유사지역_TOP5.csv"
    ),
    mime="text/csv",
)


# =========================================================
# 계산 방식 설명
# =========================================================

with st.expander(
    "유사 지역 계산 방법 보기",
    expanded=False,
):
    st.markdown(
        """
### 계산 과정

1. 각 지역의 0세부터 100세 이상까지 인구수를 추출합니다.
2. 각 연령 인구수를 해당 지역의 전체 연령 인구수로 나눕니다.
3. 각 지역을 101개 연령 비율로 이루어진 벡터로 만듭니다.
4. 선택 지역과 전국 비교 지역 사이의 코사인 유사도를 계산합니다.
5. 자기 자신을 제외하고 유사도가 높은 순서대로 5곳을 표시합니다.

### 점수 해석

- **100점에 가까울수록** 연령별 인구분포의 모양이 비슷합니다.
- 총인구가 비슷하다는 의미는 아닙니다.
- 인구가 매우 적은 지역은 연령별 분포가 불안정할 수 있으므로 사이드바에서 최소 인구수를 설정할 수 있습니다.
        """
    )
