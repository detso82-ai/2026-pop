import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="지역별 연령 인구구조",
    page_icon="👥",
    layout="wide",
)

DATA_FILENAME = "202606_202606_연령별인구현황_월간.csv"
DATA_PATH = Path(__file__).resolve().parent / DATA_FILENAME


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data(show_spinner="인구 데이터를 불러오는 중입니다.")
def load_data(file_path: Path) -> pd.DataFrame:
    """
    행정안전부 연령별 인구현황 CSV 파일을 불러옵니다.
    파일은 app.py와 같은 폴더에 있어야 합니다.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {file_path.name}"
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

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        "지원하는 인코딩으로 CSV 파일을 읽을 수 없습니다.",
    ) from last_error


# =========================================================
# 데이터 정리 함수
# =========================================================

def normalize_text(value) -> str:
    """문자열의 앞뒤 공백과 연속된 공백을 정리합니다."""

    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def remove_region_code(region_name: str) -> str:
    """
    행정구역 이름 뒤의 행정구역 코드를 제거합니다.

    예:
    서울특별시 종로구 (1111000000)
    → 서울특별시 종로구
    """

    region_name = normalize_text(region_name)

    return re.sub(
        r"\s*\(\d+\)\s*$",
        "",
        region_name,
    ).strip()


def extract_region_code(region_name: str) -> str:
    """행정구역 이름에 포함된 숫자 코드를 추출합니다."""

    match = re.search(
        r"\((\d+)\)\s*$",
        normalize_text(region_name),
    )

    if match:
        return match.group(1)

    return ""


def convert_to_number(value) -> int:
    """
    쉼표가 들어간 인구수 문자열을 정수로 변환합니다.

    예:
    '1,152' → 1152
    """

    if pd.isna(value):
        return 0

    text = str(value).strip()
    text = text.replace(",", "")

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


def find_column(
    columns,
    sex_code: str,
    category: str,
) -> str | None:
    """
    연월 부분과 관계없이 열 이름의 끝부분을 이용해 열을 찾습니다.

    예:
    2026년06월_계_30세
    2026년06월_남_총인구수
    """

    suffix = f"_{sex_code}_{category}"

    for column in columns:
        if str(column).strip().endswith(suffix):
            return column

    return None


def get_population_value(
    row: pd.Series,
    columns,
    sex_code: str,
    category: str,
) -> int:
    """선택한 성별 및 항목의 인구수를 가져옵니다."""

    column = find_column(
        columns=columns,
        sex_code=sex_code,
        category=category,
    )

    if column is None:
        return 0

    return convert_to_number(row[column])


def extract_age_population(
    row: pd.Series,
    columns,
) -> pd.DataFrame:
    """
    선택 지역의 0세부터 100세 이상까지
    전체, 남성, 여성 인구를 긴 형식 데이터로 변환합니다.
    """

    records = []

    sex_mapping = {
        "계": "전체",
        "남": "남성",
        "여": "여성",
    }

    for sex_code, sex_name in sex_mapping.items():

        for age in range(100):
            category = f"{age}세"

            population = get_population_value(
                row=row,
                columns=columns,
                sex_code=sex_code,
                category=category,
            )

            records.append(
                {
                    "연령순서": age,
                    "연령": category,
                    "구분": sex_name,
                    "인구수": population,
                }
            )

        population_100_plus = get_population_value(
            row=row,
            columns=columns,
            sex_code=sex_code,
            category="100세 이상",
        )

        records.append(
            {
                "연령순서": 100,
                "연령": "100세 이상",
                "구분": sex_name,
                "인구수": population_100_plus,
            }
        )

    return pd.DataFrame(records)


def calculate_age_group_population(
    age_data: pd.DataFrame,
    minimum_age: int,
    maximum_age: int | None = None,
) -> int:
    """전체 인구 중 지정한 연령 구간의 인구수를 계산합니다."""

    filtered = age_data[
        age_data["구분"] == "전체"
    ]

    if maximum_age is None:
        filtered = filtered[
            filtered["연령순서"] >= minimum_age
        ]

    else:
        filtered = filtered[
            filtered["연령순서"].between(
                minimum_age,
                maximum_age,
            )
        ]

    return int(filtered["인구수"].sum())


# =========================================================
# Plotly 그래프 함수
# =========================================================

def create_population_chart(
    chart_data: pd.DataFrame,
    selected_categories: list[str],
    chart_mode: str,
    total_population: int,
    region_name: str,
    show_markers: bool,
    line_width: int,
) -> go.Figure:
    """연령별 인구구조 꺾은선 그래프를 생성합니다."""

    filtered_data = chart_data[
        chart_data["구분"].isin(selected_categories)
    ].copy()

    if chart_mode == "인구 비율 (%)":
        denominator = max(total_population, 1)

        filtered_data["그래프값"] = (
            filtered_data["인구수"]
            / denominator
            * 100
        )

        y_axis_title = "전체 인구 대비 비율 (%)"
        hover_template = (
            "<b>%{x}</b><br>"
            "%{fullData.name}: %{y:.3f}%"
            "<extra></extra>"
        )

    else:
        filtered_data["그래프값"] = (
            filtered_data["인구수"]
        )

        y_axis_title = "인구수 (명)"
        hover_template = (
            "<b>%{x}</b><br>"
            "%{fullData.name}: %{y:,.0f}명"
            "<extra></extra>"
        )

    figure = go.Figure()

    line_dash_mapping = {
        "전체": "solid",
        "남성": "dash",
        "여성": "dot",
    }

    for category in selected_categories:
        category_data = filtered_data[
            filtered_data["구분"] == category
        ].sort_values("연령순서")

        mode = (
            "lines+markers"
            if show_markers
            else "lines"
        )

        figure.add_trace(
            go.Scatter(
                x=category_data["연령"],
                y=category_data["그래프값"],
                mode=mode,
                name=category,
                line={
                    "width": line_width,
                    "dash": line_dash_mapping.get(
                        category,
                        "solid",
                    ),
                },
                marker={
                    "size": 5,
                },
                customdata=category_data[
                    ["인구수"]
                ],
                hovertemplate=hover_template,
            )
        )

    x_tick_values = [
        f"{age}세"
        for age in range(0, 100, 5)
    ]

    x_tick_values.append("100세 이상")

    figure.update_layout(
        title={
            "text": f"{region_name} 연령별 인구구조",
            "x": 0.02,
            "xanchor": "left",
            "font": {
                "size": 24,
            },
        },
        xaxis_title="연령",
        yaxis_title=y_axis_title,
        hovermode="x unified",
        height=650,
        margin={
            "l": 30,
            "r": 30,
            "t": 80,
            "b": 40,
        },
        legend={
            "title": {
                "text": "구분",
            },
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    figure.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=[
            f"{age}세"
            for age in range(100)
        ] + ["100세 이상"],
        tickmode="array",
        tickvals=x_tick_values,
        ticktext=x_tick_values,
        showgrid=False,
        rangeslider={
            "visible": True,
            "thickness": 0.06,
        },
    )

    figure.update_yaxes(
        rangemode="tozero",
        separatethousands=True,
        gridcolor="rgba(128, 128, 128, 0.18)",
        zeroline=True,
        zerolinecolor="rgba(128, 128, 128, 0.35)",
    )

    return figure


# =========================================================
# 데이터 불러오기 및 검증
# =========================================================

try:
    population_df = load_data(DATA_PATH)

except FileNotFoundError:
    st.error(
        "데이터 파일을 찾을 수 없습니다."
    )

    st.code(
        DATA_FILENAME,
        language="text",
    )

    st.write(
        "위 CSV 파일을 `app.py`와 같은 폴더에 넣어 주세요."
    )

    st.stop()

except Exception as error:
    st.error(
        "데이터 파일을 읽는 과정에서 오류가 발생했습니다."
    )

    st.exception(error)
    st.stop()


if "행정구역" not in population_df.columns:
    st.error(
        "CSV 파일에서 `행정구역` 열을 찾을 수 없습니다."
    )

    st.write(
        "행정안전부 연령별 인구현황 CSV 파일인지 확인해 주세요."
    )

    st.stop()


# =========================================================
# 행정구역 정보 정리
# =========================================================

population_df = population_df.copy()

population_df["행정구역_원본"] = (
    population_df["행정구역"]
    .apply(normalize_text)
)

population_df["지역명"] = (
    population_df["행정구역_원본"]
    .apply(remove_region_code)
)

population_df["행정구역코드"] = (
    population_df["행정구역_원본"]
    .apply(extract_region_code)
)

population_df = population_df[
    population_df["지역명"] != ""
].reset_index(drop=True)


# 같은 지역명이 중복될 수 있으므로 원본 행정구역 값을 선택값으로 사용
region_options = (
    population_df["행정구역_원본"]
    .drop_duplicates()
    .tolist()
)


def format_region_option(region_original: str) -> str:
    """선택 상자에 표시할 지역 이름을 만듭니다."""

    region_name = remove_region_code(
        region_original
    )

    region_code = extract_region_code(
        region_original
    )

    if region_code:
        return f"{region_name} · {region_code}"

    return region_name


# =========================================================
# 화면 제목
# =========================================================

st.title("👥 지역별 연령 인구구조")

st.write(
    "지역명을 검색하고 선택하면 0세부터 100세 이상까지의 "
    "전체·남성·여성 인구구조를 꺾은선 그래프로 확인할 수 있습니다."
)


# =========================================================
# 사이드바 설정
# =========================================================

with st.sidebar:
    st.header("그래프 설정")

    chart_mode = st.radio(
        "표시 기준",
        options=[
            "인구수 (명)",
            "인구 비율 (%)",
        ],
        index=0,
    )

    selected_categories = st.multiselect(
        "표시할 자료",
        options=[
            "전체",
            "남성",
            "여성",
        ],
        default=[
            "전체",
            "남성",
            "여성",
        ],
    )

    show_markers = st.checkbox(
        "연령별 점 표시",
        value=False,
    )

    line_width = st.slider(
        "선 굵기",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
    )

    st.divider()

    st.caption(
        f"데이터 파일: {DATA_FILENAME}"
    )

    st.caption(
        f"지역 수: {len(population_df):,}개"
    )


# =========================================================
# 지역 검색 및 선택
# =========================================================

st.subheader("지역 선택")

search_keyword = st.text_input(
    "지역명 검색",
    placeholder="예: 서울, 종로구, 청운효자동",
    help=(
        "지역명의 일부를 입력할 수 있습니다. "
        "여러 단어를 띄어 쓰면 모든 단어가 포함된 지역만 표시됩니다."
    ),
)


search_terms = [
    word.strip().lower()
    for word in search_keyword.split()
    if word.strip()
]


if search_terms:
    filtered_region_options = []

    for region_original in region_options:
        searchable_text = (
            f"{remove_region_code(region_original)} "
            f"{extract_region_code(region_original)}"
        ).lower()

        if all(
            term in searchable_text
            for term in search_terms
        ):
            filtered_region_options.append(
                region_original
            )

else:
    filtered_region_options = region_options


if not filtered_region_options:
    st.warning(
        "검색 조건과 일치하는 지역이 없습니다. "
        "검색어를 짧게 입력해 보세요."
    )

    st.stop()


selected_region_original = st.selectbox(
    "검색 결과에서 지역 선택",
    options=filtered_region_options,
    format_func=format_region_option,
)


selected_rows = population_df[
    population_df["행정구역_원본"]
    == selected_region_original
]


if selected_rows.empty:
    st.error(
        "선택한 지역의 데이터를 찾을 수 없습니다."
    )

    st.stop()


selected_row = selected_rows.iloc[0]
selected_region_name = selected_row["지역명"]


# =========================================================
# 선택 지역 데이터 추출
# =========================================================

age_population_df = extract_age_population(
    row=selected_row,
    columns=population_df.columns,
)


total_population = get_population_value(
    row=selected_row,
    columns=population_df.columns,
    sex_code="계",
    category="총인구수",
)

male_population = get_population_value(
    row=selected_row,
    columns=population_df.columns,
    sex_code="남",
    category="총인구수",
)

female_population = get_population_value(
    row=selected_row,
    columns=population_df.columns,
    sex_code="여",
    category="총인구수",
)


# 총인구수 열을 찾지 못한 경우 연령별 합계 사용
if total_population == 0:
    total_population = int(
        age_population_df[
            age_population_df["구분"] == "전체"
        ]["인구수"].sum()
    )

if male_population == 0:
    male_population = int(
        age_population_df[
            age_population_df["구분"] == "남성"
        ]["인구수"].sum()
    )

if female_population == 0:
    female_population = int(
        age_population_df[
            age_population_df["구분"] == "여성"
        ]["인구수"].sum()
    )


population_0_14 = calculate_age_group_population(
    age_data=age_population_df,
    minimum_age=0,
    maximum_age=14,
)

population_15_64 = calculate_age_group_population(
    age_data=age_population_df,
    minimum_age=15,
    maximum_age=64,
)

population_65_plus = calculate_age_group_population(
    age_data=age_population_df,
    minimum_age=65,
)


# =========================================================
# 주요 지표
# =========================================================

st.subheader(f"{selected_region_name} 주요 지표")

metric_columns = st.columns(6)

metric_columns[0].metric(
    label="총인구",
    value=f"{total_population:,}명",
)

metric_columns[1].metric(
    label="남성",
    value=f"{male_population:,}명",
)

metric_columns[2].metric(
    label="여성",
    value=f"{female_population:,}명",
)

metric_columns[3].metric(
    label="0~14세",
    value=f"{population_0_14:,}명",
    delta=(
        f"{population_0_14 / total_population * 100:.1f}%"
        if total_population > 0
        else None
    ),
    delta_color="off",
)

metric_columns[4].metric(
    label="15~64세",
    value=f"{population_15_64:,}명",
    delta=(
        f"{population_15_64 / total_population * 100:.1f}%"
        if total_population > 0
        else None
    ),
    delta_color="off",
)

metric_columns[5].metric(
    label="65세 이상",
    value=f"{population_65_plus:,}명",
    delta=(
        f"{population_65_plus / total_population * 100:.1f}%"
        if total_population > 0
        else None
    ),
    delta_color="off",
)


# =========================================================
# 그래프
# =========================================================

st.subheader("연령별 인구구조")

if not selected_categories:
    st.warning(
        "왼쪽 사이드바에서 표시할 자료를 하나 이상 선택해 주세요."
    )

else:
    population_figure = create_population_chart(
        chart_data=age_population_df,
        selected_categories=selected_categories,
        chart_mode=chart_mode,
        total_population=total_population,
        region_name=selected_region_name,
        show_markers=show_markers,
        line_width=line_width,
    )

    st.plotly_chart(
        population_figure,
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
                    f"{selected_region_name}_연령별_인구구조"
                ),
                "height": 700,
                "width": 1400,
                "scale": 2,
            },
        },
    )


# =========================================================
# 연령별 표
# =========================================================

with st.expander(
    "연령별 인구 데이터 표 보기",
    expanded=False,
):
    population_table = (
        age_population_df
        .pivot_table(
            index=[
                "연령순서",
                "연령",
            ],
            columns="구분",
            values="인구수",
            aggfunc="sum",
        )
        .reset_index()
        .sort_values("연령순서")
    )

    population_table.columns.name = None

    population_table = population_table.drop(
        columns=["연령순서"]
    )

    column_order = [
        column
        for column in [
            "연령",
            "전체",
            "남성",
            "여성",
        ]
        if column in population_table.columns
    ]

    population_table = population_table[
        column_order
    ]

    st.dataframe(
        population_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "연령": st.column_config.TextColumn(
                "연령"
            ),
            "전체": st.column_config.NumberColumn(
                "전체",
                format="%d명",
            ),
            "남성": st.column_config.NumberColumn(
                "남성",
                format="%d명",
            ),
            "여성": st.column_config.NumberColumn(
                "여성",
                format="%d명",
            ),
        },
    )

    download_csv = population_table.to_csv(
        index=False,
    ).encode("utf-8-sig")

    safe_region_name = re.sub(
        r'[\\/:*?"<>|]',
        "_",
        selected_region_name,
    )

    st.download_button(
        label="연령별 데이터 CSV 다운로드",
        data=download_csv,
        file_name=(
            f"{safe_region_name}_연령별인구.csv"
        ),
        mime="text/csv",
    )


# =========================================================
# 원본 데이터 정보
# =========================================================

with st.expander(
    "원본 데이터 정보",
    expanded=False,
):
    st.write(
        f"파일명: `{DATA_FILENAME}`"
    )

    st.write(
        f"전체 행 수: {len(population_df):,}개"
    )

    st.write(
        f"전체 열 수: {len(population_df.columns):,}개"
    )

    st.write(
        f"선택한 행정구역 코드: "
        f"`{selected_row['행정구역코드']}`"
    )

    st.dataframe(
        population_df.head(),
        use_container_width=True,
    )
