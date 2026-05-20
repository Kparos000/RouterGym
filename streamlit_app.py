from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "RouterGym" / "results" / "analysis_outputs"
PLOTS_DIR = ANALYSIS_DIR / "plots"
MANUAL_DIR = ANALYSIS_DIR / "manual_audit"
GOLD_DIR = ANALYSIS_DIR / "gold_resolution_eval"
GOLD_PLOTS_DIR = GOLD_DIR / "plots"
SAMPLE_PATH = ANALYSIS_DIR / "streamlit_ticket_sample.csv"

CONFIG_ORDER = [
    "llm_only__base_llm1__mem_rag_bm25",
    "llm_only__base_llm2__mem_rag_bm25",
    "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25",
    "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25",
    "slm_only__base_slm1__mem_rag_bm25",
    "slm_only__base_slm2__mem_rag_bm25",
]

CONFIG_LABELS = {
    "llm_only__base_llm1__mem_rag_bm25": "LLM-only / LLM1",
    "llm_only__base_llm2__mem_rag_bm25": "LLM-only / LLM2",
    "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25": "SLM-dom / SLM1 -> LLM2",
    "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25": "SLM-dom / SLM2 -> LLM2",
    "slm_only__base_slm1__mem_rag_bm25": "SLM-only / SLM1",
    "slm_only__base_slm2__mem_rag_bm25": "SLM-only / SLM2",
}

ROUTER_LABELS = {
    "llm_only": "LLM-only",
    "slm_only": "SLM-only",
    "slm_dominant": "SLM-dominant",
}

ROUTER_COLORS = {
    "LLM-only": "#3b82f6",
    "SLM-only": "#10b981",
    "SLM-dominant": "#f59e0b",
}


st.set_page_config(
    page_title="RouterGym Results Explorer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .rg-subtle { color: #5f6b7a; font-size: 0.95rem; }
    .rg-callout {
        border-left: 4px solid #2563eb;
        background: #f8fafc;
        padding: 0.9rem 1rem;
        margin: 0.6rem 0 1rem 0;
    }
    .rg-warning {
        border-left: 4px solid #d97706;
        background: #fffbeb;
        padding: 0.9rem 1rem;
        margin: 0.6rem 0 1rem 0;
    }
    .rg-good {
        border-left: 4px solid #059669;
        background: #ecfdf5;
        padding: 0.9rem 1rem;
        margin: 0.6rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text)


def safe_read_csv(path: str | Path, *, warn: bool = False) -> pd.DataFrame:
    """Read a CSV if present; return an empty frame instead of crashing."""
    csv_path = _as_path(path)
    if not csv_path.exists():
        if warn:
            st.warning(f"Missing optional file: `{csv_path.relative_to(ROOT)}`")
        return pd.DataFrame()
    try:
        return _read_csv_cached(str(csv_path))
    except Exception as exc:  # pragma: no cover - Streamlit surface for bad local files.
        st.warning(f"Could not read `{csv_path.relative_to(ROOT)}`: {exc}")
        return pd.DataFrame()


def find_file(patterns: Sequence[str], roots: Sequence[Path] | None = None) -> Path | None:
    """Return the first file matching any pattern under the provided roots."""
    search_roots = roots or (ANALYSIS_DIR, PLOTS_DIR, MANUAL_DIR, GOLD_DIR, GOLD_PLOTS_DIR)
    for root in search_roots:
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def format_currency(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    if abs(number) < 0.01:
        return f"${number:,.6f}"
    return f"${number:,.2f}"


def format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.2%}"


def format_number(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}"


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def infer_router_family(config: object) -> str:
    text = str(config or "")
    if text.startswith("llm_only"):
        return "llm_only"
    if text.startswith("slm_only"):
        return "slm_only"
    if text.startswith("slm_dominant"):
        return "slm_dominant"
    return ""


def config_label(config: object) -> str:
    text = str(config or "")
    return CONFIG_LABELS.get(text, text.replace("__", " / ").replace("_", " "))


def standardize_config_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable display labels and router-family labels without mutating the input frame."""
    if df.empty:
        return df.copy()
    out = df.copy()
    config_col = first_existing_column(out, ["analysis_config", "config_identifier", "config"])
    if config_col is None:
        out["analysis_config"] = ""
    elif config_col != "analysis_config":
        out["analysis_config"] = out[config_col].astype(str)
    else:
        out["analysis_config"] = out["analysis_config"].astype(str)

    out["config_label"] = out["analysis_config"].map(config_label)
    if "router_family" not in out.columns:
        router_col = first_existing_column(out, ["router_mode", "router"])
        out["router_family"] = out[router_col].fillna("").astype(str) if router_col else ""
    out["router_family"] = out["router_family"].replace("", np.nan)
    out["router_family"] = out["router_family"].fillna(
        out["analysis_config"].map(infer_router_family)
    )
    out["router_family_label"] = (
        out["router_family"].map(ROUTER_LABELS).fillna(out["router_family"])
    )
    out["_config_rank"] = (
        out["analysis_config"]
        .map({config: index for index, config in enumerate(CONFIG_ORDER)})
        .fillna(999)
    )
    return out.sort_values(["_config_rank", "config_label"]).reset_index(drop=True)


def chart_template() -> dict:
    return {
        "template": "plotly_white",
        "height": 420,
        "margin": {"l": 10, "r": 10, "t": 60, "b": 20},
        "legend_title_text": "",
    }


def show_bar(
    df: pd.DataFrame,
    y: str,
    title: str,
    y_title: str,
    *,
    tickformat: str | None = None,
) -> None:
    if df.empty or y not in df.columns:
        st.info(f"`{y}` is not available for this chart.")
        return
    chart_df = df.dropna(subset=[y]).copy()
    if chart_df.empty:
        st.info(f"No non-empty values available for `{y}`.")
        return
    fig = px.bar(
        chart_df,
        x="config_label",
        y=y,
        color="router_family_label",
        color_discrete_map=ROUTER_COLORS,
        title=title,
        labels={"config_label": "Configuration", y: y_title},
        hover_data=["analysis_config"],
    )
    fig.update_layout(**chart_template(), xaxis_tickangle=-30)
    fig.update_yaxes(tickformat=tickformat)
    st.plotly_chart(fig, use_container_width=True)


def show_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    *,
    size: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    x_tickformat: str | None = None,
    y_tickformat: str | None = None,
) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info(f"Required columns for `{title}` are not available.")
        return
    chart_df = df.dropna(subset=[x, y]).copy()
    if chart_df.empty:
        st.info(f"No non-empty values available for `{title}`.")
        return
    size_arg = size if size in chart_df.columns else None
    fig = px.scatter(
        chart_df,
        x=x,
        y=y,
        size=size_arg,
        color="router_family_label",
        color_discrete_map=ROUTER_COLORS,
        text="config_label",
        title=title,
        labels={
            x: x_title or x,
            y: y_title or y,
            "router_family_label": "Router family",
            size or "": size or "",
        },
        hover_data=["analysis_config"],
    )
    fig.update_traces(textposition="top center", marker={"sizemin": 8, "line": {"width": 1}})
    fig.update_layout(**chart_template())
    fig.update_xaxes(tickformat=x_tickformat)
    fig.update_yaxes(tickformat=y_tickformat)
    st.plotly_chart(fig, use_container_width=True)


def show_plot_image(patterns: Sequence[str], caption: str | None = None) -> None:
    path = find_file(patterns)
    if path is None:
        return
    try:
        image = Image.open(path)
        st.image(image, caption=caption or path.name, use_container_width=True)
    except Exception:
        return


@st.cache_data(show_spinner=False)
def _load_config_comparator_cached() -> pd.DataFrame:
    summary = pd.read_csv(ANALYSIS_DIR / "summary_by_config.csv")
    projected_path = ANALYSIS_DIR / "projected_47k_cost_by_config.csv"
    routing_path = ANALYSIS_DIR / "routing_escalation_summary.csv"

    out = summary.copy()
    if projected_path.exists():
        projected = pd.read_csv(projected_path)
        keep = [
            col
            for col in ["analysis_config", "projected_47k_cost_usd", "avg_cost_per_ticket_usd"]
            if col in projected.columns
        ]
        out = out.merge(projected[keep], on="analysis_config", how="left")
    if routing_path.exists():
        routing = pd.read_csv(routing_path)
        keep = [col for col in ["analysis_config", "escalation_rate"] if col in routing.columns]
        out = out.merge(routing[keep], on="analysis_config", how="left")
    return out


def load_config_comparator() -> pd.DataFrame:
    path = ANALYSIS_DIR / "summary_by_config.csv"
    if not path.exists():
        st.warning("Missing `summary_by_config.csv`; comparator charts are unavailable.")
        return pd.DataFrame()
    try:
        return standardize_config_labels(_load_config_comparator_cached())
    except Exception as exc:
        st.warning(f"Could not build configuration comparator: {exc}")
        return pd.DataFrame()


def load_quality_frontier() -> pd.DataFrame:
    summary = load_config_comparator()
    if summary.empty:
        return summary

    cols = [
        "analysis_config",
        "config_label",
        "router_family",
        "router_family_label",
        "avg_total_cost_usd",
        "mean_latency_ms",
        "analysis_usable_rate",
        "parse_error_rate",
        "validation_error_rate",
    ]
    frontier = summary[[col for col in cols if col in summary.columns]].copy()

    manual = standardize_config_labels(safe_read_csv(MANUAL_DIR / "manual_quality_by_config.csv"))
    if not manual.empty:
        manual_cols = [
            "analysis_config",
            "mean_overall_manual_quality",
            "manual_quality_pass_rate_7",
        ]
        frontier = frontier.merge(
            manual[[col for col in manual_cols if col in manual.columns]],
            on="analysis_config",
            how="left",
        )

    gold = standardize_config_labels(
        safe_read_csv(GOLD_DIR / "gold_resolution_quality_by_config.csv")
    )
    if not gold.empty:
        gold_cols = [
            "analysis_config",
            "mean_overall_gold_quality_score",
            "generated_category_accuracy",
            "pass_rate_070",
            "pass_rate_080",
        ]
        frontier = frontier.merge(
            gold[[col for col in gold_cols if col in gold.columns]],
            on="analysis_config",
            how="left",
        )
    return standardize_config_labels(frontier)


def has_text(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return text.lower() not in {"", "none", "nan", "null"}


def parse_list_value(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        return [json.dumps(parsed, ensure_ascii=False)]
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text]


def bool_label(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return "yes"
    if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
        return "no"
    return str(value)


def metric_row(items: Sequence[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def render_interpretation(title: str, body: str, kind: str = "callout") -> None:
    css_class = {"callout": "rg-callout", "warning": "rg-warning", "good": "rg-good"}[kind]
    st.markdown(
        f"""<div class="{css_class}"><strong>{title}</strong><br>{body}</div>""",
        unsafe_allow_html=True,
    )


def page_executive_snapshot() -> None:
    st.title("RouterGym Results Explorer")
    st.subheader(
        "From LLM-First to SLM-Dominant: "
        "A Router-Memory Co-Design and Conversion Benchmark for Agentic Systems"
    )
    st.markdown(
        """
        RouterGym evaluates whether enterprise support-ticket agents can move from an
        expensive LLM-first design to an SLM-dominant design with routing, BM25 retrieval
        memory, structured output validation, and selective escalation.
        """
    )

    render_interpretation(
        "Key conclusion",
        "SLM-dominant routing reduces cost and latency substantially, but production "
        "viability depends on generated-answer quality, parse reliability, escalation "
        "behaviour, and human audit.",
        "good",
    )

    metric_row(
        [
            ("Generated outputs", "60,000"),
            ("Configurations", "6"),
            ("Tickets per config", "10,000"),
        ]
    )
    metric_row(
        [
            ("Memory mode", "BM25-RAG"),
            ("Gold-scored outputs", "456"),
            ("Manual audit", "Completed"),
        ]
    )

    comparator = load_config_comparator()
    if not comparator.empty:
        slm_dom = comparator[
            comparator["analysis_config"] == "slm_dominant__base_slm2__esc_llm2__mem_rag_bm25"
        ]
        llm1 = comparator[comparator["analysis_config"] == "llm_only__base_llm1__mem_rag_bm25"]
        if not slm_dom.empty and not llm1.empty:
            left = llm1.iloc[0]
            right = slm_dom.iloc[0]
            st.markdown("#### Conservative baseline vs strongest operational trade-off")
            metric_row(
                [
                    ("LLM1 avg cost", format_currency(left.get("avg_total_cost_usd"))),
                    ("SLM-dom SLM2 avg cost", format_currency(right.get("avg_total_cost_usd"))),
                    ("SLM-dom usable rate", format_percent(right.get("analysis_usable_rate"))),
                ]
            )
            metric_row(
                [
                    ("LLM1 mean latency", f"{format_number(left.get('mean_latency_ms'), 0)} ms"),
                    (
                        "SLM-dom SLM2 mean latency",
                        f"{format_number(right.get('mean_latency_ms'), 0)} ms",
                    ),
                    (
                        "SLM-dom escalation rate",
                        format_percent(right.get("escalation_rate")),
                    ),
                ]
            )

    columns = st.columns([1.2, 1])
    with columns[0]:
        st.markdown("#### What the app lets you inspect")
        st.markdown(
            """
            - operational metrics across the six final benchmark configurations
            - deterministic gold-resolution quality and generated-category accuracy
            - blinded manual-audit scores and component-level failure modes
            - cost-latency-quality frontier plots
            - concrete ticket examples with per-configuration generated answers
            """
        )
    with columns[1]:
        show_plot_image(
            ["slm_dominant_vs_llm_baseline_tradeoff.png", "usable_rate_vs_cost.png"],
            "Benchmark trade-off plot from committed analysis outputs.",
        )


def page_configuration_comparator() -> None:
    st.title("Configuration Comparator")
    st.caption("Operational metrics from the balanced 60k BM25-RAG benchmark.")
    df = load_config_comparator()
    if df.empty:
        return

    table = pd.DataFrame(
        {
            "Config": df["config_label"],
            "Router family": df["router_family_label"],
            "Avg cost/ticket": df.get("avg_total_cost_usd", pd.Series(index=df.index)).map(
                format_currency
            ),
            "Projected 47k cost": df.get("projected_47k_cost_usd", pd.Series(index=df.index)).map(
                format_currency
            ),
            "Mean latency": df.get("mean_latency_ms", pd.Series(index=df.index)).map(
                lambda x: f"{format_number(x, 0)} ms"
            ),
            "Median latency": df.get("median_latency_ms", pd.Series(index=df.index)).map(
                lambda x: f"{format_number(x, 0)} ms"
            ),
            "P95 latency": df.get("p95_latency_ms", pd.Series(index=df.index)).map(
                lambda x: f"{format_number(x, 0)} ms"
            ),
            "Usable rate": df.get("analysis_usable_rate", pd.Series(index=df.index)).map(
                format_percent
            ),
            "Parse error rate": df.get("parse_error_rate", pd.Series(index=df.index)).map(
                format_percent
            ),
            "Validation error rate": df.get("validation_error_rate", pd.Series(index=df.index)).map(
                format_percent
            ),
            "Escalation rate": df.get("escalation_rate", pd.Series(index=df.index)).map(
                format_percent
            ),
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        show_bar(df, "avg_total_cost_usd", "Average cost per ticket", "USD")
    with chart_cols[1]:
        show_bar(df, "mean_latency_ms", "Mean latency by config", "Milliseconds")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        show_bar(
            df,
            "analysis_usable_rate",
            "Usable output rate by config",
            "Usable output rate",
            tickformat=".1%",
        )
    with chart_cols[1]:
        show_bar(
            df,
            "parse_error_rate",
            "Parse error rate by config",
            "Parse error rate",
            tickformat=".1%",
        )


def page_quality_evaluation() -> None:
    st.title("Gold and Manual Quality Evaluation")
    render_interpretation(
        "Why this page matters",
        "Classifier-derived category accuracy only checks whether a category label matches. "
        "It does not prove that the generated answer is actionable, complete, policy-grounded, "
        "or structurally usable.",
    )
    st.markdown(
        """
        The deterministic gold subset contains 76 overlapping tickets across the six final
        configurations, giving 456 generated outputs. The manual audit is a blinded review of
        the generated support resolutions.
        """
    )

    gold = standardize_config_labels(
        safe_read_csv(GOLD_DIR / "gold_resolution_quality_by_config.csv")
    )
    manual = standardize_config_labels(safe_read_csv(MANUAL_DIR / "manual_quality_by_config.csv"))
    components = standardize_config_labels(
        safe_read_csv(MANUAL_DIR / "manual_quality_component_summary_by_config.csv")
    )
    manual_frontier = standardize_config_labels(
        safe_read_csv(MANUAL_DIR / "manual_quality_vs_cost_latency.csv")
    )

    if not gold.empty:
        st.markdown("#### Deterministic gold quality")
        cols = st.columns(2)
        with cols[0]:
            show_bar(
                gold,
                "mean_overall_gold_quality_score",
                "Gold quality by config",
                "Mean gold quality score",
            )
        with cols[1]:
            show_bar(
                gold,
                "generated_category_accuracy",
                "Generated category accuracy by config",
                "Generated category accuracy",
                tickformat=".1%",
            )
        display_cols = [
            "config_label",
            "rows",
            "mean_overall_gold_quality_score",
            "pass_rate_070",
            "pass_rate_080",
            "generated_category_accuracy",
            "avg_cost_usd",
            "avg_latency_ms",
        ]
        st.dataframe(
            gold[[col for col in display_cols if col in gold.columns]],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("Gold quality summary is not available.")

    if not manual.empty:
        st.markdown("#### Blinded manual audit")
        cols = st.columns(2)
        with cols[0]:
            show_bar(
                manual,
                "mean_overall_manual_quality",
                "Manual quality by config",
                "Mean manual quality score",
            )
        with cols[1]:
            show_bar(
                manual,
                "manual_quality_pass_rate_7",
                "Manual pass rate >= 7",
                "Pass rate",
                tickformat=".1%",
            )

    if not components.empty:
        component_cols = [
            col for col in components.columns if col.startswith("mean_") and col.endswith("_manual")
        ]
        if component_cols:
            component_df = components.melt(
                id_vars=["analysis_config", "config_label", "router_family_label"],
                value_vars=component_cols,
                var_name="Component",
                value_name="Mean score",
            )
            component_df["Component"] = (
                component_df["Component"]
                .str.replace("mean_", "", regex=False)
                .str.replace("_manual", "", regex=False)
                .str.replace("_", " ")
                .str.title()
            )
            fig = px.bar(
                component_df,
                x="config_label",
                y="Mean score",
                color="Component",
                barmode="group",
                title="Manual component scores by config",
                labels={"config_label": "Configuration"},
            )
            fig.update_layout(**chart_template(), xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    if not manual_frontier.empty:
        st.markdown("#### Manual quality against cost and latency")
        cols = st.columns(2)
        with cols[0]:
            show_scatter(
                manual_frontier,
                "avg_cost_usd",
                "mean_overall_manual_quality",
                "Manual quality vs cost",
                size="manual_quality_pass_rate_7",
                x_title="Average cost per ticket",
                y_title="Mean manual quality",
            )
        with cols[1]:
            show_scatter(
                manual_frontier,
                "avg_latency_ms",
                "mean_overall_manual_quality",
                "Manual quality vs latency",
                size="manual_quality_pass_rate_7",
                x_title="Average latency (ms)",
                y_title="Mean manual quality",
            )


def page_frontier() -> None:
    st.title("Cost-Latency-Quality Frontier")
    st.caption("Frontier plots combine operational summaries with gold/manual quality scores.")

    frontier = load_quality_frontier()
    if frontier.empty:
        return

    quality_options: list[tuple[str, str, str]] = []
    if (
        "mean_overall_manual_quality" in frontier.columns
        and frontier["mean_overall_manual_quality"].notna().any()
    ):
        quality_options.append(
            ("Manual quality", "mean_overall_manual_quality", "Mean manual score")
        )
    if (
        "mean_overall_gold_quality_score" in frontier.columns
        and frontier["mean_overall_gold_quality_score"].notna().any()
    ):
        quality_options.append(
            ("Gold quality", "mean_overall_gold_quality_score", "Mean gold score")
        )
    if not quality_options:
        st.warning("No quality metric is available for frontier charts.")
        return

    selected_label = st.radio(
        "Quality signal",
        [label for label, _, _ in quality_options],
        horizontal=True,
    )
    _, quality_col, quality_axis = next(
        item for item in quality_options if item[0] == selected_label
    )

    cols = st.columns(2)
    with cols[0]:
        show_scatter(
            frontier,
            "avg_total_cost_usd",
            quality_col,
            f"{selected_label} vs average cost",
            size="analysis_usable_rate",
            x_title="Average cost per ticket",
            y_title=quality_axis,
        )
    with cols[1]:
        show_scatter(
            frontier,
            "mean_latency_ms",
            quality_col,
            f"{selected_label} vs mean latency",
            size="analysis_usable_rate",
            x_title="Mean latency (ms)",
            y_title=quality_axis,
        )

    cols = st.columns(2)
    with cols[0]:
        show_scatter(
            frontier,
            "parse_error_rate",
            quality_col,
            f"{selected_label} vs parse error rate",
            size="analysis_usable_rate",
            x_title="Parse error rate",
            y_title=quality_axis,
            x_tickformat=".1%",
        )
    with cols[1]:
        y_col = (
            "analysis_usable_rate" if "analysis_usable_rate" in frontier.columns else quality_col
        )
        show_scatter(
            frontier,
            "parse_error_rate",
            y_col,
            "Structural reliability frontier",
            size=quality_col,
            x_title="Parse error rate",
            y_title="Usable output rate" if y_col == "analysis_usable_rate" else quality_axis,
            x_tickformat=".1%",
            y_tickformat=".1%" if y_col == "analysis_usable_rate" else None,
        )

    st.markdown("#### Interpretation")
    col1, col2 = st.columns(2)
    with col1:
        render_interpretation(
            "LLM1",
            "Strongest conservative quality signal, but slowest and most expensive.",
        )
        render_interpretation(
            "SLM1",
            "Strong quality-per-dollar, but the production run exposes a parse reliability weakness.",
            "warning",
        )
    with col2:
        render_interpretation(
            "SLM2",
            "Cleaner structurally in the 60k run, but weaker manual answer-quality scores.",
            "warning",
        )
        render_interpretation(
            "SLM-dom SLM2 -> LLM2",
            "Strongest operational trade-off when cost, latency, usable rate, and escalation behaviour "
            "are considered together.",
            "good",
        )


def page_ticket_explorer() -> None:
    st.title("Ticket Explorer")
    sample = standardize_config_labels(safe_read_csv(SAMPLE_PATH))
    if sample.empty:
        st.warning(
            "The curated ticket sample is missing. Run "
            "`python RouterGym/analysis/build_streamlit_ticket_sample.py` to generate it."
        )
        return

    sample["ticket_id"] = sample["ticket_id"].astype(str)
    category_col = first_existing_column(sample, ["gold_label", "topic_group"])
    config_col = "config_label"
    router_col = "router_family_label"

    filter_cols = st.columns([1.2, 1.2, 1.2, 0.8])
    with filter_cols[0]:
        categories = ["All"]
        if category_col:
            categories += sorted(sample[category_col].dropna().astype(str).unique().tolist())
        selected_category = st.selectbox("Category", categories)
    with filter_cols[1]:
        routers = sorted(sample[router_col].dropna().astype(str).unique().tolist())
        selected_routers = st.multiselect("Router family", routers, default=routers)
    with filter_cols[2]:
        configs = sample[[config_col, "analysis_config", "_config_rank"]].drop_duplicates()
        configs = configs.sort_values(["_config_rank", config_col])
        config_labels = configs[config_col].tolist()
        selected_configs = st.multiselect("Configuration", config_labels, default=config_labels)
    with filter_cols[3]:
        blind_mode = st.toggle("Blind mode", value=False)

    filtered = sample.copy()
    if selected_category != "All" and category_col:
        filtered = filtered[filtered[category_col].astype(str) == selected_category]
    if selected_routers:
        filtered = filtered[filtered[router_col].isin(selected_routers)]
    if selected_configs:
        filtered = filtered[filtered[config_col].isin(selected_configs)]

    if filtered.empty:
        st.info("No rows match the current filters.")
        return

    ticket_options = (
        filtered[["ticket_id", category_col if category_col else "ticket_id"]]
        .drop_duplicates()
        .sort_values("ticket_id", key=lambda s: pd.to_numeric(s, errors="coerce").fillna(10**9))
    )
    labels = []
    label_to_id: dict[str, str] = {}
    for _, row in ticket_options.iterrows():
        label = (
            f"{row['ticket_id']} - {row[category_col]}" if category_col else str(row["ticket_id"])
        )
        labels.append(label)
        label_to_id[label] = str(row["ticket_id"])
    selected_ticket_label = st.selectbox("Ticket ID", labels)
    selected_ticket_id = label_to_id[selected_ticket_label]
    ticket_rows = filtered[filtered["ticket_id"] == selected_ticket_id].sort_values(
        ["_config_rank", "config_label"]
    )

    first_row = ticket_rows.iloc[0]
    st.markdown("#### Ticket")
    metric_row(
        [
            ("Ticket ID", selected_ticket_id),
            ("Gold label", str(first_row.get("gold_label", "n/a"))),
            ("Rows shown", str(len(ticket_rows))),
        ]
    )
    st.write(first_row.get("ticket_text", first_row.get("original_query", "")))

    if has_text(first_row.get("gold_resolution_summary")) or has_text(
        first_row.get("gold_resolution_steps")
    ):
        with st.expander("Gold resolution reference", expanded=False):
            if has_text(first_row.get("gold_resolution_summary")):
                st.markdown(f"**Summary:** {first_row.get('gold_resolution_summary')}")
            steps = parse_list_value(first_row.get("gold_resolution_steps"))
            if steps:
                st.markdown("**Expected steps**")
                for step in steps:
                    st.markdown(f"- {step}")
            acceptance = parse_list_value(first_row.get("gold_acceptance_criteria"))
            if acceptance:
                st.markdown("**Acceptance criteria**")
                for item in acceptance:
                    st.markdown(f"- {item}")

    config_to_system = {
        config: f"System {chr(65 + index)}"
        for index, config in enumerate(
            ticket_rows.sort_values("_config_rank")["analysis_config"].drop_duplicates().tolist()
        )
    }

    st.markdown("#### Generated outputs")
    for _, row in ticket_rows.iterrows():
        anonymous = row.get("anonymous_system_id")
        system_name = (
            str(anonymous)
            if blind_mode and has_text(anonymous)
            else config_to_system.get(row["analysis_config"], "System")
            if blind_mode
            else row["config_label"]
        )
        with st.expander(system_name, expanded=not blind_mode):
            metric_items = [
                ("Generated category", str(row.get("generated_predicted_category", "n/a"))),
                ("Escalated", bool_label(row.get("escalated"))),
                ("Generation valid", bool_label(row.get("generation_valid"))),
                ("Usable output", bool_label(row.get("usable_output"))),
            ]
            metric_row(metric_items)
            metric_row(
                [
                    ("Cost", format_currency(row.get("cost_usd"))),
                    ("Latency", f"{format_number(row.get('latency_ms'), 0)} ms"),
                    ("Tokens", format_number(row.get("total_tokens"), 0)),
                    (
                        "Gold / manual score",
                        f"{format_number(row.get('deterministic_gold_score'), 3)} / "
                        f"{format_number(row.get('manual_quality_score'), 1)}",
                    ),
                ]
            )

            if has_text(row.get("parse_error")) or has_text(row.get("validation_error")):
                render_interpretation(
                    "Structural issue",
                    f"parse_error={row.get('parse_error', '')}; "
                    f"validation_error={row.get('validation_error', '')}",
                    "warning",
                )

            st.markdown("**Generated answer**")
            st.write(row.get("final_answer", ""))

            steps = parse_list_value(row.get("resolution_steps"))
            if steps:
                st.markdown("**Generated steps**")
                for step in steps:
                    st.markdown(f"- {step}")

            if has_text(row.get("reasoning")):
                with st.expander("Reasoning", expanded=False):
                    st.write(row.get("reasoning"))

            if not blind_mode:
                st.caption(
                    f"Config: {row.get('analysis_config', '')} | "
                    f"Base: {row.get('base_model_key', '')} | "
                    f"Escalation: {row.get('escalation_model_key', '') or 'none'} | "
                    f"Memory: {row.get('memory_mode', '')}"
                )


def page_methodology() -> None:
    st.title("Methodology and Reproducibility")

    st.markdown("#### Architecture summary")
    st.code(
        """Support ticket
  -> classifier
  -> router policy
  -> BM25-RAG memory retrieval
  -> SLM, LLM, or SLM with selective LLM escalation
  -> structured JSON output
  -> schema validation
  -> operational, gold, and manual quality scoring""",
        language="text",
    )

    st.markdown("#### Final experimental matrix")
    matrix = pd.DataFrame(
        [
            ("LLM-only", "LLM1", "none", "BM25-RAG"),
            ("LLM-only", "LLM2", "none", "BM25-RAG"),
            ("SLM-only", "SLM1", "none", "BM25-RAG"),
            ("SLM-only", "SLM2", "none", "BM25-RAG"),
            ("SLM-dominant", "SLM1", "LLM2", "BM25-RAG"),
            ("SLM-dominant", "SLM2", "LLM2", "BM25-RAG"),
        ],
        columns=["Router family", "Base model", "Escalation model", "Memory"],
    )
    st.dataframe(matrix, hide_index=True, use_container_width=True)

    render_interpretation(
        "Why BM25-RAG was fixed",
        "The dissertation's final production-scale comparison isolates router/model strategy "
        "under a consistent operational memory layer. Memory-mode ablation is supported by the "
        "codebase but treated as future work for the production-scale claim.",
    )

    st.markdown("#### Data construction pipeline")
    st.markdown(
        """
        raw chunked outputs -> recovered/merged outputs -> available-row audit ->
        balanced 10k extraction -> balanced_10k_all_configs.jsonl -> 60k analysis dataset ->
        operational metrics, gold scoring, and blinded manual audit.
        """
    )

    st.markdown("#### Inference environment")
    metric_row(
        [
            ("RunPod GPUs", "4 x H200 SXM"),
            ("GPU rate", "$2.99 / GPU-hour"),
            ("Final 60k run", "~24 hours"),
        ]
    )
    metric_row(
        [
            ("Estimated final run cost", "$287.04"),
            ("Projected full outputs", "287,022"),
            ("Projected full run", "~114.8 hours / ~$1,373"),
        ]
    )

    st.markdown("#### Persistent storage story")
    st.markdown(
        """
        The production run used a network volume mounted under `/workspace` for model caches,
        tokenizer and config files, the RouterGym repo, vLLM logs, chunked outputs,
        manifests/status files, recovered results, and final bundles.
        """
    )

    st.markdown("#### Static demo constraints")
    st.markdown(
        """
        This Streamlit app is intentionally static and results-based. It does not run live
        model inference and does not require GPUs, API keys, vLLM, FAISS, sentence-transformers,
        Hugging Face tokens, or model downloads.
        """
    )


PAGES = {
    "Executive Snapshot": page_executive_snapshot,
    "Configuration Comparator": page_configuration_comparator,
    "Gold and Manual Quality": page_quality_evaluation,
    "Cost-Latency-Quality Frontier": page_frontier,
    "Ticket Explorer": page_ticket_explorer,
    "Methodology and Reproducibility": page_methodology,
}


def main() -> None:
    with st.sidebar:
        st.markdown("## RouterGym")
        st.caption("Static dissertation benchmark explorer")
        page_name = st.radio("Page", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption("No live inference. No model downloads. No API keys.")

    PAGES[page_name]()


if __name__ == "__main__":
    main()
