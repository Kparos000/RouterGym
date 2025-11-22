"""Statistical analysis utilities for RouterGym evaluations."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
except Exception:  # pragma: no cover
    sm = None
    ols = None

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_two_way_anova(df: pd.DataFrame, metric: str, factorA: str, factorB: str) -> pd.DataFrame:
    """Run two-way ANOVA on a metric with two factors."""
    if df.empty or sm is None or ols is None:
        return pd.DataFrame()
    formula = f"{metric} ~ C({factorA}) + C({factorB}) + C({factorA}):C({factorB})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table


def export_anova_results(df: pd.DataFrame, filename: str = "anova_results.csv") -> Path:
    """Run ANOVA across common factors and export to CSV."""
    if df.empty:
        path = RESULTS_DIR / filename
        pd.DataFrame().to_csv(path, index=False)
        return path
    anova_df = run_two_way_anova(df, metric="accuracy", factorA="router", factorB="memory")
    path = RESULTS_DIR / filename
    anova_df.to_csv(path)
    return path


__all__ = ["run_two_way_anova", "export_anova_results"]
