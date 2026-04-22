"""
generate_tables.py
==================
Read results/comparison/metrics_table.csv and write ready-to-\\input{} LaTeX
tables into results/tables/, matching the presentation style exactly.

Usage
-----
    python3 scripts/generate_tables.py

Output files (all in results/tables/)
--------------------------------------
    tvd_main.tex      TVD table, 5 scenarios (no correlated)
    tvd_all.tex       TVD table, all 6 scenarios
    rhat_main.tex     R-hat table, 5 scenarios
    rhat_all.tex      R-hat table, all 6 scenarios
    ess_all.tex       ESS | ESS/sec combined table, all 6 scenarios
    runtime_all.tex   Wall-clock runtime table, all 6 scenarios

In your .tex file, replace a hardcoded tabular with e.g.:
    \\input{results/tables/tvd_main.tex}
"""

import csv
import math
import os

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(ROOT, "results", "comparison", "metrics_table.csv")
OUTPUT_DIR = os.path.join(ROOT, "results", "tables")

# ── constants ──────────────────────────────────────────────────────────────────
METHODS     = ["teleporting", "hybrid", "parallel_tempering", "vanilla"]
METHOD_HDRS = [
    r"\textbf{Teleport}",
    r"\textbf{Hybrid}",
    r"\textbf{PT}",
    r"\textbf{NUTS}",
]

# Short LaTeX labels for each scenario (keys match the CSV 'scenario' column).
SCENARIO_LABELS = {
    "Standard Gaussian":
        "Standard Gaussian",
    "Correlated Gaussian (\u03c1=0.9)":      # ρ
        r"Correlated ($\rho{=}0.9$)",
    "Bimodal \u2014 moderate (\u03bc=\u00b15)":   # — μ ±
        r"Bimodal moderate ($\mu{=}\pm5$)",
    "Bimodal \u2014 large (\u03bc=\u00b115)":
        r"Bimodal large ($\mu{=}\pm15$)",
    "Unequal-weight bimodal (0.9 / 0.1)":
        r"Unequal weight (0.9/0.1)",
    "Different-scale bimodal (\u03c3=0.5 vs \u03c3=2)":  # σ
        r"Different scale ($\sigma{=}0.5$ vs $\sigma{=}2$)",
}

GREEN = r"\cellcolor{green!25}"
RED   = r"\cellcolor{red!25}"

# Rhat threshold above which ESS is considered "inflated" (stuck in one mode).
RHAT_STUCK = 1.1


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_float(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _fmt(v, fmt_spec):
    return r"\text{--}" if v is None else format(v, fmt_spec)


def _color_cells(values, fmt_spec, lower_better, footnote_mask=None):
    """
    Return a list of formatted, colour-coded cell strings.

    footnote_mask : list of bool, same length as values.
        When True for index i, appends $^*$ to the cell (e.g. inflated ESS).
    """
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return [_fmt(v, fmt_spec) for v in values]

    best_i  = min(valid, key=lambda x: x[1] if lower_better else -x[1])[0]
    worst_i = max(valid, key=lambda x: x[1] if lower_better else -x[1])[0]

    cells = []
    for i, v in enumerate(values):
        s = _fmt(v, fmt_spec)
        if footnote_mask and footnote_mask[i]:
            s += r"$^*$"
        if i == best_i:
            s = fr"{GREEN}\textbf{{{s}}}"
        elif i == worst_i:
            s = fr"{RED}{s}"
        cells.append(s)
    return cells


def _preamble(fontsize=r"\scriptsize", arraystretch="1.15", tabcolsep="4pt"):
    return (
        f"{fontsize}\n"
        fr"\renewcommand{{\arraystretch}}{{{arraystretch}}}"
        "\n"
        fr"\setlength{{\tabcolsep}}{{{tabcolsep}}}"
        "\n"
    )


def _tabular(rows):
    """Wrap rows in a booktabs tabular (lcccc)."""
    hdr = " & ".join([r"\textbf{Scenario}"] + METHOD_HDRS) + r" \\"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (
        r"\begin{tabular}{lcccc}" + "\n"
        r"\toprule" + "\n"
        + hdr + "\n"
        r"\midrule" + "\n"
        + body + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}"
    )


def _label(scenario):
    return SCENARIO_LABELS.get(scenario, scenario)


# ── table generators ───────────────────────────────────────────────────────────

def tvd_table(data, include_correlated=True):
    rows = []
    for sc, row in data.items():
        if not include_correlated and "Correlated" in sc:
            continue
        values = [_safe_float(row.get(f"tvd_{m}")) for m in METHODS]
        cells  = _color_cells(values, ".3f", lower_better=True)
        rows.append([_label(sc)] + cells)
    caption = r"\textbf{Total Variation Distance} (lower is better)\\[3pt]"
    return _preamble() + caption + "\n" + _tabular(rows)


def rhat_table(data, include_correlated=True):
    rows = []
    for sc, row in data.items():
        if not include_correlated and "Correlated" in sc:
            continue
        values = [_safe_float(row.get(f"rhat_{m}")) for m in METHODS]
        cells  = _color_cells(values, ".3f", lower_better=True)
        rows.append([_label(sc)] + cells)
    caption = (
        r"\textbf{$\hat{R}$} "
        r"(closer to 1 is better; ${>}1.1$ indicates non-convergence)\\[3pt]"
    )
    return _preamble() + caption + "\n" + _tabular(rows)


def ess_table(data, include_correlated=True):
    """
    Combined ESS | ESS/sec table.  Cells where R-hat > RHAT_STUCK get $^*$.
    Footnote explains the asterisk.
    """
    rows = []
    has_footnote = False
    for sc, row in data.items():
        if not include_correlated and "Correlated" in sc:
            continue
        rhats = [_safe_float(row.get(f"rhat_{m}")) for m in METHODS]
        stuck = [r is not None and r > RHAT_STUCK for r in rhats]
        if any(stuck):
            has_footnote = True

        cells = [_label(sc)]
        for i, m in enumerate(METHODS):
            ess = _safe_float(row.get(f"ess_{m}"))
            eps = _safe_float(row.get(f"ess_per_sec_{m}"))
            ess_s = _fmt(ess, ".0f")
            eps_s = _fmt(eps, ".0f")
            combined = fr"{ess_s} \textbar{{}} {eps_s}"
            if stuck[i]:
                combined += r"$^*$"
            cells.append(combined)
        rows.append(cells)

    caption = r"\textbf{ESS \textbar{} ESS\,/\,sec} (higher is better)\\[3pt]"
    out = _preamble(tabcolsep="3pt") + caption + "\n" + _tabular(rows)
    if has_footnote:
        out += (
            "\n\n"
            r"{\footnotesize $^*$ESS inflated: $\hat{R}>1.1$, "
            r"sampler stuck in one mode.}"
        )
    return out


def runtime_table(data, include_correlated=True):
    rows = []
    for sc, row in data.items():
        if not include_correlated and "Correlated" in sc:
            continue
        values = [_safe_float(row.get(f"time_{m}")) for m in METHODS]
        cells  = _color_cells(values, ".2f", lower_better=True)
        rows.append([_label(sc)] + cells)
    caption = r"\textbf{Wall-clock runtime (seconds)} (lower is better)\\[3pt]"
    return _preamble() + caption + "\n" + _tabular(rows)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read CSV
    data = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["scenario"]] = row

    tables = {
        "tvd_main.tex":     tvd_table(data, include_correlated=False),
        "tvd_all.tex":      tvd_table(data, include_correlated=True),
        "rhat_main.tex":    rhat_table(data, include_correlated=False),
        "rhat_all.tex":     rhat_table(data, include_correlated=True),
        "ess_all.tex":      ess_table(data, include_correlated=True),
        "ess_main.tex":     ess_table(data, include_correlated=False),
        "runtime_all.tex":  runtime_table(data, include_correlated=True),
    }

    for fname, content in tables.items():
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Saved {path}")

    print(f"\nDone. {len(tables)} table files written to {OUTPUT_DIR}/")
    print("\nIn your .tex file, use e.g.:")
    print(r"  \input{results/tables/tvd_main.tex}")


if __name__ == "__main__":
    main()
