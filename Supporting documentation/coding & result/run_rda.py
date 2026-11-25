#!/usr/bin/env python3
"""Reproduce the 31-park OBJ–RAM redundancy analysis.

Inputs are the supplied, already-standardised OBJ values and long-form RAM
Top55 tag frequencies, exported losslessly to CSV. No reference results are
read during fitting. See docs/METHODS.md for all equations and assumptions.

Quick start:
    python -m pip install -r requirements.txt
    python run_rda.py --sensitivity-hg1

Python >=3.11. Dependencies: NumPy and Matplotlib only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import platform
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent


@dataclass
class Inputs:
    codes: list[str]
    names: list[str]
    obj_names: list[str]
    X: np.ndarray
    frequencies: np.ndarray
    totals: np.ndarray
    proportions: np.ndarray
    Y: np.ndarray
    zero_filled: list[str]
    mapping: list[dict[str, str]]
    qc: dict[str, Any]


@dataclass
class Fit:
    n: int
    predictor_rank: int
    residual_df: int
    Xc: np.ndarray
    Yc: np.ndarray
    Q: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    eigenvalues: np.ndarray
    species: np.ndarray
    sites_wa: np.ndarray
    sites_lc: np.ndarray
    obj_biplot: np.ndarray
    vif: np.ndarray
    total_inertia: float
    constrained_inertia: float
    residual_inertia: float
    r2: float
    adjusted_r2: float
    pseudo_f: float
    constant: float


def read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    """Read UTF-8 CSV; fail early on absent or duplicated headers."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError(f"Duplicate CSV headers in {path.name}")
        missing = set(required) - set(headers)
        if missing:
            raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty input file: {path}")
    return rows


def number(value: Any, context: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing/non-numeric value at {context}: {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"Non-finite value at {context}: {value!r}")
    return out


def validate_config(config: dict[str, Any]) -> None:
    cols, labels, groups = (config[k] for k in ("obj_columns", "obj_labels", "ram_groups"))
    if not cols or len(cols) != len(labels):
        raise ValueError("OBJ column and display-label lists must have equal nonzero length")
    if len(set(cols)) != len(cols) or len(set(groups)) != len(groups) or not groups:
        raise ValueError("OBJ columns and RAM groups must be unique")
    if not set(groups).issubset(config["ram_colours"]):
        raise ValueError("Every RAM group must have an explicit colour")
    if not set(config.get("allowed_zero_fill_groups", [])).issubset(groups):
        raise ValueError("Unknown zero-fill group in configuration")
    if any(float(config.get(key, 1.0)) != 1.0 for key in ("obj_arrow_multiplier", "ram_arrow_multiplier")):
        raise ValueError("This reproduction fixes both arrow multipliers at 1.0")


def load_inputs(data_dir: Path, config: dict[str, Any]) -> Inputs:
    """Strict name/code matching; fill only explicitly allowed verified zeros."""
    validate_config(config)
    obj_cols, groups = config["obj_columns"], config["ram_groups"]
    mapping = read_csv(data_dir / "park_mapping.csv",
                       ["park_code", "ram_name", "obj_name", "matching_status", "note"])
    for key in ("park_code", "ram_name", "obj_name"):
        vals = [row[key].strip() for row in mapping]
        if any(not x for x in vals) or len(vals) != len(set(vals)):
            raise ValueError(f"Blank or duplicate mapping field: {key}")
        for row, value in zip(mapping, vals):
            row[key] = value
    codes = [row["park_code"] for row in mapping]
    names = [row["ram_name"] for row in mapping]
    obj_names = [row["obj_name"] for row in mapping]
    obj_rows = read_csv(data_dir / "obj_zscores.csv", ["ParkName", *obj_cols])
    obj_lookup = {}
    for row in obj_rows:
        name = row["ParkName"].strip()
        if name in obj_lookup:
            raise ValueError(f"Duplicate OBJ park name: {name}")
        obj_lookup[name] = row
    if set(obj_lookup) != set(obj_names):
        raise ValueError("OBJ/mapping mismatch. Missing: "
                         f"{sorted(set(obj_names)-set(obj_lookup))}; unmapped: "
                         f"{sorted(set(obj_lookup)-set(obj_names))}")
    X = np.array([[number(obj_lookup[name][col], f"{name}/{col}")
                   for col in obj_cols] for name in obj_names], dtype=float)
    ram_rows = read_csv(data_dir / "ram_long.csv",
                       ["Attraction", "Category", "Frequency", "TotalAttractionTop55Frequency", "Percentage"])
    seen: dict[tuple[str, str], float] = {}
    denominator: dict[str, float] = {}
    for row in ram_rows:
        name, group = row["Attraction"].strip(), row["Category"].strip()
        if name not in names or group not in groups:
            raise ValueError(f"Unmapped RAM park/group: {name}/{group}")
        if (name, group) in seen:
            raise ValueError(f"Duplicate RAM park/group: {name}/{group}")
        count = number(row["Frequency"], f"{name}/{group}/Frequency")
        total = number(row["TotalAttractionTop55Frequency"], f"{name}/total")
        percentage = number(row["Percentage"], f"{name}/{group}/Percentage")
        if count < 0 or total <= 0 or count > total:
            raise ValueError(f"Invalid RAM count/denominator at {name}/{group}")
        if not count.is_integer() or not total.is_integer():
            raise ValueError(f"Expected integer tag counts at {name}/{group}")
        if name in denominator and total != denominator[name]:
            raise ValueError(f"Inconsistent RAM denominator for {name}")
        if not np.isclose(100 * count / total, percentage, rtol=1e-10, atol=1e-8):
            raise ValueError(f"Frequency/percentage mismatch at {name}/{group}")
        seen[(name, group)] = count
        denominator[name] = total
    missing_parks = set(names) - set(denominator)
    if missing_parks:
        raise ValueError(f"Missing complete RAM profiles: {sorted(missing_parks)}")
    frequencies, zero_filled = [], []
    allowed = set(config.get("allowed_zero_fill_groups", []))
    for code, name in zip(codes, names):
        missing_groups = [g for g in groups if (name, g) not in seen]
        if not set(missing_groups).issubset(allowed):
            raise ValueError(f"Unapproved missing RAM groups in {code}: {missing_groups}")
        counts = [seen.get((name, g), 0.) for g in groups]
        if sum(counts) != denominator[name]:
            raise ValueError(f"Tag counts do not sum to original Top55 denominator in {code}; "
                             "missing entries cannot be assumed zero")
        frequencies.append(counts)
        zero_filled.extend(f"{code}:{g}" for g in missing_groups)
    frequencies = np.asarray(frequencies, dtype=float)
    totals = np.asarray([denominator[name] for name in names])
    proportions = frequencies / totals[:, None]
    Y = np.sqrt(proportions)
    means = X.mean(axis=0)
    sds = X.std(axis=0, ddof=1)
    if np.any(sds <= 0):
        raise ValueError("Constant OBJ predictor; remove or investigate it explicitly")
    standardised = bool(np.allclose(means, 0, atol=1e-8) and np.allclose(sds, 1, atol=1e-8))
    if not standardised:
        warnings.warn("OBJ values are not sample z-scores. They are retained unchanged; "
                      "this script never silently restandardises the supplied input.")
    qc = {
        "n_parks": len(codes), "raw_ram_rows": len(ram_rows),
        "filled_zero_cells": zero_filled,
        "obj_mean": dict(zip(obj_cols, means.tolist())),
        "obj_sample_sd": dict(zip(obj_cols, sds.tolist())),
        "obj_is_sample_zscore": standardised,
        "maximum_proportion_sum_error": float(abs(proportions.sum(axis=1)-1).max()),
        "tag_denominator_min": float(totals.min()), "tag_denominator_max": float(totals.max()),
        "provisional_matches": [r for r in mapping if r["matching_status"] == "provisional"],
    }
    return Inputs(codes, names, obj_names, X, frequencies, totals, proportions, Y,
                  zero_filled, mapping, qc)


def subset_inputs(data: Inputs, excluded: list[str]) -> Inputs:
    if len(excluded) != len(set(excluded)):
        raise ValueError("Duplicate exclusion code")
    unknown = set(excluded) - set(data.codes)
    if unknown:
        raise ValueError(f"Unknown park code(s) to exclude: {sorted(unknown)}")
    mask = np.array([c not in excluded for c in data.codes])
    kept = np.flatnonzero(mask)
    return Inputs(
        [data.codes[i] for i in kept], [data.names[i] for i in kept],
        [data.obj_names[i] for i in kept], data.X[mask], data.frequencies[mask],
        data.totals[mask], data.proportions[mask], data.Y[mask],
        [z for z in data.zero_filled if z.split(":")[0] not in excluded],
        [data.mapping[i] for i in kept], {**data.qc, "excluded_codes": excluded,
                                        "n_parks_after_exclusion": len(kept)})


def fit_rda(X: np.ndarray, Y: np.ndarray) -> Fit:
    """Unweighted covariance RDA; vegan-compatible positive scaling=2.

    The response is already Hellinger-transformed. Only column centring is
    applied here. Predictor rank, not the nominal number of columns, is used
    for degrees of freedom. Rank-deficient X is handled by an SVD basis.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be 2D matrices with the same rows")
    n = X.shape[0]
    if n < 3 or not X.shape[1] or not Y.shape[1]:
        raise ValueError("Insufficient rows/columns for constrained ordination")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Missing/non-finite values are not permitted in RDA")
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    ux, sx, _ = np.linalg.svd(Xc, full_matrices=False)
    rank = int(np.count_nonzero(sx > np.finfo(float).eps * max(Xc.shape) * sx[0]))
    residual_df = n - rank - 1
    if rank == 0 or residual_df <= 0:
        raise ValueError("Zero predictor rank or no residual degrees of freedom")
    # QR preserves the original full-rank implementation. SVD handles aliases.
    Q = np.linalg.qr(Xc, mode="reduced")[0] if rank == X.shape[1] else ux[:, :rank]
    fitted = Q @ (Q.T @ Yc)
    residuals = Yc - fitted
    total_ss = float(np.sum(Yc ** 2))
    constrained_ss = float(np.sum(fitted ** 2))
    residual_ss = float(np.sum(residuals ** 2))
    if total_ss <= np.finfo(float).tiny:
        raise ValueError("No variation in the response matrix")
    if constrained_ss <= 100 * np.finfo(float).eps * total_ss:
        raise ValueError("No numerically identifiable constrained axes")
    if residual_ss <= 100 * np.finfo(float).eps * total_ss:
        raise ValueError("Perfect fit; a finite pseudo-F test is not defined")
    u, singular, vt = np.linalg.svd(fitted, full_matrices=False)
    keep = singular > np.finfo(float).eps * max(fitted.shape) * singular[0]
    u, singular, v = u[:, keep], singular[keep], vt.T[:, keep]
    signs = np.where(v[np.argmax(abs(v), axis=0), np.arange(v.shape[1])] < 0, -1., 1.)
    v, u = v * signs, u * signs
    eigenvalues = singular ** 2 / (n - 1)
    constant = total_ss ** 0.25
    species = v * (singular / np.sqrt(total_ss)) * constant
    sites_lc = u * constant
    sites_wa = (Yc @ v / singular) * constant
    norms = np.linalg.norm(Xc, axis=0)
    if np.any(norms <= np.finfo(float).tiny):
        raise ValueError("Constant predictor detected after subsetting")
    obj_biplot = (Xc.T @ u) / norms[:, None]
    vif = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        others = np.delete(Xc, j, axis=1)
        fitted_x = others @ np.linalg.lstsq(others, Xc[:, j], rcond=None)[0]
        rss = float(np.sum((Xc[:, j] - fitted_x) ** 2))
        vif[j] = np.inf if rss <= np.finfo(float).eps * norms[j]**2 else norms[j]**2 / rss
    r2 = constrained_ss / total_ss
    return Fit(n, rank, residual_df, Xc, Yc, Q, fitted, residuals, eigenvalues,
               species, sites_wa, sites_lc, obj_biplot, vif,
               total_ss / (n - 1), constrained_ss / (n - 1), residual_ss / (n - 1),
               r2, 1 - (1 - r2) * (n - 1) / residual_df,
               (constrained_ss / rank) / (residual_ss / residual_df), constant)


def permutation_test(fit: Fit, count: int = 9999, seed: int = 20261110,
                     tolerance: float = 1e-12) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Permute whole park-level response rows, holding X fixed.

    Each call starts a fresh PCG64 stream. Returned indices are ZERO-based;
    CSV export adds 1 for direct use with R/vegan. No spatial restrictions.
    """
    if count < 0 or seed < 0 or tolerance < 0:
        raise ValueError("Permutation count, seed and tolerance must be non-negative")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = np.empty((count, fit.n), dtype=np.int32)
    fvalues = np.empty(count, dtype=float)
    total_ss = float(np.sum(fit.Yc ** 2))
    for b in range(count):
        indices[b] = rng.permutation(fit.n)
        projected = fit.Q.T @ fit.Yc[indices[b]]
        explained = float(np.sum(projected ** 2))
        unexplained = max(total_ss - explained, 0.)
        fvalues[b] = ((explained / fit.predictor_rank) / (unexplained / fit.residual_df)
                      if unexplained > 0 else np.inf)
    extreme = int(np.count_nonzero(fvalues >= fit.pseudo_f - tolerance))
    info = {"permutations": count, "seed": seed, "generator": "NumPy PCG64",
            "permuted_F_ge_observed": extreme,
            "p_value": (extreme + 1) / (count + 1) if count else None,
            "comparison_tolerance": tolerance,
            "scheme": "Unrestricted whole-response row permutations; X fixed; no spatial blocks",
            "p_formula": "(extreme + 1) / (permutations + 1)"}
    return info, indices, fvalues


def dominant_groups(data: Inputs, config: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    """Colour by the largest untransformed within-park frequency proportion.

    Exact/near ties are reported as Mixed, not silently assigned to the first
    column. There are no ties in the supplied data.
    """
    groups = config["ram_groups"]
    labels = []
    for row in data.proportions:
        winners = np.flatnonzero(np.isclose(row, row.max(), rtol=0, atol=1e-12))
        labels.append(groups[int(winners[0])] if len(winners) == 1 else "Mixed")
    counts = Counter(labels)
    result = {g: int(counts[g]) for g in groups}
    if counts["Mixed"]:
        result["Mixed"] = int(counts["Mixed"])
    return labels, result


def write_csv(path: Path, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def model_summary(fit: Fit, perm: dict[str, Any]) -> dict[str, Any]:
    return {"n_parks": fit.n, "predictor_rank": fit.predictor_rank,
            "residual_df": fit.residual_df, "R2": fit.r2,
            "adjusted_R2": fit.adjusted_r2, "pseudo_F": fit.pseudo_f,
            "total_inertia": fit.total_inertia, "constrained_inertia": fit.constrained_inertia,
            "residual_inertia": fit.residual_inertia, "score_constant": fit.constant,
            "eigenvalues": fit.eigenvalues.tolist(),
            "axis_fraction_constrained": (fit.eigenvalues / fit.constrained_inertia).tolist(),
            "axis_fraction_total": (fit.eigenvalues / fit.total_inertia).tolist(),
            "site_scores_in_plot": "WA response-based; positive scaling=2",
            "axis_sign": "Largest absolute response loading is positive on every axis",
            "OBJ_arrow_multiplier": 1.0, "RAM_arrow_multiplier": 1.0,
            "permutation_test": perm}


def export_tables(out: Path, data: Inputs, fit: Fit, config: dict[str, Any],
                  perm: dict[str, Any], indices: np.ndarray, fvalues: np.ndarray) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    axes = [f"RDA{i+1}" for i in range(len(fit.eigenvalues))]
    groups, cols, obj_labels = config["ram_groups"], config["obj_columns"], config["obj_labels"]
    dom, counts = dominant_groups(data, config)
    summary = model_summary(fit, perm)
    summary["dominant_group_counts"] = counts
    summary["excluded_codes"] = data.qc.get("excluded_codes", [])
    summary["provisional_matches_in_model"] = [r for r in data.mapping if r["matching_status"] == "provisional"]
    write_json(out / "model_summary.json", summary)
    write_json(out / "input_quality_checks.json", data.qc)
    pid = list(zip(data.codes, data.names))
    def matrix_csv(name: str, headers: list[str], matrix: np.ndarray) -> None:
        write_csv(out / name, ["ParkCode", "ParkName", *headers],
                  [(*ids, *values) for ids, values in zip(pid, matrix.tolist())])
    matrix_csv("OBJ_inputs_aligned.csv", cols, data.X)
    matrix_csv("RAM_frequencies.csv", groups + ["Top55Total"],
               np.column_stack([data.frequencies, data.totals]))
    matrix_csv("RAM_proportions.csv", groups, data.proportions)
    matrix_csv("RAM_Hellinger.csv", groups, data.Y)
    matrix_csv("Park_scores_WA_all_axes.csv", axes, fit.sites_wa)
    matrix_csv("Park_scores_LC_all_axes.csv", axes, fit.sites_lc)
    write_csv(out / "OBJ_vectors_all_axes.csv", ["InputHeader", "ObjectiveIndicator", *axes, "VIF"],
              [(col, label, *values, float(vif)) for col, label, values, vif in
               zip(cols, obj_labels, fit.obj_biplot.tolist(), fit.vif)])
    write_csv(out / "RAM_scores_all_axes.csv", ["ResponseVariable", *axes],
              [(g, *row) for g, row in zip(groups, fit.species.tolist())])
    write_csv(out / "Axis_variance.csv", ["Axis", "Eigenvalue", "PercentConstrained", "PercentTotal"],
              [(a, v, 100*v/fit.constrained_inertia, 100*v/fit.total_inertia)
               for a, v in zip(axes, fit.eigenvalues)])
    fc = summary["axis_fraction_constrained"]
    A = [["R²", fit.r2], ["Adjusted R²", fit.adjusted_r2],
         ["RDA1 (% constrained variance)", 100*fc[0]],
         ["RDA2 (% constrained variance)", 100*fc[1] if len(fc)>1 else None],
         ["Number of parks", fit.n]]
    B = [[label, *xy[:2]] for label, xy in zip(obj_labels, fit.obj_biplot.tolist())]
    C = [[g, *xy[:2]] for g, xy in zip(groups, fit.species.tolist())]
    if len(axes) < 2:
        B = [r + [None] for r in B]
        C = [r + [None] for r in C]
    write_csv(out / "A_Model_summary.csv", ["Metric", "Value"], A)
    write_csv(out / "B_Objective_vectors.csv", ["Objective indicator", "RDA1", "RDA2"], B)
    write_csv(out / "C_RAM_group_scores.csv", ["Response variable", "RDA1", "RDA2"], C)
    write_csv(out / "D_Park_scores.csv", ["ParkCode", "ParkName", "RDA1", "RDA2"],
              [(*ids, row[0], row[1] if len(row)>1 else None) for ids, row in zip(pid, fit.sites_wa.tolist())])
    colours = config["ram_colours"]
    write_csv(out / "Park_dominance.csv",
              ["ParkCode", "ParkName", "DominantGroup", "DominantProportion", "ColourHex", *groups],
              [(code, name, group, row.max(), colours.get(group, "#888888"), *row)
               for (code, name), group, row in zip(pid, dom, data.proportions)])
    write_csv(out / "permutations_1based.csv", [f"Position{i+1}" for i in range(fit.n)], indices+1)
    write_csv(out / "permutation_statistics.csv", ["Permutation", "PseudoF"], enumerate(fvalues, 1))
    # A single three-section table matching the requested manuscript layout.
    def fmt(v: Any, integer: bool = False) -> str:
        if v is None: return "NA"
        if isinstance(v, (int, float)): return f"{int(v)}" if integer else f"{v:.3f}"
        return str(v)
    sections = [("A. Model summary", ["Metric", "Value", ""], [r+[""] for r in A]),
                ("B. Objective-variable vectors", ["Objective indicator", "RDA1", "RDA2"], B),
                ("C. RAM group scores", ["Response variable", "RDA1", "RDA2"], C)]
    markdown = ["# OBJ–RAM RDA", ""]
    hrows = []
    for title, heads, rows in sections:
        markdown.extend([f"## {title}", "", "| " + " | ".join(heads) + " |", "|---|---:|---:|"])
        hrows.append(f'<tr class="section"><th colspan="3">{html.escape(title)}</th></tr>')
        hrows.append("<tr>" + "".join(f"<th>{html.escape(x)}</th>" for x in heads) + "</tr>")
        for row in rows:
            vals = [fmt(v, row[0] == "Number of parks" and j == 1) for j, v in enumerate(row)]
            markdown.append("| " + " | ".join(vals) + " |")
            hrows.append("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in vals) + "</tr>")
        markdown.append("")
    note = ("Percentages use constrained variance. RAM uses Top55 tag-frequency composition, "
            "not image fractions. OBJ vectors and RAM scores use positive scaling 2; "
            "no display multiplier. HG1 is a provisional name/spatial-unit match when included. "
            "Coordinates retain full precision in the CSV files; this table displays 3 decimals.")
    markdown.append(note)
    (out / "ABC_table.md").write_text("\n".join(markdown)+"\n", encoding="utf-8")
    (out / "ABC_table.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>OBJ–RAM RDA table</title>'
        '<style>body{max-width:850px;margin:32px auto;font-family:"Times New Roman",serif;}'
        'table{border-collapse:collapse;width:100%;font-size:18px}th,td{border:1px solid #555;'
        'padding:5px 9px;text-align:left}.section{background:#dce9f5}td:first-child{width:51%}'
        'p{font-size:14px;line-height:1.5}</style><table>'+"\n".join(hrows)+
        '</table><p>'+html.escape(note)+'</p></html>', encoding="utf-8")
    return summary


def plot_rda(out: Path, data: Inputs, fit: Fit, config: dict[str, Any]) -> None:
    """Save vector SVG and PNG. Only annotation positions are displaced.

    The layout is regenerated; mathematical coordinates and dominant-group
    colours reproduce the supplied figure. This does not export native PSD.
    """
    if len(fit.eigenvalues) < 2:
        warnings.warn("Only one constrained axis: skipping two-dimensional plot")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(9.2, 9.6), dpi=150)
    fig.subplots_adjust(left=.13, right=.96, bottom=.16, top=.92)
    sites, ram, obj = fit.sites_wa[:, :2], fit.species[:, :2], fit.obj_biplot[:, :2]
    points = np.vstack([sites, ram, obj, [[0, 0]]])
    lows, highs = points.min(axis=0), points.max(axis=0)
    span = np.maximum(highs-lows, .1)
    ax.set_xlim(lows[0]-.17*span[0], highs[0]+.30*span[0])
    ax.set_ylim(lows[1]-.10*span[1], highs[1]+.16*span[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#eeeeee", linewidth=.45)
    ax.axhline(0, color="#aaaaaa", ls="--", lw=.65)
    ax.axvline(0, color="#aaaaaa", ls="--", lw=.65)
    dom, counts = dominant_groups(data, config)
    colours = config["ram_colours"]
    for code, xy, group in zip(data.codes, sites, dom):
        ar = ax.scatter(*xy, s=17, color=colours.get(group, "#888888"), edgecolor="white",
                        linewidth=.3, zorder=4)
        ar.set_gid(f"park-{code}")
    for group, xy in zip(config["ram_groups"], ram):
        ar = FancyArrowPatch((0, 0), xy, arrowstyle="-|>", mutation_scale=9,
                             color=colours[group], linewidth=1.35, shrinkA=0, shrinkB=0, zorder=3)
        ar.set_gid(f"RAM-{group}")
        ax.add_patch(ar)
    for label, xy in zip(config["obj_labels"], obj):
        ar = FancyArrowPatch((0, 0), xy, arrowstyle="->", mutation_scale=8,
                             color="black", linewidth=.9, shrinkA=0, shrinkB=0, zorder=2)
        ar.set_gid(f"OBJ-{label}")
        ax.add_patch(ar)
    legend_handles = [Line2D([], [], marker="o", color=colours[g], linestyle="None",
                             markersize=4, label=g) for g in config["ram_groups"]]
    if "Mixed" in counts:
        legend_handles.append(Line2D([], [], marker="o", color="#888888", ls="", label="Mixed"))
    legend = ax.legend(handles=legend_handles, loc="upper right", frameon=False,
                       title="Dominant RAM group", fontsize=8, title_fontsize=8)
    frac = fit.eigenvalues/fit.constrained_inertia*100
    ax.set_xlabel(f"RDA1 ({frac[0]:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"RDA2 ({frac[1]:.1f}%)", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)
    fig.text(.13, .95, "OBJ–RAM redundancy analysis", fontsize=15, fontweight="bold")
    fig.text(.13, .105, f"n = {fit.n}    R² = {fit.r2:.3f}    Adjusted R² = {fit.adjusted_r2:.3f}", fontsize=9)
    fig.text(.13, .078, "Axis percentages: constrained variance. Park colours: largest original G1–G8 proportion.", fontsize=8)
    fig.text(.13, .055, "RAM: Hellinger-transformed tag-frequency composition. Park scores: WA, scaling 2.", fontsize=8)
    fig.text(.13, .032, "HG1 is a provisional spatial-unit match when included; see data/park_mapping.csv.", fontsize=8)
    # Place text around fixed coordinates; deterministic overlap avoidance.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer)
    nodes = ax.transData.transform(points)
    sample = np.vstack([np.linspace([0., 0.], z, 50) for z in np.vstack([obj, ram])])
    arrowpx = ax.transData.transform(sample)
    occupied = [np.array(legend.get_window_extent(renderer).extents)]
    records = []
    candidates = []
    for label, xy in zip(config["obj_labels"], obj):
        candidates.append((label, xy, 8.4, "bold", "black", "OBJ"))
    for group, xy in zip(config["ram_groups"], ram):
        candidates.append((group, xy, 9.5, "bold", colours[group], "RAM"))
    dist = np.linalg.norm(sites[:, None, :]-sites[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    for i in np.argsort(dist.min(axis=1)):
        candidates.append((data.codes[i], sites[i], 7.0, "normal", "#333333", "Park"))
    def overlap(a: np.ndarray, b: np.ndarray) -> float:
        return max(0., min(a[2],b[2])-max(a[0],b[0]))*max(0., min(a[3],b[3])-max(a[1],b[1]))
    for label, xy, size, weight, colour, category in candidates:
        anchor = ax.transData.transform(xy)
        width, height, descent = renderer.get_text_width_height_descent(
            label, FontProperties(size=size, weight=weight), ismath=False)
        hw, hh = width/2+2, max(height+descent, size*fig.dpi/72)/2+1
        preferred = np.arctan2(xy[1], xy[0]) if category != "Park" else np.pi/4
        best = None
        for gap in (4, 8, 13, 20, 30, 42):
            for angle in preferred + np.linspace(-np.pi, np.pi, 32, endpoint=False):
                unit = np.array([np.cos(angle), np.sin(angle)])
                radius = min(hw/max(abs(unit[0]),1e-9), hh/max(abs(unit[1]),1e-9))
                center = anchor + unit*(radius+gap)
                box = np.array([center[0]-hw, center[1]-hh, center[0]+hw, center[1]+hh])
                outside = (max(0,bbox.x0-box[0])+max(0,bbox.y0-box[1])+
                           max(0,box[2]-bbox.x1)+max(0,box[3]-bbox.y1))
                area = sum(overlap(box,b) for b in occupied)
                nodehits = sum(overlap(box,np.r_[p-4,p+4]) for p in nodes)
                arrowhits = np.count_nonzero((arrowpx[:,0]>box[0])&(arrowpx[:,0]<box[2])&
                                             (arrowpx[:,1]>box[1])&(arrowpx[:,1]<box[3]))
                cost = area*100 + nodehits*60 + arrowhits*10 + outside*10000 + np.linalg.norm(center-anchor)
                if best is None or cost < best[0]:
                    best = (cost, center, box, gap)
        _, center, box, gap = best
        occupied.append(box)
        tx, ty = ax.transData.inverted().transform(center)
        if gap >= 13:
            ax.annotate("",xy=xy,xytext=(tx,ty),arrowprops={"arrowstyle":"-","lw":.35,
                        "color":colour,"alpha":.5,"shrinkA":5,"shrinkB":3},zorder=3)
        text = ax.text(tx,ty,label,fontsize=size,fontweight=weight,color=colour,
                       ha="center",va="center",zorder=5)
        text.set_gid(f"label-{category}-{label}")
        records.append([category,label,float(xy[0]),float(xy[1]),float(tx),float(ty)])
    write_csv(out/"plot_label_positions.csv",["Category","Label","TrueX","TrueY","LabelX","LabelY"],records)
    fig.savefig(out/"OBJ_RAM_RDA.svg",metadata={"Title":"OBJ–RAM RDA; original park coordinates; dominant-group colours"})
    fig.savefig(out/"OBJ_RAM_RDA.png",dpi=350)
    plt.close(fig)


def one_run(data: Inputs, config: dict[str, Any], out: Path, args: argparse.Namespace) -> dict[str, Any]:
    fit = fit_rda(data.X, data.Y)
    perm, indices, fvalues = permutation_test(fit,args.permutations,args.seed,config["comparison_tolerance"])
    result = export_tables(out,data,fit,config,perm,indices,fvalues)
    if not args.no_plot:
        plot_rda(out,data,fit,config)
    print(f"{out.name}: n={fit.n}; R2={fit.r2:.12f}; adjusted R2={fit.adjusted_r2:.12f}; "
          f"pseudo-F={fit.pseudo_f:.12f}; p={perm['p_value']}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs")
    parser.add_argument("--permutations",type=int,default=None)
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--exclude",nargs="*",default=[],help="Explicit park codes to exclude")
    parser.add_argument("--sensitivity-hg1",action="store_true",help="Also rerun without HG1, using a fresh RNG")
    parser.add_argument("--no-plot",action="store_true",help="Produce all numerical outputs without figures")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        args.permutations = config["permutations"] if args.permutations is None else args.permutations
        args.seed = config["seed"] if args.seed is None else args.seed
        data = load_inputs(args.data_dir,config)
        data = subset_inputs(data,args.exclude)
        provisional = [r["park_code"] for r in data.mapping if r["matching_status"]=="provisional"]
        if provisional:
            warnings.warn(f"Provisional name/spatial-unit matching retained: {provisional}. See park_mapping.csv.")
        args.output.mkdir(parents=True,exist_ok=True)
        one_run(data,config,args.output/"main",args)
        if args.sensitivity_hg1:
            if "HG1" in data.codes:
                one_run(subset_inputs(data,["HG1"]),config,args.output/"without_HG1",args)
            else:
                warnings.warn("HG1 is already absent; no additional sensitivity run was made")
        import matplotlib
        files = [args.config,*[args.data_dir/f for f in ("obj_zscores.csv","ram_long.csv","park_mapping.csv")]]
        write_json(args.output/"run_metadata.json",{
            "created_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
            "numpy":np.__version__,"matplotlib":matplotlib.__version__,"platform":platform.platform(),
            "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "input_sha256":{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "note":"Reference outputs are not consulted during analysis; no spatial adjustment, no causal inference."})
    except (ValueError,KeyError,OSError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
