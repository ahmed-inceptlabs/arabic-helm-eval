#!/usr/bin/env python3
"""Generate PDF figures summarizing the dedup pipeline results.

Reads all report JSONs and produces a set of publication-quality figures.

Usage:
    python 07_visualize.py
    python 07_visualize.py --run-name my-experiment
    python 07_visualize.py --report-dir reports --data-dir data
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_DIR = Path(__file__).parent
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_FIGURES_DIR = SCRIPT_DIR / "figures"

COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#DC2626",
    "green": "#059669",
    "orange": "#D97706",
    "gray": "#6B7280",
    "light": "#DBEAFE",
    "removed": "#FCA5A5",
    "kept": "#86EFAC",
}


def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })


def fig_pipeline_funnel(profile, exact_report, semantic_clusters, leakage_report):
    """Waterfall chart showing rows at each pipeline stage."""
    fig, ax = plt.subplots(figsize=(10, 6))

    source = profile.get("total_rows", 0) if profile else 0
    exact_kept = exact_report.get("dedup_output", {}).get("kept", 0) if exact_report else 0
    exact_removed = exact_report.get("dedup_output", {}).get("removed", 0) if exact_report else 0

    sem_removed = 0
    if semantic_clusters:
        sem_removed = sum(len(c.get("removed", [])) for c in semantic_clusters)
    sem_kept = exact_kept - sem_removed

    leak_high = 0
    if leakage_report:
        leak_high = len(set(
            leakage_report.get("direct_benchmark_mentions", {}).get("sample_indices", []) +
            leakage_report.get("rubric_leak_rows", {}).get("sample_indices", [])
        ))

    stages = ["Source", "After Exact\nDedup", "After Semantic\nDedup", "After Leakage\nRemoval"]
    values = [source, exact_kept, sem_kept, sem_kept - leak_high]
    removed = [0, exact_removed, sem_removed, leak_high]

    bars = ax.bar(stages, values, color=COLORS["primary"], width=0.6, zorder=3)

    for i, (bar, val, rem) in enumerate(zip(bars, values, removed)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        if rem > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"−{rem:,}", ha="center", va="center", color="white",
                    fontweight="bold", fontsize=10)

    ax.set_ylabel("Rows")
    ax.set_title("Pipeline Funnel: Rows Retained at Each Stage")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_ylim(0, source * 1.12)
    fig.tight_layout()
    return fig


def fig_language_distribution(profile):
    """Pie chart of language distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title in [(ax1, "language_user", "User Prompts"),
                           (ax2, "language_assistant", "Assistant Responses")]:
        data = profile.get(key, {})
        labels = list(data.keys())
        sizes = list(data.values())
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["orange"], COLORS["gray"]][:len(labels)]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
            colors=colors, startangle=90, pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight("bold")

        ax.legend(
            [f"{l} ({v:,})" for l, v in zip(labels, sizes)],
            loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=9, ncol=2,
        )
        ax.set_title(title)

    fig.suptitle("Language Distribution")
    fig.tight_layout()
    return fig


def fig_question_type(profile):
    """Bar chart of question types + CoT prevalence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    qt = profile.get("question_type", {})
    ax1.bar(["MCQ", "Open-ended"], [qt.get("mcq", 0), qt.get("open_ended", 0)],
            color=[COLORS["primary"], COLORS["secondary"]], width=0.5, zorder=3)
    ax1.bar_label(ax1.containers[0], fmt="{:,.0f}", fontsize=10, fontweight="bold", padding=4)
    ax1.set_title("Question Type")
    ax1.set_ylabel("Rows")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    tt = profile.get("think_tag", {})
    ax2.bar(["With <think>", "Without <think>"],
            [tt.get("rows_with_think", 0), tt.get("rows_without_think", 0)],
            color=[COLORS["green"], COLORS["gray"]], width=0.5, zorder=3)
    ax2.bar_label(ax2.containers[0], fmt="{:,.0f}", fontsize=10, fontweight="bold", padding=4)
    ax2.set_title("Chain-of-Thought Prevalence")
    ax2.set_ylabel("Rows")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle("Dataset Composition")
    fig.tight_layout()
    return fig


def fig_length_distributions(profile):
    """Histogram of user prompt and assistant response lengths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title, color in [
        (ax1, "user_length", "User Prompt Length (chars)", COLORS["primary"]),
        (ax2, "assistant_length", "Assistant Response Length (chars)", COLORS["secondary"]),
    ]:
        hist = profile.get(key, {}).get("histogram_100", {})
        bins = sorted(int(k) for k in hist.keys())
        counts = [hist[str(b)] for b in bins]
        ax.bar(bins, counts, width=90, color=color, alpha=0.85, zorder=3)
        ax.set_xlabel("Character length")
        ax.set_ylabel("Rows")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        stats = profile.get(key, {})
        ax.axvline(stats.get("mean", 0), color=COLORS["accent"], linestyle="--", linewidth=1.5, label=f"mean={stats.get('mean', 0):.0f}")
        ax.legend(fontsize=9)

    fig.suptitle("Text Length Distributions")
    fig.tight_layout()
    return fig


def fig_exact_dedup_levels(exact_report):
    """Bar chart comparing duplicate counts across normalization levels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = exact_report.get("levels", {})
    names = list(levels.keys())
    short_names = [n.replace("_", "\n") for n in names]
    removable = [levels[n].get("duplicate_rows_removable", 0) for n in names]

    colors = [COLORS["green"] if r == 0 else COLORS["orange"] if r < 50 else COLORS["accent"] for r in removable]
    bars = ax.bar(short_names, removable, color=colors, width=0.6, zorder=3)
    ax.bar_label(bars, fmt="{:,.0f}", fontsize=11, fontweight="bold", padding=4)

    ax.set_ylabel("Removable Duplicates")
    ax.set_title("Exact Dedup: Duplicates Found at Each Normalization Level")
    ax.set_ylim(0, max(removable) * 1.3 if max(removable) > 0 else 10)
    fig.tight_layout()
    return fig


def fig_similarity_histogram(sim_hist):
    """Bar chart of cosine similarity distribution with threshold line."""
    fig, ax = plt.subplots(figsize=(12, 6))

    hist = sim_hist.get("histogram", {})
    bands = list(hist.keys())
    counts = list(hist.values())

    threshold_idx = None
    colors = []
    for i, band in enumerate(bands):
        lo = float(band.split("-")[0])
        if lo >= 0.93:
            colors.append(COLORS["accent"])
            if threshold_idx is None:
                threshold_idx = i
        else:
            colors.append(COLORS["primary"])

    bars = ax.bar(range(len(bands)), counts, color=colors, width=0.8, zorder=3)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                    f"{count:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Max Cosine Similarity to Nearest Neighbor")
    ax.set_ylabel("Number of Rows")
    ax.set_title("Similarity Distribution (red = above threshold, removed as duplicates)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    if threshold_idx is not None:
        ax.axvline(threshold_idx - 0.5, color=COLORS["accent"], linestyle="--", linewidth=2,
                   label="threshold = 0.93")
        ax.legend(fontsize=11, loc="upper right")

    stats = sim_hist.get("stats", {})
    stat_text = (f"mean={stats.get('mean', 0):.3f}  median={stats.get('median', 0):.3f}\n"
                 f"p90={stats.get('p90', 0):.3f}  p95={stats.get('p95', 0):.3f}  "
                 f"p99={stats.get('p99', 0):.3f}")
    ax.text(0.02, 0.95, stat_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    fig.tight_layout()
    return fig


def fig_cluster_sizes(semantic_clusters):
    """Distribution of semantic duplicate cluster sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sizes = [c["size"] for c in semantic_clusters]
    size_counts = Counter(sizes)

    small = {k: v for k, v in sorted(size_counts.items()) if k <= 15}
    ax1.bar(list(small.keys()), list(small.values()), color=COLORS["primary"], zorder=3)
    ax1.set_xlabel("Cluster Size")
    ax1.set_ylabel("Number of Clusters")
    ax1.set_title("Cluster Size Distribution (size <= 15)")
    ax1.bar_label(ax1.containers[0], fmt="{:,.0f}", fontsize=8, padding=2)

    large = sorted([s for s in sizes if s > 15], reverse=True)[:30]
    if large:
        ax2.barh(range(len(large)), large, color=COLORS["secondary"], zorder=3)
        ax2.set_xlabel("Cluster Size")
        ax2.set_ylabel("Cluster Rank")
        ax2.set_title(f"Top {len(large)} Largest Clusters")
        ax2.invert_yaxis()
        for i, v in enumerate(large):
            ax2.text(v + max(large) * 0.01, i, f"{v:,}", va="center", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No clusters > 15", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Large Clusters")

    fig.suptitle(f"Semantic Duplicate Clusters ({len(sizes):,} total, {sum(sizes):,} rows)")
    fig.tight_layout()
    return fig


def fig_leakage_by_benchmark(leakage_report):
    """Horizontal bar chart of leakage flags per benchmark."""
    fig, ax = plt.subplots(figsize=(10, 6))

    flags = leakage_report.get("benchmark_flags", {})
    benchmarks = sorted(flags.keys(), key=lambda k: flags[k].get("flagged_count", 0))
    counts = [flags[b].get("flagged_count", 0) for b in benchmarks]

    colors = [COLORS["accent"] if c > 10000 else COLORS["orange"] if c > 100 else COLORS["gray"] for c in counts]
    bars = ax.barh(benchmarks, counts, color=colors, zorder=3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Rows Flagged (heuristic)")
    ax.set_title("Leakage Check: Flags per Benchmark")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    direct = leakage_report.get("direct_benchmark_mentions", {}).get("count", 0)
    rubric = leakage_report.get("rubric_leak_rows", {}).get("count", 0)
    note = f"High-confidence signals: {direct:,} direct mentions, {rubric:,} rubric leaks"
    ax.text(0.02, -0.12, note, transform=ax.transAxes, fontsize=9, style="italic", color=COLORS["gray"])

    fig.tight_layout()
    return fig


def fig_removal_breakdown():
    """Pie chart of removal reasons."""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["Semantic duplicates", "Leakage (high-conf)", "Exact duplicates"]
    sizes = [10439, 200, 17]
    colors = [COLORS["accent"], COLORS["orange"], COLORS["gray"]]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.1f}%\n({int(p * sum(sizes) / 100):,})" if p > 0.5 else "",
        colors=colors, startangle=90, pctdistance=0.65,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")

    ax.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=10, ncol=3)
    ax.set_title(f"Removal Breakdown ({sum(sizes):,} rows removed, 5.04%)")
    fig.tight_layout()
    return fig


def fig_arabic_normalization(profile):
    """Bar chart of Arabic normalization feature prevalence."""
    fig, ax = plt.subplots(figsize=(10, 5))

    stats = profile.get("arabic_normalization_stats", {})
    total = profile.get("total_rows", 1)

    features = {
        "Alef variants": stats.get("rows_with_alef_variants", 0),
        "Ta marbuta": stats.get("rows_with_ta_marbuta", 0),
        "Tashkeel\n(diacritics)": stats.get("rows_with_tashkeel", 0),
        "Ya variants": stats.get("rows_with_ya_variants", 0),
    }

    bars = ax.bar(list(features.keys()), list(features.values()),
                  color=COLORS["primary"], width=0.5, zorder=3)

    for bar, v in zip(bars, features.values()):
        pct = 100 * v / total
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{pct:.1f}%", ha="center", fontweight="bold", fontsize=11)

    ax.set_ylabel("Rows Containing Feature")
    ax.set_title("Arabic Orthographic Feature Prevalence")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_ylim(0, total * 1.08)
    fig.tight_layout()
    return fig


def fig_think_length_distribution(profile):
    """Histogram of <think> block lengths."""
    fig, ax = plt.subplots(figsize=(10, 5))

    hist = profile.get("think_tag", {}).get("think_block_length", {}).get("histogram_100", {})
    if not hist:
        return None

    bins = sorted(int(k) for k in hist.keys())
    counts = [hist[str(b)] for b in bins]

    ax.bar(bins, counts, width=90, color=COLORS["secondary"], alpha=0.85, zorder=3)
    ax.set_xlabel("Think Block Length (chars)")
    ax.set_ylabel("Rows")
    ax.set_title("<think> Block Length Distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    stats = profile.get("think_tag", {}).get("think_block_length", {})
    ax.axvline(stats.get("mean", 0), color=COLORS["accent"], linestyle="--", linewidth=1.5,
               label=f"mean={stats.get('mean', 0):.0f}")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def visualize(report_dir=None, data_dir=None, figures_dir=None, run_name=None):
    report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    figures_base = Path(figures_dir) if figures_dir else DEFAULT_FIGURES_DIR

    if not run_name:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    out_dir = figures_base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    print("Loading reports...", flush=True)
    profile = load_json(report_dir / "01_profile_report.json")
    exact_report = load_json(report_dir / "03_exact_dedup_report.json")
    sim_hist = load_json(report_dir / "04_similarity_histogram.json")
    leakage_report = load_json(report_dir / "05_leakage_report.json")
    semantic_clusters = load_json(data_dir / "04_semantic_clusters.json")

    figures = []

    if profile and exact_report:
        print("  [1/9] Pipeline funnel...", flush=True)
        fig = fig_pipeline_funnel(profile, exact_report, semantic_clusters, leakage_report)
        figures.append(("01_pipeline_funnel", fig))

    if profile:
        print("  [2/9] Language distribution...", flush=True)
        figures.append(("02_language_distribution", fig_language_distribution(profile)))

        print("  [3/9] Question type + CoT...", flush=True)
        figures.append(("03_question_type_cot", fig_question_type(profile)))

        print("  [4/9] Length distributions...", flush=True)
        figures.append(("04_length_distributions", fig_length_distributions(profile)))

        print("  [5/9] Arabic normalization...", flush=True)
        figures.append(("05_arabic_normalization", fig_arabic_normalization(profile)))

        print("  [6/9] Think block lengths...", flush=True)
        fig = fig_think_length_distribution(profile)
        if fig:
            figures.append(("06_think_lengths", fig))

    if exact_report:
        print("  [7/9] Exact dedup levels...", flush=True)
        figures.append(("07_exact_dedup_levels", fig_exact_dedup_levels(exact_report)))

    if sim_hist:
        print("  [8/9] Similarity histogram...", flush=True)
        figures.append(("08_similarity_histogram", fig_similarity_histogram(sim_hist)))

    if semantic_clusters:
        print("  [9/9] Cluster sizes...", flush=True)
        figures.append(("09_cluster_sizes", fig_cluster_sizes(semantic_clusters)))

    if leakage_report:
        print("  [+] Leakage by benchmark...", flush=True)
        figures.append(("10_leakage_benchmarks", fig_leakage_by_benchmark(leakage_report)))

    print("  [+] Removal breakdown...", flush=True)
    figures.append(("11_removal_breakdown", fig_removal_breakdown()))

    for name, fig in figures:
        pdf_path = out_dir / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
        png_path = out_dir / f"{name}.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    Saved {name}.pdf + .png")

    combined_path = out_dir / "all_figures.pdf"
    with PdfPages(combined_path) as pdf:
        for name, _ in figures:
            fig_path = out_dir / f"{name}.png"
            img = plt.imread(str(fig_path))
            fig_c, ax_c = plt.subplots(figsize=(img.shape[1] / 150, img.shape[0] / 150))
            ax_c.imshow(img)
            ax_c.axis("off")
            pdf.savefig(fig_c, bbox_inches="tight")
            plt.close(fig_c)

    print(f"\n{'='*50}")
    print(f"  {len(figures)} figures saved to: {out_dir}")
    print(f"  Combined PDF: {combined_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDF figures from dedup pipeline results")
    parser.add_argument("--run-name", default=None,
                        help="Name for the output folder (default: run_YYYYMMDD_HHMMSS)")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                        help="Reports directory (default: %(default)s)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Data directory (default: %(default)s)")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR),
                        help="Base directory for figure outputs (default: %(default)s)")
    args = parser.parse_args()
    visualize(report_dir=args.report_dir, data_dir=args.data_dir,
              figures_dir=args.figures_dir, run_name=args.run_name)
