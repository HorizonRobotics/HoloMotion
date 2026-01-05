import argparse
import itertools
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import mannwhitneyu
import textwrap

DEFAULT_METRICS_TO_ANALYZE = [
    "mpjpe_g",
    "mpjpe_l",
    "whole_body_joints_dist",
    "root_vel_error",
    "root_r_error",
    "root_p_error",
    "root_y_error",
    "root_height_error",
]

RADAR_METRICS = [
    "mpjpe_g",
    "mpjpe_l",
    "whole_body_joints_dist",
    "root_vel_error",
]
DEFAULT_RADAR_MAPPING = {m: m for m in RADAR_METRICS}

DEFAULT_ALPHA = 0.05


class AnalysisReportGenerator:
    """Load per-clip JSON metrics, run analysis, generate plots + markdown report."""

    def __init__(
        self,
        json_dir: str,
        plots_dir: str,
        dataset_name: str,
        metrics_to_analyze: List[str],
        radar_metric_mapping: Dict[str, str],
        metric_types_for_radar: Dict[str, str],
        alpha: float = DEFAULT_ALPHA,
        plot_quantile_cutoff: float = 0.99,
        kde_linewidth: float = 2.5,
        min_normalized_value: float = 0.2,
        radar_chart_filename: str = "radar_chart_comparison.png",
    ) -> None:
        self.json_dir = Path(json_dir)
        self.plots_dir = Path(plots_dir)
        self.dataset_name = dataset_name
        self.metrics_to_analyze = metrics_to_analyze
        self.radar_metric_mapping = radar_metric_mapping.copy()
        self.metric_types_for_radar = metric_types_for_radar.copy()
        self.alpha = alpha
        self.plot_quantile_cutoff = plot_quantile_cutoff
        self.kde_linewidth = kde_linewidth
        self.min_normalized_value = min_normalized_value
        self.radar_chart_filename = radar_chart_filename

        self.df: Optional[pd.DataFrame] = None
        self.models: List[str] = []

    def run(self) -> None:
        self.plots_dir.mkdir(exist_ok=True, parents=True)

        self.df = self._load_and_prepare_data()
        if self.df is None or self.df.empty:
            logger.warning("No valid data loaded; aborting analysis.")
            return

        self.models = sorted(self.df["model"].unique().tolist())
        if len(self.models) < 1:
            logger.warning("No models found in data; aborting analysis.")
            return

        self._create_matplotlib_radar_chart()
        markdown_content = self._generate_markdown_report()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_md = self.plots_dir / f"analysis_report_{self.dataset_name}_{ts}.md"
        out_md.write_text(markdown_content, encoding="utf-8")
        logger.info(f"Markdown report written to: {out_md}")

    def _load_and_prepare_data(self) -> Optional[pd.DataFrame]:
        if not self.json_dir.is_dir():
            logger.error(f"json_dir '{self.json_dir}' is not a directory.")
            return None

        json_files = list(self.json_dir.glob("*.json"))
        if not json_files:
            logger.error(f"No .json files found in '{self.json_dir}'.")
            return None

        all_clips: List[Dict[str, Any]] = []

        for jf in json_files:
            model_name = jf.stem

            data = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "per_clip" not in data:
                logger.warning(
                    f"Skipping non-eval JSON file '{jf.name}' "
                    f"(top-level type={type(data)}, has_per_clip={'per_clip' in data if isinstance(data, dict) else False})."
                )
                continue

            per_clip = data.get("per_clip")
            if not per_clip:
                logger.warning(f"File '{jf.name}' has empty 'per_clip'; skipping.")
                continue

            for clip in per_clip:
                clip["model"] = model_name
                all_clips.append(clip)

        if not all_clips:
            logger.error("No per_clip data found in any JSON files.")
            return None

        df = pd.DataFrame(all_clips)
        logger.info(f"Loaded {len(df)} clip records from {len(json_files)} JSON files.")
        return df

    def _create_kde_plot(self, metric: str, save_path: Path) -> None:
        if self.df is None or metric not in self.df.columns:
            return
        if self.df[metric].isnull().all():
            return
        q_high = self.df[metric].quantile(self.plot_quantile_cutoff)
        df_filtered = self.df[self.df[metric] <= q_high]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.kdeplot(
            data=df_filtered,
            x=metric,
            hue="model",
            hue_order=self.models,
            ax=ax,
            fill=False,
            common_norm=False,
            palette="tab10",
            linewidth=self.kde_linewidth,
        )
        ax.set_title(
            f'Error Distribution for "{metric}" on {self.dataset_name}', fontsize=16, weight="bold"
        )
        ax.set_xlabel(f"Error Value ({metric})", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(left=0)

        legend = ax.get_legend()
        if legend:
            legend.set_title("Models", prop={"size": 14, "weight": "bold"})
            for text in legend.get_texts():
                text.set_fontsize(14)

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _create_matplotlib_radar_chart(self) -> None:
        if self.df is None:
            return

        original_metrics = list(self.radar_metric_mapping.keys())

        raw_labels = [self.radar_metric_mapping[m] for m in original_metrics]
        display_labels = [
            textwrap.fill(label, width=20, break_long_words=False) for label in raw_labels
        ]
        num_metrics = len(original_metrics)

        median_df = self.df.groupby("model")[original_metrics].median()
        rounded_median_df = median_df.round(2)

        normalized_df = pd.DataFrame(
            index=self.models, columns=original_metrics, dtype=float
        )
        scale = 1.0 - self.min_normalized_value

        for metric in original_metrics:
            medians = rounded_median_df[metric].dropna()
            if medians.empty:
                normalized_df[metric] = self.min_normalized_value
                continue

            min_val, max_val = medians.min(), medians.max()
            rng = max_val - min_val if max_val > min_val else 1.0

            for model in self.models:
                val = rounded_median_df.loc[model, metric]
                if pd.isna(val):
                    normalized_df.loc[model, metric] = self.min_normalized_value
                    continue

                lower_better = self.metric_types_for_radar.get(metric, "lower") == "lower"
                if lower_better:
                    base = (max_val - val) / rng
                else:
                    base = (val - min_val) / rng

                norm = self.min_normalized_value + base * scale
                normalized_df.loc[model, metric] = norm

        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw=dict(projection="polar")
        )
        cmap = plt.get_cmap("tab10")
        colors = {m: cmap(i % 10) for i, m in enumerate(self.models)}

        for model in self.models:
            vals = normalized_df.loc[model].tolist()
            vals += vals[:1]
            ax.fill(angles, vals, color=colors[model], alpha=0.25)
            ax.plot(
                angles,
                vals,
                color=colors[model],
                linewidth=2.5,
                label=model,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1,
            )

        for j, metric in enumerate(original_metrics):
            angle = angles[j]
            groups: Dict[str, List[float]] = defaultdict(list)
            for model in self.models:
                orig_val = rounded_median_df.loc[model, metric]
                norm_val = normalized_df.loc[model, metric]
                groups[f"{orig_val:.2f}"].append(norm_val)

            for label_text, norm_vals in groups.items():
                avg_norm = float(np.mean(norm_vals))
                offset = 0.05
                ax.text(
                    angle,
                    avg_norm + offset,
                    label_text,
                    ha="center",
                    va="center",
                    color="black",
                    weight="bold",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="square,pad=0.3",
                        fc="white",
                        ec="none",
                        alpha=0.8,
                    ),
                )

        ax.set_thetagrids(np.degrees(angles[:-1]), display_labels, fontsize=16)
        ax.tick_params(axis="x", pad=30)
        ax.set_rgrids([0.4, 0.6, 0.8, 1.0], labels=[])
        ax.set_ylim(0, 1.25)
        ax.spines["polar"].set_visible(False)
        ax.grid(color="grey", linestyle="--", linewidth=0.5)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend_map = dict(zip(labels, handles))
            ordered_labels = [m for m in self.models if m in legend_map]
            ordered_handles = [legend_map[m] for m in ordered_labels]
            ax.legend(
                handles=ordered_handles,
                labels=ordered_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=len(ordered_handles),
                fontsize=14,
                frameon=False,
            )

        fig.suptitle(
            f"Model Comparison on {self.dataset_name} Dataset",
            fontsize=20,
            weight="bold",
            y=1.05,
        )

        save_path = self.plots_dir / self.radar_chart_filename
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Radar chart saved to: {save_path}")

    def _generate_markdown_report(self) -> str:
        if self.df is None or len(self.models) < 2:
            return ""

        parts: List[str] = [
            f"**Dataset**: {self.dataset_name}",
            f"**Models**: {', '.join(self.models)}",
            f"**Significance level (alpha)**: {self.alpha}",
            "### Pairwise metric comparisons and distributions",
        ]

        two_models = len(self.models) == 2
        if two_models:
            model1, model2 = self.models[0], self.models[1]

        for metric in self.metrics_to_analyze:
            if metric not in self.df.columns:
                continue

            p_value_str = ""
            if two_models:
                d1 = self.df.loc[self.df["model"] == model1, metric].dropna()
                d2 = self.df.loc[self.df["model"] == model2, metric].dropna()
                if not d1.empty and not d2.empty:
                    _, p_val = mannwhitneyu(d1, d2, alternative="two-sided")
                    p_value_str = f" (p = {p_val:.3g})"

            parts.append(f"#### Metric: `{metric}`{p_value_str}")

            metric_stats: List[Dict[str, Any]] = []
            for name in self.models:
                data = self.df.loc[self.df["model"] == name, metric].dropna()
                if data.empty:
                    continue
                metric_stats.append(
                    {
                        "Model": name,
                        "Median": data.median(),
                        "Q1 (25%)": data.quantile(0.25),
                        "Q3 (75%)": data.quantile(0.75),
                    }
                )

            if metric_stats:
                stats_df = (
                    pd.DataFrame(metric_stats)
                    .sort_values(by="Median")
                    .reset_index(drop=True)
                )
                parts.append(stats_df.to_markdown(index=False, floatfmt=".4f"))

            findings: List[str] = []
            lower_better = self.metric_types_for_radar.get(metric, "lower") == "lower"

            for m1, m2 in itertools.combinations(self.models, 2):
                d1 = self.df.loc[self.df["model"] == m1, metric].dropna()
                d2 = self.df.loc[self.df["model"] == m2, metric].dropna()
                if d1.empty or d2.empty:
                    continue
                _, p_val = mannwhitneyu(d1, d2, alternative="two-sided")
                if p_val >= self.alpha:
                    continue

                m1_med, m2_med = d1.median(), d2.median()
                better, worse = (m1, m2) if m1_med < m2_med else (m2, m1)
                if not lower_better:
                    better, worse = worse, better

                findings.append(
                    f"- **{better}** is significantly better than **{worse}** "
                    f"(p < {self.alpha})."
                )

            if findings:
                parts.append("\n".join(findings))
            else:
                parts.append("No statistically significant differences between models.")
            safe_metric = metric.replace(" ", "_")
            plot_filename = f"{safe_metric}.png"
            plot_path = self.plots_dir / plot_filename
            self._create_kde_plot(metric, plot_path)
            parts.append(
                f"##### Distribution plot\n"
                f"![{metric} distribution on {self.dataset_name}]({plot_filename})"
            )

        return "\n\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-clip JSON metrics, generate plots and markdown report."
    )
    parser.add_argument("--json_dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    json_dir = Path(args.json_dir).resolve()
    name = json_dir.name
    for prefix in ("metrics_output_",):
        if name.startswith(prefix) and len(name) > len(prefix):
            name = name[len(prefix):]
            break
    dataset_name = name  # e.g. "AMASS"
    plots_dir = json_dir / f"analysis_plots_{dataset_name}"

    metric_types_for_radar = {m: "lower" for m in DEFAULT_METRICS_TO_ANALYZE}

    analyzer = AnalysisReportGenerator(
        json_dir=args.json_dir,
        plots_dir=str(plots_dir),
        dataset_name=dataset_name,
        metrics_to_analyze=DEFAULT_METRICS_TO_ANALYZE,
        radar_metric_mapping=DEFAULT_RADAR_MAPPING,
        metric_types_for_radar=metric_types_for_radar,
        alpha=DEFAULT_ALPHA,
    )
    analyzer.run()
