from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from depth_reference_common import load_json, write_simple_csv


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize multiple depth-reference evaluation outputs into ranking tables")
    parser.add_argument("--metrics_json", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--rank_threshold_m", type=float, default=0.25)
    return parser


def main() -> None:
    args = _argparser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = [load_json(Path(p).resolve()) for p in args.metrics_json]
    has_agreement_metrics = any(
        any("DepthAgreementRate" in entry for entry in summary["threshold_metrics"])
        for summary in summaries
    )
    ranking_rows: List[List[Any]] = []
    curve_rows: List[List[Any]] = []
    agreement_ranking_rows: List[List[Any]] = []
    for summary in summaries:
        method_name = str(summary["method_name"])
        by_threshold = {float(x["threshold_m"]): x for x in summary["threshold_metrics"]}
        if float(args.rank_threshold_m) not in by_threshold:
            raise ValueError(f"{method_name} does not contain rank threshold {args.rank_threshold_m}")
        rank_entry = by_threshold[float(args.rank_threshold_m)]
        rank_agreement = rank_entry.get("DepthAgreementRate", float("nan"))
        ranking_rows.append(
            [
                method_name,
                f"{rank_entry['FrontIntrusionRate']:.12f}",
                f"{rank_entry['FrontIntrusionMagnitude']:.12f}",
                f"{rank_entry['TooDeepRate']:.12f}",
                f"{summary['secondary_metrics']['MissingRate']:.12f}",
                f"{summary['secondary_metrics']['AbsDepthError_Mean']:.12f}",
                f"{summary['secondary_metrics']['SignedDepthBias_Mean']:.12f}",
                *(
                    [f"{rank_agreement:.12f}"]
                    if has_agreement_metrics
                    else []
                ),
            ]
        )
        if has_agreement_metrics:
            agreement_ranking_rows.append(
                [
                    method_name,
                    f"{rank_agreement:.12f}",
                    f"{rank_entry['FrontIntrusionRate']:.12f}",
                    f"{rank_entry['TooDeepRate']:.12f}",
                    f"{summary['secondary_metrics']['MissingRate']:.12f}",
                    f"{summary['secondary_metrics']['AbsDepthError_Mean']:.12f}",
                    f"{summary['secondary_metrics']['SignedDepthBias_Mean']:.12f}",
                ]
            )
        for entry in summary["threshold_metrics"]:
            entry_agreement = entry.get("DepthAgreementRate", float("nan"))
            curve_row = [
                method_name,
                f"{float(entry['threshold_m']):.2f}",
                f"{entry['FrontIntrusionRate']:.12f}",
                f"{entry['FrontIntrusionMagnitude']:.12f}",
                f"{entry['TooDeepRate']:.12f}",
            ]
            if has_agreement_metrics:
                curve_row.append(f"{entry_agreement:.12f}")
            curve_rows.append(curve_row)

    ranking_rows.sort(key=lambda row: float(row[1]))
    write_simple_csv(
        out_dir / "ranking_at_threshold.csv",
        [
            "method_name",
            f"FrontIntrusionRate@{args.rank_threshold_m:.2f}m",
            f"FrontIntrusionMagnitude@{args.rank_threshold_m:.2f}m",
            f"TooDeepRate@{args.rank_threshold_m:.2f}m",
            "MissingRate",
            "AbsDepthError_Mean",
            "SignedDepthBias_Mean",
            *(
                [f"DepthAgreementRate@{args.rank_threshold_m:.2f}m"]
                if has_agreement_metrics
                else []
            ),
        ],
        ranking_rows,
    )
    write_simple_csv(
        out_dir / "front_intrusion_curve_all_methods.csv",
        [
            "method_name",
            "threshold_m",
            "FrontIntrusionRate",
            "FrontIntrusionMagnitude",
            "TooDeepRate",
            *(
                ["DepthAgreementRate"]
                if has_agreement_metrics
                else []
            ),
        ],
        curve_rows,
    )
    if has_agreement_metrics:
        agreement_ranking_rows.sort(key=lambda row: float(row[1]), reverse=True)
        write_simple_csv(
            out_dir / "agreement_ranking_at_threshold.csv",
            [
                "method_name",
                f"DepthAgreementRate@{args.rank_threshold_m:.2f}m",
                f"FrontIntrusionRate@{args.rank_threshold_m:.2f}m",
                f"TooDeepRate@{args.rank_threshold_m:.2f}m",
                "MissingRate",
                "AbsDepthError_Mean",
                "SignedDepthBias_Mean",
            ],
            agreement_ranking_rows,
        )


if __name__ == "__main__":
    main()
