from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ALPHA = 0.05

INPUT_PATH = Path("reports/experiment1/paired_final_comparison.csv")
OUTPUT_PATH = Path("reports/experiment1/exp1_statistical_tests.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "seed",
        "maddpg_final_return",
        "independent_ddpg_final_return",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    maddpg = df["maddpg_final_return"].to_numpy(dtype=float)
    iddpg = df["independent_ddpg_final_return"].to_numpy(dtype=float)

    if len(maddpg) != len(iddpg):
        raise ValueError("MADDPG and IDDPG results must have the same number of matched seeds.")

    differences = maddpg - iddpg
    n = len(differences)

    # Descriptive statistics
    maddpg_mean = np.mean(maddpg)
    maddpg_std = np.std(maddpg, ddof=1)

    iddpg_mean = np.mean(iddpg)
    iddpg_std = np.std(iddpg, ddof=1)

    mean_difference = np.mean(differences)
    std_difference = np.std(differences, ddof=1)

    maddpg_wins = int(np.sum(differences > 0))
    iddpg_wins = int(np.sum(differences < 0))
    ties = int(np.sum(differences == 0))

    # Paired t-test
    t_result = stats.ttest_rel(maddpg, iddpg)

    # Wilcoxon signed-rank test.
    # Two-sided because the statistical question is whether the methods differ,
    # rather than assuming MADDPG must be better.
    wilcoxon_result = stats.wilcoxon(
        maddpg,
        iddpg,
        alternative="two-sided",
        method="auto",
    )

    # 95% confidence interval for the mean paired difference.
    standard_error = std_difference / np.sqrt(n)
    t_critical = stats.t.ppf(1 - ALPHA / 2, df=n - 1)
    ci_low = mean_difference - t_critical * standard_error
    ci_high = mean_difference + t_critical * standard_error

    results = pd.DataFrame(
        [
            {
                "num_matched_seeds": n,
                "maddpg_mean_return": maddpg_mean,
                "maddpg_std_return": maddpg_std,
                "idddpg_mean_return": iddpg_mean,
                "idddpg_std_return": iddpg_std,
                "mean_paired_difference_maddpg_minus_idddpg": mean_difference,
                "paired_difference_95ci_low": ci_low,
                "paired_difference_95ci_high": ci_high,
                "maddpg_wins": maddpg_wins,
                "idddpg_wins": iddpg_wins,
                "ties": ties,
                "paired_t_statistic": t_result.statistic,
                "paired_t_pvalue": t_result.pvalue,
                "wilcoxon_statistic": wilcoxon_result.statistic,
                "wilcoxon_pvalue": wilcoxon_result.pvalue,
                "alpha": ALPHA,
                "statistically_significant_wilcoxon": wilcoxon_result.pvalue < ALPHA,
                "conclusion": (
                    "statistically significant difference"
                    if wilcoxon_result.pvalue < ALPHA
                    else "no statistically significant difference"
                ),
            }
        ]
    )

    results.to_csv(OUTPUT_PATH, index=False)

    print("Experiment 1: MADDPG vs Independent DDPG")
    print("=" * 50)
    print(f"Matched seeds: {n}")
    print(f"MADDPG mean +/- SD: {maddpg_mean:.4f} +/- {maddpg_std:.4f}")
    print(f"IDDPG  mean +/- SD: {iddpg_mean:.4f} +/- {iddpg_std:.4f}")
    print(f"MADDPG wins: {maddpg_wins}/{n}")
    print(f"IDDPG wins: {iddpg_wins}/{n}")
    print(f"Mean paired difference: {mean_difference:+.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print()
    print(
        f"Paired t-test: t={t_result.statistic:.4f}, "
        f"p={t_result.pvalue:.6f}"
    )
    print(
        f"Wilcoxon signed-rank: W={wilcoxon_result.statistic:.4f}, "
        f"p={wilcoxon_result.pvalue:.6f}"
    )
    print()
    print(
        f"Conclusion at alpha={ALPHA}: "
        f"{results.iloc[0]['conclusion']}."
    )
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

