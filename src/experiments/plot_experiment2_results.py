from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPORT_DIR = Path("reports/experiment2")


def load_summary(filename: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / filename)


def pretty_algorithm(name: str) -> str:
    mapping = {
        "maddpg": "MADDPG",
        "independent_ddpg": "Independent DDPG",
        "handcrafted": "Handcrafted",
    }
    return mapping.get(name, name)


def plot_success_rates() -> None:
    exp2a = load_summary(
        "exp2a_common_aggregate.csv"
    )
    exp2b = load_summary(
        "exp2b_common_aggregate.csv"
    )

    algorithms = [
        "maddpg",
        "independent_ddpg",
        "handcrafted",
    ]

    exp2a_success = []
    exp2b_success = []

    for algorithm in algorithms:
        row_a = exp2a[
            exp2a["algorithm"] == algorithm
        ].iloc[0]

        row_b = exp2b[
            exp2b["algorithm"] == algorithm
        ].iloc[0]

        exp2a_success.append(
            100.0 * float(row_a["mean_success_rate"])
        )
        exp2b_success.append(
            100.0 * float(row_b["mean_success_rate"])
        )

    x = range(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_a = ax.bar(
        [value - width / 2 for value in x],
        exp2a_success,
        width,
        label="Exp 2A — Nominal",
    )

    bars_b = ax.bar(
        [value + width / 2 for value in x],
        exp2b_success,
        width,
        label="Exp 2B — Narrow Passage",
    )

    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)

    ax.set_xticks(
        list(x),
        [pretty_algorithm(name) for name in algorithms],
    )

    ax.set_title(
        "Common-Test Success Rate by Controller"
    )

    ax.legend()

    ax.bar_label(
        bars_a,
        fmt="%.0f%%",
        padding=3,
    )

    ax.bar_label(
        bars_b,
        fmt="%.0f%%",
        padding=3,
    )

    fig.tight_layout()

    output = REPORT_DIR / "exp2_success_rates.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {output}")


def plot_mean_returns() -> None:
    exp2a = load_summary(
        "exp2a_common_aggregate.csv"
    )
    exp2b = load_summary(
        "exp2b_common_aggregate.csv"
    )

    algorithms = [
        "maddpg",
        "independent_ddpg",
        "handcrafted",
    ]

    exp2a_returns = []
    exp2b_returns = []

    for algorithm in algorithms:
        row_a = exp2a[
            exp2a["algorithm"] == algorithm
        ].iloc[0]

        row_b = exp2b[
            exp2b["algorithm"] == algorithm
        ].iloc[0]

        exp2a_returns.append(
            float(row_a["mean_return"])
        )

        exp2b_returns.append(
            float(row_b["mean_return"])
        )

    x = range(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_a = ax.bar(
        [value - width / 2 for value in x],
        exp2a_returns,
        width,
        label="Exp 2A — Nominal",
    )

    bars_b = ax.bar(
        [value + width / 2 for value in x],
        exp2b_returns,
        width,
        label="Exp 2B — Narrow Passage",
    )

    ax.set_ylabel("Mean Episode Return")

    ax.set_xticks(
        list(x),
        [pretty_algorithm(name) for name in algorithms],
    )

    ax.set_title(
        "Common-Test Mean Return by Controller"
    )

    ax.legend()

    ax.bar_label(
        bars_a,
        fmt="%.2f",
        padding=3,
    )

    ax.bar_label(
        bars_b,
        fmt="%.2f",
        padding=3,
    )

    fig.tight_layout()

    output = REPORT_DIR / "exp2_mean_returns.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {output}")


def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_success_rates()
    plot_mean_returns()


if __name__ == "__main__":
    main()