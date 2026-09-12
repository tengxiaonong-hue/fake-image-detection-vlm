import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("github_results/final_5models_comparison.csv")

name_map = {
    "Qwen2.5-VL-3B": "Qwen-3B",
    "Qwen2.5-VL-7B": "Qwen-7B",
    "Idefics3-8B": "Idefics3-8B",
    "LLaVA-OneVision-7B": "LLaVA-7B",
    "BLIP2-OPT-2.7B": "BLIP2"
}

strategy_map = {
    "baseline": "Baseline",
    "cause_to_effect": "CtE",
    "effect_to_cause": "EtC"
}

df["Model"] = df["Model"].replace(name_map)
df["Strategy"] = df["Strategy"].replace(strategy_map)

Path("figures").mkdir(exist_ok=True)

def make_chart(metric, ylabel, title, filename, percent=False):
    pivot = df.pivot(index="Model", columns="Strategy", values=metric)

    order = ["Baseline", "CtE", "EtC"]
    pivot = pivot[[c for c in order if c in pivot.columns]]

    ax = pivot.plot(
        kind="bar",
        figsize=(11, 6),
        width=0.8
    )

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.3)

    if percent:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


make_chart(
    "Accuracy",
    "Accuracy",
    "Fake Image Detection Accuracy",
    "accuracy_comparison.png",
    percent=True
)

make_chart(
    "F1",
    "F1 Score",
    "Fake Image Detection F1 Score",
    "f1_comparison.png",
    percent=True
)

make_chart(
    "Mean_Latency_s",
    "Mean Latency (seconds)",
    "Mean Inference Latency",
    "latency_comparison.png"
)

print("Saved:")
print("figures/accuracy_comparison.png")
print("figures/f1_comparison.png")
print("figures/latency_comparison.png")
