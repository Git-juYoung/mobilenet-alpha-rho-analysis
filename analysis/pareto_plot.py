from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    results_dir = Path("results")
    mn_path = results_dir / "mobilenet_results.csv"
    sc_path = results_dir / "standard_results.csv"

    mn = pd.read_csv(mn_path)
    sc = pd.read_csv(sc_path)

    mn["compute"] = (mn["alpha"] ** 2) * (mn["p"] ** 2)

    sc_compute = 1.0
    sc_acc = float(sc.loc[0, "test_acc"])

    plt.figure(figsize=(7, 5))

    plt.scatter(mn["compute"], mn["test_acc"], label="MobileNet (alpha, p)")
    for _, row in mn.iterrows():
        plt.text(row["compute"], row["test_acc"], f"a={row['alpha']}, p={row['p']}", fontsize=8)

    plt.scatter([sc_compute], [sc_acc], label="StandardCNN (baseline)")
    plt.text(sc_compute, sc_acc, "StandardCNN", fontsize=9)

    plt.xlabel("Relative Compute (alpha^2 * p^2)  [StandardCNN fixed at 1.0]")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Compute (MobileNet grid + StandardCNN baseline)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_dir = Path("assets")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "pareto_with_standard.png"
    plt.savefig(out_path, dpi=200)
    print(f"[Saved] {out_path}")

    plt.show()


if __name__ == "__main__":
    main()