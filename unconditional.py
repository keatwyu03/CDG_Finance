import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config
from data import DataProcessor
from models import DiffusionModel

config = get_default_config()
data_processor = DataProcessor(
    csv_path=config.data.csv_path,
    tickers=config.data.tickers,
    weekday_col=config.data.weekday_col,
    seq_len=config.data.seq_len,
    test_days=config.data.test_days,
    winsorize_lower=config.data.winsorize_lower,
    winsorize_upper=config.data.winsorize_upper,
)
data_processor.process_all()

tickers = config.data.tickers

# Real marginal distributions: use df_z — the data processor's standardized output
# (winsorized → weekday-demeaned → divided by per-asset sigma).
# The diffusion model was trained on this same space, so both sides are directly comparable.
df_z       = data_processor.df_z
df_z_train = df_z.iloc[: -config.data.test_days]
df_z_test  = df_z.iloc[-config.data.test_days :]

real = {
    t: {
        "train": df_z_train[t].values,
        "test":  df_z_test[t].values,
    }
    for t in tickers
}

# ── Unconditional generation ──────────────────────────────────────────────────
diffusion_model = DiffusionModel(
    in_channels=config.diffusion.in_channels,
    out_channels=config.diffusion.out_channels,
    sample_size=config.diffusion.sample_size,
    layers_per_block=config.diffusion.layers_per_block,
    block_out_channels=config.diffusion.block_out_channels,
    b_min=config.diffusion.b_min,
    b_max=config.diffusion.b_max,
    device=config.diffusion.device,
)
diffusion_model.load("checkpoints/diffusion_model.pt")

N_samples = 500
print(f"Generating {N_samples} unconditional samples...")
uncond = diffusion_model.sample(
    batch_size=N_samples,
    num_steps=config.diffusion.num_steps,
    stoch=0.2,
).cpu()  # (N_samples, channels, seq_len)

gen = {
    t: uncond[:, i, :].numpy().flatten()
    for i, t in enumerate(tickers)
}

# ── Marginal distribution plots ───────────────────────────────────────────────
fig, axes = plt.subplots(len(tickers), 2, figsize=(14, 5 * len(tickers)))

for row, ticker in enumerate(tickers):
    for col, split in enumerate(["train", "test"]):
        ax = axes[row, col]
        real_vals = real[ticker][split]
        gen_vals  = gen[ticker]

        x_lo = min(real_vals.min(), gen_vals.min()) - 0.5
        x_hi = max(real_vals.max(), gen_vals.max()) + 0.5
        x    = np.linspace(x_lo, x_hi, 500)

        for vals, color, label in [
            (real_vals, "darkorange", f"Real {split} (n={len(real_vals)})"),
            (gen_vals,  "steelblue",  f"Generated (n={len(gen_vals)})"),
        ]:
            kde = gaussian_kde(vals, bw_method="silverman")
            ax.plot(x, kde(x), color=color, linewidth=2, label=label)
            ax.hist(vals, bins=40, density=True, alpha=0.2, color=color)

        split_label = "In-Sample (Train)" if split == "train" else "Out-of-Sample (Test)"
        ax.set_title(f"{ticker.upper()} — {split_label}", fontsize=11)
        ax.set_xlabel("Standardized Return (df_z space)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

fig.suptitle(
    "Unconditional Generation: Marginal Distributions vs. Real Data (Stopping 20 Steps Early)",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/score_function_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
