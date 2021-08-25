import lib
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

format = "pdf"
runs = lib.get_runs(["cfq_mcd"], filters={"config.cfq.split.value": "mcd1", "config.transformer.variant.value": "relative"})

os.makedirs("out/loss_analysis", exist_ok=True)

for r in runs:
    bdir = f"out/loss_analysis/{r.id}"
    if not os.path.isdir(bdir):
        os.makedirs(f"out/loss_analysis/{r.id}")
        r.file("export/loss_details/test.pth").download(root=bdir, replace=True)

    hist = r.history(keys=["validation/test/loss", "iteration"], pandas=False)
    x = [p["iteration"] for p in hist]

    data = torch.load(f"{bdir}/export/loss_details/test.pth")

    good_mask = np.sum(data["oks"], 0) > 0
    # good_mask=data["oks"][-1]
    good = data["losses"][:, good_mask]
    bad = data["losses"][:, ~good_mask]

    fig = plt.figure(figsize=[4.5,1.5])
    plt.plot(x, np.mean(good, -1))
    plt.plot(x, np.mean(bad, -1))
    plt.plot(x, np.mean(data["losses"], -1))
    plt.legend(["``Good''", "``Bad''", "Total"], loc="upper left")
    fig.axes[0].xaxis.set_major_formatter(lambda x, _: f"{x//1000:.0f}k" if x >= 1000 else f"{x:.0f}")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.xlim(0, x[-1])
    fig.savefig(f"out/loss_decomposed.{format}", bbox_inches='tight', pad_inches=0.01)

    fig = plt.figure(figsize=[4.5,1.5])
    plt.hist(good[0], 40, alpha=0.8, zorder=2, range=(0,15))
    plt.hist(good[-1], 40, alpha=0.8, zorder=1, range=(0,15))
    plt.legend([f"Training step {x[0]//1000}k", f"Training step {x[-1]//1000}k"])
    plt.xlabel("Loss")
    plt.ylabel("Count")
    fig.savefig(f"out/loss_good_hist.{format}", bbox_inches='tight', pad_inches=0.01)

    fig = plt.figure(figsize=[4.5,1.5])
    plt.hist(bad[0], 40, alpha=0.8, zorder=2, range=(0,15))
    plt.hist(bad[-1], 40, alpha=0.8, zorder=1, range=(0,15))
    plt.legend([f"Training step {x[0]//1000}k", f"Training step {x[-1]//1000}k"])
    plt.xlabel("Loss")
    plt.ylabel("Count")
    fig.savefig(f"out/loss_bad_hist.{format}", bbox_inches='tight', pad_inches=0.01)

    break