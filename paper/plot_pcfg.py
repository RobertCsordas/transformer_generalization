import lib
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import OrderedDict

format = "pdf"
os.makedirs("out", exist_ok=True)

runs = lib.get_runs(["pcfg_nosched_productivity"])
runs = lib.common.group(runs, ['transformer.variant'])
del runs["transformer.variant_opennmt"]
print(list(runs.keys()))

def get(runs):
    acc = {}
    loss = {}

    for r in runs:
        hist = r.history(keys=["validation/val/accuracy/total", "validation/val/loss", "iteration"], pandas=False)
        for p in hist:
            i = p["iteration"]
            if i not in acc:
                acc[i] = lib.StatTracker()
                loss[i] = lib.StatTracker()

            loss[i].add(p["validation/val/loss"])
            acc[i].add(p["validation/val/accuracy/total"] * 100)

    x = list(sorted(acc.keys()))
    acc = [acc[i].get() for i in x]
    loss = [loss[i].get() for i in x]

    return x, acc, loss

data = {k: get(v) for k, v in runs.items()}

d = OrderedDict()
d["Standard"] = "transformer.variant_scaledinit"
d["Uni."] = "transformer.variant_universal_noscale"
d["Rel. Uni."] = "transformer.variant_relative_universal"

fig = plt.figure(figsize=[4.5,1.5])
for k in d.values():
    plt.plot(data[k][0], [y.mean for y in data[k][1]])
    plt.fill_between(data[k][0], [a.mean - a.std for a in data[k][1]], [a.mean + a.std for a in data[k][1]], alpha=0.3)

plt.legend(d.keys())
fig.axes[0].xaxis.set_major_formatter(lambda x, _: f"{x//1000:.0f}k" if x >= 1000 else f"{x:.0f}")
plt.xlabel("Training steps")
plt.ylabel("Accuracy [\\%]")
plt.xlim(0,300000)
plt.ylim(0,90)
fig.savefig(f"out/pcfg_accuracy.{format}", bbox_inches='tight', pad_inches=0.01)


fig = plt.figure(figsize=[4.5,1.5])
for k in d.values():
    plt.plot(data[k][0], [y.mean for y in data[k][2]])
    plt.fill_between(data[k][0], [a.mean - a.std for a in data[k][2]], [a.mean + a.std for a in data[k][2]], alpha=0.3)

plt.legend(d.keys(), ncol=3)
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.xlim(0,300000)
fig.axes[0].xaxis.set_major_formatter(lambda x, _: f"{x//1000:.0f}k" if x >= 1000 else f"{x:.0f}")
fig.savefig(f"out/pcfg_loss.{format}", bbox_inches='tight', pad_inches=0.01)
