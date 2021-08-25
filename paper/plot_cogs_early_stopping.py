import lib
import matplotlib.pyplot as plt
import os
import statistics
from collections import OrderedDict

format = "pdf"
os.makedirs("out", exist_ok=True)

runs = lib.get_runs(["cogs_trafo"])
runs = lib.common.group(runs, ['transformer.variant'])
runs["theirs"] = lib.get_runs(["cogs_trafo_official"])

window = 2500


fig = plt.figure(figsize=[4.5,1.5])

def download(run, *args, **kwargs):
    hist = run.history(*args, **kwargs, pandas=False)
    points = {p["iteration"]: p for p in hist}
    iters = list(sorted(points.keys()))
    return iters, points

def plot_runs(runs):
    data = []
    for r in runs:
        iters, points = download(r, keys=["validation/gen/accuracy/total", "iteration"])
        acc = [points[i]["validation/gen/accuracy/total"] for i in iters]

        # They might be recorded with different frequencies, so query them twice
        iters2, points = download(r, keys=["validation/val/time_since_best_loss", "iteration"])

        stop_point = iters2[-1]
        for i in iters2:
            if points[i]["validation/val/time_since_best_loss"] >= window:
                stop_point = i
                break
        data.append((iters, acc, stop_point - window))

    for d in data:
        assert d[0] == data[0][0]

    ystat = [lib.StatTracker() for _ in data[0][0]]
    for d in data:
        for s, v in zip(ystat, d[1]):
            s.add(v)
    ystat = [s.get() for s in ystat]
    mean = [s.mean*100 for s in ystat]
    std = [s.std*100 for s in ystat]

    median_stop_point = statistics.median([d[2] for d in data])

    p = plt.plot(data[0][0], mean)
    plt.fill_between(data[0][0], [m-s for m, s in zip(mean, std)], [m+s for m, s in zip(mean, std)], alpha=0.3)
    color = p[-1].get_color()
    return lambda: plt.axvline(x=median_stop_point, color=color, zorder=-100, linestyle="--")


plt.xlabel("Training steps")
plt.ylabel("Gen. accuracy [$\\%$]")

d = OrderedDict()
d["No scaling"] = "transformer.variant_noscale"
d["Token Emb. Up., Noam"] = "theirs"
d["Position Emb. Down."] = "transformer.variant_scaledinit"

print(list(runs.keys()))
plot_markers = []
for n in d.values():
    plot_markers.append(plot_runs(runs[n]))

# Markers must be plotted after the lines because legend will not work otherwise
for pm in plot_markers:
    pm()

plt.legend(list(d.keys()))
fig.axes[0].xaxis.set_major_formatter(lambda x, _: f"{x//1000:.0f}k" if x >= 1000 else f"{x:.0f}")
plt.xlim(0,50000)
fig.savefig(f"out/cogs_early_stop.{format}", bbox_inches='tight', pad_inches=0.01)
