import lib
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import statistics

format = "pdf"
os.makedirs("out", exist_ok=True)


columns = OrderedDict()
columns["Trafo"] = ["scaledinit", "noscale", "opennmt"]
columns["Uni. Trafo"] = ["universal_scaledinit", "universal_noscale", "universal_opennmt"]
columns["Rel. Trafo"] = ["relative"]
columns["Rel. Uni. Trafo"] = ["relative_universal"]


variants = OrderedDict()
# Opennmt variant unstable, skip
variants["PCFG"] = [r for r in lib.get_runs(["pcfg_nosched_productivity"]) if r.config["transformer.variant"] != "opennmt"], "val"
cfq_runs = lib.get_runs(["cfq_mcd", "cfq_mcd_2seed", "cfq_mcd_universal", "cfq_mcd_universal_2seed"])
variants["CFQ MCD 1"] = [r for r in cfq_runs if r.config["cfq.split"] == "mcd1"], "test"
variants["CFQ MCD 2"] = [r for r in cfq_runs if r.config["cfq.split"] == "mcd2"], "test"
variants["CFQ MCD 3"] = [r for r in cfq_runs if r.config["cfq.split"] == "mcd3"], "test"
variants["COGS"] = lib.get_runs(["cogs_trafo"]), "gen"
math_runs = lib.get_runs(["dm_math"])
variants["Math: add\\_or\\_sub"] = [r for r in math_runs if r.config["dm_math.task"] == "arithmetic__add_or_sub"], "extrapolate"
# Relative variant crashes, skip it
variants["Math: place\\_value"] = [r for r in math_runs if r.config["dm_math.task"] == "numbers__place_value" and r.config["transformer.variant"] != "relative"], "extrapolate"


def calculate_converged_iter(runs, key="val", threshold = 0.80):
    stop_list = []
    for r in runs:
        hist = r.history(keys=[f"validation/{key}/accuracy/total", "iteration"], pandas=False)
        x = list(sorted(a["iteration"] for a in hist))
        y = [None for _ in x]

        for h in hist:
            y[x.index(h["iteration"])] = h[f"validation/{key}/accuracy/total"]

        th = max(y) * threshold
        first = None
        for i, a in enumerate(y):
            if a >= th:
                first = i
                break

        interp = x[first - 1] + (x[first] - x[first-1])/(y[first] - y[first-1]) * (th - y[first-1])
        stop_list.append(interp)

    return statistics.median(stop_list)

all_numbers = []

for vname, (runs, key) in variants.items():
    runs = lib.common.group(runs, ['transformer.variant'])

    line = []
    for varlist in columns.values():
        found = []
        for v in varlist:
            v = f"transformer.variant_{v}"
            if v in runs:
                found.append(calculate_converged_iter(runs[v], key))

        line.append(min(found) if found else None)

    all_numbers.append(line)

    best = min(l for l in line if l)
    s_line = [(f"{f/1000:.0f}k" if f>10000 else f"{f/1000:.1f}k") if f else "-" for f in line]
    s_line = [f"\\textbf{{{s}}}" if n==best else s for s, n in zip(s_line, line)]

    print(vname + " & " + " & ".join(s_line))


fig = plt.figure(figsize=[4.5,2])

speedups = []
for vname, l in zip(variants.keys(), all_numbers):
    speedups.append((l[0] / l[2]) if l[0] and l[2] else 0)
    speedups.append((l[1] / l[3]) if l[1] and l[3] else 0)

pos = [2.5 * i for i in range(len(variants))]
plt.barh(pos, speedups[::2])
plt.barh([p + 1 for p in pos], speedups[1::2])
plt.yticks([p + 0.5 for p in pos], list(variants.keys()))
plt.legend(["Trafo", "Uni. Trafo"])
plt.axvline(x=1.0, color="red", zorder=-100, linestyle="-")
fig.savefig(f"out/convergence_transformer.{format}", bbox_inches='tight', pad_inches=0.01)
