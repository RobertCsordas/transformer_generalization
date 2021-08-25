import lib
import os
from collections import OrderedDict

format = "pdf"
os.makedirs("out", exist_ok=True)

trafos = OrderedDict()
trafos["TEU"] = "transformer.variant_opennmt"
trafos["No scaling"] = "transformer.variant_noscale"
trafos["PED"] = "transformer.variant_scaledinit"

runs = lib.get_runs(["cogs_trafo"])
runs = lib.common.group(runs, ['transformer.variant'])
runs = lib.common.calc_stat(runs, lambda n: n == "validation/gen/accuracy/total")
cogs_runs = {k: v["validation/gen/accuracy/total"].get() for k, v in runs.items()}

runs = lib.get_runs(["pcfg_nosched_productivity"])
runs = lib.common.group(runs, ['transformer.variant'])
runs = lib.common.calc_stat(runs, lambda n: n == "validation/val/accuracy/total")
pcfg_runs = {k: v["validation/val/accuracy/total"].get() for k, v in runs.items()}

all_runs = OrderedDict()
all_runs["COGS"] = cogs_runs
all_runs["PCFG"] = pcfg_runs

print(pcfg_runs)

print(" & "+" & ".join(trafos.keys())+"\\\\")
print("\\midrule")
for rname, runs in all_runs.items():
    means = [runs[k].mean for k in trafos.values()]
    maxmean = max(means)
    nums = [f"{runs[k].mean:.2f} \\pm {runs[k].std:.2f}" for k in trafos.values()]
    nums = [f"$\\mathbf{{{n}}}$" if m > maxmean-0.01 else f"${n}$" for n, m in zip(nums, means)]

    print(rname + " & " + " & ".join(nums) + "\\\\")
