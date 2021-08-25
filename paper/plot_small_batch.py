import lib
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

format = "pdf"
os.makedirs("out", exist_ok=True)

columns = OrderedDict()
columns["Trafo"] = ["scaledinit", "noscale", "opennmt"]
columns["Uni. Trafo"] = ["universal_scaledinit", "universal_noscale", "universal_opennmt"]
columns["Rel. Trafo"] = ["relative"]
columns["Rel. Uni. Trafo"] = ["relative_universal"]


cfq_big_runs = lib.get_runs(["cfq_mcd", "cfq_mcd_universal"])
cfq_small_runs = lib.get_runs(["cfq_mcd_small_batch", "cfq_mcd_small_batch_universal"])
variants = OrderedDict()
variants["CFQ MCD 1"] = [r for r in cfq_big_runs if r.config["cfq.split"] == "mcd1"], [r for r in cfq_small_runs if r.config["cfq.split"] == "mcd1"], "test"
variants["CFQ MCD 2"] = [r for r in cfq_big_runs if r.config["cfq.split"] == "mcd2"], [r for r in cfq_small_runs if r.config["cfq.split"] == "mcd2"], "test"
variants["CFQ MCD 3"] = [r for r in cfq_big_runs if r.config["cfq.split"] == "mcd3"], [r for r in cfq_small_runs if r.config["cfq.split"] == "mcd3"], "test"
variants["CFQ Out. len."] = lib.get_runs(["cfq_out_length", "cfq_out_length_universal"]), lib.get_runs(["cfq_out_length_small_batch", "cfq_out_length_universal_small_batch"]), "test"


def average_accuracy(runs, split_name):
    st = lib.StatTracker()
    runs = list(runs)
    it = max([r.summary["iteration"] for r in runs])
    for r in runs:
        st.add(r.summary[f"validation/{split_name}/accuracy/total"])
        assert r.summary["iteration"] == it
    return st.get()

drops = []
details = []

for big_runs, small_runs, split_name in variants.values():
    bgroup = lib.common.group(big_runs, ['transformer.variant'])
    sgroup = lib.common.group(small_runs, ['transformer.variant'])

    for vlist in columns.values():
        bacc = max([average_accuracy(bgroup[f"transformer.variant_{v}"], split_name) for v in vlist if f"transformer.variant_{v}" in bgroup], key=lambda x: x.mean)
        sacc = max([average_accuracy(sgroup[f"transformer.variant_{v}"], split_name) for v in vlist if f"transformer.variant_{v}" in sgroup], key=lambda x: x.mean)

        details.append((bacc,sacc))
        drops.append(sacc.mean/bacc.mean)


print("& " + " & ".join(columns.keys())+"\\\\")
print("\\midrule")
for i, vname in enumerate(variants.keys()):
    best = max(range(len(columns)), key = lambda j: drops[i * len(columns) + j])
    print(vname + " & " + " & ".join([("\\bf" if j==best else "") + f"{drops[i*len(columns) + j]:.2f}" for j in range(len(columns))])+ "\\\\")



print("& Variant & " + " & ".join(columns.keys())+"\\\\")
for i, vname in enumerate(variants.keys()):
    col = [details[i*len(columns) + j] for j in range(len(columns))]

    print("\\midrule")
    print(f"\\multirow{{3}}{{*}}{{{vname}}} & Big & "+ " & ".join([f"${c[0].mean:.2f}\\pm{c[0].std:.2f}$" for c in col])+ "\\\\")
    print(f" & Small & "+ " & ".join([f"${c[1].mean:.2f}\\pm{c[1].std:.2f}$" for c in col])+ "\\\\")

    best = max(range(len(columns)), key = lambda j: drops[i * len(columns) + j])
    print("\\cmidrule{2-6}")
    print(" & Proportion & " + " & ".join([("\\bf" if j==best else "") + f"{drops[i*len(columns) + j]:.2f}" for j in range(len(columns))])+ "\\\\")

pos = [(len(columns) + 1.5) * i for i in range(len(variants))]
fig = plt.figure(figsize=[4.5,2])
for i in range(len(columns)):
    plt.barh([p + i + 0.5 * (i//2) for p in pos], drops[i::len(columns)])

plt.yticks([p + (len(columns)-1 + 0.5)/2 for p in pos], list(variants.keys()))
plt.legend(list(columns.keys()), loc='upper left')
plt.axvline(x=1.0, color="red", zorder=-100, linestyle="-")
fig.savefig(f"out/small_batch_transformer.{format}", bbox_inches='tight', pad_inches=0.01)
