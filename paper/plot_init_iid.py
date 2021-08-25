import lib
from lib.common import calc_stat
from collections import OrderedDict

runs = lib.get_runs(["cogs_trafo"])
runs = lib.common.group(runs, ['transformer.variant'])
stats = calc_stat(runs, lambda k: k.endswith("/accuracy/total"))

runs = lib.get_runs(["pcfg_nosched_productivity"])
runs = lib.common.group(runs, ['transformer.variant'])
pcfg_gen_stats = calc_stat(runs, lambda k: k.endswith("/accuracy/total"))

runs = lib.get_runs(["pcfg_nosched_iid"])
runs = lib.common.group(runs, ['transformer.variant'])
pcfg_iid_stats = calc_stat(runs, lambda k: k.endswith("/accuracy/total"))


columns = OrderedDict()
columns["IID Validation"] = ["val"]
columns["Gen. Test"] = ["gen"]

d = OrderedDict()
d["Token Emb. Up."] = "transformer.variant_opennmt"
d["No scaling"] = "transformer.variant_noscale"
d["Pos. Emb. Down."] = "transformer.variant_scaledinit"

print(stats)

print(" & & " + " & ".join(columns) + "\\\\")
print("\\midrule")
print("\\parbox[t]{3mm}{\\multirow{3}{*}{\\rotatebox[origin=c]{90}{\\small COGS}}}")

def print_table(data):
    best = [max(v[i].mean for v in data.values()) for i in range(len(columns))]
    for vname in d.keys():
        s = data[vname]
        s = [("{\\bf" if m - a.mean < 0.005 else "") + f"{a.mean:.2f} $\\pm$ {a.std:.2f}" + ("}" if m - a.mean < 0.005 else "") for a, m in zip(s, best)]
        print(" & " + vname + " & " + " & ".join(s) +" \\\\")

print_table({vname: [stats[vcode][f"validation/{k[0]}/accuracy/total"].get() for k in columns.values()] for vname, vcode in d.items()})

print("\\midrule")
print("\\parbox[t]{3mm}{\\multirow{3}{*}{\\rotatebox[origin=c]{90}{\\small PCFG}}}")

print_table({vname: [pcfg_iid_stats[vcode]["validation/val/accuracy/total"].get(), pcfg_gen_stats[vcode]["validation/val/accuracy/total"].get()] for vname, vcode in d.items()})
