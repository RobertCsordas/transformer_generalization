import lib
from collections import OrderedDict

data = OrderedDict()
data["SCAN (length cutoff=26)"] = lib.get_runs(["scan_trafo_length_cutoff"], filters = {"config.scan.length_cutoff.value": 26}), "val", None, "$0.00^{[1]}$"
data["a"] = None, None, None, None
data["CFQ Output length"] = lib.get_runs(["cfq_out_length", "cfq_out_length_universal"]), "test", 35000, "$\\sim 0.66^{[2]}$"
cfq_runs = lib.get_runs(["cfq_mcd", "cfq_mcd_universal"])
data["CFQ MCD 1"] = [r for r in cfq_runs if r.config["cfq.split"] == "mcd1"], "test", 35000, "$0.37\\pm0.02^{[3]}$"
data["CFQ MCD 2"] = [r for r in cfq_runs if r.config["cfq.split"] == "mcd2"], "test", 35000, "$0.08\\pm0.02^{[3]}$"
data["CFQ MCD 3"] = [r for r in cfq_runs if r.config["cfq.split"] == "mcd3"], "test", 35000, "$0.11\\pm0.00^{[3]}$"
data["CFQ MCD mean"] = cfq_runs, "test", 35000, "$0.19\\pm0.01^{[2]}$"
data["b"] = None, None, None, None
data["PCFG Productivity split"] = lib.get_runs(["pcfg_nosched_productivity"]), "val", None, "$0.50\\pm0.02^{[4]}$"
data["PCFG Systematicity split"] = lib.get_runs(["pcfg_nosched_systematicity"]), "val", None, "$0.72\\pm0.00^{[4]}$"
data["c"] = None, None, None, None
data["COGS"] = lib.get_runs(["cogs_trafo"]), "gen", None, "$0.35\\pm0.06^{[5]}$"
data["d"] = None, None, None, None
math_runs = lib.get_runs(["dm_math"])
data["Math: add\\_or\\_sub"] = [r for r in math_runs if r.config["dm_math.task"] == "arithmetic__add_or_sub"], "extrapolate", None, "$\\sim0.91^{[6]*}$"
data["Math: place\\_value"] = [r for r in math_runs if r.config["dm_math.task"] == "numbers__place_value"], "extrapolate", None, "$\\sim0.69^{[6]*}$"


columns = OrderedDict()
columns["Trafo"] = ["scaledinit", "noscale", "opennmt"]
columns["Uni. Trafo"] = ["universal_scaledinit", "universal_noscale", "universal_opennmt"]
columns["Rel. Trafo"] = ["relative"]
columns["Rel. Uni. Trafo"] = ["relative_universal"]

init_type_table = {
    "scaledinit": "PED",
    "noscale": "No scaling",
    "opennmt": "TEU",
    "universal_scaledinit": "PED",
    "universal_noscale": "No scaling",
    "universal_opennmt": "TEU",
    "relative": "No scaling",
    "relative_universal": "No scaling"
}

init_order = ["PED", "TEU", "No scaling"]

def average_accuracy(runs, split_name, step) -> float:
    st = lib.StatTracker()
    runs = list(runs)
    it = max([r.summary["iteration"] for r in runs])
    for r in runs:
        if f"validation/{split_name}/accuracy/total" not in r.summary:
            continue

        if step is None:
            st.add(r.summary[f"validation/{split_name}/accuracy/total"])
            assert r.summary["iteration"] == it, f"Inconsistend final iteration for run {r.id}: {r.summary['iteration']} instead of {it}"
        else:
            hist = r.history(keys=[f"validation/{split_name}/accuracy/total", "iteration"], pandas=False)
            for h in hist:
                if h["iteration"] == step:
                    st.add(h[f"validation/{split_name}/accuracy/total"])
                    break
            else:
                assert False, f"Step {step} not found."
    return st.get()

def format_results(runs, title, split_name, step, best_other) -> str:
    run_group = lib.common.group(runs, ['transformer.variant'])
    variants = {v for k, v in init_type_table.items() if f"transformer.variant_{k}" in run_group}

    rtmp = []
    for i in init_order:
        if i not in variants:
            continue

        cols = []
        for clist in columns.values():
            cols.append(None)
            for c in clist:
                full_name = f"transformer.variant_{c}"
                if init_type_table[c] == i and full_name in run_group:
                    assert cols[-1] is None, "Can't be multiple variants"
                    cols[-1] = average_accuracy(run_group[full_name], split_name, step)

        rtmp.append((i, cols))

    cols = [[r[1][i].mean for r in rtmp if r[1][i]] for i in range(len(columns))]
    maxy = [round(max(c), 2) if c else None for c in cols]

    rows = []
    for i, cols in rtmp:
        maxx = max(c.mean for c in cols if c is not None)
        cols = [(("\\bf{" if maxy[ci] and round(c.mean, 2) == maxy[ci] else "") + \
                #("\\bf{" if abs(c.mean - maxx) < eps else "") + \
                #("\\emph{" if abs(c.mean - max_all) < eps else "") + \
                f"{c.mean:.2f} $\\pm$ {c.std:.2f}" + \
                #("}" if abs(c.mean - max_all) < eps else "") + \
                #("}" if abs(c.mean - maxx) < eps else "") + \
                ("}" if maxy[ci] and round(c.mean, 2) == maxy[ci] else "")) if c is not None else "-" \
                for ci, c in enumerate(cols)]

        rows.append(f"{i} & " + " & ".join(cols))

    res = f"\\multirow{{{len(rows)}}}{{*}}{{{title}}} & " + rows[0] + f" & \\multirow{{{len(rows)}}}{{*}}{{{best_other}}} \\\\ \n"
    for r in rows[1:]:
        res += f" & {r} & \\\\ \n"

    return res

print(" & Init & " + " & ".join(columns.keys()) + " & Reported\\\\")
print("\\midrule")
is_first_in_block = True
for dname, (runs, splitname, at_step, best_other) in data.items():
    if runs is None:
        print("\\midrule")
        is_first_in_block = True
    else:
        if not is_first_in_block:
            print("\\cmidrule{2-7}")
        is_first_in_block = False
        print(format_results(runs, dname, splitname, at_step, best_other), end="")
