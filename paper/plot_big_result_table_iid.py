import lib
from collections import OrderedDict

data = OrderedDict()

data["SCAN (length cutoff=26)"] = lib.get_runs(["scan_trafo_length_cutoff"], filters = {"config.scan.length_cutoff.value": 26}), "val", "iid", None
data["c"] = None, None, None, None
data["COGS"] = lib.get_runs(["cogs_trafo"]), "gen", "val", None
data["d"] = None, None, None, None
math_runs = lib.get_runs(["dm_math"])
data["Math: add\\_or\\_sub"] = [r for r in math_runs if r.config["dm_math.task"] == "arithmetic__add_or_sub"], "extrapolate", "interpolate", None
data["Math: place\\_value"] = [r for r in math_runs if r.config["dm_math.task"] == "numbers__place_value"], "extrapolate", "interpolate", None


columns = OrderedDict()
columns["Trafo"] = ["scaledinit", "noscale", "opennmt"]
columns["Uni. Trafo"] = ["universal_scaledinit", "universal_noscale", "universal_opennmt"]
columns["Rel. Trafo"] = ["relative"]
columns["Rel. Uni. Trafo"] = ["relative_universal"]

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

def format_results(runs, split_name, ex_split_name, step) -> str:
    run_group = lib.common.group(runs, ['transformer.variant'])

    cols = []
    ex_cols = []
    for clist in columns.values():
        found = []
        for c in clist:
            full_name = f"transformer.variant_{c}"
            if full_name in run_group:
                found.append((
                    average_accuracy(run_group[full_name], split_name, step),
                    average_accuracy(run_group[full_name], ex_split_name, step),
                ))

        if found:
            max_i = max(range(len(found)), key=lambda i: found[i][1].mean)
            cols.append(found[max_i][0])
            ex_cols.append(found[max_i][1])
        else:
            cols.append(None)
            ex_cols.append(None)

    maxval = max(c.mean for c in cols if c is not None)
    cols = [(("{\\bf" if c.mean > maxval-0.01 else "") + f"{c.mean:.2f} $\\pm$ {c.std:.2f}" +
             ("}" if c.mean > maxval-0.01 else "")) if c is not None else "-" for c in cols]
    cols = [c + (f" ({exc.mean:.2f})" if exc else "") for c, exc in zip(cols, ex_cols)]
    return " & ".join(cols)

print(" & " + " & ".join(columns.keys()) + " \\\\")
print("\\midrule")
for dname, (runs, ex_split_name, splitname, at_step) in data.items():
    if runs is None:
        print("\\midrule")
    else:
        print(f"{dname} & {format_results(runs, splitname, ex_split_name, at_step)} \\\\")
