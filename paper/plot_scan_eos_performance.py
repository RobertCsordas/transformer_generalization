import lib
from collections import OrderedDict

runs = lib.get_runs(["scan_trafo_length_cutoff"])
runs = lib.common.group(runs, ['transformer.variant', "scan.length_cutoff"])

variants = OrderedDict()
variants["Trafo"] = ["scaledinit", "noscale", "opennmt"]
variants["Uni. Trafo"] = ["universal_scaledinit", "universal_noscale", "universal_opennmt"]
variants["Rel. Trafo"] = ["relative"]
variants["Rel. Uni. Trafo"] = ["relative_universal"]

lengths = [22, 24, 25, 26, 27, 28, 30, 32, 33, 36, 40]

best = [0.58, 0.54, 0.69, 0.82, 0.88, 0.85, 0.89, 0.82, 1.00, 1.00, 1.00]

stats = lib.common.calc_stat(runs, lambda name: name.endswith("val/accuracy/total"), tracker=lib.MedianTracker)

ourtab = []
for i, (v, vlist) in enumerate(variants.items()):
    ourtab.append([])
    for l in lengths:
        all_stats = [stats.get(f"transformer.variant_{vn}/scan.length_cutoff_{l}") for vn in vlist]
        all_stats = [a for a in all_stats if a is not None]
        assert all([len(a) == 1 for a in all_stats])
        all_stats = [list(a.values())[0].get() for a in all_stats]
        ourtab[-1].append(max(all_stats))

for l in ourtab:
    for i, v in enumerate(l):
        best[i] = max(best[i], v)

for i, (v, vn) in enumerate(variants.items()):
    pstr = []
    for j, val in enumerate(ourtab[i]):
        pstr.append(("\\bf" if best[j] - val < 0.02 else "") + f"{val:.2f}")

    print(f"{' & ' if i>0 else ''}\\texttt{{{v}}}\\xspace & {' & '.join(pstr)} \\\\")
