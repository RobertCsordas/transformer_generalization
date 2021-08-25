import lib
import matplotlib.pyplot as plt
import os

format = "pdf"
os.makedirs("out", exist_ok=True)

runs = lib.get_runs(["cogs_trafo"], filters={"config.transformer.variant.value": "opennmt"})

def plot(runs, group, from_iter: int = 0, loss_group=None):
    xs = []
    ys = []
    cs = []

    loss_group = loss_group or group

    for r in runs:
        hist = r.history(keys=[f"validation/{group}/accuracy/total", f"validation/{loss_group}/loss", "iteration"], 
                         pandas=False)
        for p in hist:
            if p["iteration"] < from_iter:
                continue
            xs.append(p[f"validation/{loss_group}/loss"])
            ys.append(p[f"validation/{group}/accuracy/total"]*100)
            cs.append(p["iteration"])
    sc = plt.scatter(xs, ys, c=cs)
    plt.xscale('log')
    cbar = plt.colorbar(sc, ticks=[min(cs), max(cs)], pad=0.02)
    cbar.ax.set_yticklabels([f"{min(cs)//1000}k", f"{max(cs)//1000}k"])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size=0.25, pad=0.1)
    # plt.colorbar(im, cax)

    # plt.tick_params(axis='x', labelsize=8) 
    fig.axes[0].yaxis.set_label_coords(-0.120, 0.45)

def plot_test_axis_labels():
    plt.xlabel("Test loss")
    plt.ylabel("Test accuracy [\%]")

fig = plt.figure(figsize=[4.5,1.5])
plot(runs, "gen", 1000)
plot_test_axis_labels()
fig.axes[0].xaxis.set_minor_formatter(lambda x, _: f"{x:.2f}")
fig.savefig(f"out/cogs_loss_accuracy.{format}", bbox_inches='tight', pad_inches=0.01)

fig = plt.figure(figsize=[4.5,1.4])
plot(runs, "val", 1000)
plt.xlabel("Validation loss")
plt.ylabel("Val. accuracy [\%]")
fig.savefig(f"out/cogs_loss_accuracy_val.{format}", bbox_inches='tight', pad_inches=0.01)


runs = lib.get_runs(["cfq_mcd"])
# The API doesn't support dot in names, so filter manually.
runs = [r for r in runs if r.config["cfq.split"] == "mcd1" and r.config["transformer.variant"] == "relative"]

fig = plt.figure(figsize=[4.5,1.5])
plot(runs, "test", 1000, loss_group="val")
fig.axes[0].xaxis.set_minor_formatter(lambda x, _: f"{x:.2f}")
plt.xlabel("Validation loss")
plt.ylabel("Test accuracy [\%]")
fig.savefig(f"out/cfq_loss_accuracy.{format}", bbox_inches='tight', pad_inches=0.01)