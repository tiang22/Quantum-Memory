import sinter
import numpy as np
import sinter._plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
from sinter._plotting import _FrozenDict as FrozenDict


def get_estimated_lifetime(error, time):
    # p = 1/2 * (1 - exp(-time / life))
    if error >= 0.5:
        return 0.0
    return -time / np.log(1 - 2 * error)


cnot_scheduling_name = {
    "full": r"$k=d(d-1)$",
    "half": r"$k=d(d-1)/2$",
    "sqrt": r"$k=d-1$",
    "minimal": r"$k=2$",
    "serial": r"$k=1$",
}


def geo_mean(arr):
    arr = np.array(arr)
    if np.any(arr == 0.0):
        return 0.0
    else:
        return np.exp(np.mean(np.log(arr)))


def combine_ZX_stats(stats: list[sinter.TaskStats]):
    stats_dict = {}
    for stat in stats:
        json_metadata = deepcopy(stat.json_metadata)
        json_metadata.pop("basis")
        json_frozen = FrozenDict(json_metadata)
        if not json_frozen in stats_dict:
            stats_dict[json_frozen] = []
        stats_dict[json_frozen].append(stat)

    stats_combined = []
    for json_frozen in stats_dict:
        stats_this = stats_dict[json_frozen]
        fits = [
            sinter._plotting.fit_binomial(
                num_shots=stat.shots, num_hits=stat.errors, max_likelihood_factor=1e3
            )
            for stat in stats_this
        ]
        fit_combined = sinter._plotting.Fit(
            low=geo_mean([f.low for f in fits]),
            best=geo_mean([f.best for f in fits]),
            high=geo_mean([f.high for f in fits]),
        )
        stats_combined.append(
            sinter.TaskStats(
                "0",
                "",
                (json_frozen, fit_combined),
                sum([stat.shots for stat in stats_this]),
                sum([stat.errors for stat in stats_this]),
                0,
                0.0,
            )
        )

    return stats_combined


def y_func_error_rate(stat: sinter.TaskStats):
    result = stat.json_metadata[1]
    pieces = stat.json_metadata[0]["r"]
    values = 1.0
    result = sinter.Fit(
        low=sinter.shot_error_rate_to_piece_error_rate(
            result.low, pieces=pieces, values=values
        ),
        best=sinter.shot_error_rate_to_piece_error_rate(
            result.best, pieces=pieces, values=values
        ),
        high=sinter.shot_error_rate_to_piece_error_rate(
            result.high, pieces=pieces, values=values
        ),
    )
    return result


def y_func_lifetime(stat: sinter.TaskStats):
    fit_error = stat.json_metadata[1]
    fit_life = sinter.Fit(
        low=get_estimated_lifetime(fit_error.high, stat.json_metadata[0]["t"]),
        best=get_estimated_lifetime(fit_error.best, stat.json_metadata[0]["t"]),
        high=get_estimated_lifetime(fit_error.low, stat.json_metadata[0]["t"]),
    )
    return fit_life


def plot_comparison(path, fig_path):
    stats = sinter.read_stats_from_csv_files(path)
    stats_combined = combine_ZX_stats(stats)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True)
    fig.set_dpi(240)

    err_ref = [1e-8, 1]

    sinter._plotting.plot_custom(
        ax=ax1,
        stats=stats_combined,
        x_func=lambda stat: stat.json_metadata[0]["p"],
        y_func=y_func_error_rate,
        group_func=lambda stat: f"$d={stat.json_metadata[0]['d']}$",
        filter_func=lambda stat: stat.json_metadata[0]["mode"] == "all",
    )
    sinter._plotting.plot_custom(
        ax=ax2,
        stats=stats_combined,
        x_func=lambda stat: stat.json_metadata[0]["p"],
        y_func=y_func_error_rate,
        group_func=lambda stat: f"$d={stat.json_metadata[0]['d']}$",
        filter_func=lambda stat: stat.json_metadata[0]["mode"] == "fixed",
    )

    """
    sinter.plot_error_rate(
        ax = ax1,
        stats=stats_combined,
        x_func=lambda stat: stat.json_metadata[0]['p'],
        group_func=lambda stat: f"$d={stat.json_metadata[0]['d']}$",
        filter_func=lambda stat: stat.json_metadata[0]['mode'] == 'all',
        failure_units_per_shot_func=lambda stat: stat.json_metadata[0]['r'],
    )
    sinter.plot_error_rate(
        ax = ax2,
        stats=stats,
        x_func=lambda stat: stat.json_metadata[0]['p'],
        group_func=lambda stat: f"$d={stat.json_metadata[0]['d']}$",
        filter_func=lambda stat: stat.json_metadata[0]['mode'] == 'fixed',
        failure_units_per_shot_func=lambda stat: stat.json_metadata[0]['r'],
    )
    """

    fig.tight_layout()
    ax1.tick_params(axis="both", which="major", labelsize=15)
    ax2.tick_params(axis="both", which="major", labelsize=15)

    ax1.text(
        -0.1, 1.05, r"\textrm{(a)}", fontdict={"fontsize": 18}, transform=ax1.transAxes
    )
    ax2.text(
        -0.1, 1.05, r"\textrm{(b)}", fontdict={"fontsize": 18}, transform=ax2.transAxes
    )

    ax1.plot(err_ref, err_ref, "k--", label="$p_L=p$")
    ax1.loglog()
    ax1.set_ylim(1e-6, 1e-1)
    ax1.set_xlim(1e-7, 1e-2)
    ax1.grid()
    ax1.set_ylabel(r"$p_L$", fontsize=16)
    ax1.set_xlabel(r"$p$", fontsize=16)
    ax1.grid(visible=True, which="both")
    ax1.legend(fontsize=15)

    ax2.plot(err_ref, err_ref, "k--")
    ax2.loglog()
    ax2.set_ylim(1e-6, 1e-1)
    ax2.set_xlim(1e-7, 1e-2)
    ax2.grid()
    ax2.set_xlabel(r"$p$", fontsize=16)
    ax2.grid(visible=True, which="both")
    ax2.set_yticklabels([])

    fig.savefig(fig_path, bbox_inches="tight")


def plot_single_type_error(ax, path="./data/rotated_surface_code_one_type_err_ZX.csv"):
    stats = sinter.read_stats_from_csv_files(path)
    stats_combined = combine_ZX_stats(stats)

    sinter._plotting.plot_custom(
        ax=ax,
        stats=stats_combined,
        x_func=lambda stat: stat.json_metadata[0]["p"],
        y_func=y_func_error_rate,
        group_func=lambda stat: f'\\textrm{{{stat.json_metadata[0]["noise_type"]}}}',
    )

    error_list = [1e-8, 1]
    ax.text(
        -0.1, 1.05, r"\textrm{(a)}", fontdict={"fontsize": 18}, transform=ax.transAxes
    )

    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.plot(error_list, error_list, "k--", label=r"$p_L=p$")
    ax.loglog()
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlim(1e-6, 1e-2)
    # ax.set_ylim(1e-8, 1e-1)
    # ax.set_xlim(1e-6, 1e-2)
    ax.grid()
    ax.set_ylabel(r"$p_L$", fontsize=16)
    ax.set_xlabel(r"$p$", fontsize=16)
    ax.grid(visible=True, which="both")
    ax.legend(fontsize=15)


def plot_crosstalk_error(
    ax, fig, path="./data/rotated_surface_code_crosstalk_noise_ZX.csv"
):
    stats = sinter.read_stats_from_csv_files(path)
    combined_stats = combine_ZX_stats(stats)

    sinter._plotting.plot_custom(
        ax=ax,
        stats=combined_stats,
        x_func=lambda stat: stat.json_metadata[0]["p"],
        y_func=y_func_error_rate,
        group_func=lambda stat: {
            "ms_realistic": "\\textrm{Realistic MS gate}",
            "depolarize": "\\textrm{2 qubit depolarizing channel}",
        }[stat.json_metadata[0]["crosstalk_noise"]],
    )

    error_list = [1e-8, 1]
    # ax.text(-0.1, 1.05, r'\textrm{(a)}', fontdict={'fontsize':18}, transform=ax.transAxes)

    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.plot(error_list, error_list, "k--", label=r"$p_L=p$")
    ax.loglog()
    ax.set_ylim(1e-8, 1e-1)
    ax.set_xlim(1e-7, 1e-2)
    ax.grid()
    ax.set_ylabel(r"$p_L$", fontsize=16)
    ax.set_xlabel(r"$p$", fontsize=16)
    ax.grid(visible=True, which="both")
    fig.legend(fontsize=15, loc="outside lower center")


def plot_logical_coherence_full(ax):
    stats = sinter.read_stats_from_csv_files(
        "./data/rotated_surface_code_coherence_full_ZX.csv"
    )
    stats_combined = combine_ZX_stats(stats)
    print(len(stats_combined))

    ax.tick_params(axis="both", which="major", labelsize=15)

    cmap = mpl.colormaps["viridis"]
    cmap_low_noise = 0.2
    cmap_high_noise = 1.0

    log_noise_min = -7.0
    log_noise_max = -4.0

    get_color = lambda x: mpl.colors.to_hex(
        cmap(
            (np.log10(x) - log_noise_min)
            / (log_noise_max - log_noise_min)
            * (cmap_high_noise - cmap_low_noise)
            + cmap_low_noise
        )
    )

    def group_func(stat: sinter.TaskStats):
        if stat.json_metadata[0]["group"] == "serial":
            return {
                "label": r"\textrm{serial}",
                "sort": -2,
                "color": "r",
                "linestyle": "-",
                "marker": ".",
            }
        elif stat.json_metadata[0]["crosstalk"] == 0.0:
            return {
                "label": r"$p_c = 0$",
                "sort": -1,
                "color": mpl.colors.to_hex(cmap(0.0)),
                "linestyle": "-",
                "marker": ".",
            }
        return {
            "label": f'$p_c = 10^{{{np.log10(stat.json_metadata[0]["crosstalk"]):.0f}}}$',
            "sort": stat.json_metadata[0]["group"],
            "color": get_color(stat.json_metadata[0]["crosstalk"]),
            "linestyle": "--",
            "marker": ".",
        }

    def x_func(stat: sinter.TaskStats):
        return stat.json_metadata[0]["tc"]

    def y_func_break_even(stat: sinter.TaskStats):
        return stat.json_metadata[0]["tc"]

    def plot_args_func(index, group_key, group_stats):
        return {
            "color": group_key["color"],
            "marker": group_key["marker"],
            "linestyle": group_key["linestyle"],
        }

    def filter_func(stat: sinter.TaskStats):
        if stat.json_metadata[0]["group"] == "null":
            return False
        if stat.errors < 20:
            return False
        if stat.errors * 2 >= stat.shots:
            return False
        return True

    sinter._plotting.plot_custom(
        ax=ax,
        stats=stats_combined,
        x_func=x_func,
        y_func=y_func_break_even,
        group_func=lambda stat: {
            "label": r"\textrm{break even}",
            "sort": -3,
            "color": "k",
            "linestyle": ":",
            "marker": "",
        },
        filter_func=lambda stat: stat.json_metadata[0]["group"] == -1,
        plot_args_func=plot_args_func,
    )

    sinter._plotting.plot_custom(
        ax=ax,
        stats=stats_combined,
        x_func=x_func,
        y_func=y_func_lifetime,
        group_func=group_func,
        filter_func=filter_func,
        plot_args_func=plot_args_func,
    )

    ax.text(
        -0.1, 1.05, r"\textrm{(b)}", fontdict={"fontsize": 18}, transform=ax.transAxes
    )

    ax.loglog()
    ax.set_xlim(1e2, 1e5)
    ax.set_ylim(3e2, 1e7)
    ax.set_ylabel(r"$T_L$", fontsize=16)
    ax.set_xlabel(r"$T$", fontsize=16)
    ax.grid(visible=True, which="both")
    ax.legend(loc="upper left", fontsize=15)


def plot_logical_coherence_schedule(
    ax, path="./data/rotated_surface_code_coherence_schedule_ZX.csv"
):
    stats = sinter.read_stats_from_csv_files(path)
    stats_combined = combine_ZX_stats(stats)

    ax.tick_params(axis="both", which="major", labelsize=15)

    cmap = mpl.colormaps["plasma"]
    cmap_low_noise = 0.2
    cmap_high_noise = 1.0

    log_noise_min = -7.0
    log_noise_max = -4.0

    color_schedule = {
        "full": cmap(0.0),
        "half": cmap(0.25),
        "sqrt": cmap(0.5),
        "minimal": cmap(0.75),
        "serial": cmap(1.0),
    }

    sort_schedule = {
        "full": 1,
        "half": 2,
        "sqrt": 3,
        "minimal": 4,
        "serial": 5,
    }

    get_color = lambda x: color_schedule[x]
    get_sort = lambda x: sort_schedule[x]

    def group_func(stat: sinter.TaskStats):
        if stat.json_metadata[0]["group"] == "serial":
            return {
                "label": cnot_scheduling_name["serial"],
                "sort": get_sort(stat.json_metadata[0]["cnot_scheduling"]),
                "color": get_color(stat.json_metadata[0]["cnot_scheduling"]),
                "linestyle": "-",
                "marker": ".",
            }
        return {
            "label": cnot_scheduling_name[stat.json_metadata[0]["cnot_scheduling"]],
            "sort": get_sort(stat.json_metadata[0]["cnot_scheduling"]),
            "color": get_color(stat.json_metadata[0]["cnot_scheduling"]),
            "linestyle": "--",
            "marker": ".",
        }

    def x_func(stat: sinter.TaskStats):
        return stat.json_metadata[0]["tc"]

    def y_func_break_even(stat: sinter.TaskStats):
        return stat.json_metadata[0]["tc"]

    def plot_args_func(index, group_key, group_stats):
        return {
            "color": group_key["color"],
            "marker": group_key["marker"],
            "linestyle": group_key["linestyle"],
        }

    def filter_func(stat: sinter.TaskStats):
        if stat.json_metadata[0]["group"] == "null":
            return False
        if stat.json_metadata[0]["group"] == -1:
            return False
        if stat.errors < 20:
            return False
        if stat.errors * 2 >= stat.shots:
            return False
        return True

    sinter._plotting.plot_custom(
        ax=ax,
        stats=stats_combined,
        x_func=x_func,
        y_func=y_func_break_even,
        group_func=lambda stat: {
            "label": r"\textrm{break even}",
            "sort": -3,
            "color": "k",
            "linestyle": ":",
            "marker": "",
        },
        filter_func=lambda stat: stat.json_metadata[0]["group"] == -1,
        plot_args_func=plot_args_func,
    )

    sinter._plotting.plot_custom(
        ax=ax,
        stats=stats_combined,
        x_func=x_func,
        y_func=y_func_lifetime,
        group_func=group_func,
        filter_func=filter_func,
        plot_args_func=plot_args_func,
    )

    ax.text(
        -0.1, 1.05, r"\textrm{(a)}", fontdict={"fontsize": 18}, transform=ax.transAxes
    )

    ax.loglog()
    ax.set_xlim(1e2, 1e5)
    ax.set_ylim(3e2, 1e7)
    ax.set_ylabel(r"$T_L$", fontsize=16)
    ax.set_xlabel(r"$T$", fontsize=16)
    ax.grid(visible=True, which="both")
    ax.legend(loc="upper left", fontsize=15)


def plot_logical_coherence_crosstalk(
    ax, path="./data/rotated_surface_code_coherence_crosstalk_ZX.csv"
):
    stats = sinter.read_stats_from_csv_files(path)
    stats_combined = combine_ZX_stats(stats)

    ax.tick_params(axis="both", which="major", labelsize=15)

    cmap = mpl.colormaps["plasma"]
    cmap_low_noise = 0.2
    cmap_high_noise = 1.0

    log_noise_min = -7.0
    log_noise_max = -4.0

    color_schedule = {
        "full": cmap(0.0),
        "half": cmap(0.25),
        "sqrt": cmap(0.5),
        "minimal": cmap(0.75),
        "serial": cmap(1.0),
    }

    sort_schedule = {
        "full": 1,
        "half": 2,
        "sqrt": 3,
        "minimal": 4,
        "serial": 5,
    }

    get_color = lambda x: color_schedule[x]
    get_sort = lambda x: sort_schedule[x]

    def group_func(stat: sinter.TaskStats):
        if stat.json_metadata[0]["group"] == "serial":
            return {
                "label": cnot_scheduling_name["serial"],
                "sort": get_sort(stat.json_metadata[0]["cnot_scheduling"]),
                "color": get_color(stat.json_metadata[0]["cnot_scheduling"]),
                "linestyle": "-",
                "marker": ".",
            }
        return {
            "label": cnot_scheduling_name[stat.json_metadata[0]["cnot_scheduling"]],
            "sort": get_sort(stat.json_metadata[0]["cnot_scheduling"]),
            "color": get_color(stat.json_metadata[0]["cnot_scheduling"]),
            "linestyle": "--",
            "marker": ".",
        }

    def x_func(stat: sinter.TaskStats):
        return stat.json_metadata[0]["crosstalk"]

    def plot_args_func(index, group_key, group_stats):
        return {
            "color": group_key["color"],
            "marker": group_key["marker"],
            "linestyle": group_key["linestyle"],
        }

    def filter_func(stat: sinter.TaskStats):
        if stat.json_metadata[0]["group"] == "null":
            return False
        if stat.json_metadata[0]["group"] == -1:
            return False
        if stat.json_metadata[0]["group"] == "serial":
            return False
        if stat.errors < 20:
            return False
        if stat.errors * 2 >= stat.shots:
            return False
        return True

    tc, t_serial = 0.0, 0.0
    for stat in stats_combined:
        if stat.json_metadata[0]["group"] == 0:
            tc = stat.json_metadata[0]["tc"]
            break
    for stat in stats_combined:
        if stat.json_metadata[0]["group"] == "serial":
            t_serial = get_estimated_lifetime(
                stat.errors / stat.shots, stat.json_metadata[0]["t"]
            )
            break

    ax.hlines(
        y=tc,
        xmin=1e-7,
        xmax=1e-3,
        label=r"\textrm{break even}",
        colors="k",
        linestyles=":",
    )

    sinter._plotting.plot_custom(
        ax=ax,
        stats=stats_combined,
        x_func=x_func,
        y_func=y_func_lifetime,
        group_func=group_func,
        filter_func=filter_func,
        plot_args_func=plot_args_func,
    )

    ax.hlines(
        y=t_serial,
        xmin=1e-7,
        xmax=1e-3,
        label=cnot_scheduling_name["serial"],
        colors=get_color("serial"),
        linestyles="-",
    )

    ax.text(
        -0.1, 1.05, r"\textrm{(b)}", fontdict={"fontsize": 18}, transform=ax.transAxes
    )

    ax.loglog()
    ax.set_xlim(1e-7, 1e-3)
    ax.set_ylim(3e2, 3e5)
    ax.set_ylabel(r"$T_L$", fontsize=16)
    ax.set_xlabel(r"$p_c$", fontsize=16)
    ax.grid(visible=True, which="both")
    # ax.legend(loc='lower left', fontsize=15)


# plot_comparison(path="results/rotated_sim.csv", fig_path="results/rotated_sim.png")
# plot_comparison(path="results/toric_sim.csv", fig_path="results/toric_sim.png")
# plot_comparison(path="results/surface_sim.csv", fig_path="results/surface_sim.png")
plot_comparison(
    path="results/bb_sim_no_pc_108_8_10_50000_100.csv",
    fig_path="results/bb_sim_no_pc_108_8_10_50000_100.png",
)


"""fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
plot_single_type_error(ax1)
plot_logical_coherence_full(ax2)
fig.savefig("./figures/rotated_surface_code/fig_full.pdf", bbox_inches="tight")


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
plot_logical_coherence_schedule(ax1)
plot_logical_coherence_crosstalk(ax2)
fig.savefig("./figures/rotated_surface_code/fig_partial.pdf", bbox_inches="tight")


# padded time
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
plot_logical_coherence_schedule(
    ax1, path="./data/rotated_surface_code_coherence_schedule_padded2.csv"
)
plot_logical_coherence_crosstalk(
    ax2, path="./data/rotated_surface_code_coherence_crosstalk_padded2.csv"
)


# plot crosstalk noise

fig, ax = plt.subplots(nrows=1, ncols=1, layout="constrained", figsize=(5, 5))
plot_crosstalk_error(ax, fig, path="./data/rotated_surface_code_crosstalk_noise2.csv")
fig.savefig(
    "./figures/rotated_surface_code/fig_crosstalk_noise2.pdf", bbox_inches="tight"
)"""
