"""The [110] phase diagram from the scan: the ground state at each strain (space group, polar direction,
tilt axis, polar energy gain), beside the same model's [001] results from the study.

Ground state, as in the study: the lowest run; if it is polar (uniform Ti off-centring > 0.005 A and
more than 0.1 meV/f.u. below every centrosymmetric state), the most symmetric polar run within
0.1 meV/f.u. of it. Unlike [001], centrosymmetric states found by the free (polar-start) runs count as
references too: here a polar start can end in a lower tilt pattern without polarising.

Writes results_110.csv and results_110.md here, and the figure to figures/fig_phases_110.png (and .pdf).

    ~/.sto-envs/mace/bin/python orientations/110/analyse_110.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from ase.io import read

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sto110 as S  # noqa: E402

SYMPREC, SYMPREC_LOOSE = 5e-3, 1e-2  # A, as in the study's figures
AXES = ("001", "1-10", "110")  # film axes: in plane, in plane, normal
LABEL = {"001": "[001] (in plane)", "1-10": "[1-10] (in plane)", "110": "[110] (normal)"}
COLOUR = {"001": "#2a78d6", "1-10": "#eb6834", "110": "#1baf7a"}  # validated categorical slots 1-3
MARKER = {"001": "o", "1-10": "s", "110": "^"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIRECTIONS = {"[001]": (1, 0, 0), "[1-10]": (0, 1, 0), "[110]": (0, 0, 1),  # (|d_001|, |d_1-10|, |d_110|)
              "[1-11]": (3 ** -0.5, (2 / 3) ** 0.5, 0), "[111]": (3 ** -0.5, 0, (2 / 3) ** 0.5),
              "[100]": (0, 0.5 ** 0.5, 0.5 ** 0.5)}


def direction_name(d) -> str:
    """Nearest low-index direction to a polar vector given along the film axes, within 3 degrees."""
    v = np.abs(np.asarray(d, float))
    if np.linalg.norm(v) < S.config.D_TOL:
        return "-"
    v /= np.linalg.norm(v)
    best = max(DIRECTIONS, key=lambda k: v @ np.array(DIRECTIONS[k]))
    ang = np.degrees(np.arccos(np.clip(v @ np.array(DIRECTIONS[best]), -1, 1)))
    return best if ang < 3 else f"{best} +{ang:.0f} deg"


def tilt_axis_name(rec) -> str:
    t = np.array([rec[f"tilt_about_{k}"] for k in AXES])
    if np.linalg.norm(t) < 0.3:
        return "none"
    return direction_name(t)


def ground_states(outdir: Path) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(p.read_text()) for p in sorted(outdir.glob("*.json"))])
    rows = []
    for s, g in df.groupby("strain_pct"):
        tag = S.config.strain_tag(s)
        sym = {seed: S.symmetry(read(outdir / f"{tag}_{seed}.extxyz"), SYMPREC) for seed in g.seed}
        g = g.assign(polar=g.ti_disp_polar > S.config.D_TOL, nops=[sym[x][1] for x in g.seed],
                     sg=[sym[x][0] for x in g.seed]).sort_values(["e_per_fu", "seed"])
        npl, pol = g[~g.polar], g[g.polar]
        gain = 1000 * (npl.e_per_fu.min() - pol.e_per_fu.min()) if len(pol) else 0.0
        if len(pol) and gain > S.config.E_TOL_MEV:
            pool = pol[1000 * (pol.e_per_fu - pol.e_per_fu.min()) <= S.config.E_TOL_MEV]
        else:
            pool = npl[1000 * (npl.e_per_fu - npl.e_per_fu.min()) <= S.config.E_TOL_MEV]
        gr = pool.iloc[int(np.argmax(pool.nops.to_numpy()))]  # pool is sorted by energy: lowest of the most symmetric
        cs_pool = npl[1000 * (npl.e_per_fu - npl.e_per_fu.min()) <= S.config.E_TOL_MEV]
        cs = cs_pool.iloc[int(np.argmax(cs_pool.nops.to_numpy()))]
        a = read(outdir / f"{tag}_{gr.seed}.extxyz")
        rows.append({
            "strain_pct": s, "a_inplane": gr.a_inplane, "space_group": gr.sg,
            "space_group_loose": S.symmetry(a, SYMPREC_LOOSE)[0], "polar": bool(gr.polar),
            "gain_mev_per_fu": max(gain, 0.0), "polar_direction": direction_name([gr[f"d_{k}"] for k in AXES]),
            "ti_disp": gr.ti_disp_polar, **{f"ti_{k}": abs(gr[f"d_{k}"]) for k in AXES},
            "tilt_axis": tilt_axis_name(gr), **{f"tilt_{k}": gr[f"tilt_about_{k}"] for k in AXES},
            "c_over_a": gr.c_over_a, "ground_seed": gr.seed,
            "centrosym_space_group": cs.sg, "centrosym_tilt_axis": tilt_axis_name(cs), "centrosym_seed": cs.seed,
            "n_runs": len(g), "all_converged": bool(g.converged.all())})
    return pd.DataFrame(rows)


def study_001() -> pd.DataFrame:
    df = pd.read_csv(S.REPO / "results" / "summary_points.csv", dtype={"orientation": str})
    df["orientation"] = df.orientation.map(S.config.orientation_str)
    return df[(df.model == S.MODEL) & (df.orientation == "001")].sort_values("strain_pct")


def markdown(t: pd.DataFrame) -> str:
    head = ("| strain (%) | a (A) | ground state | polar direction | gain (meV/f.u.) | Ti [001] / [1-10] / [110] (A) "
            "| tilt axis | tilt about [001] / [1-10] / [110] (deg) | c/a | lowest centrosymmetric |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in t.to_dict("records"):
        sg = r["space_group"] + (f" ({r['space_group_loose']} at 0.01 A)" if r["space_group_loose"] != r["space_group"] else "")
        ti = " / ".join(f"{r[f'ti_{k}']:.3f}" for k in AXES)
        tl = " / ".join(f"{r[f'tilt_{k}']:.1f}" for k in AXES)
        lines.append(f"| {r['strain_pct']:+.2f} | {r['a_inplane']:.3f} | {sg} | {r['polar_direction']} "
                     f"| {r['gain_mev_per_fu']:.2f} | {ti} | {r['tilt_axis']} | {tl} | {r['c_over_a']:.3f} "
                     f"| {r['centrosym_space_group']}, tilt {r['centrosym_tilt_axis']} |")
    return head + "\n".join(lines) + "\n"


def figure(t: pd.DataFrame, d001: pd.DataFrame, path: Path):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.6, 8.8), dpi=160, facecolor=SURFACE)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.42, 2.1, 1.7, 1.7], hspace=0.12, top=0.95, bottom=0.1)
    top, ax_g, ax_p, ax_t = (fig.add_subplot(gs[i]) for i in range(4))
    for ax in (ax_p, ax_t):
        ax.sharex(ax_g)
    x = t.strain_pct.to_numpy()

    top.set_xlim(x.min() - 0.7, x.max() + 0.7)  # phase labels, one row per orientation
    top.axis("off")
    for row, (name, xs, sgs) in enumerate((("[110]", x, t.space_group), ("[001]", d001.strain_pct, d001.space_group))):
        yy = 0.72 - 0.55 * row
        top.text(x.min() - 0.75, yy, name, ha="right", va="center", fontsize=8.5, color=INK, fontweight="bold")
        for xi, sg in zip(xs, sgs):
            top.text(xi, yy, sg, ha="center", va="center", fontsize=7.5, color=INK if row == 0 else INK2)
    top.set_ylim(0, 1)

    ax_g.plot(d001.strain_pct, d001.gain_mev_per_fu, "--", color=INK2, lw=1.4, marker="o", mfc=SURFACE, ms=5,
              label="[001] (same model, from the study)")
    ax_g.plot(x, t.gain_mev_per_fu, "-", color=INK, lw=2, marker="o", ms=5.5, label="[110] (this scan)")
    ax_g.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax_g.axhline(S.config.E_QF_MEV, color=INK2, lw=0.8, ls=":")
    ax_g.text(x.max() + 0.6, S.config.E_QF_MEV * 1.15, "2 meV/f.u.", ha="right", va="bottom", fontsize=7, color=INK2)
    ax_g.set_ylabel("polar energy gain\n(meV per f.u., symlog)")
    ax_g.legend(frameon=False, fontsize=8, loc="upper center")

    for ax, key, ylabel, top_ in ((ax_p, "ti", "Ti off-centring (Å)", 1.25), (ax_t, "tilt", "octahedral tilt (deg)", 1.3)):
        ends = {}
        for k in AXES:
            y = t[f"{key}_{k}"].to_numpy()
            ax.plot(x, y, color=COLOUR[k], lw=2, marker=MARKER[k], ms=5, mec=SURFACE, mew=0.8, label=LABEL[k])
            ends[k] = y[-1]
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", title="about the film axis" if key == "tilt"
                  else "along the film axis", title_fontsize=7.5)
        ax.set_ylim(0, max(0.05, ax.get_ylim()[1]) * top_)
        span, placed = ax.get_ylim()[1], []  # direct labels at the line ends, kept clear of each other and of 0
        for k, y in sorted(ends.items(), key=lambda kv: kv[1]):
            yy = max(y, 0.035 * span, (placed[-1] + 0.07 * span) if placed else 0)
            placed.append(yy)
            ax.text(x.max() + 0.12, yy, k.join("[]"), color=INK2, fontsize=7.5, va="center")
    ax_t.set_xlabel("in-plane strain from the model's cubic a0 (%)")
    for ax in (ax_g, ax_p, ax_t):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(["0" if abs(v) < 0.05 else f"{v:+.1f}" for v in x])
    for ax in (ax_g, ax_p):
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.suptitle("SrTiO$_3$ under [110] epitaxial strain at 0 K: MACE MH-1, r2SCAN head", fontsize=10.5, y=0.985)
    fig.text(0.5, 0.03, "Ground state at each strain from 8 centrosymmetric references, 12 polar starts and a phonon "
             "check; space groups at 0.005 Å. [001] from the published study.", ha="center", fontsize=7, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), facecolor=SURFACE, bbox_inches="tight")


def main():
    outdir = S.HERE / (sys.argv[1] if len(sys.argv) > 1 else "results")
    t = ground_states(outdir)
    d001 = study_001()
    t.to_csv(S.HERE / "results_110.csv", index=False)
    (S.HERE / "results_110.md").write_text(markdown(t))
    figure(t, d001, S.REPO / "figures" / "fig_phases_110.png")
    print(markdown(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
