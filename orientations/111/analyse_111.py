"""[111] phase diagram: the model (MACE MH-1, r2SCAN head) against the PBEsol DFT, both described with
the same cell-independent measures (film_measure.py).

Ground state on each side, as in the study: the lowest run; if it is polar (uniform Ti off-centring
> 0.005 A and more than 0.1 meV/f.u. below every centrosymmetric state), the most symmetric polar run
within 0.1 meV/f.u. of it. Gain = lowest centrosymmetric - lowest polar. Strain is from each side's own
cubic a0 (model 3.918 A, DFT 3.8997 A).

Writes results_111.csv, states_111.csv (each DFT state against the model) and results_111.md here, and the
figure to figures/fig_phases_111.png (and .pdf).

    ~/.sto-envs/mace/bin/python orientations/111/analyse_111.py
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
import sto111 as S  # noqa: E402
import film_measure as F  # noqa: E402

SYMPREC = 5e-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOUR = {"inplane": "#2a78d6", "normal": "#1baf7a"}
MARKER = {"inplane": "o", "normal": "^"}
LABEL = {"inplane": "in the plane", "normal": "along the normal [111]"}
KEEP = ("ti_disp", "ti_normal", "ti_inplane", "ti_inplane_angle", "tilt_total", "tilt_normal", "tilt_inplane",
        "tilt_inplane_angle", "phases", "c_over_a", "a_inplane")


def rows_model() -> pd.DataFrame:
    out = []
    grid = {json.loads(p.read_text())["strain_pct"] for p in S.RESULTS.glob("*_np_untilted.json")}
    for p in sorted(S.RESULTS.glob("*.json")):
        r = json.loads(p.read_text())
        if r["strain_pct"] not in grid:
            continue
        a = read(str(p)[:-5] + ".extxyz")
        d = F.describe(a)
        sg, nops, _ = S.symmetry(a, SYMPREC)
        out.append({"strain_pct": r["strain_pct"], "run": r["seed"], "e_per_fu": r["energy"] / d["n_fu"],
                    "sg": sg, "nops": nops, "converged": bool(r["converged"]), "dft_path": r.get("dft_path"),
                    **{k: d[k] for k in KEEP}})
    return pd.DataFrame(out)


def rows_dft() -> pd.DataFrame:
    out = []
    for a in read(S.HERE / "dft" / "finals.extxyz", ":"):
        if "path" not in a.info:  # the two bulk 'Relaxed' runs (no IOPTCELL, so their fields shifted)
            continue
        lab = str(a.info["path"]).split("/")[1]  # the folder name; extxyz reads the label 3.9_2 back as 3.92
        d = F.describe(a)
        sg, nops, _ = S.symmetry(a, SYMPREC)
        a_in = float(lab.split("_")[0])
        out.append({"strain_pct": round((a_in / S.config.DFT_A0_CUBIC - 1) * 100, 3), "run": f"{a.info['phase']}/{lab}",
                    "e_per_fu": a.info["e0"] / d["n_fu"], "sg": sg, "nops": nops,
                    "converged": bool(a.info["converged"]), **{k: d[k] for k in KEEP}})
    return pd.DataFrame(out)


def ground_states(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s, g in df.groupby("strain_pct"):
        g = g.assign(polar=g.ti_disp > S.config.D_TOL).sort_values(["e_per_fu", "run"])
        npl, pol = g[~g.polar], g[g.polar]
        gain = 1000 * (npl.e_per_fu.min() - pol.e_per_fu.min()) if len(pol) and len(npl) else 0.0
        src = pol if len(pol) and (gain > S.config.E_TOL_MEV or npl.empty) else npl
        pool = src[1000 * (src.e_per_fu - src.e_per_fu.min()) <= S.config.E_TOL_MEV]
        gr = pool.iloc[int(np.argmax(pool.nops.to_numpy()))]
        out.append({"strain_pct": s, "space_group": gr.sg, "gain": max(gain, 0.0), "run": gr.run,
                    "converged": bool(gr.converged), "n_runs": len(g), **{k: gr[k] for k in KEEP}})
    return pd.DataFrame(out)


def character(r) -> tuple:
    """Which distortions a state has: Ti off-centring along the normal and in the plane (> 0.02 A), and tilt
    about the normal and about an in-plane axis (> 1 deg)."""
    return (r["ti_normal"] > 0.02, r["ti_inplane"] > 0.02, r["tilt_normal"] > 1.0, r["tilt_inplane"] > 1.0)


def states(rm: pd.DataFrame, rd: pd.DataFrame) -> pd.DataFrame:
    """Each DFT state beside the same structure relaxed by the model (dft_seeds_111.py), each side
    measured from its own lowest energy at that strain. 'same_state' says whether the model kept the state's
    distortions. The DFT runs kept their symmetry and the model's relaxations had none, so a state that is a
    saddle in the model falls into another one there."""
    out = []
    for s, g in rm[rm.dft_path.notna()].groupby("strain_pct"):
        m0, ds = rm[rm.strain_pct == s].e_per_fu.min(), rd[rd.strain_pct == s]
        for _, r in g.iterrows():
            x = ds[ds.run == r.dft_path].iloc[0]
            out.append({"strain_pct": s, "state": x.run, "dft_converged": bool(x.converged),
                        "dft_mev": 1000 * (x.e_per_fu - ds.e_per_fu.min()), "model_mev": 1000 * (r.e_per_fu - m0),
                        "same_state": character(x) == character(r), "dft_ti": x.ti_disp, "model_ti": r.ti_disp,
                        "dft_tilt": x.tilt_total, "model_tilt": r.tilt_total})
    return pd.DataFrame(out).sort_values(["strain_pct", "dft_mev"]).reset_index(drop=True)


def states_markdown(t: pd.DataFrame) -> str:
    head = ("\n## Each DFT state against the model\n\nEnergy above the lowest state at that strain on each side "
            "(meV/f.u.). The model relaxed each DFT final structure without symmetry; 'same' means it kept "
            "the state's distortions (Ti off-centring > 0.02 A along the normal or in the plane, tilt > 1 deg about "
            "the normal or an in-plane axis).\n\n| strain (%) | DFT run | converged | DFT | model | same | "
            "Ti DFT / model (A) | tilt DFT / model (deg) |\n|---|---|---|---|---|---|---|---|\n")
    lines = [f"| {r.strain_pct:+.2f} | {r.state} | {'yes' if r.dft_converged else 'no'} | {r.dft_mev:.2f} | "
             f"{r.model_mev:.2f} | {'yes' if r.same_state else 'no'} | {r.dft_ti:.3f} / {r.model_ti:.3f} | "
             f"{r.dft_tilt:.1f} / {r.model_tilt:.1f} |" for r in t.itertuples()]
    return head + "\n".join(lines) + "\n"


def markdown(m: pd.DataFrame, d: pd.DataFrame) -> str:
    head = ("| strain (%) | | ground state | gain (meV/f.u.) | Ti normal / in-plane (A) | tilt normal / in-plane (deg) "
            "| in-plane tilt axis | c/a |\n|---|---|---|---|---|---|---|---|\n")

    def axis(r):
        if r["tilt_inplane"] < 0.3:
            return "-"
        return "[1-10]-type" if r["tilt_inplane_angle"] < 10 else "[11-2]-type" if r["tilt_inplane_angle"] > 20 else f"{r['tilt_inplane_angle']:.0f} deg"

    lines = []
    for s in sorted(set(m.strain_pct) | set(d.strain_pct)):
        for name, t in (("DFT", d), ("model", m)):
            sel = t[t.strain_pct == s]
            if sel.empty:
                continue
            r = sel.iloc[0]
            lines.append(f"| {s:+.2f} | {name} | {r['space_group']} | {r['gain']:.2f} | {r['ti_normal']:.3f} / {r['ti_inplane']:.3f} "
                         f"| {r['tilt_normal']:.1f} / {r['tilt_inplane']:.1f} | {axis(r)} | {r['c_over_a']:.3f} |")
    return head + "\n".join(lines) + "\n"


def figure(m: pd.DataFrame, d: pd.DataFrame, path: Path):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.6, 8.8), dpi=160, facecolor=SURFACE)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.42, 2.1, 1.7, 1.7], hspace=0.12, top=0.95, bottom=0.1)
    top, ax_g, ax_p, ax_t = (fig.add_subplot(gs[i]) for i in range(4))
    for ax in (ax_p, ax_t):
        ax.sharex(ax_g)
    x = m.strain_pct.to_numpy()
    top.set_xlim(x.min() - 0.7, x.max() + 0.7)
    top.set_ylim(0, 1)
    top.axis("off")
    for row, (name, t) in enumerate((("DFT", d), ("model", m))):
        yy = 0.72 - 0.55 * row
        top.text(x.min() - 0.75, yy, name, ha="right", va="center", fontsize=8.5, color=INK, fontweight="bold")
        for xi, sg in zip(t.strain_pct, t.space_group):
            top.text(xi, yy, sg, ha="center", va="center", fontsize=7.5, color=INK2 if row == 0 else INK)

    ax_g.plot(d.strain_pct, d.gain, "--", color=INK2, lw=1.4, marker="o", mfc=SURFACE, ms=5, label="DFT (PBEsol)")
    ax_g.plot(x, m.gain, "-", color=INK, lw=2, marker="o", ms=5.5, label="model (MH-1, r2SCAN head)")
    ax_g.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax_g.axhline(S.config.E_QF_MEV, color=INK2, lw=0.8, ls=":")
    ax_g.text(x.max() + 0.6, S.config.E_QF_MEV * 1.15, "2 meV/f.u.", ha="right", va="bottom", fontsize=7, color=INK2)
    ax_g.set_ylabel("polar energy gain\n(meV per f.u., symlog)")
    ax_g.set_ylim(bottom=-0.15)
    ax_g.legend(frameon=False, fontsize=8, loc="upper center")

    for ax, key, ylabel in ((ax_p, "ti", "Ti off-centring (Å)"), (ax_t, "tilt", "octahedral tilt (deg)")):
        for part in ("inplane", "normal"):
            ax.plot(d.strain_pct, d[f"{key}_{part}"], "--", color=COLOUR[part], lw=1.3, marker=MARKER[part], mfc=SURFACE, ms=5)
            ax.plot(x, m[f"{key}_{part}"], "-", color=COLOUR[part], lw=2, marker=MARKER[part], ms=5, mec=SURFACE, mew=0.8,
                    label=LABEL[part])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(0.05, ax.get_ylim()[1]) * 1.3)
        ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center",
                  title="solid: model    dashed, open: DFT", title_fontsize=7.5)
    ax_t.set_xlabel("in-plane strain from each side's own cubic a0 (%)")
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
    fig.suptitle("SrTiO$_3$ under [111] epitaxial strain at 0 K: model against DFT", fontsize=10.5, y=0.985)
    fig.text(0.5, 0.045, "Model: 7 centrosymmetric references, 12 polar starts and the relaxed DFT structures per strain, "
             "with a phonon check. Space groups at 0.005 Å.\nUnder compressive strain C2/m tilts about an in-plane [1-10]-type "
             "axis and C2/c about an [11-2]-type axis (every DFT run); in the model they differ by 0.01–0.35 meV/f.u.",
             ha="center", va="top", fontsize=7, color=INK2, linespacing=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), facecolor=SURFACE, bbox_inches="tight")


def main():
    rm, rd = rows_model(), rows_dft()
    m, d = ground_states(rm), ground_states(rd)
    out = pd.concat([m.assign(source="model"), d.assign(source="DFT")])
    out.to_csv(S.HERE / "results_111.csv", index=False)
    st = states(rm, rd)
    st.to_csv(S.HERE / "states_111.csv", index=False)
    text = markdown(m, d) + states_markdown(st)
    (S.HERE / "results_111.md").write_text(text)
    figure(m, d, S.REPO / "figures" / "fig_phases_111.png")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
