"""Step 6: tables and figures from results/ (needs only numpy, pandas, matplotlib).

    ~/.sto-envs/mace/bin/python scripts/06_figures.py [--strain-ref own]

Figures (PNG + PDF in figures/):
  fig0_summary_<o>.*       the headline: polar energy lowering of unstrained STO, one bar per model, DFT as a line
  fig1_gain_<o>.*          polar energy gain vs strain, one panel per model, DFT dashed; log axis above
                           0.1 meV/f.u., with lines at the polar threshold and the quantum-fluctuation scale
  fig2_displacement_<o>.*  uniform Ti displacement of the ground state vs strain
  fig3_tilt_<o>.*          octahedral tilt angle of the ground state vs strain
  fig4_phases_<o>.*        space group of the ground state at each strain, DFT and every model
  fig5_phonons_001.*       soft polar modes against DFPT (if step 5 ran)
  fig6_finetune.*          fine-tuned on [001], tested on [111] (if notebook results exist)
Tables (results/):
  summary_points.csv       one row per model x orientation x strain (incl. ground-state space group)
  summary_critical.csv/md  critical strains at the polar threshold (0.1 meV/f.u.) and at 2 meV/f.u.
  summary_phonons.csv/md   lowest polar / tilt mode per model and strain (if step 5 ran)

Ground state: the lowest polar run if it beats the lowest nonpolar reference by more than 0.1 meV/f.u.
and its Ti moves more than 0.005 A, else that reference. Polar runs within 0.1 meV/f.u. of the lowest
are one state at this precision, so the most symmetric of them stands for it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from sto import analysis as an  # noqa: E402
from sto import config, models  # noqa: E402

# Palette: reference categorical slots 1-2 (validated for CVD), neutral inks for everything else.
C = {"001": "#2a78d6", "111": "#eb6834", "model": "#2a78d6", "dft": "#52514e",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
     "surface": "#fcfcfb"}
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["axis"], "axes.linewidth": 0.8, "axes.labelcolor": C["ink2"],
    "axes.titlecolor": C["ink"], "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.grid": True,
    "grid.color": C["grid"], "grid.linewidth": 0.6, "xtick.color": C["muted"], "ytick.color": C["muted"],
    "xtick.labelcolor": C["ink2"], "ytick.labelcolor": C["ink2"], "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
    "text.color": C["ink"], "font.size": 9, "lines.linewidth": 1.6, "lines.markersize": 4.5,
})
ORIENT_LABEL = {"001": "[001]", "111": "[111]"}
SYMPREC = 5e-3  # A, space-group tolerance, the same as config.D_TOL (the polar threshold)
SYMPREC_LOOSE = 1e-2  # A, a label that changes by this tolerance is marked in the phase figure


def symmetry(path, symprec=SYMPREC):
    """(Hermann-Mauguin symbol, number of symmetry operations in the cell) of a structure file;
    ('?', 0) if spglib/ASE are missing or it fails."""
    try:
        import spglib
        from ase.io import read
        a = read(str(path), format="vasp" if Path(path).name in ("CONTCAR", "POSCAR") else None)
        d = spglib.get_symmetry_dataset((a.cell, a.get_scaled_positions(), a.numbers), symprec=symprec)
        return d.international, len(d.rotations)
    except Exception:
        return "?", 0


def space_group(path, symprec=SYMPREC) -> str:
    return symmetry(path, symprec)[0]


# ------------------------------------------------------------------ loading
def load_points(ref: str) -> pd.DataFrame:
    recs = [json.loads(p.read_text()) for p in sorted((config.RESULTS / "relax" / ref).glob("*/*/*.json"))]
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    rows = []
    for (m, o, s), g in df.groupby(["model", "orientation", "strain_pct"]):
        is_np = g.seed.str.startswith("nonpolar")
        npl = g[is_np]
        if npl.empty:
            continue
        npl = npl.loc[npl.e_per_fu.idxmin()]  # lowest centrosymmetric reference (Pm-3m, I4/mcm, Imma or C2/c)
        polar = g[~is_np]
        best = polar.loc[polar.e_per_fu.idxmin()] if not polar.empty else None
        gain = an.polar_energy_gain(npl.e_per_fu, best.e_per_fu, 1) if best is not None else np.nan
        o_str = config.orientation_str(o)
        path = lambda seed: config.RESULTS / "relax" / ref / m / o_str / f"{config.strain_tag(s)}_{seed}.extxyz"  # noqa: E731
        ground = npl
        if best is not None and gain > config.E_TOL_MEV and best.ti_disp_polar > config.D_TOL:
            near = polar[(1000 * (polar.e_per_fu - best.e_per_fu) <= config.E_TOL_MEV)
                         & (polar.ti_disp_polar > config.D_TOL)].sort_values(["e_per_fu", "seed"])
            ground = near.iloc[int(np.argmax([symmetry(path(seed))[1] for seed in near.seed]))]
        sg, sg_loose = space_group(path(ground.seed)), space_group(path(ground.seed), SYMPREC_LOOSE)
        rows.append({"model": m, "orientation": o_str, "strain_pct": s, "space_group": sg, "space_group_loose": sg_loose,
                     "functional": npl.functional, "dataset": npl.dataset, "a0_model": npl.a0_model,
                     "gain_mev_per_fu": gain, "ti_disp": ground.ti_disp_polar,
                     "ti_disp_best_polar": best.ti_disp_polar if best is not None else np.nan,
                     "tilt": ground.tilt_all, "tilt_nonpolar": npl.tilt_all, "c_over_a": ground.c_over_a,
                     "ground_seed": ground.seed, "nonpolar_seed": npl.seed, "all_converged": bool(g.converged.all()), "n_runs": len(g)})
    return pd.DataFrame(rows).sort_values(["model", "orientation", "strain_pct"])


def load_dft() -> pd.DataFrame:
    """DFT polar gain and displacement per strain, where both a polar and a nonpolar final run exist."""
    if not config.DFT_FINAL_CSV.exists():
        return pd.DataFrame()
    d = pd.read_csv(config.DFT_FINAL_CSV, dtype={"orientation": str})
    d = d.dropna(subset=["strain_percent", "e_per_fu"])
    d["orientation"] = d["orientation"].map(config.orientation_str)
    if "run_type" in d:
        d = d[d.run_type == "relax"]  # DFPT and Berry-phase runs repeat their parent's structure
    rows = []
    for (o, s), g in d.groupby(["orientation", "strain_percent"]):
        npl, pol = g[g.phase_guess == "nonpolar"], g[g.phase_guess == "polar"]
        untilted_ref = False
        if len(npl) and len(pol):
            ref = npl.loc[npl.e_per_fu.idxmin()]
            gain = 1000 * (ref.e_per_fu - pol.e_per_fu.min())
            # an untilted reference (no tilted nonpolar run at this strain) also counts the tilt energy
            untilted_ref = bool(ref.tilt_all < 0.5 and pol.loc[pol.e_per_fu.idxmin()].tilt_all > 0.5)
        elif len(npl):
            gain = 0.0  # every DFT structure at this strain relaxed nonpolar
        else:
            gain = np.nan  # only polar runs: no reference energy
        low = g.loc[g.e_per_fu.idxmin()]
        contcar = Path(config.ARCHIVE) / low.path / "CONTCAR"  # dft_final.csv paths are inside the archive
        rows.append({"orientation": o, "strain_pct": s, "gain_mev_per_fu": gain, "ti_disp": low.ti_disp_polar,
                     "tilt": low.tilt_all, "c_over_a": low.get("c_over_a", np.nan),
                     "space_group": space_group(contcar), "space_group_loose": space_group(contcar, SYMPREC_LOOSE),
                     "untilted_ref": untilted_ref})
    return pd.DataFrame(rows)


def model_order(names):
    reg = [m.name for m in models.all_models()]
    func = {m.name: m.functional for m in models.all_models()}
    known = [n for n in reg if n in names]
    known.sort(key=lambda n: (func.get(n, "zz") != "PBE", reg.index(n)))
    def natural(n):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)]
    return known + sorted((n for n in names if n not in reg), key=natural)


# ------------------------------------------------------------------ figures
def _grid(n):
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    w, h = (2.55, 2.15) if n > 2 else (4.2, 3.0)  # one or two models: bigger panels
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols + 0.6, h * rows + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    axes = list(axes.flat)
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def small_multiples(pts, dft, orientation, column, ylabel, title, fname, zero_line=False, lines=(), symlog=None):
    """One panel per model. `lines`: (value, label, linestyle) reference lines; `symlog`: linear below
    this value, logarithmic above it."""
    sub = pts[pts.orientation == orientation]
    if sub.empty:
        return None
    names = model_order(sub.model.unique())
    fig, axes = _grid(len(names))
    dsub = dft[dft.orientation == orientation] if not dft.empty else dft
    have_dft = not dsub.empty and dsub[column].notna().any()
    for ax, name in zip(axes, names):
        g = sub[sub.model == name].sort_values("strain_pct")
        if zero_line:
            ax.axhline(0, color=C["axis"], lw=0.8, zorder=1)
        for value, _, ls in lines:
            ax.axhline(value, color=C["muted"], lw=0.9, ls=ls, zorder=1)
        if have_dft:
            dd = dsub.dropna(subset=[column]).sort_values("strain_pct")
            ax.plot(dd.strain_pct, dd[column], ls=(0, (4, 2)), color=C["dft"], lw=1.3, marker="o", ms=3.5,
                    mfc=C["surface"], zorder=2, label="DFT")
            if column == "gain_mev_per_fu" and dd.untilted_ref.any():
                u = dd[dd.untilted_ref]
                ax.plot(u.strain_pct, u[column], ls="none", marker="s", ms=5, mfc=C["surface"], mec=C["dft"],
                        mew=1.2, zorder=3)
        ax.plot(g.strain_pct, g[column], color=C["model"], marker="o", zorder=3, label="model")
        bad = g[~g.all_converged]
        if not bad.empty:
            ax.plot(bad.strain_pct, bad[column], ls="none", marker="x", color=C["ink2"], ms=5, zorder=4)
        ax.set_title(name, loc="left", y=1.12, pad=0)
        ax.text(0.0, 1.03, g.dataset.iloc[0] + " · " + g.functional.iloc[0], transform=ax.transAxes,
                fontsize=7, color=C["ink2"], va="bottom", ha="left")
        ax.tick_params(labelbottom=True)
    if symlog:  # the axes share y, so the first one sets it for all
        axes[0].set_yscale("symlog", linthresh=symlog, linscale=0.5)
        top = max(sub[column].max(), dsub[column].max() if have_dft else 0)
        axes[0].set_yticks([0] + [symlog * 10 ** k for k in range(int(np.log10(top / symlog)) + 1)])
        axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.supxlabel("in-plane strain (%)", fontsize=9, color=C["ink2"])
    fig.supylabel(ylabel, fontsize=9, color=C["ink2"])
    handles = [plt.Line2D([], [], color=C["model"], marker="o", label="model")]
    if have_dft:
        handles.append(plt.Line2D([], [], color=C["dft"], ls=(0, (4, 2)), marker="o", mfc=C["surface"], label="DFT"))
        if column == "gain_mev_per_fu" and dsub.dropna(subset=[column]).untilted_ref.any():
            handles.append(plt.Line2D([], [], ls="none", marker="s", mfc=C["surface"], mec=C["dft"],
                                      label="DFT, no tilted nonpolar run (gain includes tilt)"))
    if (~sub.all_converged).any():
        handles.append(plt.Line2D([], [], ls="none", marker="x", color=C["ink2"], label="not converged"))
    handles += [plt.Line2D([], [], color=C["muted"], lw=0.9, ls=ls, label=label) for _, label, ls in lines]
    # title on the first line, legend on its own line below it, panels underneath
    fig.suptitle(f"{title}, {ORIENT_LABEL[orientation]} strain", x=0.01, y=0.995, ha="left", va="top",
                 fontsize=11, color=C["ink"])
    h = fig.get_figheight()
    top = 1 - 0.48 / h
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.005, 1 - 0.30 / h),
               handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, top))
    return save(fig, fname)


LINES = {"": config.E_TOL_MEV, "_2meV": config.E_QF_MEV}  # column suffix: polar threshold (meV/f.u.)


def fig_summary(pts, dft, orientation):
    """The headline result: the polar energy lowering of unstrained STO for every model, one bar each, against DFT."""
    sub = pts[(pts.orientation == orientation) & (pts.strain_pct.abs() < 0.05)].sort_values("gain_mev_per_fu")
    if sub.empty:
        return None
    d = dft[(dft.orientation == orientation) & (dft.strain_pct.abs() < 0.05)] if not dft.empty else dft
    e_dft = float(d.gain_mev_per_fu.iloc[0]) if not d.empty else np.nan
    pbe, r2 = C["001"], C["111"]
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(sub) + 1.5))
    y = np.arange(len(sub))
    ax.barh(y, sub.gain_mev_per_fu.clip(lower=0), height=0.56, zorder=3,
            color=[pbe if f == "PBE" else r2 for f in sub.functional])
    for yi, v in zip(y, sub.gain_mev_per_fu):
        ax.text(max(v, 0) + 0.15, yi, f"{v:.2f}", va="center", fontsize=8, color=C["ink2"], zorder=5)
    ax.set_yticks(y, sub.model)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, sub.gain_mev_per_fu.max() * 1.13)
    ax.set_ylim(-0.6, len(sub) - 0.4)
    if np.isfinite(e_dft):
        ax.axvline(e_dft, color=C["ink"], lw=1.4, zorder=4)
    ax.axvline(config.E_QF_MEV, color=C["muted"], lw=1.0, ls=(0, (5, 2, 1, 2)), zorder=4)
    ax.set_xlabel("polar energy lowering (meV/f.u.)")
    ax.set_title("Unstrained STO: lowest non-polar phase minus lowest polar phase", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=pbe, label="model trained on PBE data"),
               plt.Rectangle((0, 0), 1, 1, color=r2, label="model trained on r2SCAN data"),
               plt.Line2D([], [], color=C["ink"], lw=1.4, label=f"DFT (PBEsol), {e_dft:.2f}"),
               plt.Line2D([], [], color=C["muted"], lw=1.0, ls=(0, (5, 2, 1, 2)),
                          label=f"{config.E_QF_MEV:g} meV/f.u.: quantum fluctuations (estimate)")]
    ax.legend(handles=handles, loc="lower right")
    return save(fig, f"fig0_summary_{orientation}")


def _critical(strains, gains, disps) -> dict:
    out = {}
    for suffix, e_tol in LINES.items():
        c = an.critical_strain(strains, gains, disps, e_tol=e_tol, d_tol=config.D_TOL)
        out[f"critical_compressive{suffix}_pct"], out[f"critical_tensile{suffix}_pct"] = c["compressive"], c["tensile"]
    return out


def critical_table(pts, dft):
    rows = []
    for (m, o), g in pts.groupby(["model", "orientation"]):
        rows.append({"model": m, "orientation": o, "functional": g.functional.iloc[0], "dataset": g.dataset.iloc[0],
                     "a0_model": g.a0_model.iloc[0], **_critical(g.strain_pct, g.gain_mev_per_fu.fillna(0), g.ti_disp),
                     "strain_range_pct": f"{g.strain_pct.min():+g} to {g.strain_pct.max():+g}"})
    if not dft.empty:
        for o, g in dft.groupby("orientation"):
            # where only polar DFT runs exist there is no gain, so it counts as polar at either line
            by_disp = np.where(g.ti_disp > config.D_TOL, max(LINES.values()) + 1.0, 0.0)
            gain = g.gain_mev_per_fu.where(g.gain_mev_per_fu.notna(), by_disp)
            rows.append({"model": "DFT (PBEsol)", "orientation": o, "functional": "", "dataset": "",
                         "a0_model": np.nan, **_critical(g.strain_pct, gain, g.ti_disp), "strain_range_pct": ""})
    return pd.DataFrame(rows)


# centrosymmetric space groups that occur here; anything else is a polar ground state
CENTRO = {"Pm-3m", "P4/mmm", "I4/mcm", "Imma", "C2/c", "R-3c", "R-3m", "C2/m", "Cmcm", "Fmmm", "P-1"}
PHASE_COLOURS = ["#2a78d6", "#eb6834", "#2f9e6e", "#8a5cc9", "#c9a227", "#d14d72", "#3fa7b5", "#7a6a58"]


def fig_phases(pts, dft, orientation):
    """Ground-state space group at each strain: DFT on the top row, one row per model below."""
    sub = pts[pts.orientation == orientation]
    if sub.empty:
        return None
    rows = []
    d = dft[dft.orientation == orientation] if not dft.empty else dft
    if not d.empty:
        rows.append(("DFT (PBEsol)", d))
    rows += [(n, sub[sub.model == n]) for n in model_order(sub.model.unique())]
    # one colour per polar phase, in the order they appear along the strain axis
    seen = []
    for _, g in rows:
        for sg in g.sort_values("strain_pct").space_group:
            if sg not in CENTRO and sg != "?" and sg not in seen:
                seen.append(sg)
    colour = {sg: PHASE_COLOURS[i % len(PHASE_COLOURS)] for i, sg in enumerate(seen)}
    strains = sorted(set(pd.concat([g.strain_pct for _, g in rows])))
    fig, ax = plt.subplots(figsize=(max(6.5, 0.8 * len(strains) + 2.6), 0.5 * len(rows) + 1.6))
    a0 = config.DFT_A0_CUBIC
    marked = []  # labels that read a different group at SYMPREC_LOOSE
    for y, (name, g) in enumerate(rows):
        for _, r in g.iterrows():
            polar = r.space_group in colour
            label = r.space_group
            if r.get("space_group_loose", label) not in (label, "?"):
                label += "†"
                marked.append(f"{name} at {a0 * (1 + r.strain_pct / 100):.2f} Å reads {r.space_group_loose}")
            ax.text(r.strain_pct, y, label, ha="center", va="center", fontsize=7.5,
                    color="white" if polar else C["ink2"],
                    bbox=dict(boxstyle="round,pad=0.35", fc=colour[r.space_group] if polar else C["grid"], ec="none"))
    pad = 0.6 * (strains[-1] - strains[0]) / max(len(strains) - 1, 1)
    ax.set_xlim(strains[0] - pad, strains[-1] + pad)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _ in rows])
    ax.set_xticks(strains)
    ax.set_xticklabels([f"{s:+.1f}" for s in strains])
    ax.set_xlabel("in-plane strain (%)  —  models: from their own cubic a0")
    if marked:
        ax.text(0.0, -0.11, f"Space groups at {SYMPREC:g} Å tolerance. † At {SYMPREC_LOOSE:g} Å: "
                + "; ".join(marked) + ".", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=C["ink2"])
    ax.grid(axis="y", visible=False)
    top = ax.secondary_xaxis("top", functions=(lambda s: a0 * (1 + s / 100), lambda a: (a / a0 - 1) * 100))
    top.set_xticks(config.DFT_LATTICE_PARAMS)
    top.set_xlabel("DFT in-plane a (Å)", color=C["ink2"])
    ax.set_title(f"Ground-state phase, {ORIENT_LABEL[orientation]} strain   (coloured: polar, grey: nonpolar)",
                 loc="left", pad=28)
    fig.tight_layout()
    return save(fig, f"fig4_phases_{orientation}")


def fig_finetune(crit):
    lc = config.RESULTS / "finetune" / "learning_curve.csv"
    if not lc.exists():
        return None
    d = pd.read_csv(lc).sort_values("n_train")
    zero = d[d.n_train == 0]
    d = d[d.n_train > 0]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ax = axes[0]
    ax.plot(d.n_train, d.force_mae_mev_A, color=C["111"], marker="o", label="fine-tuned on [001]")
    if not zero.empty:
        ax.axhline(zero.force_mae_mev_A.iloc[0], color=C["dft"], ls=(0, (4, 2)), lw=1.2, label="zero-shot")
    _count_axis(ax, d.n_train)
    ax.set_xlabel("[001] training frames")
    ax.set_ylabel("force MAE on [111] frames (meV/Å)")
    ax.set_title("Force error on unseen [111] strain", loc="left")
    ax.legend()
    ax = axes[1]
    ft = crit[crit.model.str.startswith("mace-ft-") & (crit.orientation == "111")].copy()
    if not ft.empty:
        ft["n"] = ft.model.map(lambda s: int(m.group(1)) if (m := re.search(r"(\d+)$", s)) else np.nan)
        ft = ft.dropna(subset=["n"]).sort_values("n")
        for side, mk in (("tensile", "o"), ("compressive", "s")):
            ax.plot(ft.n, ft[f"critical_{side}_pct"], color=C["111"], marker=mk, label=f"{side}",
                    ls="-" if side == "tensile" else (0, (1, 1.5)))
        d111 = crit[(crit.model == "DFT (PBEsol)") & (crit.orientation == "111")]
        if not d111.empty:
            for side, va_ in (("tensile", "bottom"), ("compressive", "top")):
                v = d111[f"critical_{side}_pct"].iloc[0]
                if not np.isnan(v):
                    ax.axhline(v, color=C["dft"], ls=(0, (4, 2)), lw=1.2)
                    ax.text(ft.n.max(), v, f" DFT {side} ", color=C["ink2"], fontsize=7.5, va=va_, ha="right")
        _count_axis(ax, ft.n)
        ax.legend()
    ax.set_xlabel("[001] training frames")
    ax.set_ylabel("[111] critical |strain| (%)")
    ax.set_title("Predicted [111] critical strain", loc="left")
    fig.tight_layout()
    return save(fig, "fig6_finetune")


def fig_phonons(ref):
    """Lowest polar mode of the cubic and I4/mcm references at 3.90 A (each model at its own a0) against
    the bulk DFPT, and of the polar ground state at 4.10 A against the DFT Ima2 there. One row per model."""
    recs = [json.loads(p.read_text()) for p in sorted((config.RESULTS / "phonons" / ref).glob("*/001/*.json"))]
    dft_file = config.RESULTS / "phonons" / "dft.json"
    if not recs or not dft_file.exists():
        return None
    df = pd.DataFrame([{k: r.get(k) for k in ("model", "strain_pct", "structure", "role", "lowest_polar_cm1")} for r in recs])
    dft = {(r["orientation"], str(r["strain_percent"]), r["structure"]): r["lowest_polar_cm1"]
           for r in json.loads(dft_file.read_text())}
    s0, s1 = config.STRAINS_PCT[4], config.STRAINS_PCT[-1]  # a = 3.90 and 4.10 A
    panels = [("Cubic (Pm-3m) reference, 3.90 Å", (df.strain_pct == s0) & (df.structure == "nonpolar_Pm-3m"),
               dft.get(("bulk", "", "Pm-3m")), "DFT bulk"),
              ("I4/mcm reference, 3.90 Å", (df.strain_pct == s0) & (df.structure == "nonpolar_I4mcm"),
               dft.get(("bulk", "", "I4mcm")), "DFT bulk"),
              ("Polar ground state, 4.10 Å", (df.strain_pct == s1) & (df.role == "best_polar"),
               dft.get(("001", f"{s1:g}", "Ima2")), "DFT Ima2")]
    names = model_order(df.model.unique())
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.3 * len(names) + 1.9), sharey=True)
    for ax, (title, rows, ref_value, ref_label) in zip(axes, panels):
        v = df[rows].set_index("model").lowest_polar_cm1
        ax.axvline(0, color=C["axis"], lw=0.8, zorder=1)
        if ref_value is not None:
            ax.axvline(ref_value, color=C["dft"], ls=(0, (4, 2)), lw=1.3, zorder=2)
            ax.text(ref_value, -0.75, f"{ref_label} {ref_value:.0f}".replace("-", "−"), ha="center", va="bottom",
                    fontsize=7.5, color=C["ink2"], bbox=dict(boxstyle="square,pad=0.15", fc=C["surface"], ec="none"))
        ax.plot([v.get(n, np.nan) for n in names], y, ls="none", marker="o", ms=6, color=C["model"], zorder=3)
        if v.min() > 0 and (ref_value is None or ref_value > 0):
            ax.set_xlim(0, ax.get_xlim()[1])  # all stable: start at the zero line
        ax.set_title(title, loc="left", pad=16)
        ax.grid(axis="x", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].set_ylim(len(names) - 0.5, -1.0)
    fig.supxlabel("lowest polar mode (cm⁻¹; negative = imaginary, unstable)", fontsize=9, color=C["ink2"])
    fig.suptitle("Soft polar mode against DFPT, [001] strain", x=0.01, ha="left", fontsize=11, color=C["ink"])
    fig.text(0.01, -0.02, "Models at the 3.90 Å grid point, strained from their own cubic a0. TO modes (no LO–TO splitting).",
             fontsize=7.5, color=C["ink2"], ha="left", va="top")
    fig.tight_layout()
    return save(fig, "fig5_phonons_001")


def _count_axis(ax, values):
    ax.set_xscale("log")
    ticks = sorted(set(int(v) for v in values))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def phonon_table(ref):
    recs = [json.loads(p.read_text()) for p in (config.RESULTS / "phonons" / ref).glob("*/*/*.json")]
    if not recs:
        return None
    df = pd.DataFrame([{k: r.get(k) for k in ("model", "orientation", "strain_pct", "structure", "role",
                                              "lowest_polar_cm1", "lowest_tilt_cm1", "n_imaginary")} for r in recs])
    dft = config.RESULTS / "phonons" / "dft.json"
    if dft.exists():
        for r in json.loads(dft.read_text()):
            s = r["strain_percent"]
            df.loc[len(df)] = {"model": "DFT (PBEsol)", "orientation": r["orientation"],
                               "strain_pct": float(s) if s not in ("", None) else np.nan,
                               "structure": r.get("structure", "as computed"), "role": "DFPT",
                               "lowest_polar_cm1": r["lowest_polar_cm1"], "lowest_tilt_cm1": r["lowest_tilt_cm1"],
                               "n_imaginary": r["n_imaginary"]}
    df = df.sort_values(["orientation", "strain_pct", "structure", "model"])
    df.to_csv(config.RESULTS / "summary_phonons.csv", index=False)
    md = df.assign(strain_pct=[config.strain_tag(v) if pd.notna(v) else "" for v in df.strain_pct])  # cm-1 to 0 d.p.
    (config.RESULTS / "summary_phonons.md").write_text(to_markdown(md, floatfmt="{:.0f}"))
    return df


def to_markdown(df, floatfmt="{:.2f}"):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(config.FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return config.FIGURES / f"{name}.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strain-ref", choices=("own", "dft"), default="own")
    args = ap.parse_args()
    ref = "own_a0" if args.strain_ref == "own" else "dft_a0"
    pts = load_points(ref)
    if pts.empty:
        print(f"No results in results/relax/{ref} yet. Run scripts/04_strain_relax.py first.")
        return 1
    dft = load_dft()
    pts.to_csv(config.RESULTS / "summary_points.csv", index=False)
    made = []
    for o in sorted(pts.orientation.unique()):
        lines = [(config.E_TOL_MEV, f"{config.E_TOL_MEV:g} meV/f.u.: polar threshold", (0, (1, 2))),
                 (config.E_QF_MEV, f"{config.E_QF_MEV:g} meV/f.u.: quantum fluctuations (estimate)", (0, (5, 2, 1, 2)))]
        made.append(small_multiples(pts, dft, o, "gain_mev_per_fu", "polar gain (meV/f.u.)",
                                    "Energy gained by going polar", f"fig1_gain_{o}", zero_line=True,
                                    lines=lines, symlog=config.E_TOL_MEV))
        made.append(small_multiples(pts, dft, o, "ti_disp", "Ti displacement (Å)",
                                    "Uniform Ti off-centring of the ground state", f"fig2_displacement_{o}"))
        made.append(small_multiples(pts, dft, o, "tilt", "tilt angle (°)",
                                    "Octahedral tilt of the ground state", f"fig3_tilt_{o}"))
        made.append(fig_phases(pts, dft, o))
        made.append(fig_summary(pts, dft, o))
    crit = critical_table(pts, dft)
    crit.to_csv(config.RESULTS / "summary_critical.csv", index=False)
    order = {m: i for i, m in enumerate(model_order([m for m in crit.model.unique() if m != "DFT (PBEsol)"]) + ["DFT (PBEsol)"])}
    crit = crit.sort_values(["orientation", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
    show = crit[["model", "orientation", "functional", "a0_model"]
                + [f"critical_{side}{suffix}_pct" for suffix in LINES for side in ("compressive", "tensile")]]
    (config.RESULTS / "summary_critical.md").write_text(to_markdown(show))
    made.append(fig_finetune(crit))
    phonon_table(ref)
    made.append(fig_phonons(ref))
    print("Figures:")
    for m in made:
        if m:
            print("  ", m)
    print("Tables: results/summary_points.csv, results/summary_critical.csv/.md"
          + (", results/summary_phonons.csv/.md" if (config.RESULTS / "summary_phonons.csv").exists() else ""))
    print(f"\nCritical strains (%) at the polar threshold ({config.E_TOL_MEV:g} meV/f.u.) and at {config.E_QF_MEV:g}"
          " meV/f.u.: every point from there outwards is polar. 0 = polar already unstrained (a = 3.90 A) and"
          " on the whole side; — = not polar even at the largest strain on that side.")
    print(show.to_string(index=False, na_rep="—", float_format=lambda v: f"{v:.2f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
