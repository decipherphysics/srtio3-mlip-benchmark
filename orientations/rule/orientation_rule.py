"""Does SrTiO3's polar energy gain under epitaxial strain depend only on how far its Ti-O axes are stretched,
whatever the film orientation?

Data: the PBEsol DFT for [001] and [111] films, and MH-1 r2SCAN (matpes_r2scan head) for [001], [110] and
[111] films, each strain measured from that method's own cubic a0.

For each strain the gain is the energy of the lowest nonpolar state minus that of the lowest state (meV/f.u.);
the nonpolar reference is the lowest centrosymmetric state (every Ti within 0.005 A of its cage centre, on
average). The axes are measured in the lowest nonpolar state, the one that goes polar: the three pseudocubic
Ti-Ti axes and the three O-O axes of the TiO6 cages, each as a stretch (%) from that method's cubic a0. The rule
tested is

    gain = k * sum over the stretched axes of stretch^2,

with k fitted to the [001] films under tensile strain and then applied unchanged to the other orientations.

    ~/.sto-envs/mace/bin/python orientations/rule/orientation_rule.py       (writes rule_table.csv, rule_summary.json)
    ~/.sto-envs/mace/bin/python orientations/rule/orientation_rule.py fig   (draws figures/fig_orientation_rule.png)
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
ORIENT = REPO / "orientations"
sys.path.insert(0, str(ORIENT / "111"))
import sto111 as S  # noqa: E402
import film_measure as F  # noqa: E402

an = S.an
A0 = {"DFT": 3.8997, "model": 3.9182767914555923}
GRID = [-5.12, -3.84, -2.56, -1.27, 0.01, 1.29, 2.57, 3.85, 5.14]


def snap(s):
    g = min(GRID, key=lambda x: abs(x - s))
    return g if abs(g - s) < 0.05 else None


def cube_axes(pos, sym, cell):
    """The three pseudocubic axes as mean Ti-Ti nearest-neighbour vectors. v and -v count as one direction, so
    this works whichever axes lie in the film plane."""
    ti = np.array([i for i, s in enumerate(sym) if s == "Ti"], int)
    groups = []
    for i in ti:
        v, _ = an.neighbour_vectors(pos, cell, i, ti, 7)
        for x in v:
            if np.linalg.norm(x) < 1.0:
                continue
            for g in groups:
                c = x @ g[0] / np.linalg.norm(x) / np.linalg.norm(g[0])
                if abs(c) > 0.9:
                    g.append(x if c > 0 else -x)
                    break
            else:
                groups.append([x])
    if len(groups) != 3:
        raise RuntimeError(f"found {len(groups)} Ti-Ti directions, expected 3")
    return np.array([np.mean(g, axis=0) for g in groups])


def axes(atoms):
    """Lengths (A) of the three pseudocubic Ti-Ti axes and of the three O-O axes of the TiO6 cages."""
    pos, sym, cell = atoms.get_positions(), atoms.get_chemical_symbols(), np.array(atoms.cell)
    ax = cube_axes(pos, sym, cell)
    u = ax / np.linalg.norm(ax, axis=1)[:, None]
    ti = [i for i, s in enumerate(sym) if s == "Ti"]
    o = np.array([i for i, s in enumerate(sym) if s == "O"])
    oo = [[], [], []]
    for i in ti:
        v, _ = an.neighbour_vectors(pos, cell, i, o, 6)
        proj = v @ u.T
        for k in range(3):
            oo[k].append(np.linalg.norm(v[np.argmax(proj[:, k])] - v[np.argmin(proj[:, k])]))
    return np.linalg.norm(ax, axis=1), np.array([np.mean(x) for x in oo])


def ti_disp(atoms):
    """Mean size of the Ti off-centrings from their O6 cage centres (A). Zero only for a centrosymmetric state:
    an antipolar state, whose off-centrings cancel, is not a valid nonpolar reference."""
    d = an.ti_displacements(atoms.get_positions(), atoms.get_chemical_symbols(), np.array(atoms.cell))
    return float(np.linalg.norm(d, axis=1).mean())


def dft_001():
    rows = {r["path"]: r for r in csv.DictReader(open(REPO / "data" / "dft_final.csv"))}
    out = []
    for a in read(REPO / "data" / "dft_frames.extxyz", ":"):
        if not a.info.get("is_final"):
            continue
        r = rows[str(a.info["run_path"])]
        if r["orientation"] != "001" or r["run_type"] != "relax":
            continue
        s = snap(float(r["strain_percent"]))
        if s is not None:
            out.append({"strain": s, "e": float(r["e_per_fu"]), "atoms": a, "label": r["path"]})
    return out


def dft_111():
    out = []
    for a in read(ORIENT / "111" / "dft" / "finals.extxyz", ":"):
        if "path" not in a.info or not a.info.get("converged"):
            continue
        lab = str(a.info["path"]).split("/")[1]
        if lab == "Relaxed":
            continue
        s = snap((float(lab.split("_")[0]) / A0["DFT"] - 1) * 100)
        if s is not None:
            out.append({"strain": s, "e": a.info["e0"] / (len(a) // 5), "atoms": a, "label": a.info["path"]})
    return out


def model(folder):
    out = []
    for p in sorted(glob.glob(str(folder / "*.json"))):
        r = json.loads(Path(p).read_text())
        if not r.get("converged", True):
            continue
        s = snap(r["strain_pct"])
        if s is not None:
            out.append({"strain": s, "e": r["e_per_fu"], "atoms": read(p[:-5] + ".extxyz"), "label": Path(p).stem})
    return out


def summarise(source, orientation, runs):
    rows = []
    for s in sorted({r["strain"] for r in runs}):
        g = [r for r in runs if r["strain"] == s]
        for r in g:
            r["ti"] = ti_disp(r["atoms"])
        e_min = min(r["e"] for r in g)
        npol = [r for r in g if r["ti"] < 0.005]
        if not npol:
            continue
        ref = min(npol, key=lambda r: r["e"])
        best = min(g, key=lambda r: r["e"])
        L, O = axes(ref["atoms"])
        a0 = A0["DFT" if source == "DFT" else "model"]
        st_ti, st_oo = np.sort(100 * (L / a0 - 1))[::-1], np.sort(100 * (O / a0 - 1))[::-1]
        rows.append({"source": source, "orientation": orientation, "strain_pct": s,
                     "gain_mev": 1000 * (ref["e"] - e_min), "ti_ground_A": best["ti"],
                     "nonpolar_ref": ref["label"], "ground": best["label"],
                     **{f"titi_{k + 1}_pct": float(st_ti[k]) for k in range(3)},
                     **{f"oo_{k + 1}_pct": float(st_oo[k]) for k in range(3)}})
    return rows


def x_of(row, kind):
    v = np.array([row[f"{kind}_{k}_pct"] for k in (1, 2, 3)])
    return float(np.sum(np.clip(v, 0, None) ** 2))


def main():
    rows = (summarise("DFT", "001", dft_001()) + summarise("DFT", "111", dft_111())
            + summarise("model", "001", model(REPO / "results/relax/own_a0/mace-mh-1-r2scan/001"))
            + summarise("model", "110", model(ORIENT / "110" / "results"))
            + summarise("model", "111", model(ORIENT / "111" / "results")))
    summary = {}
    for kind in ("titi", "oo"):
        for r in rows:
            r[f"X_{kind}"] = x_of(r, kind)
        for src in ("DFT", "model"):
            fit = [r for r in rows if r["source"] == src and r["orientation"] == "001" and r["strain_pct"] > 0.5]
            X = np.array([r[f"X_{kind}"] for r in fit]); G = np.array([r["gain_mev"] for r in fit])
            k = float(X @ G / (X @ X))
            summary[f"k_{kind}_{src}"] = k
            for r in rows:
                if r["source"] == src:
                    r[f"pred_{kind}"] = k * r[f"X_{kind}"]
    with open(HERE / "rule_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (HERE / "rule_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(f"{r['source']:5s} [{r['orientation']}] {r['strain_pct']:+5.2f}%  gain {r['gain_mev']:7.2f}   "
              f"Ti-Ti {r['titi_1_pct']:+5.2f} {r['titi_2_pct']:+5.2f} {r['titi_3_pct']:+5.2f}  "
              f"O-O {r['oo_1_pct']:+5.2f} {r['oo_2_pct']:+5.2f} {r['oo_3_pct']:+5.2f}   "
              f"pred(TiTi) {r['pred_titi']:7.2f}  pred(OO) {r['pred_oo']:7.2f}   ground {r['ground']}")
    return 0


if __name__ == "__main__" and len(sys.argv) == 1:
    sys.exit(main())


# ---------------------------------------------------------------- figure
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOUR = {"001": "#2a78d6", "110": "#eb6834", "111": "#1baf7a"}  # categorical slots 1-3
MARK = {"001": "o", "110": "s", "111": "^"}


def figure(path=REPO / "figures" / "fig_orientation_rule.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(HERE / "rule_table.csv")))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except ValueError:
                pass
        r["orientation"] = {1.0: "001", 110.0: "110", 111.0: "111"}.get(r["orientation"], r["orientation"])
    summ = json.loads((HERE / "rule_summary.json").read_text())
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.3), dpi=160, facecolor=SURFACE)
    xmax = 60
    for ax, src, title in ((axs[0], "DFT", "DFT (PBEsol)"), (axs[1], "model", "Model (MH-1, r2SCAN head)")):
        k = summ[f"k_oo_{src}"]
        xs = np.linspace(0, xmax, 50)
        ax.plot(xs, k * xs, color=INK2, lw=1, ls="--", zorder=1)
        ax.text(xmax * 0.97, k * xmax * 0.97, f"gain = {k:.2f} × Σ stretch²\n(fitted to [001] tensile strain)",
                ha="right", va="bottom", fontsize=7.5, color=INK2, rotation=0)
        for o in ("001", "110", "111"):
            g = [r for r in rows if r["source"] == src and r["orientation"] == o]
            if not g:
                continue
            for part, filled in (("tensile", True), ("compressive", False)):
                h = [r for r in g if (r["strain_pct"] > 0.5) == (part == "tensile") and abs(r["strain_pct"]) > 0.5]
                if not h:
                    continue
                ax.scatter([r["X_oo"] for r in h], [r["gain_mev"] for r in h], s=34, marker=MARK[o],
                           facecolor=COLOUR[o] if filled else SURFACE, edgecolor=COLOUR[o], lw=1.4, zorder=3,
                           label=f"[{o}] {part}")
        ax.set_xlim(-2, xmax); ax.set_ylim(-5, 165)
        ax.set_xlabel("Σ (stretch of each stretched TiO$_6$ cage axis)²  (%²)")
        ax.set_ylabel("polar energy gain (meV/f.u.)")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axs[2]
    for o in ("001", "110", "111"):
        g = sorted([r for r in rows if r["source"] == "model" and r["orientation"] == o], key=lambda r: r["strain_pct"])
        x = [r["strain_pct"] for r in g]
        for j, lw in ((1, 1.8), (2, 1.0), (3, 1.0)):
            ax.plot(x, [r[f"oo_{j}_pct"] for r in g], color=COLOUR[o], lw=lw, marker=MARK[o], ms=4 if j == 1 else 3,
                    mfc=COLOUR[o] if j == 1 else SURFACE, label=f"[{o}]" if j == 1 else None, zorder=3 if j == 1 else 2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("in-plane strain (%)")
    ax.set_ylabel("stretch of the three TiO$_6$ cage axes (%)")
    ax.set_title("Why: how far each Ti–O axis is stretched (model)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", title="thick: most stretched axis", title_fontsize=7)
    for a in axs:
        a.set_facecolor(SURFACE); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), facecolor=SURFACE, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fig":
    figure()
