"""
Module 1: NVT Analysis — Figure Generation
============================================
Workflow D1.2 — UltraBat Project

Reads the NVT trajectory and generates publication-quality figures:
    Fig 1 — Thermodynamics: Energy + Temperature vs time
    Fig 2 — Radial Distribution Functions: Li-O, Li-P, Li-C
    Fig 3 — Mean Square Displacement → Li+ diffusivity
    Fig 4 — Li+ Coordination Number (chemical classification)
    Fig 5 — O2 / Peroxo formation
    Fig 6 — Li surface migration

Input:  nvt_results/interface_nvt.xyz  (extXYZ, every 100 steps = 0.1 ps)
Output: figures_nvt/  (PNG figures)
        results_nvt/  (NPZ data files)

No ASE required — pure numpy/matplotlib parsing.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re, os

# ─── Paths ────────────────────────────────────────────────────────────────────
NVT_XYZ   = 'interface_nvt.xyz'
FIG_DIR   = 'figures_nvt'
RES_DIR   = 'results_nvt'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────────────
NATOMS      = 497
DT_PS       = 0.1       # ps per saved frame (100 steps × 1 fs)
EQUIL_FRAC  = 0.1       # discard first 10% (NVT already equilibrated by NPT)
R_MAX       = 8.0       # Å — RDF cutoff
N_BINS      = 300       # RDF bins (more frames → finer bins)
COORD_CUT   = 2.8       # Å — Li-O first shell
C_O_BOND    = 1.6       # Å — C-O bond (solvent O)
TM_O_BOND   = 2.2       # Å — TM-O bond (surface O)
OO_CUT      = 1.6       # Å — O-O bond detection
LI_SURF_ZMIN = 13.0     # Å — slab bottom
LI_SURF_ZMAX = 25.0     # Å — slab top
MIGRATE_THR  = 3.0      # Å — migration threshold

# ─── Style ────────────────────────────────────────────────────────────────────
BLUE  = '#2E75B6'
RED   = '#E63946'
GREEN = '#2A9D8F'
GOLD  = '#E9C46A'
GRAY  = '#6B6B6B'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 9,  'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: PARSE TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════════
print("Parsing NVT trajectory...")

frames_meta  = []  # {lz, energy, latt}
atoms_frames = []  # {symbols, pos}

with open(NVT_XYZ) as f:
    lines = f.readlines()

frame_size = NATOMS + 2
n_frames   = len(lines) // frame_size
print(f"  Total frames : {n_frames}  ({n_frames * DT_PS:.1f} ps)")

for fi in range(n_frames):
    base    = fi * frame_size
    comment = lines[base + 1]

    # Lattice
    m = re.search(r'Lattice="([^"]+)"', comment)
    latt = list(map(float, m.group(1).split()))

    # Energy
    m2 = re.search(r'(?<![_a-z])energy=([-\d.e+]+)', comment)
    energy = float(m2.group(1)) if m2 else np.nan

    # Atoms
    syms, xs, ys, zs = [], [], [], []
    for li in range(NATOMS):
        p = lines[base + 2 + li].split()
        syms.append(p[0])
        xs.append(float(p[1]))
        ys.append(float(p[2]))
        zs.append(float(p[3]))

    frames_meta.append({'lz': latt[8], 'energy': energy, 'latt': latt})
    atoms_frames.append({'symbols': syms, 'pos': np.array([xs, ys, zs]).T, 'latt': latt})

time_ps    = np.arange(n_frames) * DT_PS
equil_f    = int(n_frames * EQUIL_FRAC)
prod_range = list(range(equil_f, n_frames))
print(f"  Equil frames : {equil_f}  ({equil_f * DT_PS:.1f} ps)")
print(f"  Prod  frames : {len(prod_range)}  ({len(prod_range) * DT_PS:.1f} ps)")

# ─── Atom index helpers ───────────────────────────────────────────────────────
d0 = atoms_frames[0]
syms0 = d0['symbols']
pos0  = d0['pos']

def idx(sym): return [i for i,s in enumerate(syms0) if s == sym]

idx_Li  = idx('Li')
idx_O   = idx('O')
idx_P   = idx('P')
idx_C   = idx('C')
idx_Mn  = idx('Mn')
idx_Co  = idx('Co')
idx_Ni  = idx('Ni')
idx_TM  = idx_Mn + idx_Co + idx_Ni

li_elec = [i for i in idx_Li if pos0[i,2] > LI_SURF_ZMAX or pos0[i,2] < LI_SURF_ZMIN]
li_surf = [i for i in idx_Li if LI_SURF_ZMIN <= pos0[i,2] <= LI_SURF_ZMAX]
print(f"  Electrolyte Li: {len(li_elec)}  Surface Li: {len(li_surf)}")

# ─── MIC distance ─────────────────────────────────────────────────────────────
def mic_dist(pa, pb, latt):
    a = np.array([latt[0], latt[4], latt[8]])
    diff = pa[:, None, :] - pb[None, :, :]
    diff -= np.round(diff / a) * a
    return np.sqrt((diff**2).sum(axis=-1))

# ─── Chemical O classification ────────────────────────────────────────────────
def classify_O(fi):
    """Returns (o_solv_idx, o_surf_idx) for frame fi."""
    d    = atoms_frames[fi]
    latt = frames_meta[fi]['latt']
    pos  = d['pos']
    po   = pos[idx_O]
    pc   = pos[idx_C]
    ptm  = pos[idx_TM]

    dOC  = mic_dist(po, pc,  latt)
    dOTM = mic_dist(po, ptm, latt)

    solv = [idx_O[i] for i in range(len(idx_O)) if dOC[i].min()  < C_O_BOND]
    surf = [idx_O[i] for i in range(len(idx_O)) if dOTM[i].min() < TM_O_BOND]
    return solv, surf

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: THERMODYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Thermodynamics...")

energies = np.array([d['energy'] for d in frames_meta])
lz_vals  = np.array([d['lz']     for d in frames_meta])

# Estimate temperature from kinetic energy fluctuations
# (if MACE stores it; otherwise just plot energy)
prod_E = energies[equil_f:]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(time_ps, energies, color=BLUE, lw=0.8, alpha=0.7, label='E_pot')
# Running average
win = max(1, n_frames//50)
E_avg = np.convolve(energies, np.ones(win)/win, mode='valid')
t_avg = time_ps[win//2: win//2 + len(E_avg)]
ax.plot(t_avg, E_avg, color=BLUE, lw=2.0, label=f'Running avg ({win} frames)')
ax.axvline(time_ps[equil_f], color='gray', ls='--', lw=1.2,
           label=f'Prod start ({time_ps[equil_f]:.1f} ps)')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Potential Energy (eV)')
ax.set_title('Potential Energy — NVT Production')
ax.legend()
ax.set_xlim(0, time_ps[-1])

ax = axes[1]
ax.plot(time_ps, lz_vals, color=RED, lw=1.0, alpha=0.5)
ax.axhline(lz_vals[equil_f:].mean(), color=RED, lw=1.5, ls='--',
           label=f'⟨Lz⟩ = {lz_vals[equil_f:].mean():.3f} Å (fixed)')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Lz (Å)')
ax.set_title('Cell Length Lz — NVT (constant volume)')
ax.legend()
ax.set_xlim(0, time_ps[-1])

fig.suptitle('NVT Simulation — LRMO(104)/LP30 Interface  (T=300 K)', fontsize=13)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig1_nvt_thermodynamics.png', bbox_inches='tight')
plt.close(fig)
print("  Saved fig1_nvt_thermodynamics.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: RDF
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 2] RDFs...")

edges  = np.linspace(0, R_MAX, N_BINS + 1)
r_mid  = 0.5 * (edges[:-1] + edges[1:])

def compute_rdf(center_idx, neighbor_idx_list, prod_frames):
    hist = np.zeros(N_BINS)
    n_nb = len(neighbor_idx_list)
    for fi in prod_frames:
        d    = atoms_frames[fi]
        latt = d['latt']
        pos  = d['pos']
        if not center_idx or not neighbor_idx_list:
            continue
        dists = mic_dist(pos[center_idx], pos[neighbor_idx_list], latt)
        h, _ = np.histogram(dists.ravel(), bins=edges)
        hist += h

    nf  = len(prod_frames)
    nc  = len(center_idx)
    d0f = atoms_frames[prod_frames[0]]
    vol = d0f['latt'][0] * d0f['latt'][4] * d0f['latt'][8]
    rho = n_nb / vol
    shell = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    gofr  = hist / (nf * nc * rho * shell)
    return r_mid, gofr

r_LiO, g_LiO = compute_rdf(li_elec, idx_O, prod_range)
r_LiP, g_LiP = compute_rdf(li_elec, idx_P, prod_range)
r_LiC, g_LiC = compute_rdf(li_elec, idx_C, prod_range)

np.savez(f'{RES_DIR}/rdf_data.npz',
         r_LiO=r_LiO, g_LiO=g_LiO,
         r_LiP=r_LiP, g_LiP=g_LiP,
         r_LiC=r_LiC, g_LiC=g_LiC)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (r, g, label, color) in zip(axes, [
        (r_LiO, g_LiO, 'Li–O', RED),
        (r_LiP, g_LiP, 'Li–P', BLUE),
        (r_LiC, g_LiC, 'Li–C', GREEN)]):
    ax.plot(r, g, color=color, lw=1.6)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('g(r)')
    ax.set_title(label)
    ax.set_xlim(0, R_MAX)
    ax.set_ylim(bottom=0)
    mask = r > 1.0
    pk_r = r[mask][np.argmax(g[mask])]
    ax.axvline(pk_r, color=color, ls=':', lw=1.2,
               label=f'peak: {pk_r:.2f} Å')
    ax.legend()

fig.suptitle('Radial Distribution Functions — Electrolyte Li⁺  (NVT, 300 K)',
             fontsize=13)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig2_nvt_rdf.png', bbox_inches='tight')
plt.close(fig)
print("  Saved fig2_nvt_rdf.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: MSD → DIFFUSIVITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 3] MSD...")

latt0  = frames_meta[0]['latt']
cell   = np.array([latt0[0], latt0[4], latt0[8]])

all_pos = np.array([atoms_frames[fi]['pos'][li_elec] for fi in range(n_frames)])
pos_uw  = all_pos.copy()
for f in range(1, n_frames):
    d = all_pos[f] - all_pos[f-1]
    for dim in range(3):
        d[:, dim] -= np.round(d[:, dim] / cell[dim]) * cell[dim]
    pos_uw[f] = pos_uw[f-1] + d

# MSD — multiple origins every 50 frames
msd = np.zeros(n_frames)
cnt = np.zeros(n_frames)
step = 50
for t0 in range(0, n_frames, step):
    for dt in range(1, n_frames - t0):
        disp = pos_uw[t0+dt] - pos_uw[t0]
        msd[dt]  += np.mean(np.sum(disp**2, axis=1))
        cnt[dt]  += 1
msd[cnt > 0] /= cnt[cnt > 0]

# Fit on production window 20-70%
fs = int(0.2 * n_frames)
fe = int(0.7 * n_frames)
coeffs = np.polyfit(time_ps[fs:fe], msd[fs:fe], 1)
slope  = coeffs[0]   # Å²/ps
D      = slope / 6.0 * 1e-20 / 1e-12   # m²/s
print(f"  D(Li+) = {D:.3e} m²/s")

np.savez(f'{RES_DIR}/msd_data.npz',
         time_ps=time_ps, msd=msd, D_m2s=D, slope=slope)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(time_ps, msd, color=GOLD, lw=1.5, alpha=0.8, label='Li⁺ (electrolyte)')
t_fit = np.linspace(time_ps[fs], time_ps[fe], 100)
ax.plot(t_fit, np.polyval(coeffs, t_fit), 'k--', lw=1.8,
        label=f'Linear fit  D = {D:.2e} m²/s')
ax.axvline(time_ps[fs], color='gray', ls=':', lw=1.0, alpha=0.6)
ax.axvline(time_ps[fe], color='gray', ls=':', lw=1.0, alpha=0.6,
           label='Fit window')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('MSD (Å²)')
ax.set_title('Mean Square Displacement — Li⁺ in LP30  (NVT, 100 ps)')
ax.legend()
ax.set_xlim(0, time_ps[-1])
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig3_nvt_msd.png', bbox_inches='tight')
plt.close(fig)
print("  Saved fig3_nvt_msd.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: COORDINATION NUMBER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 4] Coordination number...")

cn_total, cn_solv, cn_surf = [], [], []

for fi in prod_range:
    d    = atoms_frames[fi]
    latt = frames_meta[fi]['latt']
    pos  = d['pos']
    o_solv, o_surf = classify_O(fi)

    pc = pos[li_elec]
    for o_set, cn_list in [(idx_O, cn_total),
                            (o_solv,  cn_solv),
                            (o_surf,  cn_surf)]:
        if o_set:
            dists = mic_dist(pc, pos[o_set], latt)
            cn_list.extend(np.sum(dists < COORD_CUT, axis=1).tolist())
        else:
            cn_list.extend([0] * len(li_elec))

cn_total = np.array(cn_total)
cn_solv  = np.array(cn_solv)
cn_surf  = np.array(cn_surf)

np.savez(f'{RES_DIR}/coordination_data.npz',
         cn_total=cn_total, cn_solv=cn_solv, cn_surf=cn_surf,
         mean_total=cn_total.mean(), mean_solv=cn_solv.mean(),
         mean_surf=cn_surf.mean())

print(f"  Li–O total   : {cn_total.mean():.2f} ± {cn_total.std():.2f}")
print(f"  Li–O solvent : {cn_solv.mean():.2f}  ± {cn_solv.std():.2f}")
print(f"  Li–O surface : {cn_surf.mean():.2f}  ± {cn_surf.std():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (cn, label, color) in zip(axes, [
        (cn_total, f'Total Li–O\n⟨N⟩={cn_total.mean():.2f}±{cn_total.std():.2f}',  RED),
        (cn_solv,  f'Solvent O (EC/DMC)\n⟨N⟩={cn_solv.mean():.2f}±{cn_solv.std():.2f}', GREEN),
        (cn_surf,  f'Surface O (oxide)\n⟨N⟩={cn_surf.mean():.2f}±{cn_surf.std():.2f}',  BLUE)]):
    bins = np.arange(-0.5, max(cn.max() + 1.5, 7), 1)
    vals, _ = np.histogram(cn, bins=bins, density=True)
    ax.bar(bins[:-1]+0.5, vals, width=0.8, color=color, alpha=0.85,
           edgecolor='white')
    ax.axvline(cn.mean(), color='black', ls='--', lw=1.5,
               label=f'⟨N⟩ = {cn.mean():.2f}')
    ax.set_xlabel('Coordination Number')
    ax.set_ylabel('Probability')
    ax.set_title(label)
    ax.legend()

fig.suptitle(f'Li⁺ Coordination in LP30  (cutoff={COORD_CUT} Å, NVT 100 ps)',
             fontsize=13)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig4_nvt_coordination.png', bbox_inches='tight')
plt.close(fig)
print("  Saved fig4_nvt_coordination.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: O2 / PEROXO FORMATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 5] O2 formation...")

min_oo = []
n_bonds = []

for fi in range(n_frames):
    d    = atoms_frames[fi]
    latt = frames_meta[fi]['latt']
    pos  = d['pos']
    po   = pos[idx_O]
    dists = mic_dist(po, po, latt)
    np.fill_diagonal(dists, 999.0)
    min_oo.append(dists.min())
    rows, cols = np.where((dists < OO_CUT) & (dists > 0.1))
    n_bonds.append(len(rows[rows < cols]))

min_oo  = np.array(min_oo)
n_bonds = np.array(n_bonds)

np.savez(f'{RES_DIR}/o2_data.npz',
         time_ps=time_ps, min_oo=min_oo, n_bonds=n_bonds)

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax = axes[0]
ax.plot(time_ps, min_oo, color=RED, lw=1.0, alpha=0.8)
ax.axhline(1.21, color='black', ls='--', lw=1.2, label='O₂ (1.21 Å)')
ax.axhline(1.35, color=GOLD,    ls='--', lw=1.0, label='Superoxo (1.35 Å)')
ax.axhline(1.60, color=BLUE,    ls='--', lw=1.0, label='Peroxo cutoff (1.60 Å)')
ax.axhspan(0.9, 1.25, alpha=0.07, color='red')
ax.axhspan(1.25, 1.60, alpha=0.05, color='orange')
ax.set_ylabel('min d(O–O) (Å)')
ax.set_title('Minimum O–O Distance — NVT (100 ps)')
ax.set_ylim(0.9, 2.0)
ax.legend(ncol=3)

ax = axes[1]
ax.bar(time_ps, n_bonds, width=DT_PS*0.8, color=GREEN, alpha=0.8)
ax.axhline(n_bonds[equil_f:].mean(), color=GREEN, ls='--', lw=1.5,
           label=f'Mean = {n_bonds[equil_f:].mean():.1f} pairs')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('# O–O pairs')
ax.set_title(f'O–O Bond-like Pairs  (d < {OO_CUT} Å)')
ax.set_xlim(0, time_ps[-1])
ax.legend()

fig.suptitle('O₂ / Peroxo Formation — LRMO Surface  (NVT 100 ps)', fontsize=13)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig5_nvt_o2.png', bbox_inches='tight')
plt.close(fig)
print("  Saved fig5_nvt_o2.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: LI SURFACE MIGRATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 6] Li migration...")

all_li_pos = np.array([atoms_frames[fi]['pos'][li_surf] for fi in range(n_frames)])
pos_li_uw  = all_li_pos.copy()
for f in range(1, n_frames):
    d = all_li_pos[f] - all_li_pos[f-1]
    d -= np.round(d / cell) * cell
    pos_li_uw[f] = pos_li_uw[f-1] + d

z_surf  = pos_li_uw[:, :, 2]
dz_net  = z_surf[-1] - z_surf[0]
dz_max  = np.max(np.abs(z_surf - z_surf[0]), axis=0)
migr_idx = np.where(np.abs(dz_net) > MIGRATE_THR)[0]

np.savez(f'{RES_DIR}/li_migration.npz',
         time_ps=time_ps, z_surf=z_surf,
         dz_net=dz_net, dz_max=dz_max,
         migrating=migr_idx)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: spaghetti plot
ax = axes[0]
for j in range(z_surf.shape[1]):
    is_m  = j in migr_idx
    ax.plot(time_ps, z_surf[:, j],
            lw=2.0 if is_m else 0.5,
            alpha=1.0 if is_m else 0.2,
            color=RED if is_m else 'steelblue',
            label=f'Li[{li_surf[j]}] Δz={dz_net[j]:+.1f}Å' if is_m else '')
ax.axhline(LI_SURF_ZMAX, color='gray', ls='--', lw=1.0, label='Slab top')
ax.axhline(LI_SURF_ZMIN, color='gray', ls=':', lw=1.0, label='Slab bot')
ax.axhspan(LI_SURF_ZMIN, LI_SURF_ZMAX, alpha=0.05, color='blue')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('z (Å)')
ax.set_title('Surface Li — z Trajectories')
ax.legend(fontsize=8)
ax.set_xlim(0, time_ps[-1])

# Panel 2: histogram
ax = axes[1]
ax.hist(dz_net, bins=15, color=BLUE, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', ls='--', lw=1.0)
ax.axvline( MIGRATE_THR, color=RED, ls=':', lw=1.3,
            label=f'±{MIGRATE_THR} Å threshold')
ax.axvline(-MIGRATE_THR, color=RED, ls=':', lw=1.3)
ax.set_xlabel('Net Δz (Å)')
ax.set_ylabel('Count')
ax.set_title('Net z-Displacement Distribution')
ax.legend()

# Panel 3: migrating atoms detail
ax = axes[2]
if len(migr_idx) > 0:
    colors_m = [RED, GOLD, GREEN, BLUE]
    for k, j in enumerate(migr_idx):
        ax.plot(time_ps, z_surf[:, j],
                lw=1.8, color=colors_m[k % len(colors_m)],
                label=f'Li[{li_surf[j]}]  Δz={dz_net[j]:+.2f} Å')
    ax.axhline(LI_SURF_ZMAX, color='gray', ls='--', lw=1.0)
    ax.axhline(LI_SURF_ZMIN, color='gray', ls=':', lw=1.0)
    ax.axhspan(LI_SURF_ZMIN, LI_SURF_ZMAX, alpha=0.06, color='blue')
    ax.set_title('Migrating Li — Detail')
    ax.legend()
else:
    ax.text(0.5, 0.5, f'No Li migrated > {MIGRATE_THR} Å',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='gray')
    ax.set_title('Migrating Li — Detail')

ax.set_xlabel('Time (ps)')
ax.set_ylabel('z (Å)')
ax.set_xlim(0, time_ps[-1])

fig.suptitle('Li⁺ Transport at LRMO Surface  (NVT 100 ps)', fontsize=13)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig6_nvt_li_migration.png', bbox_inches='tight')
plt.close(fig)
print("  Saved fig6_nvt_li_migration.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("NVT ANALYSIS COMPLETE")
print(f"{'='*55}")
print(f"  Trajectory   : {n_frames} frames  ({n_frames * DT_PS:.1f} ps)")
print(f"  Production   : {len(prod_range)} frames  ({len(prod_range)*DT_PS:.1f} ps)")
print(f"  D(Li+)       : {D:.3e} m²/s")
print(f"  ⟨N_coord⟩    : {cn_total.mean():.2f} ± {cn_total.std():.2f} (total Li–O)")
print(f"  ⟨N_solvent⟩  : {cn_solv.mean():.2f} ± {cn_solv.std():.2f} (EC/DMC O)")
print(f"  min d(O–O)   : {min_oo.min():.3f} Å")
print(f"  Migrating Li : {len(migr_idx)}")
print(f"\n  Figures → {FIG_DIR}/")
print(f"  Data    → {RES_DIR}/")
