"""
Module 3: Delta-SCF Optical Absorption Spectrum
=================================================
Workflow D1.2 — UltraBat Project

Integrates the existing Delta-SCF analysis (energies, TDM, orbital character)
with Gaussian convolution to build the corrected optical absorption spectrum.

Computes:
    - DeltaE  = E(excited) - E(gs)       [Delta-SCF excitation energy]
    - KS gap  = E_high - E_low           [Independent-particle gap from PROCAR]
    - Eb(opt) = KS_gap - DeltaE          [Optical binding energy / excitonic shift]
    - alpha(w)= Sum_i |TDM|^2_i * G(w-DeltaE_i, sigma) [Convoluted spectrum]

Input:
    OUTCAR_gs                  -> ground state reference energy
    {band_a}_{band_b}/OUTCAR   -> excited state energies
    {band_a}_{band_b}/TDM_COMPONENTS_UP.dat -> transition dipole moments
    PROCAR (or PROCAR_gs)      -> orbital character of each band

Output:
    figures/delta_scf_spectrum.png
    figures/delta_scf_orbital_character.png
    results/delta_scf_data.npz
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

# Parameters
OUTCAR_GS   = 'OUTCAR_gs'
SIGMA       = 0.10    # eV Gaussian broadening
E_MIN       = 0.0
E_MAX       = 10.0
N_GRID      = 2000
LOPTICS_DAT = 'absorption_spectrum_absorption.dat'  # SUMO output file

BLUE  = '#2E75B6'
RED   = '#E63946'
GREEN = '#2A9D8F'
GOLD  = '#E9C46A'
GRAY  = '#888888'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)


# ==============================================================================
# SECTION 1: DATA EXTRACTION
# ==============================================================================

def get_sigma_zero_energy(file_path):
    if not os.path.exists(file_path):
        return None
    final_energy = None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if "energy(sigma->0)" in line:
                    final_energy = float(line.split()[-1])
        return final_energy
    except:
        return None


def get_tdm_total(folder_path):
    """
    Reads TDM_COMPONENTS_UP.dat.
    Skips header (#) and Kpoint lines, returns Total (column 4) of first data line.
    Units: Debye^2
    """
    file_path = os.path.join(folder_path, 'TDM_COMPONENTS_UP.dat')
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data_lines = [l for l in lines
                      if l.strip()
                      and not l.strip().startswith('#')
                      and not l.strip().startswith('Kpoint')]
        if data_lines:
            parts = data_lines[0].strip().split()
            if len(parts) >= 4:
                return float(parts[3])
    except:
        return None
    return None


def get_band_info(procar_path, target_band):
    if not os.path.exists(procar_path):
        return "No PROCAR", None
    found_band  = False
    last_tot    = None
    band_energy = None
    try:
        with open(procar_path, 'r') as f:
            for line in f:
                if "band" in line:
                    parts = line.split()
                    try:
                        if int(parts[1]) == target_band:
                            found_band  = True
                            last_tot    = None
                            for i, p in enumerate(parts):
                                if p == "energy" and i + 1 < len(parts):
                                    band_energy = float(parts[i + 1])
                        else:
                            if found_band:
                                break
                            found_band = False
                    except:
                        continue
                if found_band and line.strip().startswith("tot"):
                    data = line.split()
                    if len(data) >= 11:
                        last_tot = data
        if last_tot:
            d     = last_tot
            s_val = float(d[1])
            p_val = float(d[2]) + float(d[3]) + float(d[4])
            d_val = sum(float(x) for x in d[5:10])
            tot   = float(d[-1])
            if tot > 0.001:
                char = (f"s:{s_val/tot*100:.0f}% "
                        f"p:{p_val/tot*100:.0f}% "
                        f"d:{d_val/tot*100:.0f}%")
                return char, band_energy
    except:
        return "Err", None
    return "Not found", None


# Load ground state energy
e_gs = get_sigma_zero_energy(OUTCAR_GS)
if e_gs is None:
    raise FileNotFoundError(f"Cannot read ground state energy from '{OUTCAR_GS}'")
print(f"Ground state energy: {e_gs:.6f} eV")

# Load PROCAR reference
if   os.path.isfile('PROCAR'):    procar_ref = 'PROCAR'
elif os.path.isfile('PROCAR_gs'): procar_ref = 'PROCAR_gs'
else:
    fallback = glob.glob('*/PROCAR')
    procar_ref = fallback[0] if fallback else None
print(f"PROCAR reference: {procar_ref or 'NOT FOUND'}")

band_cache = {}
def get_cached_info(band_id):
    if not procar_ref: return "No Ref", None
    if band_id not in band_cache:
        band_cache[band_id] = get_band_info(procar_ref, int(band_id))
    return band_cache[band_id]


# Process all Delta-SCF folders
print("\nProcessing Delta-SCF folders...")
outcar_paths = sorted(glob.glob('*/OUTCAR'))
results = []

for path in outcar_paths:
    e_curr = get_sigma_zero_energy(path)
    if e_curr is None:
        continue
    folder = os.path.dirname(path)
    tdm    = get_tdm_total(folder)

    converged = False
    with open(path) as f:
        for line in f:
            if 'aborting loop because EDIFF is reached' in line:
                converged = True
                break

    try:
        # Format: transition_XX_VYYY_CZZZ
        import re
        m = re.match(r'transition_\d+_V(\d+)_C(\d+)',
                     os.path.basename(folder))
        if not m:
            print(f"  [SKIP] Unrecognised folder name: {folder}")
            continue
        b_low  = int(m.group(1))   # valence band (V)
        b_high = int(m.group(2))   # conduction band (C)
    except:
        continue

    c_high, e_high = get_cached_info(b_high)
    c_low,  e_low  = get_cached_info(b_low)

    delta_e = e_curr - e_gs
    ks_gap  = (e_high - e_low) if (e_high and e_low) else None
    eb_opt  = (ks_gap - delta_e) if ks_gap is not None else None

    results.append({
        'folder'   : folder,
        'delta_e'  : delta_e,
        'ks_gap'   : ks_gap,
        'eb_opt'   : eb_opt,
        'tdm'      : tdm,
        'b_high'   : b_high,
        'b_low'    : b_low,
        'c_high'   : c_high,
        'c_low'    : c_low,
        'converged': converged,
        'label'    : f"V{b_low}->C{b_high}",
    })

results.sort(key=lambda x: x['delta_e'])

# Print table
print(f"\n{'─'*160}")
fmt = "{:<12} {:>10} {:>10} {:>10} {:>8} {:>5} | {:<6} {:<22} | {:<6} {:<22}"
print(fmt.format("Folder", "dE(eV)", "KS(eV)", "Eb(eV)",
                 "|TDM|^2", "Conv", "Band-v", "Character-v", "Band-c", "Character-c"))
print(f"{'─'*160}")
for r in results:
    print(fmt.format(
        r['folder'],
        f"{r['delta_e']:.4f}",
        f"{r['ks_gap']:.4f}"  if r['ks_gap'] else "N/A",
        f"{r['eb_opt']:.4f}"  if r['eb_opt'] else "N/A",
        f"{r['tdm']:.4f}"     if r['tdm']    else "N/A",
        "OK" if r['converged'] else "NO",
        r['b_low'],  r['c_low'],
        r['b_high'], r['c_high'],
    ))
print(f"{'─'*160}")


# ==============================================================================
# SECTION 2: GAUSSIAN CONVOLUTION
# ==============================================================================

# ── Physical filter ──────────────────────────────────────────────────────────
# Filter criterion: dE must be in optical range (0-10 eV)
# Large negative Eb indicates reference energy issue, not used as filter
MAX_DE = 10.0   # eV

all_converged = [r for r in results
                 if r['converged'] and r['tdm'] is not None and r['delta_e'] > 0]

valid    = [r for r in all_converged if r['delta_e'] <= MAX_DE]
rejected = [r for r in all_converged if r['delta_e'] >  MAX_DE]

print(f"\nAll converged     : {len(all_converged)}/{len(results)}")
print(f"Physically valid  : {len(valid)}  (dE <= {MAX_DE} eV)")
if rejected:
    print(f"Rejected          : {len(rejected)} (dE > {MAX_DE} eV):")
    for r in rejected:
        eb_str = f"{r['eb_opt']:.2f}" if r['eb_opt'] is not None else "N/A"
        print(f"  {r['folder']:35s}  dE={r['delta_e']:.2f} eV  Eb={eb_str} eV")

e_grid   = np.linspace(E_MIN, E_MAX, N_GRID)
spectrum = np.zeros(N_GRID)
for r in valid:
    gauss     = np.exp(-0.5 * ((e_grid - r['delta_e']) / SIGMA)**2)
    spectrum += r['tdm'] * gauss

spectrum_norm = spectrum / spectrum.max() if spectrum.max() > 0 else spectrum

# Optional LOPTICS comparison (SUMO .dat file, space-delimited)
loptics_e, loptics_a = None, None
if os.path.isfile(LOPTICS_DAT):
    try:
        # SUMO dat files: space-delimited, may have header lines starting with #
        data = np.loadtxt(LOPTICS_DAT, comments='#')
        loptics_e = data[:, 0]
        loptics_a = data[:, 1] / data[:, 1].max()
        print(f"LOPTICS spectrum loaded from {LOPTICS_DAT}")
        print(f"  Energy range: {loptics_e.min():.2f} - {loptics_e.max():.2f} eV")
    except Exception as e:
        print(f"Could not load {LOPTICS_DAT}: {e}")


# ==============================================================================
# SECTION 3: FIGURES
# ==============================================================================

# Figure 1: Spectrum
fig = plt.figure(figsize=(14, 5.5))
gs_layout = GridSpec(1, 2, figure=fig, wspace=0.35)

# Panel A: absorption spectrum
ax1 = fig.add_subplot(gs_layout[0])
tdm_max = max(r['tdm'] for r in valid)
for r in valid:
    ax1.vlines(r['delta_e'], 0, r['tdm'] / tdm_max,
               color=RED if r['tdm'] > 0.5*tdm_max else BLUE,
               lw=1.8, alpha=0.65)
ax1.plot(e_grid, spectrum_norm, color=RED, lw=2.5,
         label=f'Delta-SCF  (sigma={SIGMA} eV)')
if loptics_e is not None:
    ax1.plot(loptics_e, loptics_a, color=GRAY, lw=1.5,
             ls='--', label='LOPTICS (indep. particle)')
ax1.set_xlabel('Energy (eV)')
ax1.set_ylabel('Absorption (normalized)')
ax1.set_title('Optical Absorption Spectrum\n(Delta-SCF corrected)')
ax1.set_xlim(E_MIN, E_MAX)
ax1.set_ylim(0, 1.2)
ax1.legend()
for r in sorted(valid, key=lambda r: -r['tdm'])[:3]:
    ax1.annotate(r['label'], xy=(r['delta_e'], r['tdm']/tdm_max),
                 xytext=(0, 8), textcoords='offset points',
                 ha='center', fontsize=8, color=RED)

# Panel B: DeltaE vs KS gap
ax2 = fig.add_subplot(gs_layout[1])
valid_ks = [r for r in valid if r['ks_gap'] is not None]
if valid_ks:
    ks_v  = np.array([r['ks_gap']  for r in valid_ks])
    de_v  = np.array([r['delta_e'] for r in valid_ks])
    eb_v  = np.array([r['eb_opt']  for r in valid_ks])
    tdm_v = np.array([r['tdm']     for r in valid_ks])
    sc = ax2.scatter(ks_v, de_v, c=tdm_v, cmap='RdYlBu_r', s=80, zorder=3,
                     norm=Normalize(tdm_v.min(), tdm_v.max()))
    plt.colorbar(sc, ax=ax2, label='|TDM|^2 (Debye^2)')
    lim = [min(ks_v.min(), de_v.min())-0.2, max(ks_v.max(), de_v.max())+0.2]
    ax2.plot(lim, lim, color=GRAY, ls='--', lw=1.2, label='No correction')
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel('KS Gap (eV)')
    ax2.set_ylabel('Delta-SCF DeltaE (eV)')
    ax2.set_title(f'Excitonic Correction\n'
                  f'<Eb> = {eb_v.mean():.3f} +/- {eb_v.std():.3f} eV')
    ax2.legend(fontsize=9)

fig.suptitle('Module 3: Delta-SCF Optical Spectrum -- LRMO Surface', fontsize=13)
fig.tight_layout()
fig.savefig('figures/delta_scf_spectrum.png', bbox_inches='tight')
plt.close(fig)
print("Saved: figures/delta_scf_spectrum.png")


# Figure 2: Orbital character
def parse_pct(char_str, orbital):
    try:
        return float(char_str.split(f'{orbital}:')[1].split('%')[0])
    except:
        return 0.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(valid))
labels_x = [r['label'] for r in valid]

for ax, band_key, char_key, title in [
        (axes[0], 'b_low',  'c_low',  'Valence Band (initial state)'),
        (axes[1], 'b_high', 'c_high', 'Conduction Band (final state)')]:

    s_v = np.array([parse_pct(r[char_key], 's') for r in valid])
    p_v = np.array([parse_pct(r[char_key], 'p') for r in valid])
    d_v = np.array([parse_pct(r[char_key], 'd') for r in valid])

    ax.bar(x, s_v, label='s', color='#AED6F1', edgecolor='white')
    ax.bar(x, p_v, bottom=s_v, label='p', color=GREEN, edgecolor='white')
    ax.bar(x, d_v, bottom=s_v+p_v, label='d', color=RED, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Orbital character (%)')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 110)

fig.suptitle('Orbital Character of Delta-SCF Transitions -- LRMO Surface', fontsize=13)
fig.tight_layout()
fig.savefig('figures/delta_scf_orbital_character.png', bbox_inches='tight')
plt.close(fig)
print("Saved: figures/delta_scf_orbital_character.png")


# Save data
np.savez('results/delta_scf_data.npz',
         e_grid   = e_grid,
         spectrum = spectrum_norm,
         delta_Es = np.array([r['delta_e'] for r in valid]),
         ks_gaps  = np.array([r['ks_gap'] if r['ks_gap'] else np.nan for r in valid]),
         eb_opts  = np.array([r['eb_opt'] if r['eb_opt'] else np.nan for r in valid]),
         tdm_sqs  = np.array([r['tdm']    for r in valid]),
         labels   = np.array([r['label']  for r in valid]),
         sigma    = SIGMA)

print("\n" + "="*55)
print("MODULE 3 COMPLETE")
print("="*55)
print(f"  Valid transitions : {len(valid)}/{len(results)}")
if valid_ks:
    print(f"  <Eb(opt)>         : {eb_v.mean():.3f} +/- {eb_v.std():.3f} eV")
print(f"  dE range          : {min(r['delta_e'] for r in valid):.3f} - "
      f"{max(r['delta_e'] for r in valid):.3f} eV")
print(f"  sigma             : {SIGMA} eV")
print(f"\n  figures/delta_scf_spectrum.png")
print(f"  figures/delta_scf_orbital_character.png")
print(f"  results/delta_scf_data.npz")
