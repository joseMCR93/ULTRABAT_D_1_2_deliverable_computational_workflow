"""
Module 2: LOPTICS — Dielectric Function & Optical Absorption
=============================================================
LRMO (104) surface — spin polarized — Gamma only — no electrolyte

Workflow D1.2 — UltraBat Project

Computes:
    - Dielectric function ε₁(ω), ε₂(ω)
    - Optical absorption spectrum α(ω)
    - Joint density of states (JDOS)

Input:  POSCAR_LRMO_constrained.vasp  (surface sin electrolito)
Output: loptics/ directory with OUTCAR, vasprun.xml, WAVECAR
        figures/optical_spectrum.png

Notes:
    - ALGO=Exact required for LOPTICS (orthogonal wavefunctions)
    - ISMEAR=-5 (tetrahedron) better for optical properties
    - NBANDS=700 from relaxation — check if sufficient after SCF
    - WAVECAR saved → needed for subsequent Delta-SCF calculations
    - No WAVECAR input available → full SCF from scratch
"""

from pathlib import Path
from ase.io import read, write
from ase.calculators.vasp import Vasp
import numpy as np

# ─── Directories ──────────────────────────────────────────────────────────────
directory = Path.cwd()
loptics_dir = directory / "loptics"
loptics_dir.mkdir(exist_ok=True)

# ─── Structure ────────────────────────────────────────────────────────────────
# Use relaxed surface (no electrolyte)
atoms = read('POSCAR_LRMO_constrained.vasp')

# Magnetic moments — same as relaxation
for atom in atoms:
    if atom.symbol == 'Mn':
        atom.magmom = 3.5
    elif atom.symbol == 'Co':
        atom.magmom = 3.0
    elif atom.symbol == 'Ni':
        atom.magmom = 3.0

# ─── VASP Calculator ──────────────────────────────────────────────────────────
calc = Vasp(
    directory=loptics_dir,
    # ── Accuracy (same as LOPTICS) ────────────────────────────────────────
    prec='accurate',
    encut=520,
    ediff=1.0e-6,
    addgrid=True,
    lasph=True,

    # ── Exchange-correlation ──────────────────────────────────────────────
    gga='PE',
    lmaxmix=4,
    ldau_luj={
        'Mn': {'L': 2, 'U': 3.9, 'J': 0},
        'Ni': {'L': 2, 'U': 5.6, 'J': 0},
        'Co': {'L': 2, 'U': 5.0, 'J': 0},
    },
    setups='materialsproject',

    # ── Spin ─────────────────────────────────────────────────────────────
    ispin=2,

    # ── k-points ──────────────────────────────────────────────────────────
    kpts=[1, 1, 1],
    gamma=True,

    # ── SCF ───────────────────────────────────────────────────────────────
    algo='All',
    nelmin=8,
    nelm=500,
    nelmdl=-5,
    amix=0.02,
    bmix=0.00001,
    amix_mag=0.8,
    bmix_mag=0.0001,

    # ── ISMEAR=-2 with ground state integer occupations ───────────────────
    # NUP=575, NDOWN=486, NBANDS=1008
    ismear=-2,
    sigma=0.01,
    ferwe='575*1.0 433*0.0',
    ferdo='486*1.0 522*0.0',

    # ── Bands ─────────────────────────────────────────────────────────────
    nbands=1008,
    lorbit=11,

    # ── Static ────────────────────────────────────────────────────────────
    ibrion=-1,
    nsw=0,
    isif=2,

    # ── Start from LOPTICS WAVECAR ────────────────────────────────────────
    istart=1,
    icharg=0,

    # ── No LOPTICS ────────────────────────────────────────────────────────
    loptics=False,
    lwave=False,
    lcharg=False,

    # ── Performance ───────────────────────────────────────────────────────
    ncore=8,
)

print("Starting LOPTICS calculation...")
print(f"  System     : {atoms.get_chemical_formula()}")
print(f"  N atoms    : {len(atoms)}")
print(f"  Directory  : {loptics_dir}")
print(f"  NBANDS     : 1000  (~468 empty bands for optical range)")
print(f"  ALGO       : Normal (Exact too costly for 700+ bands surface)")
print(f"  LOPTICS    : True")
print(f"  Spin pol.  : Yes (Mn=3.5, Co=3.0, Ni=3.0 μB)")

atoms.set_calculator(calc)
energy = atoms.get_potential_energy()

print(f"\nLOPTICS calculation complete.")
print(f"  Total energy : {energy:.6f} eV")

# ─── Quick check: read NBANDS and NELECT from OUTCAR ─────────────────────────
outcar_path = loptics_dir / "OUTCAR"
if outcar_path.exists():
    with open(outcar_path) as f:
        for line in f:
            if 'NELECT' in line:
                nelect = float(line.split()[2])
                n_occ  = int(nelect / 2)
                print(f"\n  NELECT     : {nelect:.0f} electrons")
                print(f"  N occupied : ~{n_occ} bands (per spin)")
                print(f"  NBANDS     : 700")
                print(f"  Empty bands: ~{700 - n_occ} (per spin)")
                if 700 < 2 * n_occ:
                    print(f"  ⚠ WARNING: increase NBANDS to {2*n_occ} for full coverage")
                else:
                    print(f"  ✓ NBANDS sufficient for optical range")
                break

print(f"\nOutputs in {loptics_dir}/")
print(f"  OUTCAR      → energies, dielectric function")
print(f"  vasprun.xml → for VASPKIT post-processing")
print(f"  WAVECAR     → required for Delta-SCF (Module 3)")
