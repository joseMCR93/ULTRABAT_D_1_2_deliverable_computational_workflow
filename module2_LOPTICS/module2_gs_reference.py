"""
Module 3: Ground State Reference Calculation (ISMEAR=-2)
=========================================================
Workflow D1.2 — UltraBat Project

Computes the ground state energy with ISMEAR=-2 (integer occupations)
to provide a consistent energy reference for Delta-SCF calculations.

This fixes the ~5 eV offset observed when using the LOPTICS OUTCAR
(ISMEAR=0) as reference for Delta-SCF (ISMEAR=-2) energies.

Usage:
    python loptics_gs_reference.py
Output:
    loptics/OUTCAR  → copy as OUTCAR_gs to the Delta-SCF directory
    loptics/PROCAR  → copy as PROCAR_gs to the Delta-SCF directory
"""

from pathlib import Path
from ase.io import read
from ase.calculators.vasp import Vasp

# ─── Directories ──────────────────────────────────────────────────────────────
directory   = Path.cwd()
loptics_dir = directory / "gs_reference"
loptics_dir.mkdir(exist_ok=True)

# ─── Occupation parameters ────────────────────────────────────────────────────
# NUP and NDOWN from the LOPTICS OUTCAR (grep NELECT or check NBANDS spin split)
NUP    = 575    # occupied spin-up bands
NDOWN  = 486    # occupied spin-down bands
NBANDS = 1008   # must match LOPTICS and Delta-SCF calculations

# Build FERWE and FERDO as Python lists of floats
# ASE requires lists — VASP compressed notation ('575*1.0') is NOT supported
ferwe = [1.0] * NUP   + [0.0] * (NBANDS - NUP)    # 1008 floats
ferdo = [1.0] * NDOWN + [0.0] * (NBANDS - NDOWN)  # 1008 floats

print(f"Occupation vectors:")
print(f"  FERWE: {NUP} occupied + {NBANDS - NUP} empty  ({len(ferwe)} total)")
print(f"  FERDO: {NDOWN} occupied + {NBANDS - NDOWN} empty  ({len(ferdo)} total)")

# ─── Structure ────────────────────────────────────────────────────────────────
atoms = read('POSCAR_LRMO_constrained.vasp')

for atom in atoms:
    if atom.symbol == 'Mn':
        atom.magmom = 3.5
    elif atom.symbol == 'Co':
        atom.magmom = 3.0
    elif atom.symbol == 'Ni':
        atom.magmom = 3.0

# ─── VASP Calculator ──────────────────────────────────────────────────────────
calc = Vasp(
    directory=str(loptics_dir),

    # ── Accuracy ──────────────────────────────────────────────────────────
    prec='accurate',
    encut=520,
    ediff=1.0e-6,
    addgrid=True,
    lasph=True,
    # ── Exchange-correlation ───────────────────────────────────────────────
    gga='PE',
    lmaxmix=4,
    ldau_luj={
        'Mn': {'L': 2, 'U': 3.9, 'J': 0},
        'Ni': {'L': 2, 'U': 5.6, 'J': 0},
        'Co': {'L': 2, 'U': 5.0, 'J': 0},
    },
    setups='materialsproject',

    # ── Spin ──────────────────────────────────────────────────────────────
    ispin=2,

    # ── k-points ──────────────────────────────────────────────────────────
    kpts=[1, 1, 1],
    gamma=True,

    # ── SCF ───────────────────────────────────────────────────────────────
    algo='All',
    nelmin=8,
    nelm=1000,
    nelmdl=-5,
    amix=0.02,
    bmix=0.00001,
    amix_mag=0.8,
    bmix_mag=0.0001,

    # ── KEY: ISMEAR=-2 with ground state integer occupations ───────────────
    # Same scheme as Delta-SCF → consistent energy reference
    # ferwe/ferdo must be Python lists of floats (not VASP compressed strings)
    ismear=-2,
    sigma=0.01,
    ferwe=ferwe,
    ferdo=ferdo,

    # ── Bands ─────────────────────────────────────────────────────────────
    nbands=NBANDS,
    lorbit=11,      # write PROCAR for orbital character

    # ── Static ────────────────────────────────────────────────────────────
    ibrion=-1,
    nsw=0,
    isif=2,

    # ── Start from LOPTICS WAVECAR (much faster convergence) ───────────────
    istart=1,
    icharg=0,

    # ── No LOPTICS needed for energy reference ────────────────────────────
    loptics=False,
    lwave=False,
    lcharg=False,

    # ── Performance ───────────────────────────────────────────────────────
    ncore=8,
)

# ─── Assign calculator and run ────────────────────────────────────────────────
atoms.set_calculator(calc)

print("\nStarting ground state reference calculation (ISMEAR=-2)...")
print(f"  System  : {atoms.get_chemical_formula()}")
print(f"  N atoms : {len(atoms)}")
print(f"  ISMEAR  : -2  (integer occupations, matches Delta-SCF)")
print(f"  ISTART  : 1  (reading WAVECAR from LOPTICS)")
print(f"  NUP     : {NUP}   NDOWN : {NDOWN}   NBANDS : {NBANDS}")

energy = atoms.get_potential_energy()

print(f"\nGround state reference complete.")
print(f"  E_gs (ISMEAR=-2) = {energy:.6f} eV")
print(f"\nNext steps:")
print(f"  cp {loptics_dir}/OUTCAR /path/to/deltascf/OUTCAR_gs")
print(f"  cp {loptics_dir}/PROCAR /path/to/deltascf/PROCAR_gs")
print(f"  python module3_deltascf_spectrum.py")
