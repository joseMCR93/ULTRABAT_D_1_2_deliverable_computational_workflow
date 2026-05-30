"""
Module 1: NVT Production Run
==============================
Workflow D1.2 — UltraBat Project

Uses the equilibrated cell volume from the NPT run to set up a
canonical (NVT) production simulation at constant volume.

Strategy:
    1. Read NPT summary file → average Lz over last 20% of trajectory
    2. Rescale simulation cell to equilibrated Lz (x,y fixed)
    3. Read last NPT frame as starting structure
    4. Run 100 ps NVT with Nosé-Hoover thermostat (Langevin)

Input:
    npt_results/summary_npt.txt   → equilibrated Lz
    npt_results/interface_npt.xyz → last frame as starting structure
    OR poscar_path directly if NPT xyz not available

Output:
    nvt_results/interface_nvt.xyz     → full trajectory (every 100 steps = 0.1 ps)
    nvt_results/summary_nvt.txt       → thermodynamic log
    nvt_results/checkpoint_XXXXX.xyz  → checkpoints every 10000 steps
"""

import torch
import os
import numpy as np
import logging
from mace.calculators import MACECalculator
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.constraints import FixAtoms
from ase import units

torch.set_default_dtype(torch.float64)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ─── Parameters ───────────────────────────────────────────────────────────────
NPT_DIR      = './npt_results'   # path to NPT output directory
NVT_DIR      = './nvt_results'   # path for NVT output
MODEL_PATH   = './mace-mh-1.model'  # download from https://github.com/ACEsuit/mace-mp
MODEL_HEAD   = 'matpes_r2scan'
DEVICE       = 'cuda'
TEMP         = 300          # K
STEPS        = 100_000      # 100 ps at 1 fs/step
TIMESTEP_FS  = 1.0          # fs
FRICTION     = 0.01         # fs⁻¹ — Langevin friction (lower than equil for NVT)
SAVE_INTERVAL = 50         # save every 100 steps = 0.1 ps → 1000 frames total
EQUIL_FRAC   = 0.2          # use last 80% of NPT for Lz average


# ─── Step 1: Read equilibrated Lz from NPT ───────────────────────────────────
def get_equilibrated_lz_from_summary(npt_summary_path, equil_frac=EQUIL_FRAC):
    """Parse Lz from summary_npt.txt (column 6)."""
    data = np.loadtxt(npt_summary_path, skiprows=1)
    lz_col  = data[:, 6]
    n_prod  = int(len(lz_col) * (1 - equil_frac))
    lz_prod = lz_col[n_prod:]
    return lz_prod.mean(), lz_prod.std(), len(data), len(lz_prod)


def get_equilibrated_lz_from_xyz(npt_xyz_path, equil_frac=EQUIL_FRAC):
    """
    Parse Lz directly from extXYZ comment lines.
    Format: Lattice="ax 0 0 bx by 0 0 0 LZ" ...
    LZ is the 9th value (index 8) in the Lattice string.
    """
    import re
    lz_values = []
    with open(npt_xyz_path) as f:
        for line in f:
            if 'Lattice=' in line:
                m = re.search(r'Lattice="([^"]+)"', line)
                if m:
                    vals = m.group(1).split()
                    lz_values.append(float(vals[8]))
    lz = np.array(lz_values)
    n_prod  = int(len(lz) * (1 - equil_frac))
    lz_prod = lz[n_prod:]
    return lz_prod.mean(), lz_prod.std(), len(lz), len(lz_prod)


def get_equilibrated_lz(npt_dir, equil_frac=EQUIL_FRAC):
    """
    Try summary file first, fall back to parsing xyz directly.
    Returns (lz_mean, lz_std).
    """
    summary = os.path.join(npt_dir, 'summary_npt.txt')
    xyz     = os.path.join(npt_dir, 'interface_npt.xyz')

    if os.path.exists(summary):
        lz_mean, lz_std, n_total, n_prod = get_equilibrated_lz_from_summary(
            summary, equil_frac)
        source = "summary_npt.txt"
    elif os.path.exists(xyz):
        logging.warning("summary_npt.txt not found — parsing Lz directly from xyz.")
        lz_mean, lz_std, n_total, n_prod = get_equilibrated_lz_from_xyz(
            xyz, equil_frac)
        source = "interface_npt.xyz"
    else:
        raise FileNotFoundError(
            f"Neither summary_npt.txt nor interface_npt.xyz found in {npt_dir}")

    logging.info(f"Lz source      : {source}")
    logging.info(f"NPT frames     : {n_total} total, {n_prod} production")
    logging.info(f"Equilibrated Lz: {lz_mean:.4f} ± {lz_std:.4f} Å  "
                 f"(last {100*(1-equil_frac):.0f}%)")
    return lz_mean, lz_std


# ─── Step 2: Rescale cell to equilibrated Lz ─────────────────────────────────
def set_equilibrated_cell(atoms, lz_target):
    """
    Set cell Lz to lz_target, keeping Lx, Ly, and all angles fixed.
    Rescales atomic positions in z proportionally.
    """
    cell = atoms.cell.copy()
    lz_current = atoms.cell.lengths()[2]
    scale = lz_target / lz_current

    # Scale cell
    new_cell = cell.copy()
    new_cell[2, 2] *= scale
    # Also scale the off-diagonal z-component if present
    new_cell[0, 2] *= scale
    new_cell[1, 2] *= scale

    # Scale z-positions of FREE atoms only (fixed atoms stay put)
    fixed_set = set()
    for c in atoms.constraints:
        if hasattr(c, 'index'):
            fixed_set.update(c.index)

    pos = atoms.positions.copy()
    for i in range(len(atoms)):
        if i not in fixed_set:
            pos[i, 2] *= scale

    atoms.set_cell(new_cell)
    atoms.positions = pos

    logging.info(f"Cell rescaled: Lz {lz_current:.4f} → {lz_target:.4f} Å  "
                 f"(scale factor {scale:.6f})")
    return atoms


# ─── Main NVT run ─────────────────────────────────────────────────────────────
def run_nvt(poscar_path=None):

    os.makedirs(NVT_DIR, exist_ok=True)

    # ── Load structure: prefer last NPT frame ─────────────────────────────────
    npt_traj = os.path.join(NPT_DIR, 'interface_npt.xyz')
    npt_summary = os.path.join(NPT_DIR, 'summary_npt.txt')

    if os.path.exists(npt_traj):
        logging.info(f"Loading last NPT frame from {npt_traj}...")
        atoms = read(npt_traj, index=-1)
        logging.info("Last NPT frame loaded as starting structure.")
    elif poscar_path and os.path.exists(poscar_path):
        logging.warning(f"NPT trajectory not found. Loading from {poscar_path}")
        atoms = read(poscar_path)
    else:
        raise FileNotFoundError(
            "Neither NPT trajectory nor POSCAR found. "
            "Provide poscar_path or run NPT first."
        )

    logging.info(f"System: {atoms.get_chemical_formula()}, {len(atoms)} atoms")

    # ── Get equilibrated Lz from NPT ─────────────────────────────────────────
    lz_mean, lz_std = get_equilibrated_lz(NPT_DIR)

    # Rescale cell to equilibrated volume
    atoms = set_equilibrated_cell(atoms, lz_mean)

    # ── Restore constraints (FixAtoms from POSCAR Selective Dynamics) ─────────
    # ASE reads constraints from extxyz if saved — verify they're present
    n_fixed = sum(len(c.index) for c in atoms.constraints if hasattr(c, 'index'))
    logging.info(f"Fixed atoms: {n_fixed}")

    if n_fixed == 0:
        logging.warning("No constraints found — re-applying from Z-range fallback.")
        positions_z = atoms.positions[:, 2]
        fixed_indices = [i for i, z in enumerate(positions_z) if 15.9 <= z <= 21.9]
        atoms.set_constraint(FixAtoms(indices=fixed_indices))
        logging.info(f"Fixed {len(fixed_indices)} slab core atoms (Z: 15.9–21.9 Å).")

    # ── MACE calculator ────────────────────────────────────────────────────────
    calc = MACECalculator(
        model_paths=MODEL_PATH,
        device=DEVICE,
        default_dtype='float64',
        head=MODEL_HEAD
    )
    atoms.calc = calc

    # ── Initialize velocities from Maxwell-Boltzmann at TEMP ──────────────────
    # Use last NPT frame velocities if available, else reinitialize
    try:
        momenta = atoms.get_momenta()
        if np.all(momenta == 0):
            raise ValueError("Zero momenta")
        logging.info("Using velocities from last NPT frame.")
    except Exception:
        logging.info(f"Initializing velocities from Maxwell-Boltzmann at {TEMP} K.")
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMP)
        Stationary(atoms)
        ZeroRotation(atoms)

    # ── NVT: Langevin thermostat ───────────────────────────────────────────────
    # Lower friction than equilibration for better dynamics sampling
    dyn = Langevin(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMP,
        friction=FRICTION / units.fs,
    )

    # ── Output files ──────────────────────────────────────────────────────────
    traj_file    = os.path.join(NVT_DIR, 'interface_nvt.xyz')
    summary_file = os.path.join(NVT_DIR, 'summary_nvt.txt')

    with open(summary_file, 'w') as f:
        f.write("Step\tTime_ps\tPot_E(eV)\tKin_E(eV)\tTot_E(eV)\t"
                "Temp(K)\tVol(A^3)\tLz(A)\tFixed_drift(A)\n")

    # Reference positions for drift monitor
    fixed_indices_all = []
    for c in atoms.constraints:
        if hasattr(c, 'index'):
            fixed_indices_all.extend(c.index)
    fixed_pos_ref = atoms.positions[fixed_indices_all].copy()

    def status_logger():
        step      = dyn.nsteps
        time_ps   = step * TIMESTEP_FS / 1000.0
        pe        = atoms.get_potential_energy()
        ke        = atoms.get_kinetic_energy()
        temp_curr = atoms.get_temperature()
        z_len     = atoms.cell.lengths()[2]
        vol       = atoms.get_volume()
        drift     = np.max(np.abs(
            atoms.positions[fixed_indices_all] - fixed_pos_ref
        )) if fixed_indices_all else 0.0

        write(traj_file, atoms, append=True, format='extxyz')

        with open(summary_file, 'a') as f:
            f.write(f"{step}\t{time_ps:.3f}\t{pe:.6f}\t{ke:.6f}\t{pe+ke:.6f}\t"
                    f"{temp_curr:.2f}\t{vol:.2f}\t{z_len:.4f}\t{drift:.6f}\n")

        logging.info(
            f"Step {step:6d} ({time_ps:5.1f} ps) | "
            f"T={temp_curr:.1f}K | E={pe+ke:.4f}eV | "
            f"Lz={z_len:.4f}Å | drift={drift:.4f}Å"
        )

        if step % 10000 == 0 and step > 0:
            ckpt = os.path.join(NVT_DIR, f'checkpoint_{step}.xyz')
            write(ckpt, atoms)
            logging.info(f"Checkpoint saved: {ckpt}")

    dyn.attach(status_logger, interval=SAVE_INTERVAL)

    # ── Run ───────────────────────────────────────────────────────────────────
    logging.info(
        f"Starting NVT production: T={TEMP}K, "
        f"steps={STEPS} ({STEPS*TIMESTEP_FS/1000:.0f} ps), "
        f"Lz={lz_mean:.4f}Å (fixed)"
    )
    dyn.run(STEPS)
    logging.info("NVT production complete.")

    # ── Final summary ─────────────────────────────────────────────────────────
    data = np.loadtxt(summary_file, skiprows=1)
    logging.info("=" * 55)
    logging.info("NVT PRODUCTION COMPLETE")
    logging.info("=" * 55)
    logging.info(f"  Total time    : {STEPS * TIMESTEP_FS / 1000:.0f} ps")
    logging.info(f"  Saved frames  : {len(data)}")
    logging.info(f"  Mean T        : {data[:,5].mean():.1f} ± {data[:,5].std():.1f} K")
    logging.info(f"  Mean E_tot    : {data[:,4].mean():.4f} ± {data[:,4].std():.4f} eV")
    logging.info(f"  Lz (fixed)    : {lz_mean:.4f} Å")
    logging.info(f"  Max drift     : {data[:,8].max():.4f} Å")
    logging.info(f"  Trajectory    : {traj_file}")


if __name__ == "__main__":
    run_nvt(poscar_path='Structure_LRMO_electrolite.vasp')
