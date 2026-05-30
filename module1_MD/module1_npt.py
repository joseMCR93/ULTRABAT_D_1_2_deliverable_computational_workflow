import torch
import os
import numpy as np
import logging
from mace.calculators import MACECalculator
from ase.io import read, write
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.constraints import FixAtoms
from ase import units

torch.set_default_dtype(torch.float64)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NPTBerendsenFixedCore(Inhomogeneous_NPTBerendsen):
    """NPT Berendsen que no desplaza los átomos fijados al reescalar la celda."""
    def scale_positions_and_cell(self):
        fixed_mask = self._get_fixed_mask()
        pos_before = self.atoms.positions.copy()
        super().scale_positions_and_cell()
        pos_after = self.atoms.positions.copy()
        pos_after[fixed_mask] = pos_before[fixed_mask]
        self.atoms.positions = pos_after

    def _get_fixed_mask(self):
        mask = np.zeros(len(self.atoms), dtype=bool)
        for constraint in self.atoms.constraints:
            if hasattr(constraint, 'index'):
                mask[constraint.index] = True
        return mask


def run_mace_interface_npt(poscar_path, model_path, device='cuda', temp=300,
                           pressure_bar=1.0, steps=50000, output_dir='npt_results',
                           model_head='matpes_r2scan', equil_steps=2000):

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load structure
    atoms = read(poscar_path)
    logging.info(f"Read structure: {atoms.get_chemical_formula()}")
    logging.info(f"Total atoms: {len(atoms)}")

    # 2. Configure MACE
    try:
        calc = MACECalculator(
            model_paths=model_path,
            device=device,
            default_dtype='float64',
            head=model_head
        )
    except Exception as e:
        logging.error(f"Failed to load MACE model: {e}")
        raise
    atoms.calc = calc

    # 3. Respetar los FixAtoms ya definidos en el POSCAR (Selective Dynamics)
    #    La estructura con electrolito ya tiene 58 átomos marcados como F F F
    #    en el VASP, y ASE los lee automáticamente como FixAtoms.
    #    NO redefinir el rango Z manualmente — usar los del POSCAR directamente.
    n_fixed = sum(
        len(c.index) for c in atoms.constraints if hasattr(c, 'index')
    )
    logging.info(f"Constraints from POSCAR (Selective Dynamics): {n_fixed} fixed atoms")

    if n_fixed == 0:
        # Fallback: fijar manualmente el núcleo del slab (TM bulk)
        logging.warning("No constraints found in POSCAR — applying manual Z-range fix.")
        positions_z = atoms.positions[:, 2]
        fixed_indices = [i for i, z in enumerate(positions_z) if 15.9 <= z <= 21.9]
        atoms.set_constraint(FixAtoms(indices=fixed_indices))
        logging.info(f"Fixed {len(fixed_indices)} slab core atoms (Z: 15.9-21.9 Å).")

    # 4. Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    Stationary(atoms)
    ZeroRotation(atoms)

    # 5. NVT equilibration — crucial con solvente para evitar explosión inicial
    logging.info(f"Running {equil_steps} steps NVT equilibration (Langevin)...")
    dyn_eq = Langevin(atoms, 0.5 * units.fs,   # timestep más corto para solvente
                      temperature_K=temp, friction=0.02)
    dyn_eq.run(equil_steps)
    logging.info("NVT equilibration complete.")

    # 6. NPT — escala solo Z para equilibrar densidad del solvente
    pressure_au = pressure_bar * units.bar
    # Compresibilidad del electrolito orgánico (~EC/DMC): ~8e-5 bar⁻¹
    compressibility_au = 8.0e-5 / units.bar

    dyn = NPTBerendsenFixedCore(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=temp,
        taut=100.0 * units.fs,
        pressure_au=pressure_au,
        taup=2000.0 * units.fs,   # más lento para solvente
        compressibility_au=compressibility_au,
        mask=(0, 0, 1)
    )

    # 7. Output
    traj_file = os.path.join(output_dir, "interface_npt.xyz")
    summary_file = os.path.join(output_dir, "summary_npt.txt")

    with open(summary_file, 'w') as f:
        f.write("Step\tPot_E(eV)\tKin_E(eV)\tTot_E(eV)\tTemp(K)\tVol(A^3)\tLz(A)\tFixed_drift(A)\n")

    # Referencia para monitorizar drift de átomos fijos
    fixed_indices_all = []
    for c in atoms.constraints:
        if hasattr(c, 'index'):
            fixed_indices_all.extend(c.index)
    fixed_pos_ref = atoms.positions[fixed_indices_all].copy()

    def status_logger():
        step = dyn.nsteps
        pe   = atoms.get_potential_energy()
        ke   = atoms.get_kinetic_energy()
        temp_curr = atoms.get_temperature()
        z_len = atoms.cell.lengths()[2]
        vol   = atoms.get_volume()
        drift = np.max(np.abs(atoms.positions[fixed_indices_all] - fixed_pos_ref)) \
                if fixed_indices_all else 0.0

        write(traj_file, atoms, append=True, format='extxyz')

        with open(summary_file, 'a') as f:
            f.write(f"{step}\t{pe:.6f}\t{ke:.6f}\t{pe+ke:.6f}\t"
                    f"{temp_curr:.2f}\t{vol:.2f}\t{z_len:.4f}\t{drift:.6f}\n")

        logging.info(f"Step {step} | T={temp_curr:.1f}K | Lz={z_len:.4f}Å | "
                     f"Vol={vol:.2f}Å³ | Fixed drift={drift:.4f}Å")

        if step % 5000 == 0 and step > 0:
            write(os.path.join(output_dir, f'checkpoint_{step}.xyz'), atoms)

    dyn.attach(status_logger, interval=500)

    # 8. Run
    logging.info(f"Starting NPT production: T={temp}K, P={pressure_bar}bar, steps={steps}")
    dyn.run(steps)


if __name__ == "__main__":
    run_mace_interface_npt(
        poscar_path='Structure_LRMO_electrolite.vasp',
        model_path='mace-mh-1.model',
        device='cuda'
    )
