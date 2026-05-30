# UltraBat D1.2 — Computational Workflow
### Automated study of transport mechanisms and optical excitations at the LRMO(104)/LP30 electrolyte interface

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Project: Horizon Europe](https://img.shields.io/badge/Horizon_Europe-101103873-blue)](https://cordis.europa.eu/project/id/101103873)

---

## Overview

This repository contains the computational workflow developed for **Deliverable D1.2** of the [UltraBat](https://cordis.europa.eu/project/id/101103873) Horizon Europe project (Grant No. 101103873), led by the Technical University of Denmark (DTU).

The workflow provides an automated, modular framework for the study of Li⁺ transport mechanisms and optical excitations at the Li₁.₂Mn₀.₅₄Co₀.₁₃Ni₀.₁₃O₂ (LRMO) surface in contact with LP30 electrolyte (1:1 EC/DMC, 1 M LiPF₆). The results serve as computational input for the XFEL experiments described in Task 1.3 and D1.4.

> **Note:** The results included in this repository are intended to demonstrate the functionality of the workflow modules. They do not represent fully converged production calculations.

---

## Workflow structure

```
Module 1 ──────────────────────────────────────────────────────────────
  NPT MD (50 ps, MACE-MH1, 300 K, 1 bar) → equilibrated Lz = 37.34 Å
       │
       ▼
  NVT MD (200 ps, fixed volume) → RDF, MSD, coordination, O₂, Li hopping
       │
       ▼ snapshot
Module 2 ──────────────────────────────────────────────────────────────
  DFT+U relaxation (VASP, PBE+U, Γ-point)
       │
       ▼
  LOPTICS → ε₁(ω), ε₂(ω), α(ω)   +   VASPKIT 713 → |TDM|²
       │                                      │
       ▼ WAVECAR + OUTCAR_gs                  │ TDM list
Module 3 ──────────────────────────────────────────────────────────────
  Δ-SCF × 20 transitions (VASP, ISMEAR=-2, FERWE/FERDO)
       │
       ▼
  Corrected α(ω)  +  O 2p → TM 3d orbital character
       │
       ▼
D1.4 — XFEL experiment simulations
```

---

## Repository structure

```
├── structure/
│   └── Structure_LRMO_electrolite.vasp   # 497-atom interface model (LRMO + LP30)
│
├── module1_MD/
│   ├── plot_md_npt.py                    # NPT simulation (MACE-MH1)
│   ├── module_nvt.py                     # NVT production run
│   ├── module_1_analysis.py              # RDF, MSD, coordination, O₂, Li migration
│   ├── plot_md_nvt.py                    # Thermodynamics plotting utility
│   ├── submit_npt.sh                     # SLURM submission script (NPT)
│   └── submit_nvt.sh                     # SLURM submission script (NVT)
│
├── module2_LOPTICS/
│   ├── module2_loptics.py                # VASP LOPTICS calculator (ASE)
│   ├── loptics_gs_reference.py           # Ground state reference (ISMEAR=-2)
│   └── module2_sumo_optical.sh           # SUMO post-processing (ε₁, ε₂, α)
│
├── module3_DeltaSCF/
│   ├── excitations_submission_fixed.sh   # SLURM: automated Δ-SCF × 20 transitions
│   ├── module3_generate_tdm.sh           # VASPKIT 713: transition dipole moments
│   └── module3_deltascf_spectrum.py      # Spectral analysis + Gaussian broadening
│
├── figures/                              # Publication-quality output figures
│   ├── fig1_nvt_thermodynamics.png
│   ├── fig2_nvt_rdf.png
│   ├── fig3_nvt_msd.png
│   ├── fig4_nvt_coordination.png
│   ├── fig5_nvt_o2.png
│   └── fig6_nvt_li_migration.png
│
└── results/                              # Numerical results (numpy .npz)
    ├── rdf_data.npz
    ├── msd_data.npz
    ├── coordination_data.npz
    ├── o2_data.npz
    ├── li_migration.npz
    └── delta_scf_data.npz
```

---

## Requirements

### Module 1 — Molecular dynamics
```
Python >= 3.10
torch >= 2.0
mace-torch
ase >= 3.22
numpy
matplotlib
```

Install:
```bash
pip install torch mace-torch ase numpy matplotlib
```

The MACE-MH1 model file (`mace-mh-1.model`) must be downloaded separately from the [MACE-MP releases](https://github.com/ACEsuit/mace-mp).

### Module 2 — Optical properties
- [VASP 6.4+](https://www.vasp.at/) with `LOPTICS = .TRUE.`
- [ASE](https://wiki.fysik.dtu.dk/ase/) for input generation
- [SUMO](https://smtg-bham.github.io/sumo/) for post-processing
- [VASPKIT 1.5+](https://vaspkit.com/) for transition dipole moments (task 713)

### Module 3 — Delta-SCF
- VASP 6.4+ (same environment as Module 2)
- ASE
- numpy, matplotlib

---

## Usage

### Module 1: Molecular dynamics

```bash
# 1. NPT equilibration (50 ps)
cd module1_MD/
sbatch submit_npt.sh          # or: python plot_md_npt.py

# 2. NVT production (200 ps)
sbatch submit_nvt.sh          # or: python module_nvt.py

# 3. Analysis
python module_1_analysis.py
```

### Module 2: Optical properties

```bash
cd module2_LOPTICS/

# Ground state DFT+U + LOPTICS
python module2_loptics.py

# Ground state reference for Delta-SCF (ISMEAR=-2)
python loptics_gs_reference.py

# Post-processing with SUMO
bash module2_sumo_optical.sh
```

### Module 3: Delta-SCF excitations

```bash
cd module3_DeltaSCF/

# Run 20 Δ-SCF calculations (SLURM)
sbatch excitations_submission_fixed.sh

# Generate transition dipole moments (run from each transition folder)
bash module3_generate_tdm.sh

# Spectral analysis
cp ../module2_LOPTICS/loptics/OUTCAR OUTCAR_gs
cp ../module2_LOPTICS/loptics/PROCAR PROCAR_gs
python module3_deltascf_spectrum.py
```

---

## Key results (demonstration)

| Observable | Value |
|---|---|
| Electrolyte structure (NVT, 200 ps) | Lz = 37.34 Å (fixed) |
| Li⁺–O first shell (RDF) | 1.93 Å |
| Li⁺–P contact-ion pair | 3.32 Å |
| Li⁺ coordination number ⟨N⟩ | 3.23 ± 0.69 (solvent O only) |
| Li⁺ diffusivity D | 1.07 × 10⁻¹¹ m²/s |
| O–O bond-like pairs (d < 1.6 Å) | 20.6 mean |
| Static dielectric ε₁(0) | ~5.9 |
| ε₂ peak | ~3.6 eV |
| Δ-SCF valid transitions | 7 (4.6–7.8 eV) |
| Dominant orbital character | O 2p → TM 3d |

---

## Citation

If you use this workflow, please cite:

```
Crespo-Otero, J. M. et al. UltraBat D1.2: Computational workflow for the 
automated study of transport mechanisms at the LRMO/LP30 electrolyte interface.
UltraBat Horizon Europe Project (Grant No. 101103873), DTU (2025).
https://github.com/joseMCR93/ULTRABAT_D_1_2_deliverable_computational_workflow
```

---

## Related references

- Batatia, I. et al. MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Adv. Neural Inf. Process. Syst.* 35 (2022).
- Kresse, G. & Furthmüller, J. Efficiency of ab-initio total energy calculations for metals and semiconductors using a plane-wave basis set. *Comput. Mater. Sci.* **6**, 15–50 (1996).
- Perdew, J. P., Burke, K. & Ernzerhof, M. Generalized gradient approximation made simple. *Phys. Rev. Lett.* **77**, 3865–3868 (1996).
- Wang, V. et al. VASPKIT: A user-friendly interface facilitating high-throughput computing and analysis using VASP code. *Comput. Phys. Commun.* **267**, 108033 (2021).
- Rossi, T. C. et al. Dynamic control of X-ray core-exciton resonances by Coulomb screening in photoexcited semiconductors. *Commun. Mater.* **6**, 191 (2025).

---

## Acknowledgements

This project has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No. 101103873 (UltraBat).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
