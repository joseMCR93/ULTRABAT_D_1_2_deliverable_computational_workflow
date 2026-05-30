#!/bin/bash
# =============================================================================
# Module 2: Optical Spectrum Analysis using SUMO
# =============================================================================
# Workflow D1.2 — UltraBat Project
#
# Generates publication-quality plots of:
#   1. Dielectric function: ε₁(ω) and ε₂(ω)
#   2. Optical absorption spectrum: α(ω)
#
# Requirements:
#   - Converged LOPTICS calculation (vasprun.xml in LOPTICS_DIR)
#   - SUMO installed at SUMO_ENV
#
# Usage:
#   bash module2_sumo_optical.sh
# =============================================================================

LOPTICS_DIR="./loptics"   # path to LOPTICS output directory
SUMO_ENV="${HOME}/sumo/bin/activate"  # path to SUMO virtualenv
OUT_DIR="${LOPTICS_DIR}/optical_output"

# ── Activate SUMO ─────────────────────────────────────────────────────────────
echo "Activating SUMO environment..."
source "${SUMO_ENV}" || { echo "[ERROR] Cannot activate ${SUMO_ENV}"; exit 1; }
echo "SUMO version: $(sumo-optplot --version 2>/dev/null || echo 'unknown')"

# ── Check inputs ──────────────────────────────────────────────────────────────
VASPRUN="${LOPTICS_DIR}/vasprun.xml"
if [[ ! -f "${VASPRUN}" ]]; then
    echo "[ERROR] vasprun.xml not found at ${VASPRUN}"
    exit 1
fi
echo "Found: ${VASPRUN}"

mkdir -p "${OUT_DIR}"
cd "${LOPTICS_DIR}"

# ── Plot 1: Real dielectric function ε₁(ω) ───────────────────────────────────
echo ""
echo "Plotting ε₁(ω) — real dielectric function..."
sumo-optplot eps_real \
    -f "${VASPRUN}" \
    --units eV \
    --xmin 0 --xmax 10 \
    --dpi 300 \
    --format png \
    -p "${OUT_DIR}/eps_real"
echo "  → ${OUT_DIR}/eps_real.png"

# ── Plot 2: Imaginary dielectric function ε₂(ω) ──────────────────────────────
echo ""
echo "Plotting ε₂(ω) — imaginary dielectric function..."
sumo-optplot eps_imag \
    -f "${VASPRUN}" \
    --units eV \
    --xmin 0 --xmax 10 \
    --dpi 300 \
    --format png \
    -p "${OUT_DIR}/eps_imag"
echo "  → ${OUT_DIR}/eps_imag.png"

# ── Plot 3: Both ε₁ and ε₂ together ──────────────────────────────────────────
echo ""
echo "Plotting ε₁(ω) + ε₂(ω) combined..."
sumo-optplot eps_real eps_imag \
    -f "${VASPRUN}" \
    --units eV \
    --xmin 0 --xmax 10 \
    --dpi 300 \
    --format png \
    -p "${OUT_DIR}/dielectric_function"
echo "  → ${OUT_DIR}/dielectric_function.png"

# ── Plot 4: Absorption spectrum α(ω) ─────────────────────────────────────────
echo ""
echo "Plotting α(ω) — absorption spectrum..."
sumo-optplot absorption \
    -f "${VASPRUN}" \
    --units eV \
    --xmin 0 --xmax 10 \
    --dpi 300 \
    --format png \
    -p "${OUT_DIR}/absorption_spectrum"
echo "  → ${OUT_DIR}/absorption_spectrum.png"

# ── Plot 5: Loss function ─────────────────────────────────────────────────────
echo ""
echo "Plotting loss function..."
sumo-optplot loss \
    -f "${VASPRUN}" \
    --units eV \
    --xmin 0 --xmax 10 \
    --dpi 300 \
    --format png \
    -p "${OUT_DIR}/loss_function"
echo "  → ${OUT_DIR}/loss_function.png"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "SUMO OPTICAL ANALYSIS COMPLETE"
echo "=============================================="
echo "Output directory: ${OUT_DIR}"
ls -lh "${OUT_DIR}"/*.png 2>/dev/null
