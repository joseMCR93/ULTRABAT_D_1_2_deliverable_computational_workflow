#!/bin/bash
# =============================================================================
# Module 3: Generate TDM files for all Delta-SCF transitions
# =============================================================================
# Workflow D1.2 — UltraBat Project
#
# vaspkit 713 is run FROM EACH TRANSITION FOLDER (which contains WAVECAR).
#
# Interactive input sequence for each transition:
#   713        <- task: TDM
#   0          <- option: TDM for selected K-points
#   val con    <- band pair (e.g. "576 579")
#   1          <- k-point index (Gamma = 1, since NKPTS=1)
#
# Usage:
#   bash module3_generate_tdm.sh
# =============================================================================

DELTASCF_DIR="."     # directory containing transition_XX_VYYY_CZZZ folders
KPOINT_IDX=1         # k-point index for Gamma (vaspkit counts from 1)
LOG_FILE="tdm_generation.log"

echo "TDM generation -- $(date)" > "$LOG_FILE"
echo ""

# ── Find all transition folders ───────────────────────────────────────────────
mapfile -t FOLDERS < <(find "${DELTASCF_DIR}" -maxdepth 1 -type d \
    -name "transition_*_V*_C*" | sort)

if [[ ${#FOLDERS[@]} -eq 0 ]]; then
    echo "[ERROR] No transition_XX_VYYY_CZZZ folders found in ${DELTASCF_DIR}"
    exit 1
fi
echo "Found ${#FOLDERS[@]} transition folders"
echo ""

SUCCESS=0
FAILED=0

for folder in "${FOLDERS[@]}"; do
    folder_name=$(basename "$folder")

    # Parse band indices from folder name
    if [[ "$folder_name" =~ transition_([0-9]+)_V([0-9]+)_C([0-9]+) ]]; then
        rank="${BASH_REMATCH[1]}"
        val="${BASH_REMATCH[2]}"
        con="${BASH_REMATCH[3]}"
    else
        echo "  [SKIP] Cannot parse: ${folder_name}"
        continue
    fi

    echo "  [${rank}] V${val} -> C${con}"

    # Check WAVECAR exists in transition folder
    if [[ ! -f "${folder}/WAVECAR" ]]; then
        echo "    [WARN] No WAVECAR in ${folder_name} -- skipping"
        echo "  FAILED [${rank}] V${val}->C${con}: no WAVECAR" >> "$LOG_FILE"
        ((FAILED++))
        continue
    fi

    # Remove stale TDM files
    rm -f "${folder}/TDM_COMPONENTS_UP.dat" \
          "${folder}/TDM_COMPONENTS_DW.dat" \
          "${folder}/SELECTED_KPTS_LIST"

    # Run vaspkit 713 from the transition folder
    # Input: 713 \n 0 \n "val con" \n kpoint_idx \n
    printf "713\n0\n%s %s\n%s\n" "$val" "$con" "$KPOINT_IDX" \
        | (cd "${folder}" && vaspkit) > /dev/null 2>&1

    exit_code=$?

    # Check output
    if [[ ! -f "${folder}/TDM_COMPONENTS_UP.dat" ]]; then
        echo "    [WARN] TDM_COMPONENTS_UP.dat not generated"
        echo "  FAILED [${rank}] V${val}->C${con}: no output" >> "$LOG_FILE"
        ((FAILED++))
        continue
    fi

    # Check file is not empty
    n_lines=$(grep -v '^#' "${folder}/TDM_COMPONENTS_UP.dat" | \
              grep -v '^Kpoint' | grep -v '^[[:space:]]*$' | wc -l)

    if [[ "$n_lines" -eq 0 ]]; then
        echo "    [WARN] TDM file is empty (check WAVECAR or band range)"
        echo "  EMPTY [${rank}] V${val}->C${con}" >> "$LOG_FILE"
        ((FAILED++))
        continue
    fi

    # Extract and print Total |TDM|^2 (column 4 of first data line)
    tdm_val=$(grep -v '^#' "${folder}/TDM_COMPONENTS_UP.dat" | \
              grep -v '^Kpoint' | grep -v '^[[:space:]]*$' | \
              awk 'NR==1{print $4}')

    echo "    |TDM|^2 = ${tdm_val} Debye^2  [${n_lines} data line(s)]"
    echo "  OK [${rank}] V${val}->C${con}: |TDM|^2=${tdm_val}" >> "$LOG_FILE"
    ((SUCCESS++))

done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "TDM GENERATION COMPLETE"
echo "=============================================="
echo "  Successful : ${SUCCESS}/${#FOLDERS[@]}"
echo "  Failed     : ${FAILED}/${#FOLDERS[@]}"
echo "  Log        : ${LOG_FILE}"
echo ""
echo "Check tdm_generation.log for details."
echo "Next step: python module3_deltascf_spectrum.py"
