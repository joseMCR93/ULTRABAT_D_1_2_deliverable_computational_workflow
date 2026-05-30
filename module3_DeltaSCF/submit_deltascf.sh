#!/bin/bash
#SBATCH --mail-type=all
#SBATCH --mail-user=your@email.com
#SBATCH --qos=gp_resa
#SBATCH --job-name=deltascf_optical
#SBATCH --output=deltascf_optical_%j.out
#SBATCH --error=deltascf_optical_%j.err
#SBATCH --ntasks=224          # 2 nodes × 112 cores/node
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00

module load vasp/6.5.1

module load miniforge
source activate ase-ipython

export ASE_VASP_COMMAND="srun -n 224 vasp_gam > vasp.out"
export VASP_PP_PATH="/gpfs/scratch/dtu1/JMCRO/New-POTCARs"
export UCX_TLS=^shm
export UCX_SHM_SEG_SIZE=256m
export UCX_MEMTYPE_CACHE=n

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================
NTASKS=224
NCORE=8
NBANDS_TARGET=1000
GS_OUTCAR="OUTCAR"
OCC_THRESHOLD=0.99

# VASP requires NBANDS divisible by NPAR = NTASKS/NCORE
NPAR=$(( NTASKS / NCORE ))
remainder=$(( NBANDS_TARGET % NPAR ))
if [[ $remainder -ne 0 ]]; then
    TOTAL_BANDS=$(( NBANDS_TARGET + NPAR - remainder ))
    echo "[INFO] NBANDS adjusted: ${NBANDS_TARGET} -> ${TOTAL_BANDS} (multiple of NPAR=${NPAR})"
else
    TOTAL_BANDS=$NBANDS_TARGET
    echo "[INFO] NBANDS=${TOTAL_BANDS} already divisible by NPAR=${NPAR}"
fi

# =============================================================================
# FUNCTION: parse occupied bands for EACH SPIN CHANNEL from OUTCAR
#
# For ISPIN=2, VASP writes TWO occupation blocks in OUTCAR:
#   Block 1 = spin-up   -> used for FERWE (with excitation applied)
#   Block 2 = spin-down -> used for FERDO (ground state, unchanged)
# Reading only the last block (spin-down) and using it for both is WRONG:
# it leaves ~89 electrons missing in spin-up -> unphysical positive energies.
# =============================================================================
parse_spin_occupations() {
    local outcar="$1"
    local threshold="$2"
    local total="$3"

    if [[ ! -f "$outcar" ]]; then
        echo "[ERROR] OUTCAR not found: $outcar" >&2
        return 1
    fi

    # Returns "NUP NDOWN" — last fully occupied band in each spin block
    awk -v thr="$threshold" -v nb="$total" '
        /band No\.  band energies/ {
            block++
            delete occ
            for (i = 1; i <= nb; i++) {
                if ((getline line) > 0) {
                    split(line, f)
                    if (f[3]+0 >= thr+0) last[block] = f[1]+0
                }
            }
        }
        END {
            # block 1 = spin-up, block 2 = spin-down
            printf "%d %d\n", last[1]+0, last[2]+0
        }
    ' "$outcar"
}

# Auto-detect NUP and NDOWN
echo "Parsing spin occupations from ${GS_OUTCAR} (threshold=${OCC_THRESHOLD})..."
read -r NUP NDOWN < <(parse_spin_occupations "$GS_OUTCAR" "$OCC_THRESHOLD" "$TOTAL_BANDS")

# Sanity check
if [[ -z "$NUP" || "$NUP" -eq 0 || -z "$NDOWN" || "$NDOWN" -eq 0 ]]; then
    echo "[WARN] Auto-detection failed -- using hardcoded values"
    NUP=575
    NDOWN=486
fi

echo "  -> NUP   (spin-up  occupied) = ${NUP}"
echo "  -> NDOWN (spin-down occupied) = ${NDOWN}"
echo "  -> TOTAL_BANDS = ${TOTAL_BANDS}"
echo "  -> NELECT check: NUP+NDOWN = $((NUP + NDOWN)) electrons"
echo ""

# =============================================================================
# TOP-20 TRANSITIONS
# =============================================================================
TOP_TRANSITIONS_CSV="optical_output/top_transitions.csv"

if [[ -f "$TOP_TRANSITIONS_CSV" ]]; then
    echo "Reading top transitions from $TOP_TRANSITIONS_CSV"
    mapfile -t TRANSITION_PAIRS < <(
        tail -n +2 "$TOP_TRANSITIONS_CSV" | awk -F',' '{print $4":"$5}'
    )
else
    echo "CSV not found -- using hardcoded transitions"
    TRANSITION_PAIRS=(
        "576:579" "576:580" "575:579" "576:581" "575:580"
        "574:579" "576:582" "575:581" "574:580" "576:583"
        "575:582" "574:581" "573:579" "576:584" "575:583"
        "573:580" "574:582" "573:581" "576:585" "575:584"
    )
fi

# =============================================================================
# FUNCTION: generate FERWE (spin-up, excited state)
# Promotes electron from val_band to con_band.
# Total spin-up electrons conserved: NUP (not NUP-1+1 = NUP).
# =============================================================================
generate_ferwe() {
    local val_band=$1
    local con_band=$2
    local ferwe=()

    # Ground state spin-up: bands 1..NUP occupied
    for ((i=1; i<=TOTAL_BANDS; i++)); do
        if (( i <= NUP )); then
            ferwe+=("1.0")
        else
            ferwe+=("0.0")
        fi
    done

    # Excitation: remove from val_band, add to con_band
    ferwe[$((val_band - 1))]="0.0"
    ferwe[$((con_band - 1))]="1.0"

    # Compress to count*value notation
    local compressed="" count=1
    for ((i=1; i<${#ferwe[@]}; i++)); do
        if [[ "${ferwe[$i]}" == "${ferwe[$((i-1))]}" ]]; then
            ((count++))
        else
            compressed+="${count}*${ferwe[$((i-1))]} "
            count=1
        fi
    done
    compressed+="${count}*${ferwe[$((TOTAL_BANDS-1))]}"
    echo "$compressed"
}

# FERDO is fixed: spin-down stays in ground state for all calculations
FERDO="${NDOWN}*1.0 $((TOTAL_BANDS - NDOWN))*0.0"

# =============================================================================
# FUNCTION: verify excitation converged correctly
# =============================================================================
verify_excitation() {
    local folder="$1" val_band=$2 con_band=$3
    local outcar="${folder}/OUTCAR"
    [[ ! -f "$outcar" ]] && { echo "  [WARN] OUTCAR not found"; return; }

    local converged
    converged=$(grep -c "aborting loop because EDIFF is reached" "$outcar" || true)
    if (( converged > 0 )); then
        echo "  CONVERGED: V${val_band}->C${con_band}" >> "$STATUS_FILE"
        echo "  [OK] Converged"
    else
        echo "  NOT CONVERGED: V${val_band}->C${con_band}" >> "$STATUS_FILE"
        echo "  [WARN] Did not reach EDIFF"
    fi
}

# =============================================================================
# MAIN LOOP
# =============================================================================
STATUS_FILE="calculation_status.txt"
{
    echo "Delta-SCF Optical Transitions -- $(date)"
    echo "NUP=${NUP}  NDOWN=${NDOWN}  NBANDS=${TOTAL_BANDS}  NPAR=${NPAR}"
    echo "Total transitions: ${#TRANSITION_PAIRS[@]}"
    echo "---------------------------------------------"
} >> "$STATUS_FILE"

rank=1
prev_folder=""
for pair in "${TRANSITION_PAIRS[@]}"; do
    val_band="${pair%%:*}"
    con_band="${pair##*:}"

    folder_name=$(printf "transition_%02d_V%d_C%d" "$rank" "$val_band" "$con_band")
    mkdir -p "$folder_name"

    echo ""
    echo "====================================================="
    echo " Rank ${rank}: V${val_band} -> C${con_band}  |  ${folder_name}"
    echo "====================================================="
    echo "Running rank ${rank}: V${val_band}->C${con_band} -- $(date)" >> "$STATUS_FILE"

    ferwe=$(generate_ferwe "$val_band" "$con_band")

    cat <<EOL > "${folder_name}/INCAR"
# Delta-SCF rank ${rank}: V${val_band} -> C${con_band}
# NUP=${NUP}  NDOWN=${NDOWN}  NBANDS=${TOTAL_BANDS}

 FERWE  = ${ferwe}
 FERDO  = ${FERDO}

 NBANDS  = ${TOTAL_BANDS}
 ISMEAR  = -2
 SIGMA   = 0.010000

 CSHIFT   = 0.100000
 ENCUT    = 520.000000
 EDIFF    = 1.00e-04
 ALGO     = All
 GGA      = PE
 PREC     = Accurate
 IBRION   = -1
 ISIF     = 2
 ISPIN    = 2
 LMAXMIX  = 4
 LORBIT   = 11
 NELM     = 500
 NELMDL   = -5
 NELMIN   = 8
 NSW      = 0
 NCORE    = ${NCORE}
 ADDGRID  = .TRUE.
 LASPH    = .TRUE.
 LCHARG   = .TRUE.
 LDAU     = .TRUE.
 LWAVE    = .TRUE.
 LDAUL    = -1 2 2 2 -1
 LDAUU    = 0.000 5.000 3.900 5.600 0.000
 LDAUJ    = 0.000 0.000 0.000 0.000 0.000
 MAGMOM   = 16*0.0000 6*3.0000 25*3.5000 6*3.0000 90*0.0000
EOL

    cp KPOINTS POSCAR POTCAR "${folder_name}/"

    if [[ -f "WAVECAR" ]]; then
        echo "  Copying ground-state WAVECAR -> ${folder_name}/"
        cp WAVECAR "${folder_name}/"
    elif [[ -n "$prev_folder" && -f "${prev_folder}/WAVECAR" ]]; then
        echo "  Copying WAVECAR from ${prev_folder} -> ${folder_name}/"
        cp "${prev_folder}/WAVECAR" "${folder_name}/"
    else
        echo "  [WARN] No WAVECAR found -- starting from scratch"
    fi

    # Skip if already converged from a previous submission
    if [[ -f "${folder_name}/OUTCAR" ]]; then
        already=$(grep -c "aborting loop because EDIFF is reached" "${folder_name}/OUTCAR" 2>/dev/null || true)
        if (( already > 0 )); then
            echo "  [SKIP] Already converged -- skipping ${folder_name}"
            echo "  SKIPPED (already converged): V${val_band}->C${con_band}" >> "$STATUS_FILE"
            prev_folder="${folder_name}"
            ((rank++))
            continue
        else
            echo "  [RESTART] OUTCAR exists but not converged -- rerunning"
        fi
    fi

    (cd "${folder_name}" && srun vasp_std > vasp.out)
    exit_code=$?

    if (( exit_code == 0 )); then
        echo "  [OK] VASP finished"
        verify_excitation "${folder_name}" "$val_band" "$con_band"
    else
        echo "  [ERROR] VASP exit code ${exit_code}" | tee -a "$STATUS_FILE"
    fi

    prev_folder="${folder_name}"
    ((rank++))
done

echo "All calculations completed -- $(date)" >> "$STATUS_FILE"
echo "All calculations completed."
