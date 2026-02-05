#!/bin/bash

# Define colors for output
GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m' # No Color

EXEC="./build/gpuls"
BENCHMARK_BASE="./benchmarks/"

# Initialize array to keep track of failed cases
FAILED_CASES=()
SAVE_FILE=""
CHECKPOINT_FILE=""
TEST_CMD="resyn2rs"
ORANGE='\033[0;33m'

# Parse command line arguments
while getopts "s:v:c:" opt; do
  case $opt in
    s) SAVE_FILE="$OPTARG" ;;
    v) CHECKPOINT_FILE="$OPTARG" ;;
    c) TEST_CMD="$OPTARG" ;;
    *) echo "Usage: $0 [-s save_result_filename] [-v checkpoint_file] [-c command]" >&2; exit 1 ;;
  esac
done

# Initialize save file if specified
if [ -n "$SAVE_FILE" ]; then
    > "$SAVE_FILE"
fi

# Load checkpoint data if specified
declare -A CHECKPOINT_AND
declare -A CHECKPOINT_LEVEL

if [ -n "$CHECKPOINT_FILE" ]; then
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "Loading checkpoint from $CHECKPOINT_FILE..."
        while read -r name and_cnt level_cnt; do
            # Skip empty lines
            [ -z "$name" ] && continue
            CHECKPOINT_AND["$name"]=$and_cnt
            CHECKPOINT_LEVEL["$name"]=$level_cnt
        done < "$CHECKPOINT_FILE"
    else
        echo -e "${RED}Checkpoint file $CHECKPOINT_FILE not found!${NC}"
        exit 1
    fi
fi

# Function to run a benchmark case
run_case() {
    local aig_file=$1
    local case_name=$(basename "$aig_file" .aig)
    local full_path="$BENCHMARK_BASE${aig_file}"
    local log_file="${case_name}_failed.log"

    # echo "----------------------------------------------------------------"
    # echo "Running $case_name..."

    # Check if input file exists
    if [ ! -f "$full_path" ]; then
        if [ ! -f "$aig_file" ]; then
             # Try looking in the benchmarks folder relative to script execution if not fully qualified
            full_path="$BENCHMARK_BASE${case_name}.aig"
            if [ ! -f "$full_path" ]; then
                echo -e "${RED}${case_name} FAILED (File not found)${NC}"
                FAILED_CASES+=("$case_name (File not found)")
                return
            fi
        else
            full_path="$aig_file"
        fi
    fi

    # Run the command with stats and timing
    # ps before to get initial stats
    # time after resyn2rs to get execution time of the optimization
    # ps after to get final stats
    if $EXEC -c "read $full_path; ps; $TEST_CMD; time; ps" > "$log_file" 2>&1; then
        
        # Parse Runtime
        # Looks for: {time} prev cmd: ... full X.XX s; ...
        local runtime=$(grep "{time}" "$log_file" | sed -n 's/.*prev cmd:.*full \([0-9.]*\) s;.*/\1/p')
        if [ -z "$runtime" ]; then runtime="N/A"; fi

        # Parse Stats
        # Looks for: AIG stats: i/o = ... and = ... level = ...
        local stats_lines=$(grep "AIG stats:" "$log_file")
        local line_before=$(echo "$stats_lines" | head -n 1)
        local line_after=$(echo "$stats_lines" | tail -n 1)
        
        local prev_and=$(echo "$line_before" | sed -n 's/.*and = \([0-9]*\).*/\1/p')
        local prev_level=$(echo "$line_before" | sed -n 's/.*level = \([0-9]*\).*/\1/p')
        
        local post_and=$(echo "$line_after" | sed -n 's/.*and = \([0-9]*\).*/\1/p')
        local post_level=$(echo "$line_after" | sed -n 's/.*level = \([0-9]*\).*/\1/p')

        echo -e "${case_name} ${GREEN}PASSED${NC} (Time: ${runtime}s; And: ${prev_and} -> ${post_and}; Level: ${prev_level} -> ${post_level})"
        
        if [ -n "$SAVE_FILE" ]; then
            echo "$case_name $post_and $post_level" >> "$SAVE_FILE"
        fi

        # Compare with checkpoint if available
        if [ -n "$CHECKPOINT_FILE" ]; then
            chk_and=${CHECKPOINT_AND[$case_name]}
            chk_level=${CHECKPOINT_LEVEL[$case_name]}
            
            if [ -n "$chk_and" ]; then
                # Check for degradation
                if [ "$post_and" -gt "$chk_and" ] || [ "$post_level" -gt "$chk_level" ]; then
                     echo -e "    ${ORANGE}Warning${NC}: Result degraded (Checkpoint: AND=${RED}${chk_and}${NC}, Level=${RED}${chk_level}${NC})"
                fi
            fi
        fi
        
        rm -f "$log_file" # Remove log file if passed
    else
        echo -e "${case_name} ${RED}FAILED${NC}"
        FAILED_CASES+=("$case_name")
        # Log file is kept
    fi
}

echo "Starting Benchmark Execution..."

# List of cases sorted by size (smallest to largest)
# Comment out lines with '#' to disable specific cases

run_case "c17.aig" # 77
run_case "c432.aig" # 844
run_case "int2float.aig" # 992
run_case "ctrl.aig" # 1.1K
run_case "c1908.aig" # 1.5K
run_case "c880.aig" # 1.6K
run_case "c499.aig" # 1.7K
run_case "ss_pcm.aig" # 1.7K
run_case "usb_phy.aig" # 1.9K
run_case "c1355.aig" # 2.0K
run_case "router.aig" # 2.2K
run_case "cavlc.aig" # 2.4K
run_case "sasc.aig" # 3.0K
run_case "c3540.aig" # 3.5K
run_case "simple_spi.aig" # 3.9K
run_case "priority.aig" # 4.0K
run_case "c2670.aig" # 4.1K
run_case "iwls05_i2c.aig" # 4.1K
run_case "c6288.aig" # 5.9K
run_case "dec.aig" # 6.5K
run_case "c7552.aig" # 7.3K
run_case "adder.aig" # 7.3K
run_case "i2c.aig" # 7.4K
run_case "c5315.aig" # 8.1K
run_case "systemcdes.aig" # 9.4K
run_case "spi.aig" # 13K
run_case "bar.aig" # 14K
run_case "des_area.aig" # 15K
run_case "sin.aig" # 15K
run_case "max.aig" # 18K
run_case "tv80.aig" # 30K
run_case "systemcaes.aig" # 41K
run_case "arbiter.aig" # 45K
run_case "voter.aig" # 45K
run_case "square.aig" # 50K
run_case "iwls05_mem_ctrl.aig" # 55K
run_case "ac97_ctrl.aig" # 60K
run_case "usb_funct.aig" # 61K
run_case "sqrt.aig" # 68K
run_case "aes_core.aig" # 72K
run_case "multiplier.aig" # 80K
run_case "DMA.aig" # 94K
run_case "log2.aig" # 95K
run_case "pci_bridge32.aig" # 97K
run_case "wb_conmax.aig" # 164K
run_case "DSP.aig" # 171K
run_case "div.aig" # 174K
run_case "mem_ctrl.aig" # 190K
run_case "des_perf.aig" # 274K
run_case "RISC.aig" # 321K
run_case "ethernet.aig" # 406K
run_case "hyp.aig" # 579K
run_case "vga_lcd.aig" # 625K
run_case "leon3mp.aig" # 3.6M
run_case "netcard.aig" # 3.9M
run_case "leon2.aig" # 5.2M
run_case "leon3.aig" # 6.2M
run_case "leon3_opt.aig" # 13M


echo "----------------------------------------------------------------"
echo "Execution Finished."

if [ ${#FAILED_CASES[@]} -eq 0 ]; then
    echo -e "${GREEN}All Level 1 cases passed!${NC}"
else
    echo -e "${RED}The following cases FAILED:${NC}"
    for case in "${FAILED_CASES[@]}"; do
        echo -e "${RED}- $case${NC}"
    done
    echo "Check the corresponding *_failed.log files for details."
fi