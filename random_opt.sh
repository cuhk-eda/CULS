#!/usr/bin/env bash
# random_opt.sh — apply N random AIG operations using gpuls
#
# Usage: ./random_opt.sh <input.aig> <output.aig> [N=1000]
#
# Example: ./random_opt.sh circuit.aig result.aig 1000

set -e

INPUT="${1:?Usage: $0 <input.aig> <output.aig> [N]}"
OUTPUT="${2:?Usage: $0 <input.aig> <output.aig> [N]}"
N="${3:-1000}"

GPULS="$(dirname "$0")/build/gpuls"
if [[ ! -x "$GPULS" ]]; then
    echo "Error: gpuls binary not found at $GPULS" >&2
    exit 1
fi

# Available single-step operations (lighter-weight, safe to repeat)
OPS=(b rw rf rs st)

# Build a semicolon-separated command string
CMD="read ${INPUT}"

for ((i = 1; i <= N; i++)); do
    # Pick a random operation
    CMD+="; ${OPS[$((RANDOM % ${#OPS[@]}))]}"

    # Print stats every 100 steps
    if (( i % 100 == 0 )); then
        CMD+="; ps"
    fi
done

CMD+="; ps; time; write ${OUTPUT}"

echo "Running $N random operations on ${INPUT} ..."
"$GPULS" -c "$CMD"
echo "Done. Result written to ${OUTPUT}"
