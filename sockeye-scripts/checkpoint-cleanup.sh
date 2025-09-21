#!/bin/bash
set -euo pipefail

# Make tmp if needed
mkdir -p tmp

# Loop over conditions testA ... testL
for letter in {A..L}; do
  prefix="test${letter}"

  # 1) For each replicate 0..9, move the latest (max generation) checkpoint
  for rep in {0..9}; do
    # List matching files; continue if none
    mapfile -t files < <(ls -1 ${prefix}${rep}_gen*.pkl 2>/dev/null || true)
    if (( ${#files[@]} == 0 )); then
      continue
    fi

    # Extract generation number, pick the max, move that file
    latest_file="$(
      printf '%s\n' "${files[@]}" \
      | sed -E 's/.*_gen([0-9]+)\.pkl/\1 &/' \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2-
    )"

    # Move without overwriting if already moved
    mv -n "$latest_file" tmp/
  done

  # 2) Move every 50000th-gen checkpoint for this condition (…50000, …100000, …150000, etc.)
  #    This matches filenames ending with ...00000.pkl or ...50000.pkl
  find . -maxdepth 1 -type f \
    \( -name "${prefix}[0-9]_gen*00000.pkl" -o -name "${prefix}[0-9]_gen*50000.pkl" \) \
    -exec mv -n {} tmp/ \;
done
