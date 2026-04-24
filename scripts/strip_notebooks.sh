#!/bin/sh
# Strip outputs and execution counts from notebooks using jq.
set -eu
if [ "$#" -eq 0 ]; then
  set -- $(find . -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' | sort)
fi
for nb in "$@"; do
  tmp="$nb.tmp"
  jq '(.cells[] | select(.cell_type == "code") | .outputs) = [] |
      (.cells[] | select(.cell_type == "code") | .execution_count) = null |
      del(.metadata.widgets) |
      (.cells[].metadata) |= with_entries(select(.key as $k | ["execution", "ExecuteTime", "collapsed", "scrolled"] | index($k) | not))' "$nb" > "$tmp"
  mv "$tmp" "$nb"
done
