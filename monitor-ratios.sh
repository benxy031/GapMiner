#!/bin/bash
# File: monitor-ratios.sh

set -u

if [ -f "tests" ]; then
  TESTS_FILE="tests"
elif [ -f "bin/tests" ]; then
  TESTS_FILE="bin/tests"
else
  TESTS_FILE="tests"
fi
CSV_FILE="ratio-monitor.csv"
SAMPLE_LINES=20000
WINDOW_LINES=500
WINDOW_CANDIDATES=60
SLEEP_SECONDS=60

print_summary() {
  local lines
  lines=$(tail -n "$SAMPLE_LINES" "$TESTS_FILE" 2>/dev/null | grep "submit-candidate" || true)

  if [ -z "$lines" ]; then
    echo "No submit-candidate entries found in $TESTS_FILE"
    return 0
  fi

  echo "$lines" | awk '
    function abs(v) { return v < 0 ? -v : v }
    {
      total++
      ratio = 0.0
      if (match($0, /ratio=[0-9.]+/)) {
        ratio_str = substr($0, RSTART + 6, RLENGTH - 6)
        ratio = ratio_str + 0.0
      }

      sum += ratio
      if (total == 1 || ratio < min) min = ratio
      if (total == 1 || ratio > max) max = ratio

      if (ratio < 0.2) b1++
      else if (ratio < 0.5) b2++
      else if (ratio < 0.8) b3++
      else if (ratio < 1.0) b4++
      else b5++

      if (ratio > 1.0) gt1++

      if (match($0, /\[[0-9-]+ [0-9:]+\]/)) {
        t = substr($0, RSTART + 1, RLENGTH - 2)
        gsub(/[-:]/, " ", t)
        ts = mktime(t)
        if (ts > 0) {
          if (first_ts == 0 || ts < first_ts) first_ts = ts
          if (ts > last_ts) last_ts = ts
        }
      }
    }
    END {
      if (total == 0) {
        print "No ratios parsed"
        exit
      }

      avg = sum / total
      pct_gt1 = (total > 0) ? (100.0 * gt1 / total) : 0.0
      duration = (last_ts > first_ts) ? (last_ts - first_ts) : 0

      eta = "n/a"
      if (gt1 > 0 && duration > 0) {
        eta_sec = duration / gt1
        if (eta_sec < 60)
          eta = sprintf("%.0fs", eta_sec)
        else if (eta_sec < 3600)
          eta = sprintf("%.1fm", eta_sec / 60.0)
        else
          eta = sprintf("%.2fh", eta_sec / 3600.0)
      } else if (duration > 0) {
        eta = "> sample window"
      }

      print "=== Session Ratio Summary ==="
      printf("Total submit-candidate: %d\n", total)
      printf("Ratio > 1.0: %d (%.2f%%)\n", gt1, pct_gt1)
      printf("Ratio min/max/avg: %.6f / %.6f / %.6f\n", min, max, avg)
      if (duration > 0)
        printf("Sample timespan: %ds (from log timestamps)\n", duration)
      else
        print "Sample timespan: n/a"
      printf("ETA next >1.0 (empirical): %s\n", eta)
      print ""
      print "Histogram:"
      printf("  <0.2      : %d\n", b1)
      printf("  0.2-0.5   : %d\n", b2)
      printf("  0.5-0.8   : %d\n", b3)
      printf("  0.8-1.0   : %d\n", b4)
      printf("  >=1.0     : %d\n", b5)
    }
  '
}

append_csv_window() {
  local last_min ratios min max avg count timestamp
  last_min=$(tail -n "$WINDOW_LINES" "$TESTS_FILE" 2>/dev/null | grep "submit-candidate" | tail -n "$WINDOW_CANDIDATES" || true)

  if [ -z "$last_min" ]; then
    return 0
  fi

  ratios=$(echo "$last_min" | awk -F'ratio=' '{print $2}' | awk '{print $1}')
  min=$(echo "$ratios" | awk 'NF{if(n==0||$1<m)m=$1; n++} END{if(n==0) print 0; else print m}')
  max=$(echo "$ratios" | awk 'NF{if(n==0||$1>M)M=$1; n++} END{if(n==0) print 0; else print M}')
  avg=$(echo "$ratios" | awk 'NF{sum+=$1; n++} END{if(n==0) print 0; else print sum/n}')
  count=$(echo "$ratios" | awk 'NF{n++} END{print n+0}')
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  echo "$timestamp,0,$min,$max,$avg,$count" >> "$CSV_FILE"
}

case "${1:-watch}" in
  --summary|summary)
    echo "Using log file: $TESTS_FILE"
    print_summary
    ;;
  --watch|watch)
    if [ ! -f "$CSV_FILE" ]; then
      echo "timestamp,ratio,ratio_min,ratio_max,ratio_avg,ratio_count" > "$CSV_FILE"
    fi
    while true; do
      append_csv_window
      clear
      echo "Using log file: $TESTS_FILE"
      print_summary
      echo ""
      echo "CSV: $CSV_FILE (window updates every ${SLEEP_SECONDS}s)"
      echo "Time: $(date)"
      sleep "$SLEEP_SECONDS"
    done
    ;;
  *)
    echo "Usage: $0 [--watch|--summary]"
    exit 1
    ;;
esac
