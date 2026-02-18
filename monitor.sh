#!/bin/bash
# monitor.sh - Track ratio improvement

while true; do
  RATIOS=$(grep "submit-candidate" tests | awk -F'ratio=' '{print $2}' | awk '{print $1}')
  COUNT=$(echo "$RATIOS" | awk 'NF{n++} END{print n+0}')
  MAX=$(echo "$RATIOS" | awk 'NF{if ($1>max || n==0) max=$1; n++} END{if (n==0) print 0; else print max}')
  AVG=$(echo "$RATIOS" | awk 'NF{sum+=$1; n++} END{if (n==0) print 0; else print sum/n}')
  ABOVE_1=$(echo "$RATIOS" | awk 'NF && $1 > 1.0 {n++} END{print n+0}')
  PCT_ABOVE_1=$(awk -v a="$ABOVE_1" -v c="$COUNT" 'BEGIN {if (c>0) printf "%.2f", (a*100)/c; else print "0.00"}')
  
  clear
  echo "=== Ratio Progress ==="
  echo "Candidates: $COUNT"
  echo "Ratio > 1.0: $ABOVE_1"
  echo "% > 1.0:    ${PCT_ABOVE_1}%"
  echo "Max ratio:  $MAX"
  echo "Avg ratio:  $AVG"
  echo "Time: $(date)"
  echo ""
  
  if (( $(echo "$MAX > 0.95" | bc -l) )); then
    echo "⚠️  VERY CLOSE! Ratio > 0.95"
  elif (( $(echo "$MAX > 0.80" | bc -l) )); then
    echo "✓ Good progress! Ratio > 0.80"
  fi
  
  sleep 10
done
