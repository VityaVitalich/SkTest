file_path=$1

awk '{for(w=1;w<=NF;w++) gsub(/[^[:alnum:]]/, "", $w)}1' "$file_path" | \
  awk '{for(w=1;w<=NF;w++) print $w}' | \
  sort | \
  uniq -c | \
  sort -nr | \
  head -n 10