file_path=$1
output=$2

mkdir -p "$output"

top_words=$(awk '{for(w=1;w<=NF;w++) gsub(/[^[:alnum:]]/, "", $w)}1' "$file_path" | \
  awk '{for(w=1;w<=NF;w++) print $w}' | \
  sort | \
  uniq -c | \
  sort -nr | \
  head -n 10 | \
   awk '{print $2}')

for word in $top_words; do
  touch "$output/$word.txt"
done

