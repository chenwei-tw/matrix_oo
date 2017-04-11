set title "Matrix Multiplication Performance"
set xlabel "Matrix size (nxn)"
set ylabel "Run Time (ms)"
set terminal png font " Times_New_Roman,12 "
set output "statistic.png"
set key left 

plot \
"data.csv" using 1:2 with linespoints linewidth 2 title "Naive", \
"data.csv" using 1:3 with linespoints linewidth 2 title "SSE", \
"data.csv" using 1:4 with linespoints linewidth 2 title "SSE prefetch", \
"data.csv" using 1:5 with linespoints linewidth 2 title "AVX", \
"data.csv" using 1:6 with linespoints linewidth 2 title "AVX prefetch" \
