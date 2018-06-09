set term postscript eps enhanced color
set output "05.eps"
set size ratio -1
set xrange [-281:251]
set yrange [-110:422]
set xlabel "x [m]"
set ylabel "z [m]"
plot "05.txt" using 1:2 lc rgb "#FF0000" title 'Ground Truth' w lines,"05.txt" using 3:4 lc rgb "#0000FF" title 'MINE' w lines,"05.txt" using 5:6 lc rgb "#00FF00" title 'ORB_SLAM2' w lines,"< head -1 05.txt" using 1:2 lc rgb "#000000" pt 4 ps 1 lw 2 title 'Sequence Start' w points
