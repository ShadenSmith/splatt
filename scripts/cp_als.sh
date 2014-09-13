#!/bin/bash

BASE=$1
OUT=timings/${BASE}-$4.txt
PARTS=$TT/parts/${BASE}.graph.part.$2

echo "*****************************************************************" | tee -a ${OUT}
date | tee -a ${OUT}
./apps/sp_cp_als $TT/${BASE}.tns --reorder=${PARTS} -t $3 --scale -r 10 | tee -a ${OUT}
#./apps/sp_cp_als $TT/${BASE}.tns -t $3 --scale -r 10 | tee -a ${OUT}
echo "*****************************************************************" | tee -a ${OUT}
echo "" | tee -a ${OUT}
