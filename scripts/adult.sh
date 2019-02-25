#!/usr/bin/env bash

for i in 1.0 2.0 5.0 10.0
do
for j in 1.0 2.0 5.0 10.0
do
for k in 10 50 100 200 500
do
for l in 1.0 2.0 5.0 10.0
do
echo "mi $i, e1, $j, e2 $k, e3 $l"
nohup python -m examples.adult --mi=$i --e1=$j --e2=$k --e3=$l  > /atlas/u/tsong/exps/scripts/lvfae.out 2>&1 &
sleep 10.1
done
done
done
done