#!/usr/bin/env bash

python3 ../../../../plot_results.py $1 --rolling 50 --steps -o steps.pdf
python3 ../../../../plot_results.py $1 --rolling 50 --loss -o loss.pdf
python3 ../../../../plot_results.py $1 --rolling 50 --collisions -o collisions.pdf
python3 ../../../../plot_results.py $1 --rolling 50 --reward -o reward.pdf

