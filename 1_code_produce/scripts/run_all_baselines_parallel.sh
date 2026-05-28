#!/bin/bash

for controller in noRM Alinea Fixed 
do
  for ep in {0..19}
  do
    python -m scripts.run_one_baseline_episode --controller $controller --episode $ep --mode mixed &
  done
done

wait
python -m scripts.merge_baseline_results

#bash scripts/run_all_baselines_parallel.sh