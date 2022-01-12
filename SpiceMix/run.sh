#!/bin/bash

exit

time python main.py \
  --path2dataset="/work/magroup/tianming/Researches/SpiceMix/version1/data/obsolete/synthetic_500_100_20_15_0_0_i4" \
  --repli_list="['0']" -K=10 --use_spatial="['True']" --lambda_Sigma_x_inv=1e-5 --max_iter=50 --init_NMF_iter=20 \
  --device='cuda:6'

time python main.py \
  --path2dataset="/work/magroup/tianming/Researches/SpiceMix/version1/data/obsolete/synthetic_500_100_20_15_0_0_i4" \
  --repli_list="['0']" -K=10 --use_spatial="['True']" --lambda_Sigma_x_inv=1e-5 --max_iter=50 --init_NMF_iter=20 \
  --device='cuda:6'

time python main.py \
  --path2dataset="/media/data/tianming/Researches/RNAseq-seqFISH/version1/data/synthetic_500_100_30_15_0_0_i2" \
  --repli_list="['0']" -K=10 --use_spatial="['True']" --lambda_Sigma_x_inv=1e-5 --max_iter=50 --init_NMF_iter=20 \
  --device='cuda:0' --random_seed=2

time python main.py \
  --path2dataset="/work/magroup/tianming/Researches/SpiceMix/version1/data/scDesign2_starmap_addedNoise_inhibitory10_downsampled600_leakage90_nPatterns5_noiseAmp75_i0" \
  --repli_list="['0']" -K=12 --use_spatial="['True']" --lambda_Sigma_x_inv=1e-5 --max_iter=100 --init_NMF_iter=50 \
  --device='cuda:6'

path2source="/work/magroup/tianming/Researches/SpiceMix/version1/data"
path2target="/work/magroup/tianming/Researches/SpiceMix/version1/data2"
for n in scDesign2_starmap_addedNoise_inhibitory10_downsampled600_leakage90_nPatterns5_noiseAmp75
do
  for i in `seq 0 19`
  do
    path2src="$path2source/${n}_i${i}/files"
    path2tar="$path2target/${n}/files"
    ln -sT "$path2src/genes_0.txt" "$path2tar/genes_$i.txt"
    ln -sT "$path2src/expression_0.pkl" "$path2tar/expression_$i.pkl"
    ln -sT "$path2src/meta_0.csv" "$path2tar/meta_$i.csv"
    ln -sT "$path2src/neighborhood_0.txt" "$path2tar/neighborhood_$i.txt"
  done
done

