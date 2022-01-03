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
  --path2dataset="/work/magroup/tianming/Researches/SpiceMix/version1/data/scDesign2_starmap_addedNoise_inhibitory10_downsampled600_leakage90_nPatterns5_noiseAmp75_i0" \
  --repli_list="['0']" -K=12 --use_spatial="['True']" --lambda_Sigma_x_inv=1e-5 --max_iter=100 --init_NMF_iter=50 \
  --device='cuda:6'
