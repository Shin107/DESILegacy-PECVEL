DIR="/user/animesh.sah/FP_CUTS/shape_combined_z_south_test/cumulative_combination_3"
for f in "$DIR"/*.fits; do
    name=$(basename "$f" .fits)
    name=${name##*_}
    python3 w_theta_specific_wrapper.py \
        --input_path "$f" \
        -i \
        --name_external "$name" \
        -p south \
        --dir /user/animesh.sah/DESI_PECVEL/south_combination_3 \
        -e jackknife
done