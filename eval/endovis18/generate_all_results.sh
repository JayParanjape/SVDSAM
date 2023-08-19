python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_1-20230624T000458Z-001/seq_1/left_frames" --data_config config_endovis18_test.yml  --model_config model_svdtuning.yml --save_path "./svdbiassam_results/seq1/surgicalinstrument" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_1-20230624T000458Z-001/seq_1/labels" --labels_of_interest "surgical instrument" --pretrained_path "svdbias_onlyshift_ev18_tal_dice_1e-4.pth"

echo "......................."

python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_2-20230624T000507Z-001/seq_2/left_frames" --data_config config_endovis18_test.yml  --model_config model_svdtuning.yml --save_path "./svdbiassam_results/seq2/surgicalinstrument" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_2-20230624T000507Z-001/seq_2/labels" --labels_of_interest "surgical instrument" --pretrained_path "svdbias_onlyshift_ev18_tal_dice_1e-4.pth"


echo "......................."

python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_3-20230624T000508Z-001/seq_3/left_frames" --data_config config_endovis18_test.yml  --model_config model_svdtuning.yml --save_path "./svdbiassam_results/seq3/surgicalinstrument" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_3-20230624T000508Z-001/seq_3/labels" --labels_of_interest "surgical instrument" --pretrained_path "svdbias_onlyshift_ev18_tal_dice_1e-4.pth"


echo "......................."

python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/endovis18/seq_4-20230624T000509Z-001/seq_4/left_frames" --data_config config_endovis18_test.yml  --model_config model_svdtuning.yml --save_path "./svdbiassam_results/seq4/surgicalinstrument" --gt_path "/media/ubuntu/New Volume/jay/endovis18/seq_4-20230624T000509Z-001/seq_4/labels" --labels_of_interest "surgical instrument" --pretrained_path "svdbias_onlyshift_ev18_tal_dice_1e-4.pth"


echo "......................."