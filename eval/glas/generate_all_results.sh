echo "Training Accuracy: "
python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/GLAS/archive/train" --data_config config_glas.yml  --model_config model_svdtuning.yml --save_path "./tmp" --gt_path "/media/ubuntu/New Volume/jay/GLAS/archive" --labels_of_interest "Glands" --pretrained_path "svdshiftscale_glas_tal_CE_1e-4.pth"

echo "Testing Accuracy: "

python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/GLAS/archive/test" --data_config config_glas.yml  --model_config model_svdtuning.yml --save_path "./svdshiftscale_glas_tal_CE_1e-4/${dataset}" --gt_path "/media/ubuntu/New Volume/jay/PolypDataset/TestDataset/TestDataset/${dataset}/masks" --labels_of_interest "Glands" --pretrained_path "svdshiftscale_glas_tal_CE_1e-4.pth"

echo "......................."