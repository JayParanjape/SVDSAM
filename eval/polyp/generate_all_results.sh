# declare -a StringArray=("CVC-300" "CVC-ClinicDB" "CVC-ColonDB" "ETIS-LaribPolypDB" "Kvasir")
declare -a StringArray=("CVC-ClinicDB")


for dataset in "${StringArray[@]}"; do
    echo "${dataset}"

    python generate_predictions.py --data_folder "/media/ubuntu/New Volume/jay/PolypDataset/TestDataset/TestDataset/${dataset}/images" --data_config config_polyp.yml  --model_config model_svdtuning.yml --save_path "./svdshiftscale_CVCclinic_tal_CE_1e-3_effort2_epoch500/${dataset}" --gt_path "/media/ubuntu/New Volume/jay/PolypDataset/TestDataset/TestDataset/${dataset}/masks" --labels_of_interest "Polyp" --pretrained_path "svdshiftscale_CVCclinic_tal_CE_1e-3_effort2_epoch500.pth"

    echo "......................."
done