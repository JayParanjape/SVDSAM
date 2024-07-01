# S-SAM
This repository contains the code for **Low-Rank Adaptation of Segment Anything Model for Surgical Scene Segmentation**

## Environment File
Create a new conda environment with the config file given in the repository as follows:
```
conda env create -f ssam_env.yml
conda activate s-sam
```

## General file descriptions
- data_transforms/*.py - data transforms defined here for different datasets.
- data_utils.py - functions to generate dataloaders for different datasets
- model.py - model architectures defined here
- prompt_adapted_segment_anything/modeling/svd_layers.py - code for the singular value tuning modifications used in the model
- train.py - code for general training, common to all datasets
- driver_scratchpad.py - driver code for training models. 
- eval/*/generate_predictions.py - code for generating results for a given dataset
- eval/*/generate_predictions.sh - script to run generate_predictions for generating results for all labels of interest.
- model_loratuning.yml - config file for defining various model hyperparameters for LoRASAM
- config_<dataset_name>.yml - config file for defining various dataset related hyperparameters
  
## Example Usage for Training
```
python driver_scratchpad.py --model_config model_svdtuning.yml --data_config config_cholec8k.yml --save_path "./temp.pth"
```
Please refer to driver_scratchpad.py for other command line options and parameters.

## Example Usage for Evaluation
```
cd eval/cholec8k

bash generate_predictions_cholec.sh
```

## Citation
```
To be added
```

Please feel free to reach out to me or raise an issue in case of trouble while running the code.
