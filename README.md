# AI4Science_hackathon_team_hum
Predicting orbital binding energies using GNN

# Preprocessing
Before model training, atomic properties need to be extracted into `atom_prop.json` by running `python 0_preprocessing.py`

# Model training
1. Activate conda environment `conda activate /project/ai4s-hackathon/hum/conda_env_hackathon`
2. Run `python 1_model_training.py`

# Model inferenece
To run model inference, do
```
python 2_inference.py --model /project/ai4s-hackathon/hum/test_pred/lightning_logs/graph_regression/epoch_384/checkpoints/best-model-epoch=317-val_loss=0.0160.ckpt --input graph_data.json --output predictions.csv
```

In general, the format is
```
python 2_inference.py --model <path_to_trained_model> --input <path_to_inference_data_json_file> --output predictions.csv
```