# AI4Science_hackathon_team_hum
Predicting orbital binding energies using GNN

# Preprocessing
Before model training, atomic properties need to be extracted into `atom_prop.json` by running `python 0_preprocessing.py`

# Model training
To train the model, run `python 1_model_training.py`

# Model inferenece
To run model inference, do `python 2_inference.py --model <path_to_trained_model> --input <path_to_inference_data_json_file> --output predictions.csv`