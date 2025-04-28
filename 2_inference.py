import torch
import json
import csv
import pickle
from helper import GraphDataset, load_data_from_file
from gnn import MPNNModel
from torch_geometric.loader import DataLoader

def predict_and_save(model_path, input_json, output_csv):
    """
    Predicts binding energies for graphs in an input JSON file and saves the results to a CSV file.
    
    Args:
        model_path (str): Path to the trained model .ckpt file.
        input_json (str): Path to the input JSON file containing graph data.
        output_csv (str): Path to the output CSV file to save the predictions.
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = MPNNModel(num_layers=4, emb_dim=64, edge_dim=3, out_dim=1)
    
    # Handle the prefixed state dict
    state_dict = checkpoint['state_dict']
    
    # Remove 'model.' prefix from all keys if present
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    # Load the modified state dict
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Load mean and std for denormalization
    train_mean, train_std = pickle.load(open('binding_e_mean_std_train.pkl', 'rb'))
    
    # Load the input data
    graph_data_json = load_data_from_file(input_json)
    smiles_list = list(graph_data_json.keys())
    graph_list = [graph_data_json[key] for key in smiles_list]
    
    # Create dataset
    dataset = GraphDataset(graph_list, target_mean=train_mean, target_std=train_std)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    # Prepare CSV writer
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Molecule', 'Element', 'Orbital', 'Binding Energy'])
        
        # Process each graph
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            with torch.no_grad():
                predictions = model(data)
            
            # Process each graph in the batch
            batch_size = 128
            for graph_idx in range(min(batch_size, len(smiles_list) - batch_idx * batch_size)):
                global_graph_idx = batch_idx * batch_size + graph_idx
                smiles = smiles_list[global_graph_idx]
                graph = graph_list[global_graph_idx]
                
                # Find node offset for this graph in the batch
                start_idx = 0
                for i in range(graph_idx):
                    prev_graph_idx = batch_idx * batch_size + i
                    if prev_graph_idx < len(graph_list):
                        prev_graph = graph_list[prev_graph_idx]
                        for node in prev_graph.nodes(data=True):
                            start_idx += len(node[1].get('orbitals', []))
                
                # Process each node in the graph
                for node_idx, (_, node_data) in enumerate(graph.nodes(data=True)):
                    atom_type = node_data.get('atom_type', '')
                    orbitals = node_data.get('orbitals', [])
                    
                    for orbital_idx, orbital in enumerate(orbitals):
                        pred_idx = start_idx + sum(len(graph.nodes[n].get('orbitals', [])) for n in range(node_idx)) + orbital_idx
                        
                        # Handle special case for -1 entries
                        if orbital == -1:
                            # Write -1 entries directly
                            writer.writerow([smiles, atom_type, -1, -1])
                        else:
                            # Get prediction and denormalize it
                            if pred_idx < len(predictions):
                                pred_energy = predictions[pred_idx].item() * train_std + train_mean
                                writer.writerow([smiles, atom_type, orbital, f"{pred_energy:.2f}"])
                            else:
                                print(f"Warning: Index out of range for prediction {pred_idx}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict binding energies and save to CSV')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model .ckpt file')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    
    args = parser.parse_args()
    
    predict_and_save(args.model, args.input, args.output)