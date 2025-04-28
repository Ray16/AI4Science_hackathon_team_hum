import json
import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
from pymatgen.core.periodic_table import Element

list_all_elements = [el.symbol for el in Element]
atom_prop = json.load(open('atom_prop.json', 'r'))

def load_data_from_file(filename):
    with open(filename, "r") as file_handle:
        string_dict = json.load(file_handle)
    return _load_data_from_string_dict(string_dict)

def load_data_from_string(json_string):
    string_dict = json.loads(json_string)
    return _load_data_from_string_dict(string_dict)

def _load_data_from_string_dict(string_dict):
    result_dict = {}
    for key in string_dict:
        graph_data = string_dict[key]
        if "edges" in graph_data and "links" not in graph_data:
            graph_data["links"] = graph_data.pop("edges")
        graph = nx.node_link_graph(graph_data)
        result_dict[key] = graph
    return result_dict

def write_data_to_json_string(graph_dict, **kwargs):
    json_string = json.dumps(graph_dict, default=nx.node_link_data, **kwargs)
    return json_string

def write_data_to_json_file(graph_dict, filename, **kwargs):
    with open(filename, "w") as file_handle:
        file_handle.write(write_data_to_json_string(graph_dict, **kwargs))

class GraphDataset(Dataset):
    def __init__(self, graph_list, transform=None, target_mean=None, target_std=None):
        super().__init__()
        self.graph_list = graph_list
        self.transform = transform
        self.target_mean = target_mean
        self.target_std = target_std
        self.all_elements = sorted([el.symbol for el in Element])
        self.element_to_index = {el: i for i, el in enumerate(self.all_elements)}
        self.bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE']
        self.bond_type_to_index = {el: i for i, el in enumerate(self.bond_types)}
        self.all_orbitals = ['1s', '2s', '2p', '2p3/2', '3s', '3p', '3p3/2', '3d', '3d5/2', '4s', '4p3/2', '4d', '4d5/2', '4f7/2', '5s', '5p3/2', '5d5/2', -1]
        self.orbital_to_index = {orb: i for i, orb in enumerate(self.all_orbitals)}

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        graph_data = self.graph_list[idx]
        num_nodes = graph_data.number_of_nodes()

        node_features_list = []
        target_property_list = []

        # Extract node features and targets for ALL nodes
        for _, node in sorted(graph_data.nodes(data=True)): # Ensure consistent order
            atom_type = node['atom_type']
            formal_charge = node['formal_charge']
            orbitals = node.get('orbitals', []) # Handle cases where orbitals might be missing
            binding_energies = node.get('binding_energies', []) # Handle missing binding energies

            atom_type_encoded = self._encode_atom_type(atom_type)
            formal_charge_encoded = [formal_charge]
            electronegativity = atom_prop.get(atom_type, {}).get('electronegativity')
            electronegativity_encoded = [electronegativity if electronegativity is not None else 0.0]
            vdw_radius_encoded = [atom_prop.get(atom_type, {}).get('vdw_radius', 0.0)]
            eff_nuclear_charge_encoded = [atom_prop.get(atom_type, {}).get('eff_nuclear_charge', 0.0)]
            base_node_feature = atom_type_encoded + formal_charge_encoded + electronegativity_encoded + vdw_radius_encoded + eff_nuclear_charge_encoded

            # Create features and targets for each orbital/energy pair
            for i in range(max(len(orbitals), len(binding_energies))):
                orbital = orbitals[i] if i < len(orbitals) else '1s' # Default orbital if missing
                energy = binding_energies[i] if i < len(binding_energies) else -1.0 # Default energy if missing

                orbital_encoded = self._encode_orbital(orbital)
                node_feature = base_node_feature + orbital_encoded
                node_features_list.append(node_feature)
                target_property_list.append(float(energy))

        x = torch.tensor(node_features_list, dtype=torch.float) if node_features_list else torch.empty((0, self._get_node_feature_dim()), dtype=torch.float)
        y = torch.tensor(target_property_list, dtype=torch.float).unsqueeze(1) if target_property_list else torch.empty((0, 1), dtype=torch.float)

        if self.target_mean is not None and self.target_std is not None:
            y = (y - self.target_mean) / self.target_std

        edge_list = []
        edge_features_list = []
        for u, v, edge in graph_data.edges(data=True):
            edge_list.append((u, v))
            edge_features_list.append(self._encode_bond_type(edge['bond_type']))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float) if edge_features_list else torch.empty((0, len(self.bond_types)), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        if self.transform:
            data = self.transform(data)

        return data

    def _encode_atom_type(self, atom_type):
        encoding = [0.0] * len(self.all_elements)
        if atom_type in self.element_to_index:
            index = self.element_to_index[atom_type]
            encoding[index] = 1.0
        return encoding

    def _encode_bond_type(self, bond_type):
        encoding = [0.0] * len(self.bond_types)
        if bond_type in self.bond_type_to_index:
            index = self.bond_type_to_index[bond_type]
            encoding[index] = 1.0
        return encoding

    def _encode_orbital(self, orbital):
        encoding = [0.0] * len(self.all_orbitals)
        if orbital in self.orbital_to_index:
            index = self.orbital_to_index[orbital]
            encoding[self.orbital_to_index[orbital]] = 1.0
            #print(f"Encoding orbital '{orbital}' at index {index}: {encoding}") # Debug print
        return encoding

    def _get_node_feature_dim(self):
        return len(self.all_elements) + 1 + len(self.all_orbitals)