# save atomic properties to atom_prop dictionary
import json
from tqdm import tqdm
from pymatgen.core.periodic_table import Element
from mendeleev import element

atom_prop = {}

list_all_elements = [el.symbol for el in Element]

for ele in tqdm(list_all_elements):
    atom_prop[ele] = {}
    atom_prop[ele]['electronegativity'] = element(ele).electronegativity()
    atom_prop[ele]['vdw_radius'] = element(ele).vdw_radius
    atom_prop[ele]['eff_nuclear_charge'] = element(ele).zeff()

with open('atom_prop.json', 'w') as f:
        json.dump(atom_prop, f, indent=4)