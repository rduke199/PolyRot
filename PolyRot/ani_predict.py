import torch
import torchani
from pymatgen.io.ase import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ani_model = torchani.models.ANI2x(periodic_table_index=True).to(device)

def aseatoms2ani(molecule):
    coordinates = torch.tensor([molecule.arrays["positions"]], requires_grad=True, device=device).float()
    species = torch.tensor([molecule.arrays["numbers"]], dtype=torch.long, device=device)
    return species, coordinates

def ani2x_energy(x):
  return ani_model(aseatoms2ani(x)).energies.item()

def molecule(x):
  molecule = Molecule.from_str(x,fmt="xyz")
  return AseAtomsAdaptor().get_atoms(molecule)