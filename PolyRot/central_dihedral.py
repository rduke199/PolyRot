from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.structure import Molecule
from pymatgen.io.gaussian import GaussianInput

from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem import MolFromSmiles, AllChem


def get_central_dihed(rdk_mol):
    """
    function for recording central dihedral atom numbers
    """
    # find all potential center bonds (single bonds between carbons)
    potential_cntbond = []
    for bond in rdk_mol.GetBonds():
        if str(bond.GetBondType()) == 'SINGLE':
            atom_a = bond.GetBeginAtom()
            atom_b = bond.GetEndAtom()
            bond_atoms = [atom_a, atom_b]
            if atom_a.GetAtomicNum() == 6 and atom_b.GetAtomicNum() == 6:
                potential_cntbond.append(bond_atoms)

    # find central bond
    num_cnt_bond = int((len(potential_cntbond) - 1) / 2)
    cnt_bond = potential_cntbond[num_cnt_bond]

    cntatom_a = cnt_bond[0]
    cntatom_b = cnt_bond[1]
    dihed1 = []
    dihed2 = []

    # assemble list of atoms in first dihedral angle
    for n in cntatom_a.GetNeighbors():
        if n.GetIdx() != cntatom_b.GetIdx():
            dihed1.append(n.GetIdx())
            dihed1.extend((cntatom_a.GetIdx(), cntatom_b.GetIdx()))
            break
    for n in cntatom_b.GetNeighbors():
        if n.GetIdx() != cntatom_a.GetIdx():
            dihed1.append(n.GetIdx())
            break
    # assemble list of atoms in second dihedral angle
    for n in cntatom_a.GetNeighbors():
        dihed2.append(n.GetIdx()) if n.GetIdx() not in dihed1 else dihed1
    dihed2.extend((cntatom_a.GetIdx(), cntatom_b.GetIdx()))
    for n in cntatom_b.GetNeighbors():
        dihed2.append(n.GetIdx()) if n.GetIdx() not in dihed1 else dihed1

    return [dihed1, dihed2]


def rotated_gaus_input(smiles, dihed_degree, paramset, out_file=None):
    rdkmol = MolFromSmiles(smiles)
    rdkmol_hs = AddHs(rdkmol)
    init_structure = AllChem.rdmolfiles.MolToXYZBlock(rdkmol_hs)
    mol = Molecule.from_str(init_structure, 'xyz')

    # Get dihedral angle atoms to be rotated and frozen
    dihed_idxs = [i + 1 for i in get_central_dihed(rdkmol)[0]]

    # generate the input for gaussian energy run
    paramset.route_parameters.update(
        {"opt": "modredundant", 'SCF': '(MaxCycle=512)', 'Int': '(Grid=SuperFine)'})
    gauss_inp = generate_gaussian_input(paramset=paramset, mol=mol)
    out_file = out_file or "mol_rot_{:.2f}.log".format(dihed_degree)
    gauss_inp.write_file(out_file, cart_coords=True)

    # Freeze dihedral angle in com file
    with open(out_file, 'r') as com:
        lines = com.readlines()[:-2]
    lines.append("D {} {} {} {} ={} B\n".format(dihed_idxs[0], dihed_idxs[1], dihed_idxs[2], dihed_idxs[3], dihed_degree))
    lines.append("D {} {} {} {} F\n\n".format(dihed_idxs[0], dihed_idxs[1], dihed_idxs[2], dihed_idxs[3]))
    with open(out_file, 'w') as com:
        com.writelines(lines)
    return out_file


def generate_gaussian_input(paramset=None, mol=None, dieze_tag="#P"):
    route_parameters = paramset.route_parameters
    input_parameters = paramset.input_parameters
    link0_parameters = paramset.link0_parameters
    charge = paramset.charge
    multiplicity = paramset.multiplicity
    functional = paramset.functional
    basis_set = paramset.basis_set

    ginput = GaussianInput(mol=mol, charge=charge, spin_multiplicity=multiplicity,
                           title=None, functional=functional, basis_set=basis_set, link0_parameters=link0_parameters,
                           route_parameters=route_parameters, input_parameters=input_parameters, dieze_tag=dieze_tag)
    return ginput


class GaussianParameters(MSONable):
    def __init__(self, route_parameters=None, charge=None, multiplicity=None, basis_set=None,
                 functional=None, input_parameters=None, link0_parameters=None):
        self.route_parameters = route_parameters or {}
        self.input_parameters = input_parameters or {}
        self.link0_parameters = link0_parameters or {}
        self.charge = charge or 0
        self.multiplicity = multiplicity or 1
        self.basis_set = basis_set
        self.functional = functional


class GausParamSet(MSONable):
    def __init__(self, name=None, **kwargs):
        self.name = name or ""
        for calc, params in kwargs.items():
            exec("self.{} = sett(params, GaussianParameters())".format(calc))

    @staticmethod
    def from_json(jsonfile):
        d = loadfn(jsonfile)
        return GausParamSet.from_dict(d)