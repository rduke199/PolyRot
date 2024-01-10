import torch
import pandas as pd
from tqdm import tqdm
from pymatgen.io.ase import *

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalForceFields, rdMolTransforms
from rdkit.Chem.Draw import IPythonConsole

IPythonConsole.ipython_3d = True


class CentDihedPred:
    """
    Class to predict central dihedral PES
    """

    def __init__(self, smiles, device, ani_model, num_confs=50, max_iters=5000, verbose=1):
        """
        :param smiles: str, SMILES string
        :param device: 
        :param ani_model: 
        :param num_confs: int, number of conformers to search
        :param max_iters: int, maximum number of iterations 
        """
        self.smiles = smiles
        self.device = device
        self.ani_model = ani_model
        self.num_confs = num_confs
        self.max_iters = max_iters
        self.verbose = verbose

        self.rdk_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        self.best_conf_id = self.find_lowest_e_conf_id()
        self.best_structure = Chem.rdmolfiles.MolToXYZBlock(self.rdk_mol, confId=self.best_conf_id)

    def aseatoms2ani(self, molecule):
        """
        """
        coordinates = torch.tensor([molecule.arrays["positions"]], requires_grad=True, device=self.device).float()
        species = torch.tensor([molecule.arrays["numbers"]], dtype=torch.long, device=self.device)
        return species, coordinates

    def ani2x_energy(self, x):
        """
        """
        return self.ani_model(self.aseatoms2ani(x)).energies.item()

    @staticmethod
    def ase_molecule(xyz_coord: str):
        """
        """
        molecule = Molecule.from_str(xyz_coord, fmt="xyz")
        return AseAtomsAdaptor().get_atoms(molecule)

    @property
    def central_dihed(self):
        """
        function for recording central dihedral atom numbers
        """
        # find all potential center bonds (single bonds between carbons)
        potential_cntbond = []
        for bond in self.rdk_mol.GetBonds():
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

    def dihed_rotator(self, degree: float, central_dihed_idx: int = 0):
        """
        function for rotating a given dihedral angle to a given degree
        """
        rdMolTransforms.SetDihedralDeg(self.rdk_mol.GetConformer(), *self.central_dihed[central_dihed_idx],
                                       float(degree))
        AllChem.EmbedMolecule(self.rdk_mol)
        return AllChem.rdmolfiles.MolToXYZBlock(self.rdk_mol)

    def find_lowest_e_conf_id(self):
        """
        Find the lowest energy conformer for a molecule with RDKit
        :return: int, lowest energy conformer id
        """
        results = {}
        AllChem.EmbedMultipleConfs(self.rdk_mol, numConfs=self.num_confs, params=AllChem.ETKDG())
        results_MMFF = AllChem.MMFFOptimizeMoleculeConfs(self.rdk_mol, maxIters=self.max_iters)
        for i, result in tqdm(enumerate(results_MMFF)):
            results[i] = result[1]
        best_idx = min(results, key=results.get)
        return best_idx

    def find_torsion_conf(self, dihedral_angle: float, draw_structure: bool = False, central_dihed_idx: int = 0):
        """
        Find the lowest energy conformer for a molecule with RDKit while holding specified dihedral angle
        :param dihedral_angle: float, target dihedral angle in degrees
        :param draw_structure: bool, whether to draw the lowest energy structure
        :return: str (xyz coordinates if not drawing), RDKit Image object if drawing
        """

        dihed_idxs = self.central_dihed[central_dihed_idx]

        # Set up constraints for dihedral angle
        print("Setting dihedral") if self.verbose > 1 else None
        conf = self.rdk_mol.GetConformer(self.best_conf_id)
        rdMolTransforms.SetDihedralDeg(conf, *dihed_idxs, dihedral_angle)
        mp = ChemicalForceFields.MMFFGetMoleculeProperties(self.rdk_mol)
        ff = ChemicalForceFields.MMFFGetMoleculeForceField(self.rdk_mol, mp)
        ff.MMFFAddTorsionConstraint(*dihed_idxs, False, dihedral_angle - 0.5, dihedral_angle + 0.5, 100.0)

        # Optimize conformers
        print("Optimizing") if self.verbose > 1 else None
        r = ff.Minimize()
        final_structure = Chem.rdmolfiles.MolToXYZBlock(self.rdk_mol, confId=self.best_conf_id)

        # Check dihedral angle
        print("Dihedral is: ", rdMolTransforms.GetDihedralDeg(conf, *dihed_idxs)) if self.verbose > 1 else None

        # Draw
        if draw_structure:
            IPythonConsole.drawMol3D(self.rdk_mol, confId=self.best_conf_id)

        return final_structure

    def pred_pes_structures(self, min_angle: int = 0, max_angle: int = 180, angle_increment: int = 10):
        """
        """
        degrees = np.arange(min_angle, max_angle + 1, angle_increment).tolist()
        return [(d, self.find_torsion_conf(dihedral_angle=d)) for d in degrees]

    def pred_pes_energies(self, structures: list, return_column: str = "pred_ani2x_energy_kcal_zero"):
        """
        :param structures: list, list of tuples where the first item in each tuple is the degree and the second item is the structrue xyz coordinates 
        :param return_column: bool,
        """

        energy_df = pd.DataFrame(structures, columns=["degrees", "structure"])
        energy_df.set_index("degrees", inplace=True)

        energy_df["pred_struc"] = energy_df.apply(lambda x: self.find_torsion_conf(dihedral_angle=float(x.degrees)),
                                                  axis=1)
        energy_df["pred_molecule"] = energy_df["pred_struc"].apply(lambda x: self.ase_molecule(x))
        energy_df["pred_ani2x_energy"] = energy_df["pred_molecule"].apply(lambda x: self.ani2x_energy(x))
        energy_df["pred_ani2x_energy_kcal"] = energy_df['pred_ani2x_energy'] * 627.5
        energy_df["pred_ani2x_energy_kcal_zero"] = energy_df["pred_ani2x_energy_kcal"] - energy_df[
            "pred_ani2x_energy_kcal"].min()
        energy_df.index = energy_df.index.astype(float)
        return energy_df[return_column].to_list() if return_column else energy_df



