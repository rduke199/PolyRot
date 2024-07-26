import os
import pandas as pd
from PolyRot.utils import *
from PolyRot.chain_dimensions import PolymerRotate
from PolyRot.pes_classification import predict_class, predict_subclass

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import ChemicalForceFields, rdMolTransforms

IPythonConsole.ipython_3d = True


class CentDihed:
    """
    Class to predict central dihedral PES
    """

    def __init__(self, smiles: str, num_confs: int = 5, max_iters: int = 5000, verbose: int = 1):
        """
        Initializes the CentDihedPred object.

        :param smiles: str, SMILES string of the molecule.
        :param num_confs: int, number of conformers to search.
        :param max_iters: int, maximum number of iterations for optimization.
        :param verbose: int, verbosity level for printing (default=1).
        """
        self.verbose = verbose  # Store the verbosity level
        self.smiles = smiles  # Store the SMILES string
        print(f"Initializing CentDihed for {smiles}") if self.verbose > 2 else None
        self.num_confs = num_confs  # Store the number of conformers to search
        self.max_iters = max_iters  # Store the maximum iterations for optimization

        # Create an RDKit molecule from the provided SMILES string
        self.rdk_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        self.best_conf_id = None

    @property
    def best_structure(self):
        """
        Return XYZ structure of conformer id with the lowest energy. May take time to run.
        """
        print("Getting best structure") if self.verbose > 2 else None
        self.best_conf_id = self.best_conf_id or self.get_best_conf_id()
        return Chem.rdmolfiles.MolToXYZBlock(self.rdk_mol, confId=self.best_conf_id)

    @property
    def central_dihed_idx(self):
        """
        Identifies atoms for central dihedral angle calculation.
        """
        print("Finding central dihedral angle idx") if self.verbose > 2 else None
        # find all potential center bonds (single bonds between carbons)
        potential_cnt_bond = []
        for bond in self.rdk_mol.GetBonds():
            if str(bond.GetBondType()) == 'SINGLE':
                atom_a = bond.GetBeginAtom()
                atom_b = bond.GetEndAtom()
                bond_atoms = [atom_a, atom_b]
                if atom_a.GetAtomicNum() == 6 and atom_b.GetAtomicNum() == 6:
                    potential_cnt_bond.append(bond_atoms)

        # find central bond
        num_cnt_bond = int((len(potential_cnt_bond) - 1) / 2)
        cnt_bond = potential_cnt_bond[num_cnt_bond]

        cnt_atom_a = cnt_bond[0]
        cnt_atom_b = cnt_bond[1]
        dihed1 = []
        dihed2 = []

        # assemble list of atoms in first dihedral angle
        for n in cnt_atom_a.GetNeighbors():
            if n.GetIdx() != cnt_atom_b.GetIdx():
                dihed1.append(n.GetIdx())
                dihed1.extend((cnt_atom_a.GetIdx(), cnt_atom_b.GetIdx()))
                break
        for n in cnt_atom_b.GetNeighbors():
            if n.GetIdx() != cnt_atom_a.GetIdx():
                dihed1.append(n.GetIdx())
                break
        # assemble list of atoms in second dihedral angle
        for n in cnt_atom_a.GetNeighbors():
            dihed2.append(n.GetIdx()) if n.GetIdx() not in dihed1 else dihed1
        dihed2.extend((cnt_atom_a.GetIdx(), cnt_atom_b.GetIdx()))
        for n in cnt_atom_b.GetNeighbors():
            dihed2.append(n.GetIdx()) if n.GetIdx() not in dihed1 else dihed1

        return [dihed1, dihed2]

    def dihed_rotator(self, degree: float, central_dihed_idx: int = 0) -> object:
        """
        Rotates a given dihedral angle to a specified degree.

        :param degree: float, target dihedral angle in degrees.
        :param central_dihed_idx: int, index for central dihedral.
        :return: XYZ coordinates after rotation.
        """
        print("Rotating dihedral") if self.verbose > 2 else None
        rdMolTransforms.SetDihedralDeg(self.rdk_mol.GetConformer(), *self.central_dihed_idx[central_dihed_idx],
                                       float(degree))
        AllChem.EmbedMolecule(self.rdk_mol)
        return AllChem.rdmolfiles.MolToXYZBlock(self.rdk_mol)

    def get_best_conf_id(self):
        """
        Finds the lowest energy conformer ID for the molecule. MAY TAKES TIME TO RUN

        :return: int, ID of the lowest energy conformer.
        """
        print("Finding lowest energy conformer") if self.verbose > 1 else None
        results = {}
        AllChem.EmbedMultipleConfs(self.rdk_mol, numConfs=self.num_confs, params=AllChem.ETKDG())
        results_MMFF = AllChem.MMFFOptimizeMoleculeConfs(self.rdk_mol, maxIters=self.max_iters)
        for i, result in enumerate(results_MMFF):
            results[i] = result[1]
        best_idx = min(results, key=results.get)
        return best_idx

    def find_torsion_conf(self, dihedral_angle: float, draw_structure: bool = False, central_dihed_idx: int = 0):
        """
        Finds the lowest energy conformer while holding specified dihedral angle.

        :param dihedral_angle: float, target dihedral angle in degrees.
        :param draw_structure: bool, whether to draw the lowest energy structure.
        :param central_dihed_idx: int, index for central dihedral.
        :return: XYZ coordinates of the optimized conformer.
        """

        dihed_idxs = self.central_dihed_idx[central_dihed_idx]
        self.best_conf_id = self.best_conf_id or self.get_best_conf_id()

        # Set up constraints for dihedral angle
        print("Setting dihedral") if self.verbose > 1 else None
        conf = self.rdk_mol.GetConformer(self.best_conf_id)
        rdMolTransforms.SetDihedralDeg(conf, *dihed_idxs, dihedral_angle)
        print("Pre-minimization dihedral is: ", rdMolTransforms.GetDihedralDeg(conf, *dihed_idxs)) if self.verbose > 2 else None
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
        Predicts potential energy surface structures for given dihedral angles.

        :param min_angle: int, minimum dihedral angle in degrees (default=0).
        :param max_angle: int, maximum dihedral angle in degrees (default=180).
        :param angle_increment: int, increment value for dihedral angle (default=10).
        :return: List of tuples containing dihedral angle and corresponding XYZ coordinates.
        """
        print("Predicting PES structures") if self.verbose > 2 else None
        degrees = np.arange(min_angle, max_angle + 1, angle_increment).tolist()
        return [(d, self.find_torsion_conf(dihedral_angle=d)) for d in degrees]

    def pred_pes_energies(self, ani_model, device, structures: list = None, return_column: str = "pred_ani2x_energy_kcal_zero"):
        """
        Predicts potential energy surface energies for structures.

        :param device: device used for torch (e.g., "cuda" or "cpu").
        :param ani_model: PyTorch model for ANI prediction.
        :param structures: list, list of tuples with dihedral angle and corresponding structure.
        :param return_column: str, column name to return (default="pred_ani2x_energy_kcal_zero").
        :return: List of predicted energies or DataFrame with energy data.
        """
        print("Predicting PES energies") if self.verbose > 2 else None
        if not structures:
            structures = self.pred_pes_structures()

        energy_df = pd.DataFrame(structures, columns=["degrees", "pred_struc"])
        energy_df.set_index("degrees", inplace=True)

        energy_df["pred_molecule"] = energy_df["pred_struc"].apply(lambda x: ase_molecule(x))
        energy_df["pred_ani2x_energy"] = energy_df["pred_molecule"].apply(lambda x: ani2x_energy(x, ani_model, device))
        energy_df["pred_ani2x_energy_kcal"] = energy_df['pred_ani2x_energy'] * 627.5
        energy_df["pred_ani2x_energy_kcal_zero"] = energy_df["pred_ani2x_energy_kcal"] - energy_df[
            "pred_ani2x_energy_kcal"].min()
        energy_df.index = energy_df.index.astype(float)
        return energy_df[return_column].to_list() if return_column else energy_df

    def predict_class(self):
        """
        Use classification models to predict PES class from monomer SMILES string.

        :return: str, curve class
        """
        return predict_class(self.smiles)

    def predict_subclass(self):
        """
        Use classification models to predict PES subclass from monomer SMILES string.

        :return: str, curve subclass
        """
        # Set up model and label encoder
        return predict_subclass(self.smiles)
