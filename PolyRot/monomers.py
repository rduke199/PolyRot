import torch
import torchani

import pandas as pd
import networkx as nx
from PolyRot.utils import *
from PolyRot.chain_dimensions import PolymerRotate
from PolyRot.pes_classification import predict_class, predict_subclass

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import ChemicalForceFields, rdMolTransforms

IPythonConsole.ipython_3d = True


class Monomer:
    def __init__(self, smiles: str, num_confs: int = 5, max_iters: int = 5000, verbose: int = 1):
        self.verbose = verbose
        self.smiles = smiles
        print(f"Initializing FindDihed for {smiles}") if self.verbose > 2 else None
        self.num_confs = num_confs
        self.max_iters = max_iters
        self.rdk_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        self.best_conf_id = None

    def molecule_to_graph(self):
        """
        Convert RDKit molecule to NetworkX graph.
        """
        mol = self.rdk_mol
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=str(bond.GetBondType()))

        return G

    def find_bridges(self, G):
        """
        Find bridges in the graph.
        """
        bridges = list(nx.bridges(G))
        return bridges

    def label_edges(self, G, bridges):
        """
        Label all edges in the graph as bridges or non-bridges.
        """
        edge_labels = {}
        for edge in G.edges():
            if edge in bridges or (edge[1], edge[0]) in bridges:
                edge_labels[edge] = 'bridge'
            else:
                edge_labels[edge] = 'non-bridge'
        return edge_labels

    def find_dihedral_edges(self, G, bridges):
        """
        Identify edges linking two rings.
        """
        dihedral_edges = []

        for edge in bridges:
            v1, v2 = edge
            v1_neighbors = list(G.neighbors(v1))
            v2_neighbors = list(G.neighbors(v2))

            v1_non_bridge_neighbors = [n for n in v1_neighbors if (v1, n) not in bridges and (n, v1) not in bridges]
            v2_non_bridge_neighbors = [n for n in v2_neighbors if (v2, n) not in bridges and (n, v2) not in bridges]

            if v1_non_bridge_neighbors and v2_non_bridge_neighbors:
                dihedral_edges.append(edge)

        return dihedral_edges

    def find_other_dihedral_atoms(self, G, dihedral_edges, bridges):
        """
        Find the other two atoms for the dihedral angle.
        """
        dihedral_atoms = []
        for edge in dihedral_edges:
            v1, v2 = edge
            v1_neighbors = [n for n in G.neighbors(v1) if (v1, n) not in bridges and (n, v1) not in bridges]
            v2_neighbors = [n for n in G.neighbors(v2) if (v2, n) not in bridges and (n, v2) not in bridges]

            v1_neighbors.sort(key=lambda x: G.nodes[x]['atomic_num'], reverse=True)
            v2_neighbors.sort(key=lambda x: G.nodes[x]['atomic_num'], reverse=True)

            if v1_neighbors and v2_neighbors:
                dihedral_atoms.append((v1_neighbors[0], v1, v2, v2_neighbors[0]))

        return dihedral_atoms

    @property
    def dihedral_atoms(self):
        """
        Identifies atoms for central dihedral angle calculation using graph theory.
        """
        G = self.molecule_to_graph()
        bridges = self.find_bridges(G)
        edge_labels = self.label_edges(G, bridges)
        dihedral_edges = self.find_dihedral_edges(G, bridges)
        dihedral_atoms = self.find_other_dihedral_atoms(G, dihedral_edges, bridges)

        if self.verbose > 2:
            print("Graph nodes:", G.nodes(data=True))
            print("Graph edges:", G.edges(data=True))
            print("Bridges:", bridges)
            print("Dihedral edges:", dihedral_edges)
            print("Dihedral atoms:", dihedral_atoms)

        return dihedral_atoms

    @staticmethod
    def visualize_dihedral_edges(smiles, dihedral_atoms):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        # Draw the molecule
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:', '')
    
        # Convert to NetworkX graph
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=str(bond.GetBondType()))
    
        # Get positions for the nodes
        pos = nx.spring_layout(G)
    
        # Draw the graph
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, font_weight='bold')
    
        # Highlight dihedral edges and nodes
        for dihedral in dihedral_atoms:
            # Dihedral atoms
            atoms = [dihedral[0], dihedral[1], dihedral[2], dihedral[3]]
            edges = [(dihedral[0], dihedral[1]), (dihedral[1], dihedral[2]), (dihedral[2], dihedral[3])]
    
            # Highlight nodes
            nx.draw_networkx_nodes(G, pos, nodelist=atoms, node_color='red')
    
            # Highlight edges
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
    
        plt.title('Molecule with Highlighted Dihedrals')
        plt.show()
    
        return svg

    @property
    def best_structure(self):
        """
        Return XYZ structure of conformer id with the lowest energy. May take time to run.
        """
        print("Getting best structure") if self.verbose > 2 else None
        self.best_conf_id = self.best_conf_id or self.get_best_conf_id()
        return Chem.rdmolfiles.MolToXYZBlock(self.rdk_mol, confId=self.best_conf_id)

    def dihed_rotator(self, degree: float, dihed_indices: list) -> object:
        """
        Rotates a given dihedral angle to a specified degree.

        :param degree: float, target dihedral angle in degrees.
        :param dihed_indices: list, list of dihedral atom indices
        :return: XYZ coordinates after rotation.
        """
        print("Rotating dihedral") if self.verbose > 2 else None
        rdMolTransforms.SetDihedralDeg(self.rdk_mol.GetConformer(), *dihed_indices, float(degree))
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

    def find_torsion_conf(self, dihedral_angle: float, dihed_indices: list, draw_structure: bool = False):
        """
        Finds the lowest energy conformer while holding specified dihedral angle.

        :param dihedral_angle: float, target dihedral angle in degrees.
        :param dihed_indices: list, list of dihedral atom indices
        :param draw_structure: bool, whether to draw the lowest energy structure.
        :return: XYZ coordinates of the optimized conformer.
        """

        self.best_conf_id = self.best_conf_id or self.get_best_conf_id()

        # Set up constraints for dihedral angle
        print("Setting dihedral") if self.verbose > 1 else None
        conf = self.rdk_mol.GetConformer(self.best_conf_id)
        rdMolTransforms.SetDihedralDeg(conf, *dihed_indices, dihedral_angle)
        print("Pre-minimization dihedral is: ", rdMolTransforms.GetDihedralDeg(conf, *dihed_indices)) if self.verbose > 2 else None
        mp = ChemicalForceFields.MMFFGetMoleculeProperties(self.rdk_mol)
        ff = ChemicalForceFields.MMFFGetMoleculeForceField(self.rdk_mol, mp)
        ff.MMFFAddTorsionConstraint(*dihed_indices, False, dihedral_angle - 0.5, dihedral_angle + 0.5, 100.0)

        # Optimize conformers
        print("Optimizing") if self.verbose > 1 else None
        r = ff.Minimize()
        final_structure = Chem.rdmolfiles.MolToXYZBlock(self.rdk_mol, confId=self.best_conf_id)

        # Check dihedral angle
        print("Dihedral is: ", rdMolTransforms.GetDihedralDeg(conf, *dihed_indices)) if self.verbose > 1 else None

        # Draw
        if draw_structure:
            IPythonConsole.drawMol3D(self.rdk_mol, confId=self.best_conf_id)

        return final_structure

    def pred_pes_structures(self, dihed_idx: int = 0, min_angle: int = 0, max_angle: int = 180, angle_increment: int = 10):
        """
        Predicts potential energy surface structures for given dihedral angles.

        :param dihed_idx: list, index of dihedral angle to be used. 
        :param min_angle: int, minimum dihedral angle in degrees (default=0).
        :param max_angle: int, maximum dihedral angle in degrees (default=180).
        :param angle_increment: int, increment value for dihedral angle (default=10).
        :return: List of tuples containing dihedral angle and corresponding XYZ coordinates.
        """
        print("Predicting PES structures") if self.verbose > 2 else None
        degrees = np.arange(min_angle, max_angle + 1, angle_increment).tolist()
        dihed_indices = list(self.dihedral_atoms[dihed_idx])
        return [(d, self.find_torsion_conf(dihedral_angle=d, dihed_indices=dihed_indices)) for d in degrees]

    def pred_pes_energies(self, ani_model, device, structures: list, return_column: str = "pred_ani2x_energy_kcal_zero"):
        """
        Predicts potential energy surface energies for structures.

        :param device: device used for torch (e.g., "cuda" or "cpu").
        :param ani_model: PyTorch model for ANI prediction.
        :param structures: list, list of tuples with dihedral angle and corresponding structure.
        :param return_column: str, column name to return (default="pred_ani2x_energy_kcal_zero").
        :return: List of predicted energies or DataFrame with energy data.
        """
        print("Predicting PES energies") if self.verbose > 2 else None

        energy_df = pd.DataFrame(structures, columns=["degrees", "pred_struc"])
        energy_df.set_index("degrees", inplace=True)

        energy_df["pred_molecule"] = energy_df["pred_struc"].apply(lambda x: ase_molecule(x))
        energy_df["pred_ani2x_energy"] = energy_df["pred_molecule"].apply(lambda x: ani2x_energy(x, ani_model, device))
        energy_df["pred_ani2x_energy_kcal"] = energy_df['pred_ani2x_energy'] * 627.5
        energy_df["pred_ani2x_energy_kcal_zero"] = energy_df["pred_ani2x_energy_kcal"] - energy_df[
            "pred_ani2x_energy_kcal"].min()
        energy_df.index = energy_df.index.astype(float)
        return energy_df[return_column].to_list() if return_column else energy_df

    def pred_dihed_pes(self, dihed_idx: int, ani_model, device, return_column: str = "pred_ani2x_energy_kcal_zero", **kwargs):
        """
        Predicts potential energy surface energies for structures.

        :param dihed_idx: list, index of dihedral angle to be used.
        :param device: device used for torch (e.g., "cuda" or "cpu").
        :param ani_model: PyTorch model for ANI prediction.
        :param return_column: str, column name to return (default="pred_ani2x_energy_kcal_zero").
        :return: List of predicted energies or DataFrame with energy data.
        """
        print("Predicting PES energies") if self.verbose > 2 else None
        structures = self.pred_pes_structures(dihed_idx, **kwargs)
        return self.pred_pes_energies(ani_model, device, structures, return_column=return_column)

    def predict_class(self):
        """
        Use classification models to predict PES class from monomer SMILES string.

        :return: str, curve class
        """
        warnings.warn("USE WITH CAUTION. This ML model is for use only in a small chemical space. See "
                      "https://doi.org/10.1021/acs.macromol.3c00824 for more detials. ")
        return predict_class(self.smiles)

    def predict_subclass(self):
        """
        Use classification models to predict PES subclass from monomer SMILES string.

        :return: str, curve subclass
        """
        warnings.warn("USE WITH CAUTION. This ML model is for use only in a small chemical space. See "
                      "https://doi.org/10.1021/acs.macromol.3c00824 for more detials. ")
        return predict_subclass(self.smiles)

    def generate_PolyRot(self, temperature: float = 700):
        from PolyRot.central_dihedral import CentDihed

        # Setup TorchANI
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ani_model = torchani.models.ANI2x(periodic_table_index=True).to(device)

        # Produce PES prediction
        monomer = CentDihed(smiles="CCCC", num_confs=5)
        monomer.pred_pes_energies(device=device, ani_model=ani_model)
        bond_lengths = []  # TODO
        ring_lengths = []  # TODO
        deflection_angles = []  # TODO
        dihed_energies = [self.pred_pes_energies(i, ani_model, device) for i, _ in enumerate(self.dihedral_atoms)]

        polymer = PolymerRotate(bond_lengths=bond_lengths, ring_lengths=ring_lengths,
                                deflection_angles=deflection_angles, dihed_energies=dihed_energies, temp=temperature)

        return polymer
