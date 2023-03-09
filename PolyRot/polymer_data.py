import os
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
from pymatgen.io.xyz import XYZ


class MoleculeRot:
    """
    MoleculeRot Object

    Creates an object for each molecule with a variety of properties including a dictionary (and a normalized
    dictionary) for its energies at each dihedral rotation. Each object also has descriptive properties such as name,
    smile ring number, unit number, polymer number, etc. There are only a couple class functions, but these draw the
    molecule structure and plot the PE curve.

    """

    def __init__(
            self,
            name,
            master_dir,
            unified_unconst=False,
            chromophore_fn="master_chromophore.json",
            rings_fn="master_rings.json",
            sidechain_fn="master_sidechain.json",
            # shape_fn="master_shapes.json",
            # shapeclass_fn="master_shapeclass.json",
            energy_fn="master_energies.json",
            homo_fn="master_homos.json",
            lumo_fn="master_lumos.json",
            unconst_fn="master_geomopt.json",
            omega_fn="master_omega.json",
            smiles_fn="master_smiles.json",
            centdihed_fn="master_centdihed_atoms.json",
            structures_fn="master_structures.json",
            anti_syn_fn="master_anti_syn.json",
            backbone_len_fn="master_backbone_length.json",
            length_masters_dir="lengths"
    ):
        # Note: when entering this analysis, all energies should be in eV
        self.name = name
        print(self.name)
        self.master_dir = master_dir
        self.unified_unconst = unified_unconst

        self.ring_num = int(self.name.split('_')[1])
        self.unit_num = int(self.name.split('_')[2])
        self.polymer_num = int(self.name.split('_')[3])
        self.substituents = (self.name.split('_')[4]).upper() if len(name.split('_')) > 4 else ""
        self.molecule = "{}_{}".format(self.ring_num, self.polymer_num)
        self.anti_syn = self.get_data_from_master_1unit(os.path.join(master_dir, anti_syn_fn))

        self.rings = self.get_data_from_master_1unit(os.path.join(master_dir, rings_fn))
        self.chromophore = str(self.get_data_from_master_1unit(os.path.join(master_dir, chromophore_fn)))
        self.side_chains = tuple(self.get_data_from_master_1unit(os.path.join(master_dir, sidechain_fn)))
        self.side_chains_str = str(self.side_chains)
        # self.curve_shape = self.get_data_from_master(os.path.join(master_dir, shape_fn))
        # self.curve_class = self.get_data_from_master(os.path.join(master_dir, shapeclass_fn))
        self.energy_dict = self.make_dict_floats(self.get_data_from_master(os.path.join(master_dir, energy_fn))) or {}
        self.homo_dict = self.make_dict_floats(self.get_data_from_master(os.path.join(master_dir, homo_fn))) or {}
        self.lumo_dict = self.make_dict_floats(self.get_data_from_master(os.path.join(master_dir, lumo_fn))) or {}
        self.tuned_omega_orig = self.get_data_from_master(os.path.join(master_dir, omega_fn))
        self.smiles = self.get_data_from_master(os.path.join(master_dir, smiles_fn))
        self.unconst_path = os.path.join(master_dir, unconst_fn)
        self.unconst_data = self.get_data_from_master(self.unconst_path)
        self.cent_diheds = self.get_data_from_master(os.path.join(master_dir, centdihed_fn))
        self.structures_data = self.make_dict_floats(self.get_data_from_master(os.path.join(master_dir, structures_fn)), values=False) or {}
        self.backbone_length = self.make_dict_floats(self.get_data_from_master(os.path.join(master_dir, backbone_len_fn)), values=False) or {}
        self.central_bond_length_dict = self.get_central_bond_length_dict() or {}
        self.norm_energy_dict = self.get_norm_energy() or {}
        self.probability = self.get_probability() or {}

        self.lm_bond_lengths = self.read_json(os.path.join(master_dir, length_masters_dir, "bond_lengths.json"))
        self.lm_ring_lengths = self.read_json(os.path.join(master_dir, length_masters_dir, "ring_lengths.json"))
        self.lm_deflection_angles = self.read_json(os.path.join(master_dir, length_masters_dir, "deflection_angles.json"))
        self.lm_dihed_energies = self.read_json(os.path.join(master_dir, length_masters_dir, "dihed_energies.json"))

    def __str__(self):
        return f'name: {self.name}\n{self.ring_num} ring type, {self.unit_num} monomer units, {self.substituents} ' \
               f'substituents\nenergy dictionary: {self.norm_energy_dict}'

    def __all_props__(self):
        attrs = [k for k, v in vars(type(self)).items() if isinstance(v, property)]
        props = {a: getattr(self, a) for a in attrs}
        props.update(self.__dict__)
        return props

    def get_data_from_master(self, master_file):
        with open(master_file, 'r') as fn:
            _dict = json.load(fn)
        mol_name = "_".join(self.name.split('_')[:4])
        try:
            orig_dict = [v for k, v in _dict.items() if k.startswith(mol_name)][0]
            if isinstance(orig_dict, dict) and self.anti_syn < 0:
                return {180-float(key): value for key, value in orig_dict.items()}
            else:
                return orig_dict
        except IndexError:
            print("Error. {} not in {}".format(mol_name, master_file))
            return None

    def get_data_from_master_1unit(self, master_file):
        mol_name = 'mols_{}_1_{:02d}'.format(self.ring_num, self.polymer_num)
        _dict = self.read_json(master_file)
        try:
            return [v for k, v in _dict.items() if k.startswith(mol_name)][0]
        except IndexError:
            print("Error. {} not in {}".format(mol_name, master_file))

    @staticmethod
    def read_json(json_file):
        with open(json_file, 'r') as fn:
            _dict = json.load(fn)
        return _dict

    @staticmethod
    def make_dict_floats(_dict, values=True):
        if _dict is not None:
            if values:
                return dict(sorted({float(key): float(value) for key, value in _dict.items()}.items()))
            else:
                return dict(sorted({float(key): value for key, value in _dict.items()}.items()))

    @property
    def rdkit_mol(self):
        try:
            mol_rdkit = Chem.MolFromSmiles(self.smiles)
            mol_rdkit.SetProp("_Name", self.name)
            return mol_rdkit
        except:
            print("Unable to convert smiles to rdkit mol: ", self.smiles)

    @property
    def molecule_type(self):
        if self.chromophore and self.side_chains_str:
            return self.chromophore + self.side_chains_str

    @property
    def chromophore_side(self):
        return '{}, {}'.format(self.chromophore, self.side_chains)

    @property
    def tuned_omega(self):
        if self.tuned_omega_orig is not None:
            return float("0." + str(self.tuned_omega_orig[1:]))

    @property
    def inv_tuned_omega(self):
        if self.tuned_omega is not None:
            return 1/self.tuned_omega

    @property
    def unified_unconst_data(self):
        """
        This returns the unconstrained energy data from the minimum energy length out of all lengths of this polymer,
        instead of the unconstrained energy data from this polymer at this particular length
        """
        units = [1, 3, 5, 7]
        angle_dict, energy_dict = {}, {}
        with open(self.unconst_path, 'r') as fn:
            _dict = json.load(fn)
        for unit in units:
            mol_name = 'mols_{}_{}_{:02d}_{}'.format(self.ring_num, unit, self.polymer_num, self.substituents)
            try:
                angle_dict[unit] = _dict[mol_name][0]
                energy_dict[unit] = _dict[mol_name][1]
            except KeyError:
                try:
                    angle_dict[unit], energy_dict[unit] = [v for k, v in _dict.items() if k.startswith(mol_name)][0]
                except IndexError:
                    pass
        try:
            min_energy_unit = min(energy_dict, key=energy_dict.get)
            return [angle_dict[min_energy_unit], energy_dict[min_energy_unit]]
        except ValueError:
            return None

    @property
    def unconst_energy(self):
        if self.unified_unconst:
            _dict = self.unified_unconst_data
        else:
            _dict = self.unconst_data
        if _dict is None:
            return None
        else:
            return _dict[1]

    @property
    def min_e_angle(self):
        if self.unified_unconst:
            _dict = self.unified_unconst_data
        else:
            _dict = self.unconst_data
        _dict = self.unconst_data
        if _dict is None:
            return None
        else:
            return abs(_dict[0])

    @property
    def adjusted_min_e_angle(self):
        if self.min_e_angle:
            return abs(90 - self.min_e_angle)

    def get_central_bond_length_dict(self):
        _dict = {}
        for degree in self.structures_data.keys():
            xyz_structure = self.structures_data[degree]
            mol = XYZ._from_frame_string(xyz_structure)
            cnt_atom1 = self.cent_diheds[0][1]
            cnt_atom2 = self.cent_diheds[0][2]
            cnt_bond_length = mol.get_distance(cnt_atom1, cnt_atom2)
            _dict[float(degree)] = cnt_bond_length
        _sorted_dict = dict(sorted(_dict.items()))
        return _sorted_dict

    @staticmethod
    def classify_curve(energy_dict, subcategories=False):
        e0, e90, e180, e40, e140 = energy_dict.get(0), energy_dict.get(90), energy_dict.get(180), energy_dict.get(
            40), energy_dict.get(140)
        if None in [e0, e90, e180]:
            return None
        elif max(energy_dict.values()) < 2:
            return 'rolling_hill'
        elif e180 > e90 and e0 > e90:
            if subcategories:
                if max([e180, e0]) > 2 * e90:
                    return 'w_high'
                return 'w_small'
            return 'w_shaped'
        elif abs(e0 - e180) < 1 and abs(e40 - e140) < 1 and e0 < 1:
            if subcategories:
                if max(energy_dict.values()) > 7:
                    return 'high_peak'
                return 'small_peak'
            return 'central_peak'
        else:
            if subcategories:
                if e180 > e90 or e0 > e90:
                    return 'welled_tilted'
                return 'non_welled_tilted'
            return 'tilted'

    @property
    def curve_class(self):
        if self.norm_energy_dict:
            return self.classify_curve(self.norm_energy_dict)

    @property
    def curve_subclass(self):
        if self.norm_energy_dict:
            return self.classify_curve(self.norm_energy_dict, subcategories=True)

    @property
    def min_bond_length(self):
        if self.central_bond_length_dict:
            return min(self.central_bond_length_dict.values())

    @property
    def bond_length_variance(self):
        if self.central_bond_length_dict:
            return max(self.central_bond_length_dict.values()) - self.min_bond_length

    @property
    def bond_length_diff90(self):
        if self.central_bond_length_dict and self.central_bond_length_dict.get(90.0):
            return self.central_bond_length_dict[90.0] - self.min_bond_length

    @property
    def bond_length_diff180(self):
        if self.central_bond_length_dict:
            if max(self.central_bond_length_dict, key=self.central_bond_length_dict.get) == 180.0:
                return self.central_bond_length_dict[180.0] - self.min_bond_length
            if max(self.central_bond_length_dict, key=self.central_bond_length_dict.get) == 0.0:
                return self.central_bond_length_dict[0.0] - self.min_bond_length

    @property
    def min_eng_bl(self):
        if self.norm_energy_dict and self.central_bond_length_dict:
            degree = min(self.norm_energy_dict, key=self.norm_energy_dict.get)
            return self.central_bond_length_dict.get(degree)

    @property
    def min_is_planar(self):
        if self.norm_energy_dict and self.central_bond_length_dict:
            degree = min(self.norm_energy_dict, key=self.norm_energy_dict.get)
            return True if degree == 0 or degree == 180 else False

    @property
    def donor(self):
        homo_dict = {
            "mol_phenyl": -9.337,
            "mol_pyrimidine": -10.053,
            "mol_pyridine_F": -9.981,
            "mol_pyridine_OCH3": -9.098,
            "mol_thiazole": -9.888,
            "mol_pyridine": -10.053,
            "mol_phenyl_FF": -9.704,
            "mol_phenyl_FOCH3": -8.956,
            "mol_thiophene_F": -9.427,
            "mol_thiophene": -9.437,
            "mol_phenyl_F": -9.221,
            "mol_thiophene_OCH3": -8.146,
            "mol_phenyl_OCH3": -8.244,
            "mol_phenyl_OCH3OCH3": -7.866
        }
        ring1 = self.rings[0]
        ring2 = self.rings[1]
        if homo_dict.get(ring1) >= homo_dict.get(ring2):
            return [ring1, homo_dict.get(ring2)]
        elif homo_dict.get(ring2) >= homo_dict.get(ring1):
            return [ring2, homo_dict.get(ring2)]

    @property
    def donor_HOMO(self):
        return self.donor[1]

    @property
    def acceptor(self):
        lumo_dict = {
            "mol_phenyl": 1.901,
            "mol_pyrimidine": 1.194,
            "mol_pyridine_F": 1.303,
            "mol_pyridine_OCH3": 1.651,
            "mol_thiazole": 1.614,
            "mol_pyridine": 1.608,
            "mol_phenyl_FF": 1.72,
            "mol_phenyl_FOCH3": 2.04,
            "mol_thiophene_F": 1.784,
            "mol_thiophene": 2.011,
            "mol_phenyl_F": 1.636,
            "mol_thiophene_OCH3": 1.822,
            "mol_phenyl_OCH3": 1.881,
            "mol_phenyl_OCH3OCH3": 2.036
        }
        ring1 = self.rings[0]
        ring2 = self.rings[1]
        if lumo_dict.get(ring1) <= lumo_dict.get(ring2):
            return [ring1, lumo_dict.get(ring1)]
        elif lumo_dict.get(ring2) <= lumo_dict.get(ring1):
            return [ring2, lumo_dict.get(ring1)]

    @property
    def acceptor_LUMO(self):
        return self.acceptor[1]

    @property
    def donor_acceptor_gap(self):
        homo_dict = {
            "mol_phenyl": -9.337,
            "mol_pyrimidine": -10.053,
            "mol_pyridine_F": -9.981,
            "mol_pyridine_OCH3": -9.098,
            "mol_thiazole": -9.888,
            "mol_pyridine": -10.053,
            "mol_phenyl_FF": -9.704,
            "mol_phenyl_FOCH3": -8.956,
            "mol_thiophene_F": -9.427,
            "mol_thiophene": -9.437,
            "mol_phenyl_F": -9.221,
            "mol_thiophene_OCH3": -8.146,
            "mol_phenyl_OCH3": -8.244,
            "mol_phenyl_OCH3OCH3": -7.866
        }
        lumo_dict = {
            "mol_phenyl": 1.901,
            "mol_pyrimidine": 1.194,
            "mol_pyridine_F": 1.303,
            "mol_pyridine_OCH3": 1.651,
            "mol_thiazole": 1.614,
            "mol_pyridine": 1.608,
            "mol_phenyl_FF": 1.72,
            "mol_phenyl_FOCH3": 2.04,
            "mol_thiophene_F": 1.784,
            "mol_thiophene": 2.011,
            "mol_phenyl_F": 1.636,
            "mol_thiophene_OCH3": 1.822,
            "mol_phenyl_OCH3": 1.881,
            "mol_phenyl_OCH3OCH3": 2.036
        }
        ring1 = self.rings[0]
        ring2 = self.rings[1]
        r1_homo = homo_dict[ring1]
        r1_lumo = lumo_dict[ring1]
        r2_homo = homo_dict[ring2]
        r2_lumo = lumo_dict[ring2]
        return min(r1_lumo-r2_homo, r2_lumo-r1_homo)

    @property
    def min_e(self):
        if self.norm_energy_dict:
            return min(self.norm_energy_dict.values())

    @property
    def band_gap(self):
        if self.homo_dict and self.lumo_dict:
            gaps = {key: self.lumo_dict[key] - self.homo_dict.get(key, None) for key in self.lumo_dict.keys()}
            return min(gaps.values())

    @property
    def global_barrier(self):
        if self.norm_energy_dict:
            return max(self.norm_energy_dict.values())

    @property
    def center_barrier(self):
        if self.norm_energy_dict:
            return self.norm_energy_dict.get(90.0)

    def get_norm_energy(self):
        # This dictionary is the only one with kcal/mol energies
        if len(self.energy_dict.keys()) > 17:
            _norm_edict = {float(deg): 23.06 * (float(eng) - min(self.energy_dict.values())) for deg, eng in self.energy_dict.items()}
            _sorted_norm_edict = dict(sorted(_norm_edict.items()))
            return _sorted_norm_edict

    @property
    def avg_backbone_length(self):
        if self.backbone_length:
            _dict = {k: v for k, v in self.backbone_length.items() if v > 0}
            if len(_dict) > 0:
                return sum(_dict.values()) / len(_dict)

    @staticmethod
    def boltzman_dist(x, A=1):
        return A * np.exp(-x / (298 * 8.314))

    @staticmethod
    def planarity_func(angle, prob):
        angle_rad = angle * np.pi / 180
        return prob * np.cos(angle_rad) ** 2

    def get_probability(self):
        if self.norm_energy_dict:
            probabilities = {d: self.boltzman_dist(e) for d, e in self.norm_energy_dict.items()}
            sum_prob = sum(probabilities.values())
            return {d: e/sum_prob for d, e in probabilities.items()}
        return {}

    @property
    def inflection_points(self):
        if self.norm_energy_dict:
            # compute second derivative
            smooth_d2 = np.gradient(np.gradient(np.array(list(self.norm_energy_dict.values()))))
            # find switching points
            infls = np.where(np.diff(np.sign(smooth_d2)))[0]
            return [list(self.norm_energy_dict.keys())[i] for i in infls]
        return []

    @property
    def center_barrier_width(self):
        if len(self.inflection_points) == 2:
            return abs(self.inflection_points[0]-self.inflection_points[1])
        elif len(self.inflection_points) > 2:
            array = np.asarray(self.inflection_points)
            closest_idxs = np.argpartition((np.abs(array - 90)), 2)[:2]
            return abs(self.inflection_points[closest_idxs[0]]-self.inflection_points[closest_idxs[1]])
        return None

    @property
    def planarity(self):
        if self.probability:
            indiv_planarity = [self.planarity_func(d, e) for d, e in self.probability.items()]
            return sum(indiv_planarity)

    @property
    def poly_ring_lengths(self):
        r1_base, r2_base = self.rings[0].split("_")[1], self.rings[1].split("_")[1]
        return [self.lm_ring_lengths[r1_base], self.lm_ring_lengths[r2_base]]

    @property
    def poly_bond_lengths(self):
        r1_base, r2_base = self.rings[0].split("_")[1], self.rings[1].split("_")[1]
        return [self.lm_bond_lengths["{}__{}".format(r2_base, r1_base)], self.min_eng_bl]

    @property
    def poly_deflection_angles(self):
        r1_base, r2_base = self.rings[0].split("_")[1], self.rings[1].split("_")[1]
        return [self.lm_deflection_angles[r2_base], self.lm_deflection_angles[r2_base], self.lm_deflection_angles[r1_base], self.lm_deflection_angles[r1_base]]

    @property
    def poly_dihed_energies(self):
        back_diheds = {
            "pyrimidine": "phenyl",
            "phenyl": "phenyl",
            "thiophene": "thiophene",
            "pyridine": "phenyl",
            "thiazole": "thiophene"
        }
        r1_base, r2_base = self.rings[0].split("_")[1], self.rings[1].split("_")[1]
        e_diheds = sorted([back_diheds[r2_base], back_diheds[r1_base]])
        norm_energy_list = [(k, round(v, 4)) for k, v in self.norm_energy_dict.items()]
        return [self.lm_dihed_energies["{}__{}".format(e_diheds[0], e_diheds[1])], norm_energy_list]

    def draw_structure(self, out_dir=None):
        molecule = Chem.MolFromSmiles(self.smiles)
        AllChem.Compute2DCoords(molecule)
        if out_dir is not None:
            Draw.MolToFile(molecule, out_dir + 'img_{}.png'.format(self.name))
        return Draw.MolToImage(molecule)

    def plot_torsion_energy(self, out_dir=None):
        """
        Plots the energy vs the torsion angle for a molecule given a torsion energy
        file. Saves the plot to a png file.
        """
        fig, ax = plt.subplots()
        phi, energy = self.norm_energy_dict.keys(), self.norm_energy_dict.values()
        plt.scatter(phi, energy)

        # ax.set_xlim(0, 3)
        # ax.set_ylim(0, 3)

        plt.xlim(min(phi) - 3, max(phi) + 3)
        # plt.xticks(np.linspace(start=0, stop=180, num=7))
        plt.ylim(top=max(energy) + 5, bottom=min(energy) - 5)
        # plt.yticks(np.linspace(start=-10, stop=14, num=5))
        plt.xlabel("dihedral angle (degrees)")
        plt.ylabel("energy (kcal/mol)")
        plt.title("Energy for " + self.name)
        fig.set_facecolor('w')
        if out_dir is not None:
            plt.savefig(out_dir + 'torsionE_plt_{}.png'.format(self.name), dpi=300, bbox_inches='tight')
            plt.close('all')

    def plot_homo_lumo(self, out_dir=None):
        """
        Plots the homo and lumo vs the torsion angle for a molecule given a torsion energy
        file. Saves the plot to a png file.
        """
        fig, ax = plt.subplots()
        phi_h, homo = self.homo_dict.keys(), self.homo_dict.values()
        phi_l, lumo = self.lumo_dict.keys(), list(self.lumo_dict.values())

        # eng_max, eng_min = max(max([homo, lumo])), min(min([homo, lumo]))

        ax.scatter(phi_h, homo, label='HOMO')
        ax.plot(phi_h, homo)
        ax.scatter(phi_l, lumo, label='LUMO')
        ax.plot(phi_l, lumo)

        ax.set_xlim(-3, 183)
        ax.set_xticks(np.linspace(start=0, stop=180, num=7))
        ax.set_ylim(top=5, bottom=-20)
        ax.set_yticks(np.linspace(start=-20, stop=5, num=6))
        ax.set_xlabel("dihedral angle (degrees)")
        ax.set_ylabel("energy (kcal/mol)")
        ax.set_title("HOMO/LUMO for " + self.name)
        fig.set_facecolor('w')
        fig.legend(loc='lower left', bbox_to_anchor=(0.9, 0.2))
        if out_dir is not None:
            plt.savefig(out_dir + 'tortion_homo_lumo_{}.png'.format(self.name), dpi=300, bbox_inches='tight')
            plt.close('all')

    def get_angle_from_energy(self, energy):
        for angle, eg in self.norm_energy_dict.items():
            if energy == eg:
                return angle
        return "Dihedral angle doesn't exist for {}.".format(energy)

    @staticmethod
    def write_json(data, filename):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def write_nnff_json(self, json_outdir):
        for angle in self.norm_energy_dict.keys():
            json_data = {
                "molecule_name": self.name,
                "degree": angle,
                "smile": self.smiles,
                "energy": self.norm_energy_dict[angle]
            }
            self.write_json(json_data, "{}/{}_{}deg.json".format(json_outdir, self.name, int(angle)))
