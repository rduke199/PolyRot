import math
import random
import warnings
import itertools
import pandas as pd
from tqdm import tqdm
from numba import jit
from PolyRot.utils import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from timeit import default_timer as timer
from scipy.spatial.transform import Rotation


class PolymerRotate:
    def __init__(self,
                 bond_lengths: list,
                 ring_lengths: list,
                 deflection_angles: list,
                 dihed_energies: list,
                 theta_degrees=True
                 ):
        """
        :param bond_lengths: list, list of length of the inter-moiety bond
        :param ring_lengths: list, list of length of the ring tangent
        :deflection_angles: list, list of deflection angle between ring tangent and inter-moiety bond in radians
        :dihed_energies: set or list, list of tuples. Each tuple should contain (angle, energy).
        :param theta_degrees: bool, reads theta as degrees if True, radians if False
        """

        if not len(bond_lengths) == len(ring_lengths) == len(dihed_energies) == (len(deflection_angles) / 2):
            raise ValueError(
                "Error. bond_lengths and ring_lengths should be the same length, and deflection_angles should be twice as long. len(bond_lengths)={}, len(ring_lengths={}, dihed_energies={}, and len(deflection_angles)={}".format(
                    len(bond_lengths), len(ring_lengths), len(dihed_energies), len(deflection_angles)))

        self.bond_lengths = bond_lengths
        self.ring_lengths = ring_lengths
        self.deflection_angles = np.radians(deflection_angles) if theta_degrees else deflection_angles
        self.dihed_energies = dihed_energies
        self.theta_degrees = theta_degrees

    def std_chain(self, n):
        """
        Build a standard polymer chain from sequence of tangent vectors
        :param n: int, number of rings

        :return  list of [x,y,z] points for the chain
        """
        length_list = list(itertools.chain(*list(zip(self.bond_lengths, self.ring_lengths))))
        xyz_points = [[0, 0, 0]]
        for i in range(0, n + 1):
            l_idx = i % len(self.deflection_angles)
            length, deflection_angle = length_list[l_idx], self.deflection_angles[l_idx]

            if i == 0:
                xyz_points.append(
                    tuple([length * x for x in [math.sin(deflection_angle), 0, math.cos(deflection_angle)]]))
                continue
            previous_v = np.subtract(xyz_points[i], xyz_points[i - 1])
            previous_unit_v = previous_v / np.linalg.norm(previous_v)
            bond_pt = self.rotate((xyz_points[i] + length * previous_unit_v), xyz_points[i], deflection_angle,
                                  np.array([0, 1, 0]))

            xyz_points.append(tuple(np.round_(bond_pt, decimals=3)))

        return xyz_points[:-1]

    def rotated_chain(self, n, return_angles=False):
        """
        Build a randomly rotated polymer chain from sequence of tangent vector
        :param n: int, number of repeat units
        :param return_angles: bool, return tuple with list of points then the random angle if True

        :return  list of [x,y,z] points for the chain
        """
        cum_angle_funcs = []
        for dihed in self.dihed_energies:
            df_dihed = self.create_dihedral_df(dihed)
            cum_angle_func = interp1d(df_dihed.cum_probability.to_list() + [0], df_dihed.angle.to_list() + [0],
                                      kind='cubic')
            cum_angle_funcs.append(cum_angle_func)

        chain = self.std_chain(n)
        if return_angles:
            new_chain, dihed_angles = self.random_rotate(chain, cum_angle_funcs, return_angles=return_angles)
            return new_chain, dihed_angles
        new_chain = self.random_rotate(chain, cum_angle_funcs, return_angles=return_angles)
        return new_chain

    def create_dihedral_df(self, dihed_rot):
        """
        Create pandas DataFrame for a dihedral rotation. Columns include:
          - angle (angle of rotation)
          - energy (system energy in arbitrary unit)
          - Bolztman (Bolztman distribution of energy)
          - probability (Bolztman probability)
          - cum_probability (cumulative probability)

        :param dihed_rot: set or list, list of tuples. Each tuples should contain (angle, energy).
        """
        dihed_df = pd.DataFrame(dihed_rot, columns=["angle", "energy"])
        for i_, row in dihed_df.iterrows():
            new_row = row.to_dict()
            if new_row['angle'] != 180:
                new_row['angle'] = 360 - new_row['angle']
                dihed_df = dihed_df.append(new_row, ignore_index=True)
        dihed_df.sort_values(by="angle", inplace=True)
        dihed_df['boltzman'] = dihed_df.apply(lambda x: self.boltzman_dist(x.energy, T=700), axis=1)
        dihed_df['probability'] = dihed_df.apply(lambda x: x.boltzman / dihed_df.boltzman.sum(), axis=1)
        dihed_df['cum_probability'] = dihed_df.probability.cumsum()
        return dihed_df

    def dihedral_rotate(self, pts: list, idx: int, theta: float, theta_degrees=True):
        """
        Rotate a tangent vector corresponding to an inter-moiety bonds (first point
        should be an ODD index since python numbering starts at 0) and updates all subsequent
        points in the pts list.

        :param pts: list, list of [x,y,z] points for the chain
        :param idx: int, index for pts list for first point in notable vector
        :param theta: float, angle to rotate
        :param theta_degrees: bool, reads theta as degrees if True, radians if False

        :return:

        """
        # Return original points and raise warning if index is an even number
        if not idx % 2:
            warnings.warn("Error. The index for the function dihedral_rotate should be and odd number")
            return pts

        # Get variables for rotation matrix
        origin = pts[idx]
        rot_radians = np.radians(theta) if theta_degrees else theta
        rot_axis = np.subtract(pts[idx], pts[idx - 1])

        new_points = pts[:idx + 1] + [self.rotate(i, origin, rot_radians, rot_axis) for i in
                                      pts[-(len(pts) - (idx + 1)):]]

        return new_points

    def random_rotate(self, pts, prob_func, return_angles=False):
        """
        Rotate the dihedral angles according to the cumulative probability distribution
        :param pts: list, list of [x,y,z] points for the chain
        :param prob_func: func or list, probability function for dihedral angle or list of probability functions
        :param return_angles: bool, return tuple with list of points then the random angle if True

        :return: list of [x,y,z] points for the updated chain

        """
        # prep for random rotations
        prob_funcs = prob_func if isinstance(prob_func, list) else [prob_func]
        dihed_angles = []

        # Iterate through points by index and rotate every other line
        prob_f_idx = 0
        for i, _ in enumerate(pts):
            if i % 2:  # Only perform operation for odd indexes
                # Perform dihedral rotation on the given index and update new points
                dihed = prob_funcs[prob_f_idx % len(prob_funcs)](random.uniform(0, 1))
                dihed_angles.append(float(dihed))
                pts = self.dihedral_rotate(pts, i, dihed, theta_degrees=self.theta_degrees)
                prob_f_idx += 1
        if return_angles:
            return pts, dihed_angles
        return pts

    @staticmethod
    def boltzman_dist(x, A=1, T=298):
        return A * np.exp(-x / (T * 8.314 / 1000))  # kJ/mol

    @staticmethod
    def rotate(pt, origin, rotation_radians, rotation_axis):
        """
        Get new bond vector
        :param pt: list, point to be rotated, [x, y, z]
        :param origin: list, point to be rotated around, [x, y, z]
        :param rotation_radians: float, radians to rotate
        :param rotation_axis: list, vector axis upon which to rotate, [x, y, z]

        :return: rotated point as list, [x, y, z]
        """
        rot_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation = Rotation.from_rotvec(rotation_radians * rot_axis)
        vec = np.subtract(pt, origin)
        rotated_vec = rotation.apply(vec)
        return list(np.add(origin, rotated_vec))


# ---------------------- Analysis ----------------------

@jit(target_backend='cuda')
def multi_polymer(poly_obj, n, num_poly):
    """
    Generate list of rotated polymers
    :param poly_obj: obj, RotatePolymer object for rotation
    :param n: int, number of polymer units
    :param num_poly: int, number of polymers to create

    :return: list of rotated polymers
    """
    all_poly = list(np.zeros(num_poly))
    for i in tqdm(range(num_poly)):
        all_poly[i] = poly_obj.rotated_chain(n)
    return all_poly


def cos_vals(pts):
    """
    Compute the tangent correlation between ith inter-moiety vector and the 1st inter-moiety vector

    :param pts: list, list of [x,y,z] points for the chain

    :return: list of [x,y,z] points for the updated chain

    """
    final_corr = []
    second_third = np.subtract(pts[2], pts[1])
    for i in range(2, len(pts), 2):
        i_vector = np.subtract(pts[i], pts[i - 1])
        new_pt = np.dot(i_vector, second_third) / (np.linalg.norm(i_vector) * np.linalg.norm(second_third))
        final_corr.append(new_pt)
    return final_corr


def multi_corr_data(polymer_list, plot=True):
    """
    Run tangent-tangent correlation analysis on polymer list.
    :param polymer_list: list, list of polymer chains
    :param plot: bool, plot log of tangent-tangent correlation functions if true

    :return: multi corr data as dictionary, {"x": [...], "y": [...]}
    """
    cos_list = np.array([cos_vals(p) for p in tqdm(polymer_list)])
    corr = np.mean(cos_list, axis=0)
    corr_data = {
        "x": list(range(0, len(corr))),
        "y": [np.log(c) for _, c in enumerate(corr)]
    }
    if plot:
        plt.scatter(corr_data["x"], corr_data["y"])
        plt.xlabel("Number of Units")
        plt.ylabel("log(correlation function)")
    return corr_data


def n_p(polymer_list, plot=True):
    """
    Run tangent-tangent correlation analysis on polymer list to get Np.
    :param polymer_list: list, list of polymer chains
    :param plot: bool, plot log of tangent-tangent correlation functions if true

    :return: Np, the persistence length measured in repeat units
    """
    data = multi_corr_data(polymer_list, plot=plot)
    m, b = np.polyfit(x=data["x"], y=data["y"], deg=1)
    return -1 / m


def avg_r_2(polymer_list, in_nm_2=False):
    """
    Compute average R^2 for a list of molecules
    :param polymer_list: list, list of polymer chains
    :param in_nm_2: bool, return mean square end-to-end distance in nm^2 if True

    :return: mean square end-to-end distance
    """
    all_r_2 = [(np.linalg.norm(np.array(polymer_list[i][-1]) - np.array(polymer_list[i][0]))) ** 2 for i, _ in
               enumerate(polymer_list)]
    return np.mean(all_r_2) * 0.01 if in_nm_2 else np.mean(all_r_2) * 0.01


if __name__ == "__main__":
    # Molecule-Specific Default Variables
    DIHED_ROT = {(0, 0), (10, 0.293835), (20, 0.967939), (30, 1.86645),
                 (40, 3.177), (50, 4.91823), (60, 6.91593), (70, 8.91844), (80, 10.5465),
                 (90, 11.4081), (100, 11.2003), (110, 10.0682), (120, 8.36782), (130, 6.51926),
                 (140, 5.079767), (150, 4.36963), (160, 4.04486), (170, 3.80474), (180, 3.82803)}

    LCSC = 2.548  # length of the ring tangent
    LCC = 1.480  # length of the inter-moiety bond
    DEFLECTION = 15  # degrees

    polymer = PolymerRotate(bond_lengths=[LCSC, LCSC], ring_lengths=[LCC, LCC],
                            deflection_angles=[DEFLECTION, -DEFLECTION, -DEFLECTION, DEFLECTION],
                            dihed_energies=[DIHED_ROT, DIHED_ROT])
    ch = polymer.std_chain(50)
    new_ch = polymer.rotated_chain(50)

    draw_chain(ch, dim3=False)
    draw_chain(new_ch, dim3=True)

    # Generate 10,000 polymers
    start = timer()
    polymers = multi_polymer(polymer, n=50, num_poly=100)
    print("Generating ", len(polymers), "polymers with GPU:", timer() - start)

    # View results of tangent-tangent correlation function for polymers
    Np = n_p(polymers, plot=False)
    print("Np: ", Np)

    # Get mean square end-to-end distance for polymer with 25 units
    r_2 = avg_r_2(polymers, in_nm_2=True)
    print("R^2: ", r_2)
