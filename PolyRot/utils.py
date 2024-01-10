import torch
from pymatgen.io.ase import *
import matplotlib.pyplot as plt


def draw_chain(pts, dim3=False, figsize=(50, 50)):
    """
    Plot polymer chain
        :param pts: list, list of [x,y,z] points for the chain
        :param dim3: bool, plot in 3 dimensions if True
        :param figsize: list,

    """
    x, y, z = list(zip(*pts))
    if dim3:
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_ylim(-15, 15)
        ax.set_zlim(-15, 15)
        ax.plot(z, y, x)
        ax.scatter(z, y, x)
        ax.view_init(elev=20., azim=-75)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_ylim(round(min(x)) - 2, round(max(x)) + 2)
        ax.plot(z, x)
        ax.scatter(z, x)
        ax.set_aspect('equal', adjustable='box')
    return ax


def get_dist_btw_points(pts):
    return [round(np.linalg.norm(np.subtract(pts[i + 1], pts[i])), 3) for i, _ in enumerate(pts) if i < len(pts) - 1]


def get_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_dihedral(a, b, c, d):
    b0 = -1.0 * (np.array(b) - np.array(a))
    b1 = np.array(c) - np.array(b)
    b2 = np.array(d) - np.array(c)
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


# TORCHANI FUNCTIONS

def aseatoms2ani(device, molecule):
    """
    Converts ASE atoms to ANI format tensors.

    :param device: str, device used for torch (e.g., "cuda" or "cpu").
    :param molecule: ASE atoms object.
    :return: Tuple of species and coordinates tensors.
    """
    coordinates = torch.tensor(np.array([molecule.arrays["positions"]]), requires_grad=True, device=device).float()
    species = torch.tensor(np.array([molecule.arrays["numbers"]]), dtype=torch.long, device=device)
    return species, coordinates


def ani2x_energy(ase_atoms, ani_model, device):
    """
    Computes ANI2x energy from input.

    :param ani_model: PyTorch model for ANI prediction.
    :param ase_atoms: ASE atoms object.
    :param device: str, device used for torch (e.g., "cuda" or "cpu").
    :return: ANI2x energy value.
    """
    return ani_model(aseatoms2ani(device, ase_atoms)).energies.item()


def ase_molecule(xyz_coord: str):
    """
    Creates ASE atoms from XYZ coordinates string.

    :param xyz_coord: str, XYZ coordinates.
    :return: ASE atoms object.
    """
    molecule = Molecule.from_str(xyz_coord, fmt="xyz")
    return AseAtomsAdaptor().get_atoms(molecule)
