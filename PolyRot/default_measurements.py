"""
Default values for common ring lengths, deflection angles, bond lengths, etc. All lengths
are given in Angstroms and all angles are given in degrees. 

It is recommended that users find exact values via calculations or literature review.
"""


def bond_length(atom1="C", atom2="C", bond_order=1):
    """
    Get default value for bond length
    :param atom1: str, element abbreviation 1 ('aro' for aromatic)
    :param atom2: str, element abbreviation 2 ('aro' for aromatic)
    :param bond_order: float, bond order (1=single, 1.5=aromatic, 2=double, 3=triple)

    :return: float, bond length in Angstroms
    """
    bond_orders = {1: "_", 1.5: "--", 2: "=", 3: "-="}

    bond_id1 = "{}{}{}".format(atom1.capitalize(), bond_orders.get(bond_order), atom2.capitalize())
    bond_id2 = "{}{}{}".format(atom2.capitalize(), bond_orders.get(bond_order), atom1.capitalize())

    bond_lengths = {
        "C_C": 1.54,
        "C_N": 1.47,
        "C_O": 1.43,
        "aro_C":  1.52,
        "aro_N": 1.42,
        "C_H": 1.09,
        "N_H": 0.99,
        "C_F": 1.37,
        "C_Cl": 1.76,
        "C_Br": 1.94,
        "C_I": 2.14,
        "C_S": 1.82,
        "C_P": 1.84,
        "O_H": 0.98,
        "S_S": 2.07,

        "C--C": 1.40,
        "C--N": 1.34,
        "N--N": 1.35,

        "C=C": 1.34,
        "C=O": 1.21,
        "C=N": 1.25,
        "N=N": 1.25,

        "C-=C": 1.20,
        "C-=N": 1.16,
    }
    return bond_lengths.get(bond_id1) or bond_lengths.get(bond_id2)


def ring_length(ring_size=None, ring_name=None):
    """
    Get default value for ring length. If given ring_name not in available rings, this will
    default to ring size method.

    :param ring_size: int, number of atoms in ring (produces a very rough estimate)
    :param ring_name: str, ring name (limited to available rings)

    :return: float, ring length in Angstroms
    """
    ring_names = {
        "thiophene": 2.52,
        "thiazole": 2.36,
        "pyrimidine": 2.71,
        "phenyl": 2.85,
        "pyridine": 2.78,
    }
    ring_sizes = {
        5: 2.4,
        6: 2.8,
    }
    return ring_names.get(ring_name) or ring_sizes.get(ring_size)


def deflection_angle(ring_size=None, ring_name=None):
    """
    Get default value for ring length. If given ring_name not in available rings, this will
    default to ring size method.

    :param ring_size: int, number of atoms in ring
    :param ring_name: str, ring name (limited to available rings)

    :return: float, deflection angle in degrees
    """
    ring_names = {
      "thiazole": 13,
      "thiophene": 16,
      "phenyl": 0,
      "pyrimidine": 1,
      "pyridine": 3,
    }
    ring_sizes = {
        5: 18,
        6: 1,
    }
    return ring_names.get(ring_name) or ring_sizes.get(ring_size)


if __name__=="__main__":
    print(ring_length(ring_size=5))
    print(bond_length(atom1="C", atom2="C", bond_order=1.5))
    print(deflection_angle(ring_size=5))
