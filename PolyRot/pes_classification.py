import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model


def classify_pes(energy_dict, subcategories=False):
    """
    Classifies a PES based on the energy values.
    :param energy_dict: dict, dictionary containing pes information, {<dihedral rotation angle (degrees)>: <energy (kcal/mol)>}
        The energy dictionary should contain at least 10 energy points (though it is recommended that it contain at
        least 18). The dihedral rotation angles (dict keys) should include 0, 40, 90, 140, and 180.
    :param subcategories: bool, return subcategory classification if True, class classification if False

    :return: str, curve classification
    """
    if len(energy_dict.keys() or []) < 10:
        raise ValueError(
            "Error. All dihedral PES should have at least 10 energy points. It is recommended that each have at least "
            "18 energy points. Inputted dihedral PES has only {} energy points".format(len(energy_dict.keys() or [])))

    e0, e90, e180, e40, e140 = energy_dict.get(0), energy_dict.get(90), energy_dict.get(180), energy_dict.get(
        40), energy_dict.get(140)
    if None in [e0, e90, e180, e40, e140]:
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


def predict_class(smiles):
    """
    Use classification models to predict PES class from monomer SMILES string.
    :param smiles: str, monomer SMILES string

    :return: str, curve class
    """
    # Set up model and label encoder
    poly_rot = os.path.dirname(os.path.realpath(__file__))
    imported_model = load_model(os.path.join(poly_rot, 'classification_models', 'curve_class_model-0.3-30'))
    encoder = LabelEncoder()
    encoder.classes_ = np.load(os.path.join(poly_rot, 'classification_models', 'curve_class_transform.npy'), allow_pickle=True)

    # Perform prediction
    fingerprint = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2))
    predictions = imported_model.predict(np.array([fingerprint], ))
    results = encoder.inverse_transform([np.argmax(predictions)])

    return results[0]


def predict_subclass(smiles):
    """
    Use classification models to predict PES subclass from monomer SMILES string.
    :param smiles: str, monomer SMILES string

    :return: str, curve subclass
    """
    # Set up model and label encoder
    poly_rot = os.path.dirname(os.path.realpath(__file__))
    imported_model = load_model(os.path.join(poly_rot, 'classification_models', 'curve_subclass_model-0.3-30'))
    encoder = LabelEncoder()
    encoder.classes_ = np.load(os.path.join(poly_rot, 'classification_models', 'curve_subclass_transform.npy'), allow_pickle=True)

    # Perform prediction
    fingerprint = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2))
    predictions = imported_model.predict(np.array([fingerprint], ))
    results = encoder.inverse_transform([np.argmax(predictions)])

    return results[0]


if __name__ == "__main__":
    print(predict_class("C1=CC=C(S1)C2=CC=C(S2)"))
    print(predict_subclass("C1=CC=C(S1)C2=CC=C(S2)"))
