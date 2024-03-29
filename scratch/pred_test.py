from PolyRot.central_dihedral import *

import datetime
import torchani
import torch
import time

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = torchani.models.ANI2x(periodic_table_index=True).to(my_device)

JSON_FN = "~/PolyRot/scratch/pes_master.json"


def pes_predict_CentDihed(smiles, energy_df, ani_model=None, device=None, use_dft_structures=False, **kwargs):
    energy_df.index = energy_df.index.astype(float)
    energy_df["energy_kcal"] = energy_df['energy'] * 23
    energy_df["energy_zero"] = energy_df["energy_kcal"] - energy_df["energy_kcal"].min()

    dihed_pred = CentDihed(smiles=smiles, **kwargs)
    if use_dft_structures:
        structures = list(zip(energy_df.index, energy_df.structure))
        pred_energy_df = dihed_pred.pred_pes_energies(structures=structures, return_column="", ani_model=ani_model,
                                                      device=device)
    else:
        pred_energy_df = dihed_pred.pred_pes_energies(return_column="", ani_model=ani_model, device=device)

    combined_df = pd.concat([pred_energy_df, energy_df], axis=1)
    combined_df["error_kcal"] = combined_df["energy_zero"] - combined_df["pred_ani2x_energy_kcal_zero"]

    return combined_df


def pes_prediction(identifier, plot=False, json_fn=JSON_FN, return_error=False, verbose=1, **kwargs):
    df2 = pd.read_json(json_fn).transpose()

    smiles = df2.loc[identifier]["smiles"]
    energy_df = pd.DataFrame(df2.loc[identifier]["pes"]).transpose()
    try:
        pred = pes_predict_CentDihed(smiles, energy_df, verbose=verbose, **kwargs)
        if plot:
            pred.sort_index(inplace=True)
            pred.energy_zero.plot()
            pred.pred_ani2x_energy_kcal_zero.plot()
        return pred.error_kcal.mean() if return_error else pred
    except Exception as e:
        print("ERROR: ", e) if verbose > 0 else None


for num_confs in [1, 2, 5, 10, 20, 50]:
    print(f"Starting predictions with {num_confs} confs...")
    pred_df = pd.read_json(JSON_FN).transpose()  # .head(1)

    start_time = time.time()
    pred_df["pred_results"] = pred_df.apply(lambda x: pes_prediction(x.name, device=my_device, ani_model=my_model, num_confs=num_confs, verbose=0), axis=1)
    end_time = time.time()
    total_time = end_time - start_time

    pred_df.to_pickle(f"data/ani_pred_smiles_{num_confs:02}.pkl")

    error_df = pred_df.apply(lambda x: x.pred_results.error_kcal if x.pred_results is not None else None, axis=1)
    error_mean = error_df.stack().mean()
    error_std = error_df.stack().std()
    print(f"Error for {num_confs} confs: {error_mean:0.2f} +/- {error_std:0.2f}")
    print(f"Time for {num_confs} confs: {datetime.timedelta(seconds=round(total_time))}")
