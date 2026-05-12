import os
import numpy as np
import pandas as pd
import syllog_encoding as se

def calculate_energy(df,task_encoder,resp_encoder,p = 2,var_name = "energy_2"):
    """Calculates the local energy for each reasoner and each syllogism.
    The local energy is given by E_local(f, s) = avg_{t in N(s)} d_R(f(s), f(t))^p where
    f is the syllogism -> response map given by the reasoner
    s is the syllogism"""
    print(f"Calculating local energy for p = {p}")
    grouped = df.groupby(["id", "retest"])
    for (pid, retest_val), reasoner in grouped:
        print(f"processing {pid}, {retest_val}")
        reas_dict = se.compute_reasoner_dict(reasoner)
        for syllog in reas_dict.keys():
            syllog_energy = se.local_energy(reas_dict,syllog,task_encoder,resp_encoder,p)
            mask = (df["id"] == pid) & (df["retest"] == retest_val) & (df["enc_task"] == syllog)
            df.loc[mask, var_name] = syllog_energy
    return df

def pivot_table(df):
    df_piv = df.pivot(index=["enc_task", "id"], 
                      columns="retest", 
                      values=["correctness", "energy_sqrt", "energy_sqrt_pc", "task_success_zs", "validity", "E_One_sqrt_pc"])

    df_piv.columns = [f'{metric}_{retest}' for metric, retest in df_piv.columns]

    df_piv = df_piv.reset_index()
    return df_piv

def difficulty(df):
    grouped = df.groupby(["enc_task", "retest"])
    for (task, retest_val), task_df in grouped:
        task_success = task_df["correctness"].sum()/len(task_df) - 0.5
        mask = (df["enc_task"] == task) & (df["retest"] == retest_val)
        df.loc[mask,"task_success"] = task_success
    return df


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "dames.csv")

df = pd.read_csv(DATA_PATH)
# df = df[df["id"] == "dames-2022-time__21"]


te = se.StandardTaskEncoder()
re = se.StandardResponseEncoder()
te_oh = se.OneHotTaskEncoder(se.ALL_SYLLOGISMS)
re_oh = se.OneHotResponseEncoder(se.ALL_RESPONSES)

df = calculate_energy(df,te,re)
df = calculate_energy(df,te_oh,re_oh, var_name = "E_One")

df["energy_sqrt"] = np.sqrt(df["energy_2"])
person_means = df.groupby(["id","retest"])["energy_sqrt"].transform("mean")
df["energy_sqrt_pc"] = df["energy_sqrt"] - person_means
df["energy_sqrt_pc"] = df.groupby(["id","retest"])["energy_sqrt_pc"].transform(lambda x: x / x.std() if x.std() > 0 else 0)

df["E_One_sqrt"] = np.sqrt(df["E_One"])
person_means = df.groupby(["id","retest"])["E_One_sqrt"].transform("mean")
df["E_One_sqrt_pc"] = df["E_One_sqrt"] - person_means
df["E_One_sqrt_pc"] = df.groupby(["id","retest"])["E_One_sqrt_pc"].transform(lambda x: x / x.std() if x.std() > 0 else 0)


df = difficulty(df)
df["task_success_zs"] = (df["task_success"] - df["task_success"].mean() )/ df["task_success"].std()


df_piv = pivot_table(df)


DF_PATH = os.path.join(SCRIPT_DIR,"data","dames_with_energy.csv")
PIV_PATH = os.path.join(SCRIPT_DIR,"data","pivot_table.csv")
df.to_csv(DF_PATH, index = False)
df_piv.to_csv(PIV_PATH, index = False)

    




