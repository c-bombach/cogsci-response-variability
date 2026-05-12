import os
import pandas as pd
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "distances_test_retest.csv")
distances_df = pd.read_csv(DATA_PATH)
distances_df["Same Participant"] = distances_df["same_id"].replace({0: "No", 1: "Yes"})


cdfs = sns.displot(data = distances_df, x = "distance", hue = "Same Participant", kind = "ecdf")
cdfs.set_axis_labels("t","P(Distance < t)")
cdfs.savefig("same_id_distribution.png")