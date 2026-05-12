
import os
import pandas as pd
import syllog_encoding as se 

PRIOR_KNOWLEDGE_PARTICIPANTS = ["dames-2022-time__20",
                                "dames-2022-time__34",
                                "dames-2022-time__77",
                                "dames-2022-time__83",
                                "dames-2022-time__85"]

def test_retest_dist(df,re, p = 2):
    dist_dict = {}
    grouped = df.groupby("id")
    for pid, reasoner in grouped:
        reas_dict_0 = se.compute_reasoner_dict(reasoner.loc[reasoner["retest"] == 0])
        reas_dict_1 = se.compute_reasoner_dict(reasoner.loc[reasoner["retest"] == 1])
        dist = se.lp_distance(reas_dict_0,reas_dict_1,re)
        dist_dict.update({pid: dist})
    return dist_dict

def distances_test_retest(df,re, p =2):
    distances = []
    grouped = df.groupby("id")
    for pid1,reasoner1 in grouped:
        for pid2, reasoner2 in grouped:
            reas_dict_1_test = se.compute_reasoner_dict(reasoner1.loc[reasoner1["retest"] == 0])
            reas_dict_2_retest = se.compute_reasoner_dict(reasoner2.loc[reasoner2["retest"] == 1])
            dist = se.lp_distance(reas_dict_1_test,reas_dict_2_retest,re)
            distances.append({"id_test": pid1, "id_retest": pid2, "distance": dist, "same_id": int(pid1 == pid2)})
    return pd.DataFrame(distances)

def distances_within_session(df,re, p = 2, retest = 0):
    distances = []
    grouped = df.groupby("id")
    for pid1,reasoner1 in grouped:
        for pid2, reasoner2 in grouped:
            reas_dict_1 = se.compute_reasoner_dict(reasoner1.loc[reasoner1["retest"] == retest])
            reas_dict_2 = se.compute_reasoner_dict(reasoner2.loc[reasoner2["retest"] == retest])
            dist = se.lp_distance(reas_dict_1,reas_dict_2, re)
            distances.append({"id1": pid1, "id2": pid2, "distance": dist})
    return pd.DataFrame(distances)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "dames_with_energy.csv")


df = pd.read_csv(DATA_PATH)
re = se.StandardResponseEncoder()

#
distances_df = distances_test_retest(df,re)
dist_test = distances_within_session(df,re)
dist_retest = distances_within_session(df,re, retest = 1)

dist_test.to_csv("data/distances_test.csv",index = False)
dist_retest.to_csv("data/distances_retest.csv", index = False)
distances_df.to_csv("data/distances_test_retest.csv", index = False)


