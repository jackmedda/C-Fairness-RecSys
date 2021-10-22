import pickle
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

data_ml = pickle.load(open("data/ml-1m_five_blocks_topk_gender.pkl", "rb"))
n1, n2 = data_ml["blocks"][0]["X_train"].shape
females = (
    data_ml["data_users"]
    .userid[data_ml["data_users"].gender == "F"]
    .to_numpy()
)
males = (
    data_ml["data_users"]
    .userid[data_ml["data_users"].gender == "M"]
    .to_numpy()
)
groups = (females, males)
dir_cov = "results/"
files_dir_cov = sorted(glob.glob(dir_cov + "*.pkl"))

if files_dir_cov == []:
    assert False, "Results of experiments not found in folder" + dir_cov
    exit()

###########################################################################
########################## COMPUTE Chi2stat  ##############################
###########################################################################
res_cov = {}
for f in sorted(files_dir_cov):
    data = pickle.load(open(f, "rb"))
    data = [data] if not isinstance(data, list) else data
    for d in data:
        model = d["model"]
        nq = model["tau_1"].shape[1]
        chi2 = np.stack([model["tau_1"][gr].sum(0) for gr in groups])

        chi2_stat = (
            (
                chi2
                - (
                    chi2.sum(0).reshape(1, nq)
                    * chi2.sum(1).reshape(len(groups), 1)
                )
                / chi2.sum()
            )
            ** 2
            / (
                (
                    chi2.sum(0).reshape(1, nq)
                    * chi2.sum(1).reshape(len(groups), 1)
                )
                / chi2.sum()
            )
        ).sum()
        d["chi2_stat"] = chi2_stat
    data = sorted(
        data, key=lambda x: x["nll"]
    )  # Sort best init according to loglike
    res_cov[data[0]["block"]] = data[0]

mean_chi2_cov = np.mean([d["chi2_stat"] for d in res_cov.values()])
print(f"CHi2stat LBM with gender sensitive attribute : {mean_chi2_cov:.2f}")

###########################################################################
########################## COMPUTE NDCG####  ##############################
###########################################################################
ndcg_10_cov = []
ndcg_10_cov_group = []
for k, md in res_cov.items():
    model = md["model"]
    X_test = np.array(data_ml["blocks"][k]["X_test"].todense())
    n1 = X_test.shape[0]
    true_test = list(  # np.array(
        [X_test[i, X_test[i,].nonzero()].flatten().tolist() for i in range(n1)]
    )
    pred_cov = (
        model["tau_1"] @ model["pi"] @ model["tau_2"].T
        + model["eta_row"]
        + model["eta_col"]
    ) + 1
    pred_test_cov = list(  # np.array(
        [
            pred_cov[i, X_test[i,].nonzero()].flatten().tolist()
            for i in range(n1)
        ]
    )
    ndcg_10_cov.append(np.mean([ndcg_score([x], [y], k=10) for x, y in zip(true_test, pred_test_cov)]))
    ndcg_10_cov_females = np.mean(
        [ndcg_score([x], [y], k=10) for x, y in
         zip([true_test[idx] for idx in groups[0]], [pred_test_cov[idx] for idx in groups[0]])]
    )
    ndcg_10_cov_males = np.mean(
        [ndcg_score([x], [y], k=10) for x, y in
         zip([true_test[idx] for idx in groups[1]], [pred_test_cov[idx] for idx in groups[1]])]
    )
    ndcg_10_cov_group.append([ndcg_10_cov_females, ndcg_10_cov_males])

ndcg_10_cov_group = np.array(ndcg_10_cov_group)
print(
    f"NDCG@10 LBM with gender sensitive attribute: {np.mean(ndcg_10_cov):.3f}"
)
for idgr in range(len(groups)):
    print(
        f"NDCG@10 group {idgr} LBM with gender sensitive attribute: {np.mean(ndcg_10_cov_group[:,idgr]):.3f}"
    )
