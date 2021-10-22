import os
import argparse
import pickle
import itertools
import re
from collections import defaultdict
from typing import Sequence, NamedTuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mat_ticker
import pandas as pd
import numpy as np
import scipy.stats

import helpers.constants as constants
from helpers.logger import RcLogger

import data.utils as data_utils
from models.utils import RelevanceMatrix
from metrics import Metrics

import data.datasets.lastfm

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

arg_parser = argparse.ArgumentParser(description="Argument parser to compute metrics of reproduced works")

arg_parser.add_argument('-dataset',
                        default="movielens_1m",
                        choices=["movielens_1m", "filtered(20)_lastfm_1K"],
                        help="Dataset to use",
                        type=str)
arg_parser.add_argument('-sensitive_attribute',
                        default="Gender",
                        help="The attribute to be considered to compute the metrics",
                        type=str)
arg_parser.add_argument('--only_plot',
                        help="does not compute the metrics. It loads the specific results pickle file "
                             "and create plots and tables",
                        action="store_true")

args = arg_parser.parse_args()

# Attributes of each value of the following arrays in order:
# - target: `Ranking` or `Rating`, it is the prediction target,
# - paper: name or nickname that identify the related paper,
# - model name: the model to which the fairness approach is applied or
#               the new name of a fairness-aware baseline, e.g. ParityLBM,
# - run_id or path: the path, list of paths, callable that returns paths as a list (even with one only path),
#                   or `run_id`, which is the run id of a relevance matrix inside `data\relevance_matrices` generated
#                   with `recommender_codebase` saving function to save relevance matrices,
# - baseline: 1) same as `run_id or path` but for the baseline,
#             2) a tuple where the first value is the same for `run_id or path` and the second is the specific name of
#             the baseline, otherwise `baseline` string will be added at the end of `model name`,
# - function to retrieve data (OPTIONAL): function to read the file with predictions (use one of the functions of the
#                                         class RelevanceMatrix that start with `from_...` inside `models\utils.py`,
# - function to retrieve baseline data (OPTIONAL): the same for `function to retrieve data (OPTIONAL)`
#                                                  but for the baseline

experiments_models_gender_ml1m = [
    ('Rating', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\gender_balanced") if "UU-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "UU-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\gender_balanced") if "II-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "II-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\gender_balanced") if "MF-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "MF-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'TopPopular',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\gender_balanced") if "Pop-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "Pop-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'Mean',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\gender_balanced") if "Mean-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "Mean-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\gender_balanced") if "UU-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "UU-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\gender_balanced") if "II-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "II-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\gender_balanced") if "MF-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "MF-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Kamishima', ['PMF Mean Matching', 'PMF BDist Matching', 'PMF Mi Normal'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\movielens_1m_out_mean_matching_gender.json",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\movielens_1m_out_bdist_matching_gender.json",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\movielens_1m_out_mi_normal_gender.json"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\movielens_1m_out_pmf_baseline.json", 'PMF'),
     RelevanceMatrix.from_rec_indep_json,
     RelevanceMatrix.from_rec_indep_json),
    ('Ranking', 'User-oriented fairness', 'PMF',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\PMF_112997d298_Gender___per_user_timestamp_split[80%, 20%]_out.csv",
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\PMF\11_PMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'NeuMF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\NeuMF_112997d298_Gender___per_user_timestamp_split[80%, 20%]_out.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\NCF\11_NCF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'BiasedMF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\BiasedMF_112997d298_Gender___per_user_timestamp_split[80%, 20%]_out.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\BiasedMF\11_BiasedMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'STAMP',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\STAMP_112997d298_Gender___per_user_timestamp_split[80%, 20%]_out.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\STAMP\11_STAMP_movielens-1m_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'Co-clustering', 'Parity LBM',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Co-clustering for fair recommendation\results\movielens_1m\block_0_run_2_gender.pkl",
     (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Co-clustering for fair recommendation\results\movielens_1m\block_0_baseline.pkl", 'Standard LBM'),
     RelevanceMatrix.from_co_clustering_fair_pickle,
     RelevanceMatrix.from_co_clustering_fair_pickle),
    ('Ranking', 'Librec', 'BN-SLIM-U',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Librec-auto tutorial\movielens_1m_gender_experiment\exp00000\result\out-1.txt',
     (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Librec-auto tutorial\movielens_1m_SLIM_U\exp00000\result\out-1.txt", 'SLIM-U'),
     RelevanceMatrix.from_librec_result,
     RelevanceMatrix.from_librec_result),
    ('Rating', 'Antidote Data', 'ALS MF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\movielens_1m_antidote_group_out_als_MF_Gender.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\movielens_1m_antidote_group_out_als_MF_baseline.csv",
     RelevanceMatrix.from_antidote_data,
     RelevanceMatrix.from_antidote_data),
    ('Rating', 'Antidote Data', 'LMaFit MF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\movielens_1m_antidote_group_out_lmafit_MF_Gender.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\movielens_1m_antidote_group_out_lmafit_MF_baseline.csv",
     RelevanceMatrix.from_antidote_data,
     RelevanceMatrix.from_antidote_data),
    ('Rating', 'FairGo', 'FairGo GCN',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\movielens_1m_reproduce_data\predictions_gender.pkl',
     (r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\movielens_1m_reproduce_data\predictions_baseline.pkl', "GCN"),
     RelevanceMatrix.from_fair_go_predictions,
     RelevanceMatrix.from_fair_go_predictions),
    ('Rating', 'Haas', ['ALS BiasedMF Parity', 'ALS BiasedMF Val'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ALS_parity_adj_user_gender.csv",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ALS_val_adj_user_gender.csv"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ALS_orig_user_gender.csv", "ALS BiasedMF"),
     RelevanceMatrix.from_rating_prediction_fairness_result,
     RelevanceMatrix.from_rating_prediction_fairness_result),
    ('Rating', 'Haas', ['Item-Item Parity', 'Item-Item Val'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ItemItem_parity_adj_user_gender.csv",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ItemItem_val_adj_user_gender.csv"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ItemItem_orig_user_gender.csv", "Item-Item"),
     RelevanceMatrix.from_rating_prediction_fairness_result,
     RelevanceMatrix.from_rating_prediction_fairness_result)
]

experiments_models_age_ml1m = [
    ('Rating', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\age_balanced") if "UU-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "UU-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\age_balanced") if "II-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "II-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\age_balanced") if "MF-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "MF-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'TopPopular',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\age_balanced") if "Pop-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "Pop-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'Mean',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\age_balanced") if "Mean-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "Mean-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\age_balanced") if "UU-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "UU-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\age_balanced") if "II-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "II-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\age_balanced") if "MF-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\movielens_1m\baseline") if "MF-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Kamishima', ['PMF Mean Matching', 'PMF BDist Matching', 'PMF Mi Normal'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\movielens_1m_out_mean_matching_age.json",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\movielens_1m_out_bdist_matching_age.json",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\movielens_1m_out_mi_normal_age.json"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\movielens_1m_out_pmf_baseline.json", 'PMF'),
     RelevanceMatrix.from_rec_indep_json,
     RelevanceMatrix.from_rec_indep_json),
    ('Ranking', 'User-oriented fairness', 'PMF',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\PMF_de3706b370_Age___per_user_timestamp_split[80%, 20%]_out.csv",
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\PMF\11_PMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'NeuMF',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\NeuMF_de3706b370_Age___per_user_timestamp_split[80%, 20%]_out.csv",
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\NCF\11_NCF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'BiasedMF',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\BiasedMF_de3706b370_Age___per_user_timestamp_split[80%, 20%]_out.csv",
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\BiasedMF\11_BiasedMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'STAMP',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\STAMP_de3706b370_Age___per_user_timestamp_split[80%, 20%]_out.csv",
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\STAMP\11_STAMP_movielens-1m_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'Co-clustering', 'Parity LBM',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Co-clustering for fair recommendation\results\movielens_1m\block_0_run_15_age.pkl",
     (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Co-clustering for fair recommendation\results\movielens_1m\block_0_baseline.pkl", 'Standard LBM'),
     RelevanceMatrix.from_co_clustering_fair_pickle,
     RelevanceMatrix.from_co_clustering_fair_pickle),
    ('Ranking', 'Librec', 'BN-SLIM-U',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Librec-auto tutorial\movielens_1m_age_experiment\exp00000\result\out-1.txt",
    (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Librec-auto tutorial\movielens_1m_SLIM_U\exp00000\result\out-1.txt", 'SLIM-U'),
     RelevanceMatrix.from_librec_result,
     RelevanceMatrix.from_librec_result),
    ('Rating', 'Antidote Data', 'ALS MF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\movielens_1m_antidote_group_out_als_MF_Age.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\movielens_1m_antidote_group_out_als_MF_baseline.csv",
     RelevanceMatrix.from_antidote_data,
     RelevanceMatrix.from_antidote_data),
    ('Rating', 'Antidote Data', 'LMaFit MF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\movielens_1m_antidote_group_out_lmafit_MF_Age.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\movielens_1m_antidote_group_out_lmafit_MF_baseline.csv",
     RelevanceMatrix.from_antidote_data,
     RelevanceMatrix.from_antidote_data),
    ('Rating', 'FairGo', 'FairGo GCN',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\movielens_1m_reproduce_data\predictions_age.pkl',
     (r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\movielens_1m_reproduce_data\predictions_baseline.pkl', "GCN"),
     RelevanceMatrix.from_fair_go_predictions,
     RelevanceMatrix.from_fair_go_predictions),
    ('Rating', 'Haas', ['ALS BiasedMF Parity', 'ALS BiasedMF Val'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ALS_parity_adj_bucketized_user_age.csv",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ALS_val_adj_bucketized_user_age.csv"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ALS_orig_bucketized_user_age.csv", "ALS BiasedMF"),
     RelevanceMatrix.from_rating_prediction_fairness_result,
     RelevanceMatrix.from_rating_prediction_fairness_result),
    ('Rating', 'Haas', ['Item-Item Parity', 'Item-Item Val'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ItemItem_parity_adj_bucketized_user_age.csv",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ItemItem_val_adj_bucketized_user_age.csv"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\movielens_1m\ItemItem_orig_bucketized_user_age.csv", "Item-Item"),
     RelevanceMatrix.from_rating_prediction_fairness_result,
     RelevanceMatrix.from_rating_prediction_fairness_result)
]

experiments_models_gender_lfm1k = [
    ('Rating', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\gender_balanced") if "UU-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "UU-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\gender_balanced") if "II-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "II-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\gender_balanced") if "MF-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "MF-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'TopPopular',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\gender_balanced") if "Pop-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "Pop-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'Mean',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\gender_balanced") if "Mean-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "Mean-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\gender_balanced") if "UU-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "UU-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\gender_balanced") if "II-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "II-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\gender_balanced") if "MF-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "MF-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Kamishima', ['PMF Mean Matching', 'PMF BDist Matching', 'PMF Mi Normal'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\filtered(20)_lastfm_1K_out_mean_matching_gender.json",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\filtered(20)_lastfm_1K_out_bdist_matching_gender.json",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\filtered(20)_lastfm_1K_out_mi_normal_gender.json"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\filtered(20)_lastfm_1K_out_pmf_baseline.json", 'PMF'),
     RelevanceMatrix.from_rec_indep_json,
     RelevanceMatrix.from_rec_indep_json),
    ('Ranking', 'User-oriented fairness', 'PMF',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\filtered(20)_lastfm_1K_PMF_034b94c16d_lastfm___per_user_timestamp_split[80%, 20%]_out.csv",
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\PMF\11_PMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'NeuMF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\filtered(20)_lastfm_1K_NeuMF_034b94c16d_lastfm___per_user_timestamp_split[80%, 20%]_out.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\NCF\11_NCF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'BiasedMF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\filtered(20)_lastfm_1K_BiasedMF_034b94c16d_lastfm___per_user_timestamp_split[80%, 20%]_out.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\BiasedMF\11_BiasedMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'STAMP',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\filtered(20)_lastfm_1K_STAMP_034b94c16d_lastfm___per_user_timestamp_split[80%, 20%]_out.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\STAMP\11_STAMP_filtered(20)-lastfm-1K_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'Co-clustering', 'Parity LBM',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Co-clustering for fair recommendation\results\filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_run_14_user_gender.pkl",
     (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Co-clustering for fair recommendation\results\filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_baseline.pkl", 'Standard LBM'),
     RelevanceMatrix.from_co_clustering_fair_pickle,
     RelevanceMatrix.from_co_clustering_fair_pickle),
    ('Ranking', 'Librec', 'BN-SLIM-U',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Librec-auto tutorial\filtered(20)_lastfm_1K_gender_experiment\exp00000\result\out-1.txt',
     (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Librec-auto tutorial\filtered(20)_lastfm_1K_SLIM_U\exp00000\result\out-1.txt", 'SLIM-U'),
     RelevanceMatrix.from_librec_result,
     RelevanceMatrix.from_librec_result),
    # ('Rating', 'Antidote Data', 'ALS MF',
    #  r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\filtered(20)_lastfm_1K_antidote_group_out_als_MF_user_gender.csv',
    #  r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\filtered(20)_lastfm_1K_antidote_group_out_als_MF_baseline.csv",
    #  RelevanceMatrix.from_antidote_data,
    #  RelevanceMatrix.from_antidote_data),
    ('Rating', 'Antidote Data', 'LMaFit MF',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\filtered(20)_lastfm_1K_antidote_group_out_lmafit_MF_user_gender.csv",
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\filtered(20)_lastfm_1K_antidote_group_out_lmafit_MF_baseline.csv",
     RelevanceMatrix.from_antidote_data,
     RelevanceMatrix.from_antidote_data),
    ('Rating', 'FairGo', 'FairGo GCN',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\filtered(20)_lastfm_1K_reproduce_data\predictions_gender.pkl',
     (r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\filtered(20)_lastfm_1K_reproduce_data\predictions_baseline.pkl', "GCN"),
     RelevanceMatrix.from_fair_go_predictions,
     RelevanceMatrix.from_fair_go_predictions),
    ('Rating', 'Haas', ['ALS BiasedMF Parity', 'ALS BiasedMF Val'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ALS_parity_adj_user_gender.csv",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ALS_val_adj_user_gender.csv"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ALS_orig_user_gender.csv", "ALS BiasedMF"),
     RelevanceMatrix.from_rating_prediction_fairness_result,
     RelevanceMatrix.from_rating_prediction_fairness_result),
    ('Rating', 'Haas', ['Item-Item Parity', 'Item-Item Val'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ItemItem_parity_adj_user_gender.csv",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ItemItem_val_adj_user_gender.csv"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ItemItem_orig_user_gender.csv", "Item-Item"),
     RelevanceMatrix.from_rating_prediction_fairness_result,
     RelevanceMatrix.from_rating_prediction_fairness_result)
]

experiments_models_age_lfm1k = [
    ('Rating', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\age_balanced") if "UU-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "UU-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\age_balanced") if "II-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "II-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\age_balanced") if "MF-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "MF-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'TopPopular',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\age_balanced") if "Pop-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "Pop-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Ekstrand', 'Mean',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\age_balanced") if "Mean-E" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "Mean-E" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'User-User',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\age_balanced") if "UU-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "UU-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'Item-Item',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\age_balanced") if "II-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "II-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Ranking', 'Ekstrand', 'FunkSVD',
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\age_balanced") if "MF-B" in f.name],
     [f.path for f in os.scandir(r"D:\Reproducibility Study\All the cool kids\results\filtered(20)_lastfm_1K\baseline") if "MF-B" in f.name][0],
     RelevanceMatrix.from_cool_kids_result,
     RelevanceMatrix.from_cool_kids_result),
    ('Rating', 'Kamishima', ['PMF Mean Matching', 'PMF BDist Matching', 'PMF Mi Normal'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\filtered(20)_lastfm_1K_out_mean_matching_age.json",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\filtered(20)_lastfm_1K_out_bdist_matching_age.json",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\filtered(20)_lastfm_1K_out_mi_normal_age.json"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Recommendation Independence\filtered(20)_lastfm_1K_out_pmf_baseline.json", 'PMF'),
     RelevanceMatrix.from_rec_indep_json,
     RelevanceMatrix.from_rec_indep_json),
    ('Ranking', 'User-oriented fairness', 'PMF',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\filtered(20)_lastfm_1K_PMF_41d1bbe242_lastfm___per_user_timestamp_split[80%, 20%]_out.csv",
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\PMF\11_PMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'NeuMF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\filtered(20)_lastfm_1K_NeuMF_41d1bbe242_lastfm___per_user_timestamp_split[80%, 20%]_out.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\NCF\11_NCF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'BiasedMF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\filtered(20)_lastfm_1K_BiasedMF_41d1bbe242_lastfm___per_user_timestamp_split[80%, 20%]_out.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\BiasedMF\11_BiasedMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'User-oriented fairness', 'STAMP',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\recommender_codebase\data\user-oriented fairness files\filtered(20)_lastfm_1K_STAMP_41d1bbe242_lastfm___per_user_timestamp_split[80%, 20%]_out.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\result\STAMP\11_STAMP_filtered(20)-lastfm-1K_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy",
     RelevanceMatrix.from_user_oriented_fairness_files,
     RelevanceMatrix.from_nlr_models_result),
    ('Ranking', 'Co-clustering', 'Parity LBM',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Co-clustering for fair recommendation\results\filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_run_17_user_age.pkl",
     (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Co-clustering for fair recommendation\results\filtered(20)_lastfm_1K\filtered(20)_lastfm_1K_block_0_baseline.pkl", 'Standard LBM'),
     RelevanceMatrix.from_co_clustering_fair_pickle,
     RelevanceMatrix.from_co_clustering_fair_pickle),
    ('Ranking', 'Librec', 'BN-SLIM-U',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Librec-auto tutorial\filtered(20)_lastfm_1K_age_experiment\exp00000\result\out-1.txt",
     (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Librec-auto tutorial\filtered(20)_lastfm_1K_SLIM_U\exp00000\result\out-1.txt", 'SLIM-U'),
     RelevanceMatrix.from_librec_result,
     RelevanceMatrix.from_librec_result),
    # ('Rating', 'Antidote Data', 'ALS MF',
    #  r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\filtered(20)_lastfm_1K_antidote_group_out_als_MF_user_age.csv',
    #  r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\filtered(20)_lastfm_1K_antidote_group_out_als_MF_baseline.csv",
    #  RelevanceMatrix.from_antidote_data,
    #  RelevanceMatrix.from_antidote_data),
    ('Rating', 'Antidote Data', 'LMaFit MF',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\filtered(20)_lastfm_1K_antidote_group_out_lmafit_MF_user_age.csv',
     r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\antidote-data-framework\filtered(20)_lastfm_1K_antidote_group_out_lmafit_MF_baseline.csv",
     RelevanceMatrix.from_antidote_data,
     RelevanceMatrix.from_antidote_data),
    ('Rating', 'FairGo', 'FairGo GCN',
     r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\filtered(20)_lastfm_1K_reproduce_data\predictions_age.pkl',
     (r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\filtered(20)_lastfm_1K_reproduce_data\predictions_baseline.pkl', "GCN"),
     RelevanceMatrix.from_fair_go_predictions,
     RelevanceMatrix.from_fair_go_predictions),
    ('Rating', 'Haas', ['ALS BiasedMF Parity', 'ALS BiasedMF Val'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ALS_parity_adj_user_age.csv",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ALS_val_adj_user_age.csv"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ALS_orig_user_age.csv", "ALS BiasedMF"),
     RelevanceMatrix.from_rating_prediction_fairness_result,
     RelevanceMatrix.from_rating_prediction_fairness_result),
    ('Rating', 'Haas', ['Item-Item Parity', 'Item-Item Val'],
     [
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ItemItem_parity_adj_user_age.csv",
         r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ItemItem_val_adj_user_age.csv"
     ], (r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\RatingPredictionFairness\results\filtered(20)_lastfm_1K\ItemItem_orig_user_age.csv", "Item-Item"),
     RelevanceMatrix.from_rating_prediction_fairness_result,
     RelevanceMatrix.from_rating_prediction_fairness_result)
]

if args.sensitive_attribute == "Gender":
    sensitive_field = "user_gender"
    sensitive_values = ["Male", "Female"]

    if args.dataset == "movielens_1m":
        experiments_models_gender = experiments_models_gender_ml1m
    else:
        experiments_models_gender = experiments_models_gender_lfm1k
else:
    if args.dataset == "movielens_1m":
        sensitive_field = "bucketized_user_age"
        sensitive_values = ["1-34", "35-56+"]

        experiments_models_age = experiments_models_age_ml1m
    else:
        sensitive_field = "user_age"
        sensitive_values = ["1-24", "25+"]

        experiments_models_age = experiments_models_age_lfm1k

gender_short_values = {"Male": "M", "Female": "F"}
if args.dataset == "movielens_1m":
    age_short_values = {"1-34": "Y", "35-56+": "O"}
else:
    age_short_values = {"1-24": "Y", "25+": "O"}

# _compute_equity_score
metrics = {
    "Ranking": ["ndcg", "ks", "mannwhitneyu", "ndcg_user_oriented_fairness", "epsilon_fairness", "f1_score", "mrr"],
    "Rating": ["ks", "mannwhitneyu", "mae", "rmse"]#, "value_unfairness", "absolute_unfairness",
               #"overestimation_unfairness", "underestimation_unfairness", "rating_demographic_parity",
               #"rating_equal_opportunity", "gei", "theil"]
}
metrics_type = {
    "ndcg": "with_diff",
    "f1_score": "with_diff",
    "mrr": "no_diff",
    "ks": "no_diff",
    "mannwhitneyu": "no_diff",
    "rmse": "with_diff",
    "mae": "with_diff",
    "epsilon_fairness": "no_diff",
    "ndcg_user_oriented_fairness": "with_diff",
    "value_unfairness": "no_diff",
    "absolute_unfairness": "no_diff",
    "overestimation_unfairness": "no_diff",
    "underestimation_unfairness": "no_diff",
    "rating_demographic_parity": "no_diff",
    "rating_equal_opportunity": "no_diff"
}


def main():
    RcLogger.start_logger(level="INFO")

    metrics_inv_map = defaultdict(list)
    for target, metrs in metrics.items():
        for m in metrs:
            metrics_inv_map[m].append(target)

    fig_axs = dict.fromkeys(np.unique(np.concatenate(list(metrics.values()))))
    results = dict.fromkeys(np.unique(np.concatenate(list(metrics.values()))))
    stats = dict.fromkeys(np.unique(np.concatenate(list(metrics.values()))))
    for key in fig_axs:
        fig_axs[key] = plt.subplots(1, len([1 for ms in metrics.values() if key in ms]), figsize=(40, 15))
        results[key] = []
        stats[key] = []

    if os.path.exists(os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study", f"{args.dataset}_results_{args.sensitive_attribute}.pkl")):
        with open(os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study", f"{args.dataset}_results_{args.sensitive_attribute}.pkl"), 'rb') as pk:
            results = pickle.load(pk)

        with open(os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study", f"{args.dataset}_stats_{args.sensitive_attribute}.pkl"), 'rb') as pk:
            stats = pickle.load(pk)

    if not args.only_plot:
        model_data_type = "binary"
        if args.dataset == "movielens_1m":
            dataset_metadata = {
                'dataset': 'movielens',
                'dataset_size': '1m',
                'n_reps': 2,
                'train_val_test_split_type': 'per_user_timestamp',
                'train_val_test_split': ["70%", "10%", "20%"]
            }
            users_field = "user_id"
            items_field = "movie_id"
            rating_field = "user_rating"
        else:
            dataset_metadata = {
                'dataset': 'lastfm',
                'dataset_size': '1K',
                'n_reps': 2,
                'min_interactions': 20,
                'train_val_test_split_type': 'per_user_random',
                'train_val_test_split': ["70%", "10%", "20%"]
            }
            users_field = "user_id"
            items_field = "artist_id"
            rating_field = "user_rating"

        # It is necessary to load "orig_train", so `n_reps` and `model_data_type` are irrelevant
        orig_train, val, test = data_utils.load_train_val_test(dataset_metadata, model_data_type)
        if val is not None:
            orig_train = orig_train.concatenate(val)

        observed_items, unobserved_items, _, other_returns = data_utils.get_train_test_features(
            users_field,
            items_field,
            train_data=orig_train,
            test_or_val_data=test,
            item_popularity=False,
            sensitive_field=sensitive_field,
            rating_field=rating_field,
            other_returns=["sensitive", "test_rating_dataframe"]
        )

        sensitive_group = other_returns['sensitive']
        test_rating_dataframe = other_returns["test_rating_dataframe"]

        if args.sensitive_attribute == "Gender":
            exps = pd.DataFrame(experiments_models_gender, columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])
        else:
            exps = pd.DataFrame(experiments_models_age, columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])

        group1 = [gr for gr in sensitive_group if sensitive_group[gr]]
        group2 = [gr for gr in sensitive_group if not sensitive_group[gr]]

        relevant_matrices_files = list(os.scandir(constants.SAVE_RELEVANCE_MATRIX_PATH))

        for _, exp in exps.iterrows():
            if not isinstance(exp['id_file'], list):
                if pd.isnull(exp['id_file']):
                    continue

                if os.path.isfile(exp['id_file']):
                    filepath = [exp['id_file']]
                elif callable(exp['id_file']):
                    filepath = exp['id_file']()
                else:
                    filepath = [f.path for f in relevant_matrices_files if exp['id_file'] in f.name]
            else:
                filepath = exp['id_file']

            model_name = exp['model']
            for i, _file in enumerate(filepath):
                if _file is None:
                    continue

                if isinstance(exp['model'], list):
                    model_name = exp['model'][i]

                multiple_files = False
                if not isinstance(exp['model'], list) and isinstance(exp['id_file'], list):
                    model_name = f"{exp['model']} {i}"
                    multiple_files = True

                metrics_to_compute = check_computed_metrics(
                    metrics,
                    results,
                    exp['target'],
                    f"{exp['paper']} \n {model_name}",
                    multiple_files=multiple_files
                )

                if metrics_to_compute:
                    rel_matrix = load_specific_rel_matrix(exp['read_f'], _file, args.sensitive_attribute)

                    compute_metrics_update_results(
                        rel_matrix,
                        metrics_to_compute,
                        metrics_type,
                        exp,
                        results,
                        model_name,
                        stats,
                        observed_items=observed_items,
                        unobserved_items=unobserved_items,
                        test_rating_dataframe=test_rating_dataframe,
                        sensitive_group=sensitive_group,
                        group1=group1,
                        group2=group2
                    )

            if not isinstance(exp['model'], list) and isinstance(exp['id_file'], list):
                for m in metrics[exp['target']]:
                    res = results[m]
                    st = stats[m]

                    if not res:
                        continue

                    if metrics_type[m] == "with_diff":
                        columns = ["target", "paper/model", "value", "type"]
                    elif metrics_type[m] == "no_diff":
                        columns = ["target", "paper/model", "value"]

                    res = pd.DataFrame(res, columns=columns)
                    if metrics_type[m] == "with_diff":
                        type_gr = res.groupby("type")
                        for r_type, r_df in [(t_gr, type_gr.get_group(t_gr)) for t_gr in ['Total'] + sensitive_values + ['Diff']]:
                            multiple_rows = []
                            for _, r in r_df.iterrows():
                                if r["target"] == exp['target'] and \
                                        re.match(
                                            f"{exp['paper']} \n {exp['model']}" + r' \d+',
                                            r["paper/model"]
                                        ) is not None:
                                    multiple_rows.append(r.tolist())

                            for row in multiple_rows:
                                results[m].remove(row)
                            if multiple_rows:
                                if r_type != 'Diff':
                                    results[m].append([
                                        exp["target"],
                                        f"{exp['paper']} \n {exp['model']}",
                                        np.mean([x[2] for x in multiple_rows]),
                                        r_type
                                    ])
                                else:
                                    gr1_val = None
                                    gr2_val = None
                                    for res_r in results[m]:
                                        if res_r[0] == exp["target"] and res_r[1] == f"{exp['paper']} \n {exp['model']}":
                                            if res_r[3] == sensitive_values[0]:
                                                gr1_val = res_r[2]
                                            elif res_r[3] == sensitive_values[1]:
                                                gr2_val = res_r[2]

                                    if gr1_val is None or gr2_val is None:
                                        raise ValueError(f"One of the two sensitive values in {sensitive_values} "
                                                         f"has not been computed")

                                    results[m].append([
                                        exp["target"],
                                        f"{exp['paper']} \n {exp['model']}",
                                        gr1_val - gr2_val,
                                        r_type
                                    ])

                    elif metrics_type[m] == "no_diff":
                        multiple_rows = []
                        for _, r in res.iterrows():
                            if r["target"] == exp['target'] and \
                                    re.match(
                                        f"{exp['paper']} \n {exp['model']}" + r' \d+',
                                        r["paper/model"]
                                    ) is not None:
                                multiple_rows.append(r.tolist())

                        for row in multiple_rows:
                            results[m].remove(row)
                        if multiple_rows:
                            if m in ["ks", "mannwhitneyu"]:
                                results[m].append([
                                    exp["target"],
                                    f"{exp['paper']} \n {exp['model']}",
                                    {
                                        "statistic": np.mean([x[2]['statistic'] for x in multiple_rows]),
                                        "pvalue": np.mean([x[2]['pvalue'] for x in multiple_rows])
                                    }
                                ])
                            else:
                                results[m].append([
                                    exp["target"],
                                    f"{exp['paper']} \n {exp['model']}",
                                    np.mean([x[2] for x in multiple_rows])
                                ])

                    if st is not None:
                        st = pd.DataFrame(st, columns=["target", "paper/model", "value"])
                        multiple_rows = []
                        for _, s in st.iterrows():
                            if s["target"] == exp['target'] and \
                                    re.match(
                                        f"{exp['paper']} \n {exp['model']}" + r' \d+',
                                        s["paper/model"]
                                    ) is not None:
                                multiple_rows.append(s.tolist())

                        for row in multiple_rows:
                            stats[m].remove(row)

                        stats[m].append([
                            exp["target"],
                            f"{exp['paper']} \n {exp['model']}",
                            {
                                "statistic": np.mean([x[2]['statistic'] for x in multiple_rows]),
                                "pvalue": np.mean([x[2]['pvalue'] for x in multiple_rows])
                            }
                        ])

                if isinstance(exp['model'], list):
                    print(
                        f"{exp['target']}",
                        f"{exp['paper']}",
                        f"{model_name}" if not (not isinstance(exp['model'], list) and isinstance(exp['id_file'], list))
                        else f"{exp['model']}"
                    )

            if not pd.isnull(exp['baseline']):
                path_or_id, baseline_name = parse_baseline(exp)

                if not pd.isnull(path_or_id):
                    if callable(path_or_id):
                        path_or_id = path_or_id()
                    elif not os.path.isfile(path_or_id):
                        path_or_id = [f.path for f in relevant_matrices_files if path_or_id in f.name][0]

                    baseline_metrics = check_computed_metrics(
                        metrics,
                        results,
                        exp['target'],
                        f"{exp['paper']} \n {baseline_name}",
                        multiple_files=False
                    )

                    if baseline_metrics:
                        rel_matrix = load_specific_rel_matrix(exp['read_f_base'], path_or_id, args.sensitive_attribute)

                        compute_metrics_update_results(
                            rel_matrix,
                            baseline_metrics,
                            metrics_type,
                            exp,
                            results,
                            baseline_name,
                            stats,
                            observed_items=observed_items,
                            unobserved_items=unobserved_items,
                            test_rating_dataframe=test_rating_dataframe,
                            sensitive_group=sensitive_group,
                            group1=group1,
                            group2=group2
                        )

            if not isinstance(exp['model'], list):
                print(f"Completed: {exp['target']}", exp['paper'], model_name)
            else:
                for _mod_name in exp['model']:
                    print(f"Completed: {exp['target']}", exp['paper'], _mod_name)

            with open(os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study", f"{args.dataset}_results_{args.sensitive_attribute}.pkl"), 'wb') as pk:
                pickle.dump(results, pk, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study", f"{args.dataset}_stats_{args.sensitive_attribute}.pkl"), 'wb') as pk:
                pickle.dump(stats, pk, protocol=pickle.HIGHEST_PROTOCOL)

    # Retrieve names of all the baselines in order to use the patch in the plots
    paper_baseline_names = []
    if args.sensitive_attribute == "Gender":
        exps_df = pd.DataFrame(experiments_models_gender, columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])
    else:
        exps_df = pd.DataFrame(experiments_models_age, columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])

    for _, exp in exps_df.iterrows():
        _, baseline_name = parse_baseline(exp)
        paper_baseline_names.append(f"{exp['paper']} \n {baseline_name}")

    for key in results:
        if metrics_type[key] == "with_diff":
            results[key] = pd.DataFrame(results[key], columns=["target", "paper_model", key, args.sensitive_attribute])
        elif metrics_type[key] == "no_diff":
            results[key] = pd.DataFrame(results[key], columns=["target", "paper_model", key])

    for m in metrics_inv_map:
        for i, target in enumerate(metrics_inv_map[m]):
            if metrics_type[m] == "with_diff":
                hue = args.sensitive_attribute
                palette = dict(zip(sensitive_values + ['Diff'], ['#F5793A', '#A95AA1', '#85C0F9']))
                color = None
            elif metrics_type[m] == "no_diff":
                hue = None
                palette = None
                #color = '#601A4A'
                color = '#3A8391'
            else:
                raise ValueError(f"Metrics type `{metrics_type[m]} not supported`")

            m_df = results[m][results[m]["target"] == target]
            if hue is not None:
                m_df = m_df[m_df[hue] != "Total"]

            if m in ["ks", "mannwhitneyu"]:
                m_df[[m, "pvalue"]] = m_df[m].apply(lambda x: (x['statistic'], x['pvalue'])).to_list()

            m_max = m_df[m].max()
            ax = fig_axs[m][1][i] if isinstance(fig_axs[m][1], np.ndarray) else fig_axs[m][1]

            barplot = sns.barplot(x="paper_model", y=m, hue=hue, data=m_df, ax=ax, color=color, palette=palette)
            ax.set_title(target)
            ax.set_ylim(top=max(m_max * 1.05, ax.get_ylim()[1]) if i != 0 else m_max * 1.05)  # 5% more

            ax.minorticks_on()
            ax.yaxis.set_minor_locator(mat_ticker.AutoMinorLocator(10))
            ax.grid(axis='y', which='both', ls=':')
            ax.tick_params(axis='both', which='minor', length=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='center')
            ax.set_xlabel("")

            if hue is None:
                ax.annotate("", (-0.06, 1), (-0.06, 0),
                            xycoords='axes fraction',
                            arrowprops={'arrowstyle': '<-'})

            xticks = ax.get_xticklabels()
            patches = sorted(barplot.patches, key=lambda x: x.xy[0])
            for tick in xticks:
                if tick.get_text() in paper_baseline_names:
                    if metrics_type[m] == "no_diff":
                        patches[xticks.index(tick)].set_hatch('.')
                    else:
                        idx_baseline = xticks.index(tick) * 3
                        for idx in range(idx_baseline, idx_baseline + 3):
                            patches[idx].set_hatch('/')

    plots_path = os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study", "plots", args.dataset)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    for m, (fig, axs) in fig_axs.items():
        fig.savefig(os.path.join(plots_path, f"plot_{args.sensitive_attribute}_{m}.png"))

        plt.close(fig)

    results_to_latex_table()
    # results_to_latex_full_table()
    results_to_paper_table()
    results_to_paper_table_fused_datasets()


def compute_metrics_update_results(rel_matrix, metrics, metrics_type, exp, results, model_name, stats, **kwargs):
    predictions = rel_matrix.as_dataframe()

    observed_items = kwargs.pop("observed_items")
    unobserved_items = kwargs.pop("unobserved_items")

    test_rating_dataframe = kwargs.pop("test_rating_dataframe")

    sensitive_group = kwargs.pop("sensitive_group")
    group1 = kwargs.pop("group1")
    group2 = kwargs.pop("group2")

    metrics_handler = Metrics()
    metrics_handler.compute_metrics(
        metrics=metrics,
        cutoffs=[10],
        only=["custom"],
        **{
            "observed_items": observed_items,
            "unobserved_items": unobserved_items,
            "predictions": predictions,
            "sensitive": sensitive_group,
            "test_rating_dataframe": test_rating_dataframe.loc[
                predictions.index, predictions.columns
            ][~predictions.isna()] if exp["target"] == "Rating" else None
        }
    )

    for m in metrics:
        if metrics_type[m] == "with_diff":
            m_total = metrics_handler.get(m, k=10)
            m_gr1 = metrics_handler.get(m, k=10, user_id=group1)
            m_gr2 = metrics_handler.get(m, k=10, user_id=group2)
            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_total, "Total"])
            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_gr1, sensitive_values[0]])
            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_gr2, sensitive_values[1]])
            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_gr1 - m_gr2, "Diff"])

            gr1_vals = metrics_handler.get(m, k=10, user_id=group1, raw=True)
            gr2_vals = metrics_handler.get(m, k=10, user_id=group2, raw=True)
            stats[m].append([
                exp['target'],
                f"{exp['paper']} \n {model_name}",
                scipy.stats.mannwhitneyu(gr1_vals, gr2_vals)._asdict()
            ])
        elif metrics_type[m] == "no_diff":
            m_value: NamedTuple = metrics_handler.get(m, k=10)
            if m in ["ks", "mannwhitneyu"]:
                m_value = m_value._asdict()  # it avoids problems with pickle

            results[m].append([exp['target'], f"{exp['paper']} \n {model_name}", m_value])


def load_specific_rel_matrix(function, _file, sens_attr):
    if pd.isnull(function):
        rel_matrix = RelevanceMatrix.load(_file)
    else:
        if function == RelevanceMatrix.from_co_clustering_fair_pickle:
            co_clust_sens_attr = "gender" if sens_attr == "Gender" else "age"
            rel_matrix_args = [
                os.path.join(
                    r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Co-clustering for fair recommendation\data",
                    f"{args.dataset}_extra_data_{co_clust_sens_attr}.pkl"
                )
            ]
        elif function == RelevanceMatrix.from_nlr_models_result:
            rel_matrix_args = [
                os.path.join(
                    r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\NLR\dataset",
                    f"{args.dataset}",
                    f"{args.dataset}.test.csv"
                )
            ]
        elif function == RelevanceMatrix.from_fair_go_predictions:
            rel_matrix_args = [
                os.path.join(
                    r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO",
                    f"{args.dataset}_reproduce_data", "testing_ratings_dict.npy"
                ),
                os.path.join(
                    r"C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO",
                    f"{args.dataset}_reproduce_data", "mapping_user_item.npy"
                )
            ]
        else:
            rel_matrix_args = []

        rel_matrix = function(_file, *rel_matrix_args)

    return rel_matrix


def check_computed_metrics(metrics, results, target, paper_model, multiple_files=False):
    metrs = metrics[target].copy()

    for m in metrics[target]:
        res = results[m]

        for r in res:
            if r[0] == target and r[1] == paper_model:
                metrs.remove(m)
                break
            elif multiple_files:
                if r[0] == target and r[1] == re.sub(r'\s\d+', '', paper_model):
                    metrs.remove(m)
                    break

    return metrs


def parse_baseline(exp):
    if isinstance(exp['baseline'], Sequence) and not isinstance(exp['baseline'], str):
        path_or_id, baseline_name = exp['baseline']
    else:
        path_or_id = exp['baseline']
        if not isinstance(exp['model'], str):
            raise ValueError("Specify baseline name if model name is a sequence")
        baseline_name = exp['model'] + " Baseline"

    return path_or_id, baseline_name


def results_to_latex_table():
    repr_path = os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study")

    with open(os.path.join(repr_path, f"{args.dataset}_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results: dict = pickle.load(pk)

    tables_path = os.path.join(repr_path, "tables", args.dataset)
    if not os.path.exists(tables_path):
        os.makedirs(tables_path)

    for m, dict_df in results.items():
        formatted_m = m.replace('_', ' ').title()

        if metrics_type[m] == "with_diff":
            df = pd.DataFrame(dict_df, columns=["Target", "Paper/Model", formatted_m, args.sensitive_attribute.title()])
        elif metrics_type[m] == "no_diff":
            df = pd.DataFrame(dict_df, columns=["Target", "Paper/Model", formatted_m])

            if m in ["ks", "mannwhitneyu"]:
                df[[formatted_m, "pvalue"]] = df[formatted_m].apply(lambda x: (x['statistic'], x['pvalue'])).to_list()
        else:
            raise ValueError(f"metric `{m}` not supported")

        df["Paper/Model"] = df["Paper/Model"].str.replace('\n', '')
        print(df.to_string())
        if metrics_type[m] == "with_diff":
            df = df.pivot(index=["Target", "Paper/Model"], columns=args.sensitive_attribute.title(), values=formatted_m)

            cols = df.columns.to_list()
            diff_idx = cols.index("Diff")
            cols.pop(diff_idx)

            df = df[cols + ["Diff"]]

        with open(os.path.join(tables_path, f"table_{args.sensitive_attribute}_{m}.txt"), "w") as f:
            f.write(df.round(3).to_latex(multirow=True, caption=formatted_m))


def results_to_latex_full_table():
    repr_path = os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study")

    with open(os.path.join(repr_path, f"{args.dataset}_results_Gender.pkl"), "rb") as pk:
        results_gender: dict = pickle.load(pk)

    with open(os.path.join(repr_path, f"{args.dataset}_results_Age.pkl"), "rb") as pk:
        results_age: dict = pickle.load(pk)

    if results_age and results_gender:
        tables_path = os.path.join(repr_path, "full_tables", args.dataset)
        if not os.path.exists(tables_path):
            os.makedirs(tables_path)

        for m, gender_df in results_gender.items():
            formatted_m = m.replace('_', ' ').title()

            age_df = results_age[m]

            if metrics_type[m] == "with_diff":
                g_df = pd.DataFrame(gender_df, columns=["Target", "Paper/Model", formatted_m, "Gender"])
                a_df = pd.DataFrame(age_df, columns=["Target", "Paper/Model", formatted_m, "Age"])
            elif metrics_type[m] == "no_diff":
                g_df = pd.DataFrame(gender_df, columns=["Target", "Paper/Model", formatted_m])
                a_df = pd.DataFrame(age_df, columns=["Target", "Paper/Model", formatted_m])

                if m in ["ks", "mannwhitneyu"]:
                    g_df[[formatted_m, "pvalue"]] = g_df[formatted_m].apply(lambda x: (x['statistic'], x['pvalue'])).to_list()
                    a_df[[formatted_m, "pvalue"]] = a_df[formatted_m].apply(lambda x: (x['statistic'], x['pvalue'])).to_list()
            else:
                raise ValueError(f"metric `{m}` not supported")

            g_df["Paper/Model"] = g_df["Paper/Model"].str.replace('\n', '')
            a_df["Paper/Model"] = a_df["Paper/Model"].str.replace('\n', '')
            if metrics_type[m] == "with_diff":
                g_df = g_df.pivot(index=["Target", "Paper/Model"], columns="Gender", values=formatted_m)

                cols = g_df.columns.to_list()
                diff_idx = cols.index("Diff")
                cols.pop(diff_idx)

                g_df = g_df[cols + ["Diff"]]

                a_df = a_df.pivot(index=["Target", "Paper/Model"], columns="Age", values=formatted_m)

                cols = a_df.columns.to_list()
                diff_idx = cols.index("Diff")
                cols.pop(diff_idx)

                a_df = a_df[cols + ["Diff"]]

                g_df = g_df.rename(columns={'Diff': '$\Delta$G'})
                a_df = a_df.rename(columns={'Diff': '$\Delta$A'})

                df = g_df
                for col in a_df.columns:
                    if col not in df.columns:
                        df[col] = a_df[col]
            else:
                df = g_df
                if m in ["ks", "mannwhitneyu"]:
                    g_cols_rename = {formatted_m: "Gender Value", "pvalue": "Gender pvalue"}
                    a_cols_rename = {formatted_m: "Age Value", "pvalue": "Age pvalue"}
                else:
                    g_cols_rename = {formatted_m: "Gender Value"}
                    a_cols_rename = {formatted_m: "Age Value"}

                df = df.rename(columns=g_cols_rename)
                a_df = a_df.rename(columns=a_cols_rename)

                df["Age Value"] = a_df["Age Value"]
                if m in ["ks", "mannwhitneyu"]:
                    df["Gender pvalue"] = df["Gender pvalue"].map(str)
                    df["Age pvalue"] = a_df["Age pvalue"].map(str)

                df = df.sort_values("Target").set_index(["Target", "Paper/Model"])

            df = df.rename(columns={**gender_short_values, **age_short_values})

            with open(os.path.join(tables_path, f"full_table_{m}.txt"), "w") as f:
                f.write(df.round(3).to_latex(multirow=True, caption=f"{formatted_m} {args.dataset}"))


def results_to_paper_table():
    repr_path = os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study")

    with open(os.path.join(repr_path, f"{args.dataset}_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results: dict = pickle.load(pk)

    with open(os.path.join(repr_path, f"{args.dataset}_stats_{args.sensitive_attribute}.pkl"), "rb") as pk:
        stats: dict = pickle.load(pk)

    tables_path = os.path.join(repr_path, "paper_tables", args.dataset)
    if not os.path.exists(tables_path):
        os.makedirs(tables_path)

    paper_map = {
        'Ekstrand': 'Ekstrand et al.',
        'Librec': 'Burke et al.',
        'Antidote Data': 'Rastegarpanah et al.',
        'Kamishima': 'Kamishima et al.',
        'FairGo': 'Wu et al.',
        'Co-clustering': 'Frisch et al.',
        'User-oriented fairness': 'Li et al.',
        'Haas': 'Ashokan et al.'
    }

    model_map = {
        'BN-SLIM-U': 'SLIM-U',
        'Parity LBM': 'LBM',
        'STAMP': 'STAMP',
        'BiasedMF': 'BiasedMF',
        'PMF': 'PMF',
        'NeuMF': 'NCF',
        'FunkSVD': 'FunkSVD',
        'TopPopular': 'TopPopular',
        'User-User': 'UserKNN',
        'Item-Item': 'ItemKNN',
        'Mean': 'AvgRating',
        'PMF Mean Matching': 'PMF Mean',
        'PMF BDist Matching': 'PMF BDist',
        'PMF Mi Normal': 'PMF Mi',
        'ALS MF': 'ALS',
        'LMaFit MF': 'LMaFit',
        'FairGo GCN': 'FairGo GCN',
        'ALS BiasedMF': 'ALS BiasedMF',
        'ALS BiasedMF Parity': 'ALS BiasedMF Par',
        'ALS BiasedMF Val': 'ALS BiasedMF Val',
        'Item-Item Parity': 'ItemKNN Par',
        'Item-Item Val': 'ItemKNN Val'
    }

    if args.sensitive_attribute == "Gender":
        exps_df = pd.DataFrame(experiments_models_gender,
                               columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])
    else:
        exps_df = pd.DataFrame(experiments_models_age,
                               columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])

    paper_baseline_names_model = {}
    stat_model_baseline = {}
    for _, exp in exps_df.iterrows():
        _, baseline_name = parse_baseline(exp)
        paper_baseline_names_model[f"{exp['paper']} \n {baseline_name}"] = exp['model']
        if not isinstance(exp['model'], str):
            for m_name in exp['model']:
                stat_model_baseline[f"{exp['paper']} \n {m_name}"] = baseline_name
        else:
            stat_model_baseline[f"{exp['paper']} \n {exp['model']}"] = baseline_name

    paper_metrics = {metr: res for metr, res in results.items() if metr in ['ndcg', 'rmse', 'ks']}
    paper_stats = {metr: st for metr, st in stats.items() if metr in ['ndcg', 'rmse']}

    paper_dfs = {
        metr: pd.DataFrame(
            dict_df,
            columns=["Target", "Paper/Model", metr.replace('_', ' ').title(), args.sensitive_attribute.title()]
        ) if metrics_type[metr] == "with_diff" else pd.DataFrame(
                dict_df,
                columns=["Target", "Paper/Model", metr.replace('_', ' ').upper()]
            )
        for metr, dict_df in paper_metrics.items()
    }

    stats_dfs = {
        st: pd.DataFrame(stat_dict, columns=["Target", "Paper/Model", "Stat"]) for st, stat_dict in paper_stats.items()
    }

    for metr in paper_dfs:
        paper_dfs[metr][["Paper", "Model"]] = paper_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
        del paper_dfs[metr]["Paper/Model"]

        paper_dfs[metr]["Paper"] = paper_dfs[metr]["Paper"].str.strip()
        paper_dfs[metr]["Model"] = paper_dfs[metr]["Model"].str.strip()

        if metr != "ks":
            stats_dfs[metr][["Paper", "Model"]] = stats_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
            del stats_dfs[metr]["Paper/Model"]

            stats_dfs[metr]["Paper"] = stats_dfs[metr]["Paper"].str.strip()
            stats_dfs[metr]["Model"] = stats_dfs[metr]["Model"].str.strip()

            stats_dfs[metr][["Stat", "Pvalue"]] = [(x["statistic"], x["pvalue"]) for x in stats_dfs[metr]["Stat"]]

    for ut, targ in [["ndcg", "Ranking"], ["rmse", "Rating"]]:
        ut_df = paper_dfs[ut]
        ks_df = paper_dfs["ks"][paper_dfs["ks"]["Target"] == targ]
        st_df = stats_dfs[ut][stats_dfs[ut]["Target"] == targ]

        diffs = ut_df[ut_df[args.sensitive_attribute.title()] == "Diff"]
        ut_df = ut_df.drop(diffs.index).reset_index(drop=True)

        diffs = diffs.rename(columns={ut.title(): "Value"})
        del diffs[args.sensitive_attribute.title()]
        diffs["Metric"] = "DS"

        stat_ks_df = ks_df.copy()
        stat_ks_df["Pvalue"] = stat_ks_df["KS"].map(lambda x: x['pvalue'])

        ks_df["KS"] = ks_df["KS"].map(lambda x: x['statistic'])
        ks_df = ks_df.rename(columns={"KS": "Value"})
        ks_df["Metric"] = "KS"

        fair_df = pd.concat([ks_df, diffs])

        ut_df = ut_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
        ut_df = ut_df.rename(columns={ut.title(): "Value"})

        ut_df["Type"] = ut.upper()
        fair_df["Type"] = "Fairness"

        del ut_df["Target"]
        del fair_df["Target"]

        new_df = pd.concat([ut_df, fair_df]).reset_index(drop=True)

        new_df["Status"] = new_df.apply(
            lambda x: 'Bef' if f"{x['Paper']} \n {x['Model']}" in paper_baseline_names_model else 'Aft',
            axis=1
        ).to_list()

        new_df["Model"] = new_df.apply(
            lambda x: x["Model"] if f"{x['Paper']} \n {x['Model']}" not in paper_baseline_names_model else (
                paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"] if
                isinstance(paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"], str) else x["Model"]
            ),
            axis=1
        ).to_list()

        for bas, m_name in paper_baseline_names_model.items():
            if not isinstance(m_name, str):
                _paper, _model = [x.strip() for x in bas.split('\n')]
                if _paper in new_df["Paper"].values:
                    bas_row = new_df.loc[(new_df["Paper"] == _paper) & (new_df["Model"] == _model)].copy()

                    for b_idx in bas_row.index:
                        new_df.loc[b_idx, "Model"] = m_name[0]

                    for _, b_row in bas_row.iterrows():
                        for _mod_name in m_name[1:]:
                            new_df = new_df.append(b_row.copy(), ignore_index=True)
                            new_df.loc[new_df.index[-1], "Model"] = _mod_name
                            new_df.loc[new_df.index[-1], "Status"] = "Bef"

        new_df = new_df.reset_index(drop=True)

        # reorder columns
        new_df_grouped = new_df.groupby(["Metric", "Status"])
        # metric_order = ["Total"] + sensitive_values + ["DS", "KS"]
        metric_order = ["Total"] + ["DS", "KS"]
        status_order = itertools.product(metric_order, ["Bef", "Aft"])
        new_df = pd.concat([new_df_grouped.get_group(group_name) for group_name in status_order])

        new_df = new_df.round(3)
        new_df = new_df.astype(str)
        new_df["Value"] = new_df["Value"].map(lambda x: f'{x:<05s}' if float(x) >= 0 else f'{x:<06s}')

        sig1p = '{\\scriptsize \\^{}}'
        sig5p = '{\\scriptsize *}'

        # Add asterisks for statistical significance to DS and KS
        for stat_metric, stat_data in [("DS", st_df), ("KS", stat_ks_df)]:
            for idx, row in new_df[new_df["Metric"] == stat_metric].iterrows():
                stat_d = stat_data.loc[stat_data["Paper"] == row["Paper"]]
                if row["Status"] == "Aft":
                    model_name = row["Model"]
                else:
                    model_name = stat_model_baseline[f"{row['Paper']} \n {row['Model']}"]

                pval = stat_d.loc[stat_d["Model"] == model_name]['Pvalue'].iloc[0]
                pval = sig1p if pval < 0.01 else (sig5p if pval < 0.05 else '')

                new_df.loc[idx, "Value"] = f'{pval}{new_df.loc[idx, "Value"]}'

        new_df['Paper'] = new_df['Paper'].map(paper_map)
        new_df['Model'] = new_df['Model'].map(model_map)

        new_df = new_df.pivot(index=["Paper", "Model"], columns=["Type", "Metric", "Status"])
        new_df.columns.names = [''] * len(new_df.columns.names)

        print(new_df)

        for col in new_df.columns:
            if col[1] == "RMSE":
                best_val = str(new_df[col].astype(float).min())
            elif col[2] == "KS":
                best_val = str(new_df[col].map(lambda x: x.replace(sig1p, '').replace(sig5p, '')).astype(float).min())
            elif col[2] == "DS":
                best_val = str(new_df[col].map(lambda x: x.replace(sig1p, '').replace(sig5p, '')).astype(float).abs().min())
            else:
                best_val = str(new_df[col].astype(float).max())

            best_rows = (
                    (new_df[col] == f"-{best_val:<05s}") |
                    (new_df[col] == f"{best_val:<05s}") |
                    (new_df[col] == sig5p + f"-{best_val:<05s}") |
                    (new_df[col] == sig5p + f"{best_val:<05s}") |
                    (new_df[col] == sig1p + f"^-{best_val:<05s}") |
                    (new_df[col] == sig1p + f"^{best_val:<05s}")
            )
            new_df.loc[best_rows, col] = ['\\bftab ' + f"{val:<05s}"
                                          if float(val.replace(sig1p, '').replace(sig5p, '')) >= 0
                                          else '\\bftab ' + f"{val:<06s}"
                                          for val in new_df.loc[best_rows, col]]

        new_df.columns = new_df.columns.droplevel([0, 1])

        print(new_df)

        with open(os.path.join(tables_path, f"paper_table_{ut.upper()}_{args.sensitive_attribute}.txt"), "w") as f:
            f.write(new_df.to_latex(
                caption=f"[{args.dataset.upper()}-{'TR' if ut.upper() == 'NDCG' else 'RP'}-{args.sensitive_attribute}] \dots",
                column_format="ll|rrrrrr|rrrr",
                multicolumn_format="c",
                label=f"{ut.lower()}_{args.sensitive_attribute.lower()}_{args.dataset.lower()}",
                escape=False
            ).replace('Aft', '\\multicolumn{1}{c}{Aft}').replace('Bef', '\\multicolumn{1}{c}{Bef}'))


def results_to_paper_table_fused_datasets():
    repr_path = os.path.join(constants.BASE_PATH, "..", "projects", "reproducibility_study")

    with open(os.path.join(repr_path, f"movielens_1m_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results_ml1m: dict = pickle.load(pk)

    with open(os.path.join(repr_path, f"movielens_1m_stats_{args.sensitive_attribute}.pkl"), "rb") as pk:
        stats_ml1m: dict = pickle.load(pk)

    with open(os.path.join(repr_path, f"filtered(20)_lastfm_1K_results_{args.sensitive_attribute}.pkl"), "rb") as pk:
        results_lfm1k: dict = pickle.load(pk)

    with open(os.path.join(repr_path, f"filtered(20)_lastfm_1K_stats_{args.sensitive_attribute}.pkl"), "rb") as pk:
        stats_lfm1k: dict = pickle.load(pk)

    tables_path = os.path.join(repr_path, "paper_tables", "fused")
    if not os.path.exists(tables_path):
        os.makedirs(tables_path)

    paper_map = {
        'Ekstrand': 'Ekstrand et al.',
        'Librec': 'Burke et al.',
        'Antidote Data': 'Rastegarpanah et al.',
        'Kamishima': 'Kamishima et al.',
        'FairGo': 'Wu et al.',
        'Co-clustering': 'Frisch et al.',
        'User-oriented fairness': 'Li et al.',
        'Haas': 'Ashokan et al.'
    }

    model_map = {
        'BN-SLIM-U': 'SLIM-U',
        'Parity LBM': 'LBM',
        'STAMP': 'STAMP',
        'BiasedMF': 'BiasedMF',
        'PMF': 'PMF',
        'NeuMF': 'NCF',
        'FunkSVD': 'FunkSVD',
        'TopPopular': 'TopPopular',
        'User-User': 'UserKNN',
        'Item-Item': 'ItemKNN',
        'Mean': 'AvgRating',
        'PMF Mean Matching': 'PMF Mean',
        'PMF BDist Matching': 'PMF BDist',
        'PMF Mi Normal': 'PMF Mi',
        'ALS MF': 'ALS',
        'LMaFit MF': 'LMaFit',
        'FairGo GCN': 'FairGo GCN',
        'ALS BiasedMF': 'ALS BiasedMF',
        'ALS BiasedMF Parity': 'ALS BiasedMF Par',
        'ALS BiasedMF Val': 'ALS BiasedMF Val',
        'Item-Item Parity': 'ItemKNN Par',
        'Item-Item Val': 'ItemKNN Val'
    }

    for ut, targ in [["ndcg", "Ranking"], ["rmse", "Rating"]]:
        out_dfs = []
        for dataset, results, stats, exps_models_gender, exps_models_age in [
            ("ML1M", results_ml1m, stats_ml1m, experiments_models_gender_ml1m, experiments_models_age_ml1m),
            ("LFM1K", results_lfm1k, stats_lfm1k, experiments_models_gender_lfm1k, experiments_models_age_lfm1k)
        ]:
            if args.sensitive_attribute == "Gender":
                exps_df = pd.DataFrame(exps_models_gender,
                                       columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])
            else:
                exps_df = pd.DataFrame(exps_models_age,
                                       columns=['target', 'paper', 'model', 'id_file', 'baseline', 'read_f', 'read_f_base'])

            paper_baseline_names_model = {}
            stat_model_baseline = {}
            for _, exp in exps_df.iterrows():
                _, baseline_name = parse_baseline(exp)
                paper_baseline_names_model[f"{exp['paper']} \n {baseline_name}"] = exp['model']
                if not isinstance(exp['model'], str):
                    for m_name in exp['model']:
                        stat_model_baseline[f"{exp['paper']} \n {m_name}"] = baseline_name
                else:
                    stat_model_baseline[f"{exp['paper']} \n {exp['model']}"] = baseline_name

            paper_metrics = {metr: res for metr, res in results.items() if metr in ['ndcg', 'rmse', 'ks']}
            paper_stats = {metr: st for metr, st in stats.items() if metr in ['ndcg', 'rmse']}

            paper_dfs = {
                metr: pd.DataFrame(
                    dict_df,
                    columns=["Target", "Paper/Model", metr.replace('_', ' ').title(), args.sensitive_attribute.title()]
                ) if metrics_type[metr] == "with_diff" else pd.DataFrame(
                        dict_df,
                        columns=["Target", "Paper/Model", metr.replace('_', ' ').upper()]
                    )
                for metr, dict_df in paper_metrics.items()
            }

            stats_dfs = {
                st: pd.DataFrame(stat_dict, columns=["Target", "Paper/Model", "Stat"]) for st, stat_dict in paper_stats.items()
            }

            for metr in paper_dfs:
                paper_dfs[metr][["Paper", "Model"]] = paper_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
                del paper_dfs[metr]["Paper/Model"]

                paper_dfs[metr]["Paper"] = paper_dfs[metr]["Paper"].str.strip()
                paper_dfs[metr]["Model"] = paper_dfs[metr]["Model"].str.strip()

                if metr != "ks":
                    stats_dfs[metr][["Paper", "Model"]] = stats_dfs[metr]["Paper/Model"].str.split('\n', expand=True)
                    del stats_dfs[metr]["Paper/Model"]

                    stats_dfs[metr]["Paper"] = stats_dfs[metr]["Paper"].str.strip()
                    stats_dfs[metr]["Model"] = stats_dfs[metr]["Model"].str.strip()

                    stats_dfs[metr][["Stat", "Pvalue"]] = [(x["statistic"], x["pvalue"]) for x in stats_dfs[metr]["Stat"]]

            ut_df = paper_dfs[ut]
            ks_df = paper_dfs["ks"][paper_dfs["ks"]["Target"] == targ]
            st_df = stats_dfs[ut][stats_dfs[ut]["Target"] == targ]

            diffs = ut_df[ut_df[args.sensitive_attribute.title()] == "Diff"]
            ut_df = ut_df.drop(diffs.index).reset_index(drop=True)

            diffs = diffs.rename(columns={ut.title(): "Value"})
            del diffs[args.sensitive_attribute.title()]
            diffs["Metric"] = "DS"

            stat_ks_df = ks_df.copy()
            stat_ks_df["Pvalue"] = stat_ks_df["KS"].map(lambda x: x['pvalue'])

            ks_df["KS"] = ks_df["KS"].map(lambda x: x['statistic'])
            ks_df = ks_df.rename(columns={"KS": "Value"})
            ks_df["Metric"] = "KS"

            fair_df = pd.concat([ks_df, diffs])

            ut_df = ut_df.rename(columns={args.sensitive_attribute.title(): 'Metric'})
            ut_df = ut_df.rename(columns={ut.title(): "Value"})

            ut_df["Type"] = ut.upper()
            fair_df["Type"] = "Fairness"

            del ut_df["Target"]
            del fair_df["Target"]

            new_df = pd.concat([ut_df, fair_df]).reset_index(drop=True)

            new_df["Status"] = new_df.apply(
                lambda x: 'Bef' if f"{x['Paper']} \n {x['Model']}" in paper_baseline_names_model else 'Aft',
                axis=1
            ).to_list()

            new_df["Model"] = new_df.apply(
                lambda x: x["Model"] if f"{x['Paper']} \n {x['Model']}" not in paper_baseline_names_model else (
                    paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"] if
                    isinstance(paper_baseline_names_model[f"{x['Paper']} \n {x['Model']}"], str) else x["Model"]
                ),
                axis=1
            ).to_list()

            for bas, m_name in paper_baseline_names_model.items():
                if not isinstance(m_name, str):
                    _paper, _model = [x.strip() for x in bas.split('\n')]
                    if _paper in new_df["Paper"].values:
                        bas_row = new_df.loc[(new_df["Paper"] == _paper) & (new_df["Model"] == _model)].copy()

                        for b_idx in bas_row.index:
                            new_df.loc[b_idx, "Model"] = m_name[0]

                        for _, b_row in bas_row.iterrows():
                            for _mod_name in m_name[1:]:
                                new_df = new_df.append(b_row.copy(), ignore_index=True)
                                new_df.loc[new_df.index[-1], "Model"] = _mod_name
                                new_df.loc[new_df.index[-1], "Status"] = "Bef"

            new_df = new_df.reset_index(drop=True)

            # reorder columns
            new_df_grouped = new_df.groupby(["Metric", "Status"])
            # metric_order = ["Total"] + sensitive_values + ["DS", "KS"]
            metric_order = ["Total"] + ["DS", "KS"]
            status_order = itertools.product(metric_order, ["Bef", "Aft"])
            new_df = pd.concat([new_df_grouped.get_group(group_name) for group_name in status_order])

            new_df = new_df.round(3)
            new_df = new_df.astype(str)
            new_df["Value"] = new_df["Value"].map(lambda x: f'{x:<05s}' if float(x) >= 0 else f'{x:<06s}')

            sig1p = '{\\scriptsize \\^{}}'
            sig5p = '{\\scriptsize *}'

            # Add asterisks for statistical significance to DS and KS
            for stat_metric, stat_data in [("DS", st_df), ("KS", stat_ks_df)]:
                for idx, row in new_df[new_df["Metric"] == stat_metric].iterrows():
                    stat_d = stat_data.loc[stat_data["Paper"] == row["Paper"]]
                    if row["Status"] == "Aft":
                        model_name = row["Model"]
                    else:
                        model_name = stat_model_baseline[f"{row['Paper']} \n {row['Model']}"]

                    pval = stat_d.loc[stat_d["Model"] == model_name]['Pvalue'].iloc[0]
                    pval = sig1p if pval < 0.01 else (sig5p if pval < 0.05 else '')

                    new_df.loc[idx, "Value"] = f'{pval}{new_df.loc[idx, "Value"]}'

            new_df['Paper'] = new_df['Paper'].map(paper_map)
            new_df['Model'] = new_df['Model'].map(model_map)

            new_df = new_df.pivot(index=["Paper", "Model"], columns=["Type", "Metric", "Status"])
            new_df.columns.names = [''] * len(new_df.columns.names)

            print(new_df)

            for col in new_df.columns:
                if col[1] == "RMSE":
                    best_val = str(new_df[col].astype(float).min())
                elif col[2] == "KS":
                    best_val = str(new_df[col].map(lambda x: x.replace(sig1p, '').replace(sig5p, '')).astype(float).min())
                elif col[2] == "DS":
                    best_val = str(new_df[col].map(lambda x: x.replace(sig1p, '').replace(sig5p, '')).astype(float).abs().min())
                else:
                    best_val = str(new_df[col].astype(float).max())

                best_rows = (
                        (new_df[col] == f"-{best_val:<05s}") |
                        (new_df[col] == f"{best_val:<05s}") |
                        (new_df[col] == sig5p + f"-{best_val:<05s}") |
                        (new_df[col] == sig5p + f"{best_val:<05s}") |
                        (new_df[col] == sig1p + f"^-{best_val:<05s}") |
                        (new_df[col] == sig1p + f"^{best_val:<05s}")
                )
                new_df.loc[best_rows, col] = ['\\bftab ' + f"{val:<05s}"
                                              if float(val.replace(sig1p, '').replace(sig5p, '')) >= 0
                                              else '\\bftab ' + f"{val:<06s}"
                                              for val in new_df.loc[best_rows, col]]

            new_df.columns = new_df.columns.droplevel([1])
            new_df = new_df.rename(columns={'Value': dataset})

            out_dfs.append(new_df)

        out_dfs = pd.concat(out_dfs, axis=1, join="outer")
        out_dfs.fillna('-', inplace=True)
        print(out_dfs)

        with open(os.path.join(tables_path, f"paper_table_{ut.upper()}_{args.sensitive_attribute}.tex"), "w") as f:
            f.write(out_dfs.to_latex(
                caption=f"[FUSED-{'TR' if ut.upper() == 'NDCG' else 'RP'}-{args.sensitive_attribute}] \dots",
                column_format="ll|rrrrrr|rrrrrr",
                multicolumn_format="c",
                label=f"{ut.lower()}_{args.sensitive_attribute.lower()}_{args.dataset.lower()}",
                escape=False
            ).replace('Aft', '\\multicolumn{1}{c}{Aft}').replace('Bef', '\\multicolumn{1}{c}{Bef}'))


if __name__ == "__main__":
    main()
