import os
import inspect

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import data.utils as data_utils
from helpers.recsys_arg_parser import RecSysArgumentParser
from helpers.logger import RcLogger

"""
Commands to choose the right dataset from which the data for algorithms is generated
  
  # MovieLens 1M
      Example with 70% train set, 10% validation set, 20% test set for gender:
      python -m Preprocessing.generate_input_data -dataset movielens 
      --dataset_size 1m -model Pointwise --n_reps 2 --train_val_test_split_type per_user_timestamp
      --train_val_test_split 70% 10% 20% --users_field user_id --items_field movie_id --rating_field user_rating
      --sensitive_field user_gender
      
      Example with 70% train set, 10% validation set, 20% test set for age:
      python -m Preprocessing.generate_input_data -dataset movielens 
      --dataset_size 1m -model Pointwise --n_reps 2 --train_val_test_split_type per_user_timestamp
      --train_val_test_split 70% 10% 20% --users_field user_id --items_field movie_id --rating_field user_rating
      --sensitive_field bucketized_user_age
      
  # Last.FM 1K
      Example with 70% train set, 10% validation set, 20% test set for gender:
      python -m Preprocessing.generate_input_data -dataset lastfm --dataset_size 1K
      -model Pointwise --n_reps 2 --min_interactions 20 --train_val_test_split_type per_user_random 
      --train_val_test_split 70% 10% 20% --users_field user_id --items_field artist_id --rating_field user_rating
      --sensitive_field user_gender
      
      Example with 70% train set, 10% validation set, 20% test set for age:
      python -m Preprocessing.generate_input_data -dataset lastfm --dataset_size 1K
      -model Pointwise --n_reps 2 --min_interactions 20 --train_val_test_split_type per_user_random 
      --train_val_test_split 70% 10% 20% --users_field user_id --items_field artist_id --rating_field user_rating
      --sensitive_field user_age
  
  The specific files to create can be specified with one of the arguments inside
  RecSysArgumentParser._add_other_features_args inside helpers.recsys_arg_parser, which name ends with `input_data`
  or you can call the relative function inside data.utils
"""

if __name__ == "__main__":
    args = RecSysArgumentParser().routine_arg_parser()
    RcLogger.start_logger(level="INFO")

    metadata = vars(args).copy()

    orig_train, validation, test = data_utils.load_train_val_test(metadata)
    
    if data_utils.preprocessed_dataset_exists(metadata, "binary", split="train"):
        train = data_utils.load_tf_features_dataset(metadata, "binary", split="train")
    else:
        train = None

    base_reproduce_path = os.path.join(os.path.dirname(inspect.getsourcefile(lambda: 0)), os.pardir, 'reproducibility_study')

    if metadata['dataset'] == "movielens":
        sensitive_values = ['Male', 'Female'] if args.sensitive_field == 'user_gender' else ['1-34', '35-56+']
    
        kwargs = {
            'fairgo_sensitive_fields': ['user_gender', 'bucketized_user_age', None],
            'model_paths': {
                'PMF': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'result', r"PMF\11_PMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
                'NeuMF': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'result', r"NCF\11_NCF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
                'STAMP': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'result', r"STAMP\11_STAMP_movielens-1m_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
                'BiasedMF': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_movielens-1m_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy")
            },
            'test_path': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'dataset', r"movielens_1m\movielens_1m.test.csv"),
            'sensitive_values': sensitive_values
        }
    else:
        sensitive_values = ['Male', 'Female'] if args.sensitive_field == 'user_gender' else ['1-24', '25+']

        kwargs = {
            'fairgo_sensitive_fields': ['user_gender', 'user_age'],
            'model_paths': {
                'PMF': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'result', r"PMF\11_PMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy"),
                'NeuMF': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'result', r"NCF\11_NCF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_lay[32,16,8]_los1_lr0.001_optAdam_pla[64]_sam1.0_tes100_tra1_uve64__test.npy"),
                'STAMP': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'result', r"STAMP\11_STAMP_filtered(20)-lastfm-1K_2018_all0_att64_bat128_dro1_dro1_dro0.2_ear0_epo100_gra10_hid64_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_max30_neg0_neg1_neg0_neg[]_num1_optAdam_pla[64]_sam1.0_spa0_sup0_tes100_tra1_uve64__test.npy"),
                'BiasedMF': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'result', r"BiasedMF\11_BiasedMF_filtered(20)-lastfm-1K_2018_bat128_dro1_dro0.2_ear0_epo100_gra10_ive64_l21e-05_l2b0_l2s0.0_los1_lr0.001_optAdam_sam1.0_tes100_tra1_uve64__test.npy")
            },
            'test_path': os.path.join(base_reproduce_path, 'Li et al', 'NLR', 'dataset', r"filtered(20)_lastfm_1K\filtered(20)_lastfm_1K.test.csv"),
            'sensitive_values': sensitive_values
        }
        
    if validation is not None:
        kwargs['train_val_as_train'] = True,  # some works do not use validation, so train = train + validation if True

    data_utils.to_input_data_from_metadata(metadata, orig_train, test, train=train, validation=validation, **kwargs)
