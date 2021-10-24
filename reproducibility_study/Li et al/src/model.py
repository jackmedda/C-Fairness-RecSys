from rerankings.user_oriented_fairness.src.data_loader import DataLoader
from rerankings.user_oriented_fairness.src.utils.tools import *
import pandas as pd
import gurobipy as gp
import os
from gurobipy import GRB

### Non original project lines code ###
import inspect


class UGF(object):
    def __init__(self, data_loader, k, eval_metric_list, fairness_metric='f1',
                 epsilon=0.05, logger=None, model_name='', group_name=''):
        """
        Train fairness model
        :param data_loader: Dataloader object
        :param k: k for top-K number of items to be selected from the entire list
        :param eval_metric_list: a list contains all the metrics to report
        :param fairness_metric: a string, the metric used for fairness constraint, default='f1'
        :param epsilon: the upper bound for the difference between two groups scores
        :param logger: logger for logging info
        """
        self.data_loader = data_loader
        self.dataset_name = data_loader.path.split('/')[-1]
        self.k = k
        self.eval_metric_list = eval_metric_list
        self.fairness_metric = fairness_metric
        self.epsilon = epsilon
        self.model_name = model_name
        self.group_name = group_name
        if logger is None:
            self.logger = create_logger()
        else:
            self.logger = logger

    @staticmethod
    def _check_df_format(df):
        """
        check if the input dataframe contains all the necessary columns
        :return: None
        """
        expected_columns = ['uid', 'iid', 'score', 'label', 'q']
        for c in expected_columns:
            if c not in df.columns:
                raise KeyError('Missing column ' + c)

    @staticmethod
    def _build_fairness_optimizer(group_df_list, k, metric, name='UGF'):
        """
        Use Gurobi to build faireness optimizer
        :param group_df_list: a list contains dataframes from two groups
        :param k: an integer for the length of top-K list
        :param metric: the metric string for fairness constraint. e.g. 'f1', 'recall', 'precision'
        :param name: a string which is the name of this optimizer
        :return: the Gurobi model, a list of Qi*Si for obj function, a list of two group metric
        """
        try:
            # Create a new model
            m = gp.Model(name)
            var_score_list = []                 # store the Qi * Si
            metric_list = []                    # store the averaged metric scores for two groups
            # Create variables
            for df in group_df_list:
                df_group = df.groupby('uid')
                tmp_metric_list = []            # store the metric calculation for each user in current user group
                tmp_var_score_list = []         # store var * score for object function use
                for uid, group in df_group:
                    tmp_var_list = []           # store variables for sum(Qi) == k use
                    tmp_var_label_list = []     # store var * label for recall calculation use
                    score_list = group['score'].tolist()
                    label_list = group['label'].tolist()
                    item_list = group['iid'].tolist()

                    for i in range(len(item_list)):
                        var_name = str(uid) + '_' + str(item_list[i])   # variable name is "uid_iid"
                        v = m.addVar(vtype=GRB.BINARY, name=var_name)
                        tmp_var_list.append(v)
                        tmp_var_score_list.append(score_list[i] * v)
                        tmp_var_label_list.append(label_list[i] * v)
                    # Add first constraint: Sum(Qi)==k
                    m.addConstr(gp.quicksum(tmp_var_list) == k)
                    # calculate the corresponding measures
                    if group['label'].sum() == 0:
                        continue
                    if metric == 'recall':
                        tmp_metric_list.append(gp.quicksum(tmp_var_label_list) / group['label'].sum())
                    elif metric == 'precision':
                        tmp_metric_list.append(gp.quicksum(tmp_var_label_list) / k)
                    elif metric == 'f1':
                        f1 = 2 * gp.quicksum(tmp_var_label_list) / (group['label'].sum() + k)
                        tmp_metric_list.append(f1)
                    else:
                        raise ValueError('Unknown metric for optimizer building.')

                metric_list.append(gp.quicksum(tmp_metric_list) / len(tmp_metric_list))
                var_score_list.extend(tmp_var_score_list)
            m.update()
            return m, var_score_list, metric_list

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
        except AttributeError:
            print('Encountered an attribute error')
        pass

    @staticmethod
    def _format_result(model, df):
        """
        format the gurobi results to dataframe.
        :param model: optimized gurobi model
        :param df: the pandas dataframe to add the optimized results into
        :return: None
        """
        from tqdm import tqdm

        df['sc_lab_q'] = list(zip(df['score'], df['label'], df['q']))
        del df['score']
        del df['label']
        del df['q']

        df = df.pivot(index="uid", columns="iid", values="sc_lab_q")
        for v in tqdm(model.getVars()):
            v_s = v.varName.split('_')
            uid = int(v_s[0])
            iid = int(v_s[1])
            _data = df.loc[uid, iid]
            df.loc[uid, iid] = (_data[0], _data[1], v.x)
            # df.loc[(df['uid'] == uid) & (df['iid'] == iid), 'q'] = int(v.x)
            """
            v_s = v.varName.split('_')
            uid = int(v_s[0])
            iid = int(v_s[1])
            new_q[f"{uid}_{iid}"] = v.x
            """
        df = df.melt(ignore_index=False).reset_index()
        df = df.rename(columns={'index': 'uid', 'variable': 'iid'})

        scores = np.empty(len(df), dtype=np.float32)
        labels = np.empty(len(df), dtype=np.int32)
        qs = np.empty(len(df), dtype=np.float32)
        for i, val in enumerate(df['value']):
            scores[i] = val[0]
            labels[i] = val[1]
            qs[i] = val[2]

        df['score'] = scores
        df['label'] = labels
        df['q'] = qs

        del df['value']

        return df

    def _print_metrics(self, df, metrics, message='metric scores'):
        """
        Print out evaluation scores
        :param df: the dataframe contains the data for evaluation
        :param metrics: a list, contains the metrics to report
        :param message: a string, for print message
        :return: None
        """
        results = evaluation_methods(df, metrics=metrics)
        r_string = ""
        for i in range(len(metrics)):
            r_string = r_string + metrics[i] + "=" + '{:.4f}'.format(results[i]) + " "
        print(message + ": " + r_string)
        # write the message into the log file
        self.logger.info(message + ": " + r_string)

    def train(self):
        """
        Train fairness model
        """
        # model = read('gurobi_model.mps')

        # Prepare data
        all_df = self.data_loader.rank_df.copy(deep=True)    # the dataframe with entire test data
        self._check_df_format(all_df)   # check the dataframe format
        group_df_list = [self.data_loader.g1_df.copy(deep=True),
                         self.data_loader.g2_df.copy(deep=True)]  # group 1 (active), group 2 (inactive)

        # Print original evaluation results
        self.logger.info('Model:{} | Dataset:{} | Group:{} |  Epsilon={} | K={} | GRU_metric={}'
                         .format(self.model_name, self.dataset_name, self.group_name,
                                 self.epsilon, self.k, self.fairness_metric))
        self._print_metrics(all_df, self.eval_metric_list, 'Before optimization overall scores           ')
        self._print_metrics(group_df_list[0], self.eval_metric_list, 'Before optimization group 1 (active) scores  ')
        self._print_metrics(group_df_list[1], self.eval_metric_list, 'Before optimization group 2 (inactive) scores')

        # build optimizer
        m, var_score_list, metric_list = \
            self._build_fairness_optimizer(group_df_list, self.k, metric=self.fairness_metric, name='UGF_f1')

        # |group_1_recall - group_2_recall| <= epsilon
        m.addConstr(metric_list[0] - metric_list[1] <= self.epsilon)
        m.addConstr(metric_list[1] - metric_list[0] <= self.epsilon)

        # Set objective function
        m.setObjective(gp.quicksum(var_score_list), GRB.MAXIMIZE)

        # Optimize model
        m.optimize()

        # m.write('gurobi_model.mps')

        # Format the output results and update q column of the dataframe
        all_df = self._format_result(m, all_df)
        #####################################
        all_df.to_csv(os.path.join(
            os.path.dirname(inspect.getsourcefile(lambda: 0)),
            os.pardir,
            'out_results',
            f"{dataset_name}_{model_name}_{group_name_title}_out.csv"
        ), index=False)
        exit()
        #####################################
        group_df_list[0].drop(columns=['q'], inplace=True)
        group_df_list[0] = pd.merge(group_df_list[0], all_df, on=['uid', 'iid', 'score', 'label'], how='left')
        group_df_list[1].drop(columns=['q'], inplace=True)
        group_df_list[1] = pd.merge(group_df_list[1], all_df, on=['uid', 'iid', 'score', 'label'], how='left')

        # Print updated evaluation results
        self._print_metrics(all_df, self.eval_metric_list, 'After optimization overall metric scores     ')
        self._print_metrics(group_df_list[0], self.eval_metric_list, 'After optimization group 1 (active) scores   ')
        self._print_metrics(group_df_list[1], self.eval_metric_list, 'After optimization group 2 (inactive) scores ')
        self.logger.info('\n\n')


if __name__ == '__main__':
    """
    Please update the following block for different datasets
    """
    ############### Parameters to be changed for different datasets ###########
    epsilon = 0.0                           # fairness constraint coefficient
    dataset_folder = os.path.join(os.path.dirname(inspect.getsourcefile(lambda: 0)), os.pardir, 'data')           # dataset directory
    dataset_name = 'filtered(20)_lastfm_1K'           # dataset name
    # dataset_name = 'movielens_1m'  # dataset name
    model_name = 'BiasedMF'                      # model name (which model does this ranking file generated by)
    group_name_title = ''           # grouping method name for distinguish different experiment results
    logger_dir = os.path.join(os.path.dirname(inspect.getsourcefile(lambda: 0)), os.pardir, 'results', model_name)    # logging file path

    ### Non original project lines code ###
    # group_1 = "Male"
    # group_2 = "Female"

    if 'lastfm' in model_name:
        print()
        group_1 = "1-24"
        group_2 = "25+"
    else:
        print()
        # group_1 = "1-34"
        # group_2 = "35-56+"

    data_path = os.path.join(dataset_folder, dataset_name)
    run_id = '41d1bbe242'
    rank_file = [f for f in os.listdir(data_path) if model_name in f and 'rank.csv' in f and run_id in f][0]   # original input ranking csv file name
    group_name_title = rank_file.split('_')[1]

    group_1_file = [f for f in os.listdir(data_path) if model_name in f and group_1 in f and run_id in f][0]      # advantaged group testing file name
    group_2_file = [f for f in os.listdir(data_path) if model_name in f and group_2 in f and run_id in f][0]    # disadvantaged group testing file name
    ############################################################################

    if not os.path.exists(logger_dir):
        os.mkdir(logger_dir)
    logger_file = model_name + '_' + dataset_name + '_' + group_name_title + '_reRank_result.log'
    logger_path = os.path.join(logger_dir, logger_file)
    dl = DataLoader(data_path, rank_file=rank_file, group_1_file=group_1_file, group_2_file=group_2_file)

    logger = create_logger(name='result_logger', path=logger_path)

    metrics = ['ndcg', 'f1']
    topK = ['10']

    metrics_list = [metric + '@' + k for metric in metrics for k in topK]

    UGF_model = UGF(dl, k=10, eval_metric_list=metrics_list, fairness_metric='f1',
                    epsilon=epsilon, logger=logger, model_name=model_name, group_name=group_name_title)
    UGF_model.train()


