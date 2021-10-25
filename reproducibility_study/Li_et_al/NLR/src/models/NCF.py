# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *


class NCF(RecModel):
    @staticmethod
    def parse_model_args(parser, model_name='NCF'):
        parser.add_argument('--layers', type=str, default='[64]',
                            help="Size of mlp layers.")
        parser.add_argument('--p_layers', type=str, default='[]',
                            help="Size of prediction mlp layers.")
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, label_min, label_max, feature_num, user_num, item_num, u_vector_size, i_vector_size,
                 layers, p_layers, random_seed, model_path, **kwargs):
        self.layers = eval(layers)
        self.p_layers = eval(p_layers)
        RecModel.__init__(self, label_min=label_min, label_max=label_max, feature_num=feature_num,
                          user_num=user_num, item_num=item_num, u_vector_size=u_vector_size,
                          i_vector_size=i_vector_size, random_seed=random_seed, model_path=model_path, **kwargs)

    def _init_weights(self):
        self.gmf_uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.gmf_iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.mlp_uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.mlp_iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)

        self.mlp = nn.ModuleList([])
        pre_size = 2 * self.ui_vector_size
        for layer_size in self.layers:
            self.mlp.append(nn.Linear(pre_size, layer_size))
            pre_size = layer_size

        self.p_layer = nn.ModuleList([])
        pre_size = pre_size + self.ui_vector_size
        for layer_size in self.p_layers:
            self.p_layer.append(nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.prediction = torch.nn.Linear(pre_size, 1)
        self.l2_embeddings = []

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['uid']
        i_ids = feed_dict['iid']

        gmf_u_vectors = self.gmf_uid_embeddings(u_ids)
        gmf_i_vectors = self.gmf_iid_embeddings(i_ids)
        mlp_u_vectors = self.mlp_uid_embeddings(u_ids)
        mlp_i_vectors = self.mlp_iid_embeddings(i_ids)

        gmf = gmf_u_vectors * gmf_i_vectors

        mlp = torch.cat((mlp_u_vectors, mlp_i_vectors), dim=1)
        for layer in self.mlp:
            mlp = layer(mlp)
            mlp = F.relu(mlp)
            mlp = torch.nn.Dropout(p=feed_dict['dropout'])(mlp)

        output = torch.cat((gmf, mlp), dim=1)
        for layer in self.p_layer:
            output = layer(output)
            output = F.relu(output)
            output = torch.nn.Dropout(p=feed_dict['dropout'])(output)

        prediction = self.prediction(output).view([-1])
        check_list.append(('prediction', prediction))

        out_dict = {'prediction': prediction,
                    'check': check_list, EMBEDDING_L2: []}
        return out_dict
