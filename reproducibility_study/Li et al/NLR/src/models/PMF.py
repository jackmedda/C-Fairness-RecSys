# coding=utf-8

import torch
from models.RecModel import RecModel
from utils.global_p import *


class PMF(RecModel):
    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.l2_embeddings = []

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]

        pmf_u_vectors = self.uid_embeddings(u_ids)
        pmf_i_vectors = self.iid_embeddings(i_ids)
        prediction = (pmf_u_vectors * pmf_i_vectors).sum(dim=1).view([-1])

        out_dict = {'prediction': prediction,
                    'check': check_list, EMBEDDING_L2: []}
        return out_dict
