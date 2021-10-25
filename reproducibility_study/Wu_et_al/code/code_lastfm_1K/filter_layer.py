# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys 

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
 
import torch.nn.functional as F
import torch.autograd as autograd 

import pdb
from collections import defaultdict
import time 
from shutil import copyfile
import pickle
import data_utils


class DisClfGender(nn.Module):
    def __init__(self,embed_dim,out_dim,attribute,use_cross_entropy=True):
        super(DisClfGender, self).__init__()
        self.embed_dim = int(embed_dim) 
        self.attribute = attribute
        self.criterion = nn.CrossEntropyLoss()#torch.nn.BCELoss()#nn.CrossEntropyLoss()   
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim/4), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim/4), int(self.embed_dim/8), bias=True), 
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Linear(int(self.embed_dim /8),int(self.embed_dim /16), bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(int(self.embed_dim /16), self.out_dim , bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.out_dim, self.out_dim , bias=True),
            # nn.Sigmoid()
            )
    def forward(self, ents_emb, labels, return_loss=True):
        scores = self.net(ents_emb)
        outputs = F.softmax(scores, dim=1)#F.log_softmax(scores, dim=1)
        # outputs=scores
        # pdb.set_trace()
        if return_loss:
            loss = self.criterion(outputs, labels)
            # pdb.set_trace()
            # if loss>100:
            #     print('gender d process')
            #     pdb.set_trace()
            return loss
        else:
            return outputs,labels


class DisClfAge(nn.Module):
    def __init__(self,embed_dim,out_dim,attribute,use_cross_entropy=True):
        super(DisClfAge, self).__init__()
        self.embed_dim = int(embed_dim) 
        self.attribute = attribute
        self.criterion = nn.CrossEntropyLoss()#torch.nn.BCELoss()#nn.CrossEntropyLoss()  
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim/4), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim/4), int(self.embed_dim/8), bias=True), 
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Linear(int(self.embed_dim /8),int(self.embed_dim /16), bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(int(self.embed_dim /16), self.out_dim , bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.out_dim, self.out_dim , bias=True),
            )
    def forward(self, ents_emb, labels, return_loss=True):
        scores = self.net(ents_emb)
        outputs = F.softmax(scores, dim=1)#F.log_softmax(scores, dim=1)
        # outputs = scores
        # pdb.set_trace()
        if return_loss:
            loss = self.criterion(outputs, labels)
            # if loss>100:
            #     print('age d process')
            #     pdb.set_trace()
            # pdb.set_trace()
            return loss
        else:
            return outputs,labels




class AttributeFilter(nn.Module):
    def __init__(self, embed_dim, attribute='gender'):
        super(AttributeFilter, self).__init__()
        self.embed_dim = embed_dim
        self.attribute = attribute 
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim*2), bias=True), 
            nn.LeakyReLU(0.2,inplace=True),  
            nn.Dropout(p=0.3), 
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True), 
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim/2), int(self.embed_dim), bias=True), 
            nn.LeakyReLU(0.2,inplace=True), 
            nn.Dropout(p=0.3),
            nn.Linear(self.embed_dim, self.embed_dim , bias=True),
            nn.Linear(self.embed_dim, self.embed_dim , bias=True)
            # nn.Sigmoid()
            )
    def forward(self, ents_emb):
        # h0 = F.leaky_relu(self.W0(ents_emb))
        # h1 = F.leaky_relu(self.fc1(ents_emb))#self.fc1(ents_emb)#
        # h2 = F.leaky_relu(self.fc2(h1))
        # h3 = (self.fc3(h2))
        # h2 = self.batchnorm(h2)
        h3 = self.net(ents_emb)
        return h3

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


class LineGCN(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(LineGCN, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.user_num=user_num
        self.item_num=item_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 
        self.factor_num=factor_num  

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)  
  
    def forward(self,user_item_matrix,item_user_matrix,d_i_train,d_j_train): 
        
        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        # pdb.set_trace()
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]] 
            
        d_i_train0=torch.cuda.FloatTensor(d_i_train)
        d_j_train0=torch.cuda.FloatTensor(d_j_train)
        # print(d_i_train0.shape,d_j_train0.shape)
        # pdb.set_trace() 
        d_i_train1=d_i_train0.expand(-1,self.factor_num)
        d_j_train1=d_j_train0.expand(-1,self.factor_num)

        users_embedding=self.embed_user.weight#torch.cat((self.embed_user.weight, users_features0),1)
        items_embedding=self.embed_item.weight#torch.cat((self.embed_item.weight, items_features0),1) 

        gcn1_users_embedding = (torch.sparse.mm(user_item_matrix, items_embedding) + users_embedding.mul(d_i_train1))#*2. #+ users_embedding
        gcn1_items_embedding = (torch.sparse.mm(item_user_matrix, users_embedding) + items_embedding.mul(d_j_train1))#*2. #+ items_embedding

        gcn_users_embedding= torch.cat((users_embedding,gcn1_users_embedding),-1)#+gcn4_users_embedding
        gcn_items_embedding= torch.cat((items_embedding,gcn1_items_embedding),-1)#+gcn4_items_embedding#
        
 
        return gcn_users_embedding,gcn_items_embedding#,torch.unsqueeze(torch.cat((gcn_users_embedding, gcn_items_embedding),0), 0)






# class AttributeLineGCN(nn.Module):
#     def __init__(self, user_num, item_num, factor_num,users_features,items_features):
#         super(AttributeLineGCN, self).__init__()
#         """
#         user_num: number of users;
#         item_num: number of items;
#         factor_num: number of predictive factors.
#         """     
#         self.user_num=user_num
#         self.item_num=item_num
#         self.embed_user = nn.Embedding(user_num, factor_num)
#         self.embed_item = nn.Embedding(item_num, factor_num) 
#         self.factor_num=factor_num   

#         self.users_features = torch.cuda.FloatTensor(users_features)
#         self.items_features = torch.cuda.FloatTensor(items_features)
#         nn.init.normal_(self.embed_user.weight, std=0.01)
#         nn.init.normal_(self.embed_item.weight, std=0.01)  

#         self.fc1 = nn.Linear(30, 48, bias=True)
#         self.fc2 = nn.Linear(18, 48, bias=True)
        
#         # self.model_filter1 = AttributeFilter(112, attribute='gender')
#         # self.model_filter2 = AttributeFilter(112, attribute='age')
#         # self.model_filter3 = AttributeFilter(112, attribute='occupation')

#     def forward(self,user_item_matrix,item_user_matrix,d_i_train,d_j_train,attribute_filter=0):   
        
#         for i in range(len(d_i_train)):
#             d_i_train[i] = [d_i_train[i]]
#         # pdb.set_trace()
#         for i in range(len(d_j_train)):
#             d_j_train[i] = [d_j_train[i]] 
            
#         d_i_train0 = torch.cuda.FloatTensor(d_i_train)
#         d_j_train0 = torch.cuda.FloatTensor(d_j_train)
#         # print(d_i_train0.shape,d_j_train0.shape)
#         # pdb.set_trace() 
#         d_i_train1 = d_i_train0.expand(-1,self.factor_num)
#         d_j_train1 = d_j_train0.expand(-1,self.factor_num)
#         users_features0 = self.fc1(self.users_features)
#         items_features0 = self.fc2(self.items_features)

#         users_embedding = torch.cat((self.embed_user.weight, users_features0),1)
#         items_embedding = torch.cat((self.embed_item.weight, items_features0),1)  

#         gcn1_users_embedding = (torch.sparse.mm(user_item_matrix, items_embedding) + users_embedding)#.mul(d_i_train1))#*2. #+ users_embedding
#         gcn1_items_embedding = (torch.sparse.mm(item_user_matrix, users_embedding) + items_embedding)#.mul(d_j_train1))#*2. #+ items_embedding
        
        
#         gcn2_users_embedding = (torch.sparse.mm(user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding)#.mul(d_i_train1))#*2. + users_embedding
#         gcn2_items_embedding = (torch.sparse.mm(item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding)#.mul(d_j_train1))#*2. + items_embedding
         
#         # #user_gender,user_age,user_occpuation 2 + 7 +21
 
#         gcn_users_embedding= gcn2_users_embedding#torch.cat((users_embedding,gcn1_users_embedding,gcn2_users_embedding),-1)#+gcn4_users_embedding
#         gcn_items_embedding= gcn2_items_embedding#torch.cat((items_embedding,gcn1_items_embedding,gcn2_items_embedding),-1)#+gcn4_items_embedding#
#        # user_cat_attribute_real = torch.cat([gcn_users_embedding,self.users_features[:,:2]],-1)

#         return gcn_users_embedding,gcn_items_embedding#,user_cat_attribute_real,user_cat_attribute#,users_embedding,items_embedding


if __name__ == "__main__":
    class BPR(nn.Module):
        def __init__(self, user_num, item_num, factor_num, user_item_matrix, item_user_matrix, d_i_train, d_j_train):
            super(BPR, self).__init__()
            """
            user_num: number of users;
            item_num: number of items;
            factor_num: number of predictive factors.
            """
            self.user_item_matrix = user_item_matrix
            self.item_user_matrix = item_user_matrix
            self.embed_user = nn.Embedding(user_num, factor_num)
            self.embed_item = nn.Embedding(item_num, factor_num)
            self.mse_loss = nn.MSELoss()

            for i in range(len(d_i_train)):
                d_i_train[i] = [d_i_train[i]]
            for i in range(len(d_j_train)):
                d_j_train[i] = [d_j_train[i]]

            self.d_i_train = torch.FloatTensor(d_i_train)
            self.d_j_train = torch.FloatTensor(d_j_train)
            self.d_i_train = self.d_i_train.expand(-1, factor_num)
            self.d_j_train = self.d_j_train.expand(-1, factor_num)

            nn.init.normal_(self.embed_user.weight, std=0.01)
            nn.init.normal_(self.embed_item.weight, std=0.01)

        def forward(self, user, item, rating):

            users_embedding = self.embed_user.weight
            items_embedding = self.embed_item.weight

            gcn1_users_embedding = (
                    torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(
                self.d_i_train))  # *2. #+ users_embedding
            gcn1_items_embedding = (
                    torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(
                self.d_j_train))  # *2. #+ items_embedding

            gcn2_users_embedding = (
                        torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(
                    self.d_i_train))  # *2. + users_embedding
            gcn2_items_embedding = (
                        torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(
                    self.d_j_train))  # *2. + items_embedding

            gcn3_users_embedding = (
                        torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(
                    self.d_i_train))  # *2. + gcn1_users_embedding
            gcn3_items_embedding = (
                        torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(
                    self.d_j_train))  # *2. + gcn1_items_embedding

            gcn4_users_embedding = (
                        torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(
                    self.d_i_train))#*2. + gcn1_users_embedding
            gcn4_items_embedding = (
                        torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(
                    self.d_j_train))#*2. + gcn1_items_embedding

            gcn_users_embedding = torch.cat(
                (users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding, gcn4_users_embedding),
                -1)  # +gcn4_users_embedding
            gcn_items_embedding = torch.cat(
                (items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding, gcn4_items_embedding),
                -1)  # +gcn4_items_embedding#

            user = F.embedding(user, gcn_users_embedding)
            item = F.embedding(item, gcn_items_embedding)
            # # pdb.set_trace()
            prediction = (user * item).sum(dim=-1)
            loss_part = self.mse_loss(prediction, rating)
            l2_regulization = 0.01 * (user ** 2 + item ** 2).sum(dim=-1)
            # loss=-((rediction_i-prediction_j).sigmoid())**2#self.loss(prediction_i,prediction_j)#.sum()
            # l2_regulization = 0.01*((gcn1_users_embedding**2).sum(dim=-1).mean()+(gcn1_items_embedding**2).sum(dim=-1).mean())

            loss2 = loss_part
            # loss= loss2 + l2_regulization
            loss = loss_part + l2_regulization.mean()
            # pdb.set_trace()
            return prediction, loss, loss2, gcn_users_embedding, gcn_items_embedding

    def readD(set_matrix, num_):
        user_d = []
        for i in range(num_):
            len_set = (1.0 / (len(set_matrix[i]) + 1)) if i in set_matrix else 0
            user_d.append(len_set)
        return user_d

    dataset = "filtered(20)_lastfm_1K"

    user_num = 268
    item_num = 51609
    embs_path = "code_lastfm_1K"

    data_folder = f"{dataset}_reproduce_data"

    factor_num = 64
    batch_size = 2048 * 100

    dataset_base_path = os.path.join(
        r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO',
        data_folder
    )
    training_user_set, training_item_set, _ = np.load(os.path.join(dataset_base_path, 'training_set.npy'), allow_pickle=True)

    training_ratings_dict, train_dict_count = np.load(os.path.join(dataset_base_path, 'training_ratings_dict.npy'), allow_pickle=True)
    testing_ratings_dict, test_dict_count = np.load(os.path.join(dataset_base_path, 'testing_ratings_dict.npy'), allow_pickle=True)
    val_ratings_dict, val_dict_count = np.load(dataset_base_path + '/validation_ratings_dict.npy', allow_pickle=True)

    user_rating_set_all, _, _ = np.load(os.path.join(dataset_base_path, 'user_rating_set_all.npy'), allow_pickle=True)

    u_d = readD(training_user_set, user_num)
    i_d = readD(training_item_set, item_num)

    d_i_train = u_d
    d_j_train = i_d

    # user-item  to user-item matrix and item-user matrix
    def readTrainSparseMatrix(set_matrix, is_user):
        user_items_matrix_i = []
        user_items_matrix_v = []
        if is_user:
            d_i = u_d
            d_j = i_d
            shape = (user_num, item_num)
        else:
            d_i = i_d
            d_j = u_d
            shape = (item_num, user_num)
        for i in set_matrix:
            len_set = len(set_matrix[i])
            for j in set_matrix[i]:
                user_items_matrix_i.append([i, j])
                d_i_j = np.sqrt(d_i[i] * d_j[j])
                # 1/sqrt((d_i+1)(d_j+1))
                user_items_matrix_v.append(d_i_j)  # (1./len_set)
        user_items_matrix_i = torch.LongTensor(user_items_matrix_i)
        user_items_matrix_v = torch.FloatTensor(user_items_matrix_v)
        return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v, shape)


    sparse_u_i = readTrainSparseMatrix(training_user_set, True)
    sparse_i_u = readTrainSparseMatrix(training_item_set, False)

    train_dataset = data_utils.BPRData(
        train_raing_dict=training_ratings_dict, is_training=True, data_set_count=train_dict_count)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True)

    testing_dataset_loss = data_utils.BPRData(
        train_raing_dict=testing_ratings_dict, is_training=False, data_set_count=test_dict_count)
    testing_loader = DataLoader(testing_dataset_loss,
                                     batch_size=test_dict_count, shuffle=False)

    validation_dataset_loss = data_utils.BPRData(
        train_raing_dict=val_ratings_dict, is_training=False, data_set_count=val_dict_count)
    validation_loader = DataLoader(validation_dataset_loss,
                                batch_size=val_dict_count, shuffle=False)

    model = BPR(user_num, item_num, factor_num, sparse_u_i, sparse_i_u, d_i_train, d_j_train)

    optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.005)


    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    user_e = None
    item_e = None
    val_rmse = None
    print('--------training processing-------')
    for epoch in range(100):
        model.train()
        start_time = time.time()
        # pdb.set_trace()
        print('train data of ng_sample is end')
        # elapsed_time = time.time() - start_time
        # print(' time:'+str(round(elapsed_time,1)))
        # start_time = time.time()

        for user, rating, item in train_loader:
            model.zero_grad()
            prediction, loss, loss2, gcn_user_emb, gcn_item_emb = model(user, item, rating)
            loss.backward()
            optimizer_bpr.step()

        model.eval()

        user_e = gcn_user_emb.detach().numpy()
        item_e = gcn_item_emb.detach().numpy()

        str_print_evl = ''  # 'epoch:'+str(epoch)
        pre_all = []
        label_all = []
        for pair_i in testing_ratings_dict:
            u_id, r_v, i_id = testing_ratings_dict[pair_i]
            pre_get = np.sum(user_e[u_id] * item_e[i_id])
            pre_all.append(pre_get)
            label_all.append(r_v)

        r_test = rmse(np.array(pre_all), np.array(label_all))
        res_test = round(np.mean(r_test), 4)
        str_print_evl += "test:\trmse:" + str(res_test)

        print(str_print_evl)

        ### ADDED CODE ###
        ### VALIDATION ###
        str_print_evl = ''  # 'epoch:'+str(epoch)
        pre_all = []
        label_all = []
        for pair_i in val_ratings_dict:
            u_id, r_v, i_id = val_ratings_dict[pair_i]
            pre_get = np.sum(user_e[u_id] * item_e[i_id])
            pre_all.append(pre_get)
            label_all.append(r_v)

        r_val = rmse(np.array(pre_all), np.array(label_all))
        res_val = round(np.mean(r_val), 4)
        str_print_evl += "validation:\trmse:" + str(res_val)

        if val_rmse is not None:
            if r_val < val_rmse:
                val_rmse = r_val
                print("epoch", epoch)
                np.save(
                    os.path.join(r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\code',
                                 embs_path,
                                 r'new_gcn_embs\user_emb_epoch.npy'),
                    model.embed_user.weight.detach().numpy(), allow_pickle=True)
                np.save(
                    os.path.join(r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\FairGO\code',
                                 embs_path,
                                 r'new_gcn_embs\item_emb_epoch.npy'),
                    model.embed_item.weight.detach().numpy(), allow_pickle=True)
        else:
            val_rmse = r_val

        print(str_print_evl)

    test_prediction = []
    label_all = []
    for pair_i in testing_ratings_dict:
        u_id, r_v, i_id = testing_ratings_dict[pair_i]
        pre_get = np.sum(user_e[u_id] * item_e[i_id])
        test_prediction.append(pre_get)

    with open(os.path.join(dataset_base_path, f"predictions_baseline.pkl"), 'wb') as pk:
        pickle.dump(test_prediction, pk)

