#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from utils import *
from Graph_generate.lastfm_data_process import LastFmDataset
from Graph_generate.lastfm_star_data_process import LastFmStarDataset
from Graph_generate.lastfm_graph import LastFmGraph
from Graph_generate.yelp_data_process import YelpDataset
from Graph_generate.yelp_graph import YelpGraph
import torch
import random
import numpy as np
from time import time
import torch.nn as nn
#from data import load_dataset

#from FedRec.server import FedRecServer
#from FedRec.client import FedRecClient
import random
import torch
import torch.nn as nn
import json
import pickle
from utils import * 
import time
from torch.nn.utils.rnn import pad_sequence
import argparse
from FM.FM_model import FactorizationMachine
from FM.FM_feature_evaluate import evaluate_feature
from FM.FM_item_evaluate import evaluate_item


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


#dataset=None
#epochs=200
#batch_size=256
grad_limit=1.0
#clients_limit=0.05
#items_limit=60
#part_percent=1
#attack_lr=0.01
#attack_batch_size=256


# In[4]:


# SCPR agrs
lr=0.02
flr=0.0001
reg=0.001
decay=0.0
qonly=1
bs=64
hs=64
ip=0.01
dr=0.5
optim="Ada"
observe=25
uf=1
rd=0
useremb=1
freeze=0
command=8
seed=0
max_epoch=250
pretrain=1
load_fm_epoch=0
#data_name=LAST_FM


# In[5]:


data_name="LAST_FM_STAR"
dataset = load_dataset(data_name)
ITEM = 'item'
ITEM_FEATURE = 'belong_to'

dataset = load_dataset(data_name)
kg = load_kg(data_name)
hs=64
qonly=1
ip=0.01
dr=0.5
user_length, item_length, feature_length=int(getattr(dataset, 'user').value_len),int(getattr(dataset, 'item').value_len),int(getattr(dataset, 'feature').value_len)
bs = 64
max_epoch=250
lr = 0.02
decay=0.0
flr=0.0001
reg=0.001
observe=25
command=8
uf=1
seed=0
useremb=1
load_fm_epoch=0
PAD_IDX1 = user_length + item_length
PAD_IDX2 = feature_length
filename = 'v1-data-{}-lr-{}-flr-{}-reg-{}-bs-{}-command-{}-uf-{}-seed-{}'.format(
            data_name, lr, flr, reg, bs, command, uf, seed)


# In[6]:


TMP_DIR = {
    LAST_FM: './tmp/last_fm',
    YELP: './tmp/yelp',
    LAST_FM_STAR: './tmp/last_fm_star',
    YELP_STAR: './tmp/yelp_star',
}


# In[7]:


#getattr(dataset, 'item')#從0~7431，總共7432個


# In[8]:


#getattr(dataset, 'feature')#從0~8437，總共8438個


# In[9]:


#getattr(dataset, 'user')#0~1800 總共1801個uid


# # initialize KG

# In[10]:


DatasetDict = {
        LAST_FM: LastFmDataset,
        LAST_FM_STAR: LastFmStarDataset,#here
        YELP: YelpDataset,
        YELP_STAR: YelpDataset
    }
GraphDict = {
        LAST_FM: LastFmGraph,
        LAST_FM_STAR: LastFmGraph,
        YELP: YelpGraph,
        YELP_STAR: YelpGraph
    }


# In[11]:


import os
import json
from easydict import EasyDict as edict
data_dir ="./data/lastfm_star"
data_dir


# In[12]:


class LastFmStarDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir + '/Graph_generate_data'
        self.load_entities()
        self.load_relations()
    def get_relation(self):
        # Entities
        USER = 'user'
        ITEM = 'item'
        FEATURE = 'feature'

        # Relations
        INTERACT = 'interact'
        FRIEND = 'friends'
        LIKE = 'like'
        BELONG_TO = 'belong_to'
        relation_name = [INTERACT, FRIEND, LIKE, BELONG_TO]

        fm_relation = { 
            USER: {#user 这个entity有三个relation，分别relation什么类型的entity
                INTERACT: ITEM,
                FRIEND: USER,
                LIKE: FEATURE,
            },
            ITEM: {
                BELONG_TO: FEATURE,
                INTERACT: USER
            },
            FEATURE: {
                LIKE: USER,
                BELONG_TO: ITEM
            }
        }
        fm_relation_link_entity_type = {#the keys are the relationship types and the values are lists representing the source entity and target entity respectively
            INTERACT: [USER, ITEM],
            FRIEND: [USER, USER],
            LIKE: [USER, FEATURE],
            BELONG_TO: [ITEM, FEATURE]
        }
        return fm_relation, relation_name, fm_relation_link_entity_type

        
    def load_entities(self):
        #load entities with LAST_FM_STAR DATA
        entity_files = edict(
                    user='user_dict.json',
                    item='item_dict.json',
                    feature='original_tag_map.json',
                )
        for entity_name in entity_files:
            with open(os.path.join(self.data_dir, entity_files[entity_name]), encoding='utf-8') as f:
                mydict = json.load(f)
                #user_dict
                #friends refers to who are friends with the user .
                #"like refers to the item that user like"
                #the format:{"0":{"friends":[246, 382, 462, 676, 735],"like":[11,12,13,14,15]},"1":...}

                #if entity_name == 'item':#{'0': {'feature_index': [69, 75]}, '1': {'feature_index': [74, 77, 80, 139, 706]}
                    #print(mydict)

                if entity_name == 'feature':#Each feature is assigned a unique index
                    #print(mydict){'1': 0, '2': 1, '3': 2, '4': 3, '5': 4,...Each feature is assigned a unique index
                    entity_id = list(mydict.values())#[0, 1, 2, 3, 4, 5,... list of index

                else:
                    #uid's attribute {'id': [0, 1, 2, 3, 4, 5, 6, 7,...]'value_len': 1801}
                    #iid"s attribute: {'id': [0, 1, 2, 3, 4,...],'value_len': 7432}
                    #feature's attribute :{'id': [0, 1, 2, 3, 4,...],'value_len': 8438}
                    entity_id = list(map(int, list(mydict.keys())))#[0, 1, 2, 3, 4, 5, ...]
                setattr(self,entity_name, edict(id=entity_id, value_len=max(entity_id) + 1))
                    #print(getattr(self, entity_name))
                print('Load', entity_name, 'of size', len(entity_id))
                print(entity_name, 'of max id is', max(entity_id))
                
    #用entity来建立relation
    def load_relations(self):
        """
        relation: head entity---> tail entity
        --
        """
        LastFm_relations = edict(#根据user_item_train.json 这个file进行配对
            interact=('user_item_train.json', self.user, self.item),  # (filename, head_entity, tail_entity) self.user and self.item is uid and iid attribute
            friends=('user_dict.json', self.user, self.user),
            like=('user_dict.json', self.user, self.feature),
            belong_to=('item_dict.json', self.item, self.feature),
        )
        for name in LastFm_relations:
            #  Save tail_entity
            relation = edict(
                data=[],
            )
            #print(relation)
            #{'data': []}

            knowledge = [list([]) for i in range(LastFm_relations[name][1].value_len)] #create many list as the size of head_entity
            with open(os.path.join(self.data_dir, LastFm_relations[name][0]), encoding='utf-8') as f:
                mydict = json.load(f)
                if name in ['interact']:#interact=('user_item_train.json', self.user, self.item)
                    #print(mydict) {'0': [5780, 5781, 5782, 5783, ...],"1":[154, 155, 156, 157,...],...}#user_item interaction for train data
                    for key, value in mydict.items():
                        head_id = int(key)
                        tail_ids = value
                        knowledge[head_id] = tail_ids#knowledge[0] 用index来代表 head user id，value是[item id]来代表interaction
                elif name in ['friends']:#user 的friends是什么
                    #print(mydict) the data from friends=('user_dict.json' 
                    #from friends like:   {'0': {'friends': [246, 382, 462, ...],"like":[11, 12, 13, 14,...]},'1':...}
                    for key in mydict.keys():
                        head_str = key
                        head_id = int(key)
                        tail_ids = mydict[head_str][name]
                        knowledge[head_id] = tail_ids #index 是user head id ，value是user tail id（user head 的friends）
                elif name in ['like']:#user 喜欢什么feature/attribute
                    #print(mydict) #the data  from  like=('user_dict.json'
                    #from like:   {'0': {'friends': [246, 382, 462, ...],"like":[11, 12, 13, 14,...]},'1':...}
                    for key in mydict.keys():
                        head_str = key
                        head_id = int(key)
                        tail_ids = mydict[head_str][name]
                        knowledge[head_id] = tail_ids #index 是user head id ，value是user tail id（user head like的attribute）
                elif name in ['belong_to']:#该item belongs to 什么feature
                    #print(mydict)#from {'0': {'feature_index': [69, 75]}, '1': {'feature_index': [74, 77, 80, 139, 706]},
                    for key in mydict.keys():
                        head_str = key
                        head_id = int(key)
                        tail_ids = mydict[head_str]['feature_index']
                        knowledge[head_id] = tail_ids#index 是item head id ，value是feature tail id
                    
                relation.data = knowledge
                setattr(self, name, relation) #for example self.interact includes the {'data':[knowledge of interact]}
                tuple_num = 0
                for i in knowledge:
                    tuple_num += len(i)
                print('Load', name, 'of size', tuple_num)
            #now we have 4 attributes contains the relations of each entity
            #print(self.interact)   {'data': [[5780, 5781, 5782, 5783, ....


# In[13]:


data_name="LAST_FM_STAR"
dataset = load_dataset(data_name)


# In[14]:


kg = GraphDict[data_name](dataset)


# In[15]:


class LastFmGraph(object):

    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self._load_knowledge(dataset)
        self._clean()
    def _load_entities(self, dataset):
        print('load entities...')
        num_nodes = 0
        #dataset is a instance of  LastFmStarDataset in the file lastfm_star_data_process.py
        data_relations, _, _ = dataset.get_relation()  # entity_relations, relation_name, link_entity_type
        #print(data_relations)# how head entity relate to tail entity {'user': {'interact': 'item', 'friends': 'user', 'like': 'feature'}, 'item': {'belong_to': 'feature', 'interact': 'user'}, 'feature': {'like': 'user', 'belong_to': 'item'}}
        entity_list = list(data_relations.keys())#user item feature
        for entity in entity_list:
            self.G[entity] = {}# loop 完三次后：{'user': {}, 'item': {}, 'feature': {}}
            entity_size = getattr(dataset, entity).value_len #等于self.user.value_len

            for eid in range(entity_size):
                entity_rela_list = data_relations[entity].keys()
                self.G[entity][eid] = {r: [] for r in entity_rela_list}
            #print(self.G) if entity is user:{'user': {0: {'interact': [], 'friends': [], 'like': []},1:...第一个dic的key是 entity type，value 是entity id，entity id的value则是relation type,relation type 的value是relation entities
            num_nodes += entity_size
            print('load entity:{:s}  : Total {:d} nodes.'.format(entity, entity_size))
        print('ALL total {:d} nodes.'.format(num_nodes))
        print('===============END==============')
        #这里是每个entity的{'user': {0: {'interact': [], 'friends': [], 'like': []},1:...},'item': {0: {'belong_to': [], 'interact': []}, 1:...},'feature': {0: {'like': [], 'belong_to': []}, 1: ...}}
        #print(self.G)
    def _load_knowledge(self, dataset):
        #data_relations_name = relation_name = [INTERACT, FRIEND, LIKE, BELONG_TO]
       #link_entity_type=fm_relation_link_entity_type = {     #the keys are the relationship types and the values are lists representing the source entity and target entity respectively
            #INTERACT: [USER, ITEM],
            #FRIEND: [USER, USER],
            #LIKE: [USER, FEATURE],
           # BELONG_TO: [ITEM, FEATURE]}
        
        _, data_relations_name, link_entity_type = dataset.get_relation()
        for relation in data_relations_name:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for he_id, te_ids in enumerate(data):# head_entity_id 取出他的index，也就是head id, tail_entity_ids
                if len(te_ids) <= 0:
                    continue
                e_head_type = link_entity_type[relation][0] #找出link_entity_type中的head type and tail type :{user item feature}
                e_tail_type = link_entity_type[relation][1]
                for te_id in set(te_ids):
                    self._add_edge(e_head_type, he_id, relation, e_tail_type, te_id)#填入这些slot{'user': {0: {'interact': [], 'friends': [], 'like': []},1:...},'item': {0: {'belong_to': [], 'interact': []}, 1:...},'feature': {0: {'like': [], 'belong_to': []}, 1: ...}}
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))
        print('===============END==============')
        #KG最终形态！！！！！！！！print(self.G)#{'user': {0: {'interact': [6663, 6664, 6666, 6667, 

                
    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)
    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data


# In[16]:


abc=LastFmGraph(dataset)


# In[ ]:


def predict_feature(model,user_output, given_preference, to_test):#user_output = user id
            user_emb = model.user_emb(torch.LongTensor([0]))[..., :-1].detach().numpy() #user_output = user id
            #print("torch.LongTensor([0]) shape:",torch.LongTensor([0]).shape)
            #print("user_emb shape",user_emb.shape)
            #print("given_preference",given_preference)
            gp = model.feature_emb(torch.LongTensor(given_preference))[..., :-1].detach().numpy()
            #print("torch.LongTensor(given_preference) shape",torch.LongTensor(given_preference).shape)
            #print("gp shape",gp.shape)
            emb_weight = model.feature_emb.weight[..., :-1].detach().numpy()
            #print("emb_weight.shape",emb_weight.shape)
            result = list()

            for test_feature in to_test:#剩余的那些没有interact过并且不讨厌的feature进行计算
 
                temp = 0
                temp += np.inner(user_emb, emb_weight[test_feature])#user 跟residual 的相似度
                for i in range(gp.shape[0]):
                    temp += np.inner(gp[i], emb_weight[test_feature])#有过interaction的feature跟residual的相似度
                result.append(temp)

            return result

def topk(y_true, pred, k):
            y_true_ = y_true[:k]
            pred_ = pred[:k]
            if sum(y_true_) == 0:
                return 0
            else:
                return roc_auc_score(y_true, pred) if len(np.unique(y_true)) > 1 else None


def rank_by_batch(uid,kg, pickle_file, iter_, bs, pickle_file_length, model, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):#要用full feature
            '''
            user_output, item_p_output, i_neg2_output, preference_list = list(), list(), list(), list()
            '''
            left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)

            I = pickle_file[0][left:right]
            II = pickle_file[1][left:right]
            III = pickle_file[2][left:right]
            IV = pickle_file[3][left:right]

            i = 0
            index_none = list()

            for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
                if i_neg2_output is None or len(i_neg2_output) == 0:
                    index_none.append(i)
                i += 1

            i = 0
            result_list = list()
            for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):#user_output = user id  每一个user都做一次
                if i in index_none:
                    i += 1
                    continue
                #addition           
                if user_output == uid:
                    #print("user_output",user_output)
                    full_feature = kg.G[ITEM][item_p_output][ITEM_FEATURE]
                    preference_feature = preference_list
                    residual_preference = list(set(full_feature) - set(preference_feature))
                    residual_feature_all = list(set(list(range(feature_length - 1))) - set(full_feature))

                    if len(residual_preference) == 0:
                        continue
                    to_test = residual_feature_all + residual_preference

                    predictions = predict_feature(model, user_output, preference_feature, to_test)#user_output = user id
                    predictions = np.array(predictions)

                    #print("predictions:",predictions)
                    #print("=====================")

                    predictions = predictions.reshape((len(to_test), 1)[0])#剩余的那些没有interact过并且不讨厌的feature的分数（to test）
                    #print("predictions=predictions.reshape((len(to_test), 1)[0])",predictions)
                    y_true = [0] * len(predictions)
                    for i in range(len(residual_preference)):
                        y_true[-(i + 1)] = 1
                    tmp = list(zip(y_true, predictions))
                    #print("tmp1=list(zip(y_true, predictions))",tmp)
                    tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
                    #print("tmp2=sorted(tmp, key=lambda x: x[1], reverse=True)",tmp)
                    y_true, predictions = zip(*tmp)
                    #print("y_true",y_true)
                    #print(" predictions", predictions)
                    icon = []

                    for index, item in enumerate(y_true):
                        if item > 0:
                            icon.append(index)

                    auc = roc_auc_score(y_true, predictions) if len(np.unique(y_true)) > 1 else None
                    #print("uid",uid)
                    #print("epoch",epoch)
                    #print("auc",auc)
                    result_list.append((uid,auc, topk(y_true, predictions, 10), topk(y_true, predictions, 50)
                                        , topk(y_true, predictions, 100), topk(y_true, predictions, 200),
                                        topk(y_true, predictions, 500), len(predictions)))
                    i += 1
                
                else:
                    continue
            #print("result_list",result_list)
                
            return result_list


def evaluate_feature(kg, model,uid,filename, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):#kg, model, epoch(I set 2), filename, PAD_IDX1(9233), PAD_IDX2(8438), user_length(1801), feature_length(8438), data_name(LAST_FM_STAR), ITEM("item"), ITEM_FEATURE("'belong_to'")
            # TODO add const PAD_IDX1, PAD_IDX2, user_length, data_name, ITEM, ITEM_FEATURE
            model.eval()
            model.cpu()
            tt = time.time()
            pickle_file = load_fm_sample(dataset=data_name, mode='valid')#this function is in utils.py,set mode to valid means validation set

            print('Open evaluation pickle file: takes {} seconds, evaluation length: {}'.format(time.time() - tt, len(pickle_file[0])))
            pickle_file_length = len(pickle_file[0])

            start = time.time()
            print('Starting uid :{} '.format(uid))
            bs = 64
            max_iter = int(pickle_file_length / float(bs))
            max_iter = 100

            result = list()
            print('max_iter-----------', max_iter)#100个batch
            for iter_ in range(max_iter):
                if iter_ > 1 and iter_ % 20 == 0:
                    print('--')
                    print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                                float(iter_) * 100 / max_iter))
                result += rank_by_batch(uid,kg, pickle_file, iter_, bs, pickle_file_length, model, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)

            auc_mean = np.mean(np.array([item[1] for item in result if item[1] is not None]))
            auc_median = np.median(np.array([item[1] for item in result if item[1] is not None]))
            print("epoch",epoch)
            print("uid",uid)
            print('feature auc mean: {}'.format(auc_mean), 'feature auc median: {}'.format(auc_median))
            PATH = TMP_DIR[data_name] + '/FM-log-merge/' + filename + '.txt'
            if not os.path.isdir(TMP_DIR[data_name] + '/FM-log-merge/'):
                os.makedirs(TMP_DIR[data_name] + '/FM-log-merge/')

            with open(PATH, 'a') as f:
                with open(PATH, 'a') as f:
                    f.write('validating uid {} on feature prediction\n'.format(uid))
                    auc_mean = np.mean(np.array([item[1] for item in result if item[1] is not None]))
                    auc_median = np.median(np.array([item[1] for item in result if item[1] is not None]))
                    f.write('feature auc mean: {}\n'.format(auc_mean))
                    f.write('feature auc median: {}\n'.format(auc_median))
                    f.flush()
            model.train()
            cuda_(model)

            
#item evaluation
from sklearn.metrics import roc_auc_score
from utils import *
import time
from torch.nn.utils.rnn import pad_sequence


def topk1(y_true, pred, k):
    y_true_ = y_true[:k]
    pred_ = pred[:k]
    #print("Type of y_true:", type(y_true))
    #print("Content of y_true:", y_true)
    #print("y_true length:", len(y_true))
    #print("pred_ len:", len(pred_))
    #print("Content of pred:", pred)

    if sum(y_true_) == 0:
        return 0
    else:
        return roc_auc_score(y_true_, pred_) if len(np.unique(y_true)) > 1 else None


def rank_by_batch1(kg,uid, items_emb,feature_emb,pickle_file, iter_, bs, pickle_file_length, model, rd, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):
    '''
    user_output, item_p_output, i_neg2_output, preference_list = list(), list(), list(), list()
    '''
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    III = pickle_file[2][left:right]
    IV = pickle_file[3][left:right]

    i = 0
    index_none = list()

    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        if i_neg2_output is None or len(i_neg2_output) == 0:
            index_none.append(i)
        i += 1

    i = 0
    result_list = list()
    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        if i in index_none:
            i += 1
            continue
        if user_output == uid:
            total_list = list(i_neg2_output)[: 1000] + [item_p_output]

            user_input = [user_output] * len(total_list)

            pos_list, pos_list2 = list(), list()
            cumu_length = 0
            for instance in zip(user_input, total_list):
                new_list = list()
                new_list.append(instance[0])
                new_list.append(instance[1] + user_length)
                pos_list.append(torch.LongTensor(new_list))
                f = kg.G[ITEM][instance[1]][ITEM_FEATURE]
                if rd == 1:
                    f = list(set(f) - set(preference_list))
                cumu_length += len(f)
                pos_list2.append(torch.LongTensor(f))

            if cumu_length == 0:
                pass


            pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
            prefer_list = torch.LongTensor(preference_list).expand(len(total_list), len(preference_list))

            if cumu_length != 0:
                pos_list2.sort(key=lambda x: -1 * x.shape[0])
                pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)
            else:
                pos_list2 = torch.LongTensor([PAD_IDX2]).expand(pos_list.shape[0], 1)
            
            #for pos,pos2,pre in zip(pos_list, pos_list2, prefer_list):
                #if pos[0] == uid:
            #print("cuda_(pos_list).shape",cuda_(pos_list).unsqueeze(0).shape)
            #print("cuda_(pos_list)",cuda_(pos_list))
            #print("cuda_(pos_list2).unsqueeze(0).shape",cuda_(pos_list2).unsqueeze(0).shape)
            #print("cuda_(prefer_list).unsqueeze(0)",cuda_(prefer_list).unsqueeze(0).shape)
            #print("cuda_(prefer_list)",cuda_(prefer_list))
            #print("len(prefer_list)",len(prefer_list))
            #predictions, _, _ = model(items_emb,feature_emb,cuda_(pos).unsqueeze(0), cuda_(pos2).unsqueeze(0), cuda_(pre).unsqueeze(0))
            #print("cuda_(prefer_list).shape",cuda_(prefer_list).unsqueeze(0).shape)
            #print("cuda_(pos_list).shape",cuda_(pos_list).unsqueeze(0).shape)
            #print("cuda_(pos_list)",cuda_(pos_list).unsqueeze(0))
            predictions_list=[]
            for pos,pos2,pre in zip(pos_list,pos_list2,prefer_list):
                if pos[0] == uid:
                    predictions, _, _ = model(items_emb,feature_emb,cuda_(pos).unsqueeze(0), cuda_(pos2).unsqueeze(0), cuda_(pre).unsqueeze(0))
                    predictions_list.append(predictions)
                else:
                    #print("uid not exist in this batch")
                    continue
            
            predictions = torch.cat(predictions_list, dim=0)
            #print("predictions.shape",predictions.shape)
            #print("predictions",predictions)
            predictions = predictions.detach().cpu().numpy()
            
            mini_gtitems = [item_p_output]
            num_gt = len(mini_gtitems)
            num_neg = len(total_list) - num_gt
            #print("Shape of predictions:", predictions.shape)
                    #print("num_neg:", num_neg)
            #print(predictions)
            predictions = predictions.reshape((num_neg + 1, 1)[0])
            #print("reshape prediction:",predictions.shape)
            y_true = [0] * len(predictions)
            #print("y_true = [0] * len(predictions)",y_true)
            y_true[-1] = 1
            #print("y_true[-1] = 1",y_true)
            tmp = list(zip(y_true, predictions))
            #print("tmp = list(zip(y_true, predictions))",tmp)
            tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
            #print("tmp = sorted(tmp, key=lambda x: x[1], reverse=True)",tmp)
            y_true, predictions = zip(*tmp)
            #print("y_true, predictions = zip(*tmp)",y_true,predictions)
            auc = roc_auc_score(y_true, predictions) if len(np.unique(y_true)) > 1 else None
            #if auc == None
            #print("auc",auc)
            result_list.append((uid,auc, topk1(y_true, predictions, 10), topk1(y_true, predictions, 50)
                                        , topk1(y_true, predictions, 100), topk1(y_true, predictions, 200),
                                        topk1(y_true, predictions, 500), len(predictions)))
            #print("result_list",result_list)
            #print("finish++++++++++++++++++++++++++++++++++++++++++++++++++++")
            i += 1
        else:
            #print("uid not exist in this iteration")
            continue
            
        
    return result_list


def evaluate_item(kg,items_emb,feature_emb ,client_dic, epoch, filename, rd, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE):
    #TODO add const PAD_IDX1, PAD_IDX2, user_length, data_name, ITEM, ITEM_FEATURE
    model.eval()
    tt = time.time()
    pickle_file = load_fm_sample(dataset=data_name, mode='valid')
    print('evaluate data:{}'.format(data_name))
    print('Open evaluation pickle file: takes {} seconds, evaluation length: {}'.format(time.time() - tt, len(pickle_file[0])))
    pickle_file_length = len(pickle_file[0])
    print('ui length:{}'.format(pickle_file_length))

    start = time.time()
    #print('Starting {} epoch'.format(epoch))
    bs = 64
    max_iter = int(pickle_file_length / float(bs))
    # Only do 20 iteration for the sake of time
    max_iter = 20

    result = list()
    for iter_ in range(max_iter):
        if iter_ > 1 and iter_ % 50 == 0:
            print('--')
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                        float(iter_) * 100 / max_iter))
        result += rank_by_batch1(kg,user_id,server.items_emb,server.feature_emb, pickle_file, iter_, bs, pickle_file_length, client_dic, rd, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)
        #print(result)

    #for i in result:
        #print("item",i)
    auc_mean = np.mean(np.array([item[1] for item in result if item[1] is not None]))
    auc_median = np.median(np.array([item[1] for item in result if item[1] is not None]))
    print("epoch",epoch)
    print("uid",user_id)
    print('item auc mean: {}'.format(auc_mean), 'item auc median: {}'.format(auc_median),
          'over num {}'.format(len(result)))
    PATH = TMP_DIR[data_name] + '/FM-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[data_name] + '/FM-log-merge/'):
        os.makedirs(TMP_DIR[data_name] + '/FM-log-merge/')
    with open(PATH, 'a') as f:
        f.write('validating uid {}  on item prediction\n'.format(user_id))
        auc_mean = np.mean(np.array([item[1] for item in result if item[1] is not None]))
        auc_median = np.median(np.array([item[1] for item in result if item[1] is not None]))
        f.write('item auc mean: {}\n'.format(auc_mean))
        f.write('item auc median: {}\n'.format(auc_median))

    model.train()
    cuda_(model)
    
#rec evaluation
import torch
import numpy as np


def evaluate_recall(rating, ground_truth, top_k):
    top_k_items = sorted(rating, key=rating.get, reverse=True)[:top_k]
    #print("top_k_items",top_k_items)
    #print("ground_truth",ground_truth)
    hit = 0
    for i in top_k_items:
        if i in ground_truth:
            hit += 1

    recall = hit / len(ground_truth)
    return recall


def evaluate_ndcg(rating, ground_truth, top_k):
    top_k_items = sorted(rating, key=rating.get, reverse=True)[:top_k]
    #top_k = min(top_k, torch.tensor(rating).shape[0])
    #_, rating_k = torch.topk(torch.tensor(rating), top_k,dim=0)
    #rating_k = rating_k.cpu().tolist()
    dcg, idcg = 0., 0.

    for i, v in enumerate(top_k_items):
        if i < len(ground_truth):
            idcg += (1 / np.log2(2 + i))
        if v in ground_truth:
            dcg += (1 / np.log2(2 + i))

    ndcg = dcg / idcg
    return ndcg



# RecServer

# In[17]:


class FedRecServer(nn.Module):#initialized item embedding
    def __init__(self, emb_size, user_length, item_length, feature_length, qonly, hs, ip, dr):#3706,32
        super().__init__()
        
        self.user_length = user_length#get from global
        self.item_length  = item_length
        self.feature_length = feature_length

        self.hs = hs
        self.ip = ip
        self.dr = dr
        
        #不用
        self.qonly = qonly  # only use quadratic form
        

        # dimensions
        self.emb_size = emb_size
        self.items_emb = nn.Embedding(self.item_length, self.hs+1).to(device)
        nn.init.normal_(self.items_emb.weight, std=0.01)
        
        self.feature_emb = nn.Embedding(self.feature_length+1, self.hs + 1, padding_idx=self.feature_length, sparse=False).to(device)
        self.feature_emb.weight.data.normal_(0,self.ip)

        # _______ set the padding to zero _______
        self.feature_emb.weight.data[feature_length,:] = 0
        

        
    def train_(self,epoch, client,pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_new, index_none, residual_feature, neg_feature):#clients以及他们的index放进去
        
        #self.pos_list = pos_list
        #self.pos_list2 = pos_list2
        #self.neg_list = neg_list
        #self.neg_list2 = neg_list2
        #self.new_neg_list=new_neg_list
        #self.new_neg_list2=new_neg_list2
        #self.preference_list_1=preference_list_1
        #self.preference_list_new = preference_list_new
        #self.index_none=index_none
        #self.residual_feature=residual_feature
        #self.neg_feature=neg_feature

                
   
        batch_items_emb_grad = torch.zeros_like(self.items_emb.weight)#initialize the gradient value of item embedding
        batch_feature_emb_grad = torch.zeros_like(self.feature_emb.weight)
        batch_loss=0.
        batch_loss2=0.
        for i in range(len(pos_list)):
            user_id = pos_list[i][0].item()
            items_emb_grad,feature_emb_grad, loss, loss_2,result_pos,result_neg=client[user_id].train_(self.items_emb,self.feature_emb,pos_list[i], pos_list2[i], neg_list[i], neg_list2[i], new_neg_list[i], new_neg_list2[i],preference_list_1[i], preference_list_new[i], index_none, residual_feature[i],neg_feature[i])
            batch_loss+=loss
            batch_loss2+=loss_2
            
            with torch.no_grad():

                norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
                too_large = norm[:, 0] > grad_limit#1
                items_emb_grad[too_large] /= (norm[too_large] / grad_limit)#gradient clipping step
                batch_items_emb_grad += items_emb_grad



                norm1 = feature_emb_grad.norm(2, dim=-1, keepdim=True)
                too_large1 = norm1[:, 0] > grad_limit#1
                feature_emb_grad[too_large1] /= (norm1[too_large1] / grad_limit)#gradient clipping step
                batch_feature_emb_grad += feature_emb_grad
                
        with torch.no_grad():
            self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-lr)#update 最后的item embedding weight
            self.feature_emb.weight.data.add_(batch_feature_emb_grad, alpha=-lr)

            

        
        
   
        
        
        
        


        #if target_results is not None and target_cnt != 0:
            #return batch_loss, batch_loss2, target_results / target_cnt
        #else:
        return batch_loss, batch_loss2

    


# # FedClients

# In[18]:


class FedRecClient(nn.Module):#为了学好user embedding（但只储存在此处），item embedding的weight可以上传server
    def __init__(self,emb_size, user_length, item_length, feature_length, qonly, hs, ip, dr):
        super(FedRecClient, self).__init__()#super().__init__()
              
        self.items_emb_grad = None
        
        self.model=FactorizationMachine(emb_size=hs, user_length=user_length, item_length=item_length,feature_length=feature_length, qonly=qonly, hs=hs, ip=ip, dr=dr).to(device)

   
    
    def train_(self,items_emb,feature_emb,pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_new, index_none, residual_feature, neg_feature):#要修改compute，在最后要把item embedding 独立出来，然后再return给server
        self.pos_list = pos_list
        self.pos_list2 = pos_list2
        self.neg_list = neg_list
        self.neg_list2 = neg_list2
        self.new_neg_list=new_neg_list
        self.new_neg_list2=new_neg_list2
        self.preference_list_1=preference_list_1
        self.preference_list_new = preference_list_new
        self.index_none=index_none
        self.residual_feature=residual_feature
        self.neg_feature=neg_feature
        self.items_emb=items_emb
        self.feature_emb=feature_emb
        
     
                
        
        
        #reset the gradient
        self.model.zero_grad()

        result_pos, feature_bias_matrix_pos, nonzero_matrix_pos = self.model(self.items_emb,self.feature_emb,self.pos_list.unsqueeze(0),self.pos_list2.unsqueeze(0),self.preference_list_1.unsqueeze(0))  # (bs, 1), (bs, 2, 1), (bs, 2, emb_size)

        result_neg, feature_bias_matrix_neg, nonzero_matrix_neg = self.model(self.items_emb,self.feature_emb,self.neg_list.unsqueeze(0), self.neg_list2.unsqueeze(0), self.preference_list_1.unsqueeze(0))
        if self.model.items_emb.weight.grad is not None:
            self.model.items_emb.weight.grad.zero_()

        diff = (result_pos - result_neg)
        loss = - lsigmoid(diff).sum(dim=0)
        
        if command in [8]:
                # The second type of negative sample
                new_result_neg, new_feature_bias_matrix_neg, new_nonzero_matrix_neg = self.model(self.items_emb,self.feature_emb,self.new_neg_list.unsqueeze(0), self.new_neg_list2.unsqueeze(0),
                                                                                            self.preference_list_new.unsqueeze(0))
                # Reason for this is that, sometimes the sample is missing, so we have to also omit that in result_pos
                T = cuda_(torch.tensor([]))
                for i in range(1):
                    if i in index_none:
                        continue
                    T = torch.cat([T, result_pos[i]], dim=0)

                T = T.view(T.shape[0], -1)
                assert T.shape[0] == new_result_neg.shape[0]
                diff = T - new_result_neg
                if loss is not None:
                    loss += - lsigmoid(diff).sum(dim=0)
                else:
                    loss = - lsigmoid(diff).sum(dim=0)
        
        
        # regularization
        if reg_float != 0:
                if qonly != 1:
                    feature_bias_matrix_pos_ = (feature_bias_matrix_pos ** 2).sum(dim=1)  # (bs, 1)
                    feature_bias_matrix_neg_ = (feature_bias_matrix_neg ** 2).sum(dim=1)  # (bs, 1)
                    nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)  # (bs, 1)
                    nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)  # (bs, 1)
                    new_nonzero_matrix_neg_ = (new_nonzero_matrix_neg_ ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    regular_norm = (
                                feature_bias_matrix_pos_ + feature_bias_matrix_neg_ + nonzero_matrix_pos_ + nonzero_matrix_neg_ + new_nonzero_matrix_neg_)
                    loss += (reg * regular_norm).sum(dim=0)
                else:
                    nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    loss += (reg * nonzero_matrix_pos_).sum(dim=0)
                    loss += (reg * nonzero_matrix_neg_).sum(dim=0)
        
      
        loss1=loss.data
        loss.backward()
   
        
        user_emb_grad = self.model.user_emb.weight.grad
        self.model.user_emb.weight.data.add_(user_emb_grad, alpha=-lr)
        bias_grad = self.model.Bias.grad
        self.model.Bias.data.add_(bias_grad, alpha=-lr)
        
        
        
            
        self.items_emb_grad = self.model.items_emb.weight.grad# stores the gradient computed for the item embeddings into the self.items_emb_grad attribute of the client
        
    
        
        if uf == 1:
                # updating feature embedding
                # we try to optimize
                A = self.model.feature_emb(preference_list_1[0].unsqueeze(0)).unsqueeze(0)[..., :-1]
                #print("A Shape",A.shape)
                #print("=================================================================")
                user_emb = self.model.ui_emb[0][0].unsqueeze(0)[..., :-1].unsqueeze(dim=1).detach()
       
                #print("=================================================================")
                if useremb == 1:
                    A = torch.cat([A, user_emb], dim=1)

                B = self.model.feature_emb(residual_feature.unsqueeze(0))[..., :-1]
                C = self.model.feature_emb(neg_feature.unsqueeze(0))[..., :-1]

                D = torch.matmul(A, B.transpose(2, 1))
                E = torch.matmul(A, C.transpose(2, 1))

                p_vs_residual = D.view(D.shape[0], -1, 1)
                p_vs_neg = E.view(E.shape[0], -1, 1)

                p_vs_residual = p_vs_residual.sum(dim=1)
                p_vs_neg = p_vs_neg.sum(dim=1)
                diff = (p_vs_residual - p_vs_neg)
                temp = - lsigmoid(diff).sum(dim=0)
                loss = temp
                loss_2 = temp.data

                if self.model.feature_emb.weight.grad is not None:
                    self.model.feature_emb.weight.grad.zero_()

           
                loss.backward()

                self.feature_emb_grad = self.model.feature_emb.weight.grad#
        #print(loss)    
        

        return self.items_emb_grad,self.feature_emb_grad,loss1.cpu().item(),loss_2.cpu().item(),result_pos,result_neg

        
   


# # FM model

# In[19]:


import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import time
from utils import *

class FactorizationMachine(nn.Module):#In general, a second-order Factorization Machine models the interaction between every pair of features in the input data.
    """ the quadratic form refers to the second order interaction between features that is captured by the model. Factorization Machines capture both linear relationships (first order) and interactions between pairs of features (second order or quadratic).

Setting qonly to True would mean that the model only considers these second order or quadratic interactions, ignoring the first order linear relationships.

Why might you want to do this? It depends on the specific problem and dataset. In some situations, the interactions between features might be much more important than the individual linear effects. For instance, in recommendation systems, the interaction between a specific user and a specific item (user-item interaction) could be more important than the effect of the user or item individually."""
    def __init__(self,emb_size, user_length, item_length, feature_length, qonly, hs, ip, dr):

        super(FactorizationMachine, self).__init__()#super().__init__()
        

        self.user_length = user_length#get from global
        self.item_length  = item_length
        self.feature_length = feature_length

        self.hs = hs
        self.ip = ip
        self.dr = dr

        self.dropout2 = nn.Dropout(p=self.dr)  # dropout ratio
        self.qonly = qonly  # only use quadratic form
        

        # dimensions
        self.emb_size = emb_size

        # _______ User embedding
        self.user_emb = nn.Embedding(1,hs + 1, sparse=False).to(device)
        

        # _______ Scala Bias _______
        self.Bias = nn.Parameter(torch.randn(1).normal_(0, 0.01), requires_grad=True)

        self.init_weight()

        

    def init_weight(self):
        self.user_emb.weight.data.normal_(0, 0.01)

    '''
        param: a list of user ID and busi ID
        '''
    def forward(self,items_emb,feature_emb,ui_pair, feature_index, preference_index):
        self.items_emb=items_emb
        self.feature_emb=feature_emb
        user_idx = torch.tensor([0]).to(device)#doesnt matter每個client都是獨立的
        self.real_idx=ui_pair[0][1]-user_length
        #print("self.user_emb(user_idx),shape",self.user_emb(user_idx).shape)
        #print("self.items_emb(self.real_idx).unsqueeze(0)",self.items_emb(self.real_idx).unsqueeze(0).shape)
        if len(self.user_emb(user_idx).shape) == len(self.items_emb(self.real_idx).shape):
        
            self.ui_emb = torch.cat([self.user_emb(user_idx), self.items_emb(self.real_idx)],dim=0).to(device).unsqueeze(0)
        else:
            self.ui_emb = torch.cat([self.user_emb(user_idx), self.items_emb(self.real_idx).unsqueeze(0)],dim=0).to(device).unsqueeze(0)
            #print("self.ui_emb = torch.cat shape error")
        
        #ui_emb 要有+1 ，而不是item_emb
        feature_matrix_ui = self.ui_emb#将pos_list做uid 跟iid的embedding
        #print("feature_matrix_ui.shape",feature_matrix_ui.shape)
        
        nonzero_matrix_ui = feature_matrix_ui[..., :-1]#This line is taking all dimensions except for the last one. 
        #print("nonzero_matrix_ui.shape",nonzero_matrix_ui.shape)
        feature_bias_matrix_ui = feature_matrix_ui[..., -1:]#This line is taking just the last dimension. The -1: in indexing means "get the last dimension only".the bias term


        feature_matrix_preference = self.feature_emb(preference_index)#get preference embedding
        #print("feature_matrix_preference.shape",feature_matrix_preference.shape)
        # _______ dropout has been done already (when data was passed in) _______
        nonzero_matrix_preference = feature_matrix_preference[..., :-1]  # (bs, 2, emb_size)
        feature_bias_matrix_preference = feature_matrix_preference[..., -1:]  # (bs, 2, 1)
 
        if len(nonzero_matrix_ui.shape) == len(nonzero_matrix_preference.shape):
            nonzero_matrix = torch.cat((nonzero_matrix_ui, nonzero_matrix_preference), dim=1)
        else:
            print("nonzero_matrix,cat error")
            sys.exit()

        if len(feature_bias_matrix_ui.shape) == len(feature_bias_matrix_preference.shape):
            feature_bias_matrix = torch.cat((feature_bias_matrix_ui, feature_bias_matrix_preference), dim=1)
        else:
            print("feature_bias_matrix,cat error")
            sys.exit()

        # _______ make a clone _______
        nonzero_matrix_clone = nonzero_matrix.clone()#uid and idd and preference embedding
        feature_bias_matrix_clone = feature_bias_matrix.clone()#bias matrix
        
        #Second-order term of FM:

        # _________ sum_square part _____________
        summed_features_embedding_squared = nonzero_matrix.sum(dim=1, keepdim=True) ** 2  # (bs, 1, emb_size) it's the first half of the pairwise interaction term in the FM formula.

        # _________ square_sum part _____________
        squared_sum_features_embedding = (nonzero_matrix * nonzero_matrix).sum(dim=1, keepdim=True)  # (bs, 1, emb_size)

        # ________ FM __________
        FM = 0.5 * (summed_features_embedding_squared - squared_sum_features_embedding)  # (bs, 1, emb_size)

        # Optional: remove the inter-group interaction
        # ***---***
        #算出每个user的preference的FM
        new_non_zero_2 = nonzero_matrix_preference
        summed_features_embedding_squared_new_2 = new_non_zero_2.sum(dim=1, keepdim=True) ** 2
        squared_sum_features_embedding_new_2 = (new_non_zero_2 * new_non_zero_2).sum(dim=1, keepdim=True)
        newFM_2 = 0.5 * (summed_features_embedding_squared_new_2 - squared_sum_features_embedding_new_2)
        FM = (FM - newFM_2)
        #The intention of this is to make the model focus more on the interactions between different groups (like user-item, user-preference, item-preference)
        #rather than interactions within the same group (like preference-preference). """
        # ***---***

        FM = self.dropout2(FM)  # (bs, 1, emb_size)

        Bilinear = FM.sum(dim=2, keepdim=False)  # (bs, 1)
        result = Bilinear + self.Bias  # (bs, 1)
        #result is the predicted score
        return result, feature_bias_matrix_clone, nonzero_matrix_clone
    # end def


# # Fedmain()

# In[20]:


def translate_pickle_to_data(dataset, kg, pickle_file, iter_, bs, pickle_file_length, uf):
    '''
    user_pickle = pickle_file[0]
    item_p_pickle = pickle_file[1]
    i_neg1_pickle = pickle_file[2]
    i_neg2_pickle = pickle_file[3]
    preference_pickle = pickle_file[4]
    '''
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)
    # user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle = zip(*pickle_file[left:right])

    pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_2 = [], [], [], [], [], [], [], []

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    III = pickle_file[2][left:right]
    IV = pickle_file[3][left:right]
    V = pickle_file[4][left:right]

    residual_feature, neg_feature = None, None

    if uf == 1:
        feature_range = np.arange(feature_length).tolist()
        residual_feature, neg_feature = [], []
        for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
            gt_feature = kg.G[ITEM][item_p_pickle][ITEM_FEATURE]
            this_residual_feature = list(set(gt_feature) - set(preference_pickle))
            remain_feature = list(set(feature_range) - set(gt_feature))
            this_neg_feature = np.random.choice(remain_feature, len(this_residual_feature))
            residual_feature.append(torch.LongTensor(this_residual_feature))
            neg_feature.append(torch.LongTensor(this_neg_feature))
        residual_feature = pad_sequence(residual_feature, batch_first=True, padding_value=PAD_IDX2)
        neg_feature = pad_sequence(neg_feature, batch_first=True, padding_value=PAD_IDX2)

    i = 0
    index_none = list()
    for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
        pos_list.append(torch.LongTensor([user_pickle, item_p_pickle + user_length]))
        f = kg.G[ITEM][item_p_pickle][ITEM_FEATURE]
        pos_list2.append(torch.LongTensor(f))
        neg_list.append(torch.LongTensor([user_pickle, i_neg1_pickle + user_length]))
        f = kg.G[ITEM][i_neg1_pickle][ITEM_FEATURE]
        neg_list2.append(torch.LongTensor(f))

        preference_list_1.append(torch.LongTensor(preference_pickle))
        if i_neg2_pickle is None:
            index_none.append(i)
        i += 1

    i = 0
    for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
        if i in index_none:
            i += 1
            continue
        new_neg_list.append(torch.LongTensor([user_pickle, i_neg2_pickle + user_length]))
        f = kg.G[ITEM][i_neg2_pickle][ITEM_FEATURE]
        new_neg_list2.append(torch.LongTensor(f))
        preference_list_2.append(torch.LongTensor(preference_pickle))
        i += 1


    pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
    pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)
    neg_list = pad_sequence(neg_list, batch_first=True, padding_value=PAD_IDX1)
    neg_list2 = pad_sequence(neg_list2, batch_first=True, padding_value=PAD_IDX2)
    new_neg_list = pad_sequence(new_neg_list, batch_first=True, padding_value=PAD_IDX1)
    new_neg_list2 = pad_sequence(new_neg_list2, batch_first=True, padding_value=PAD_IDX2)
    preference_list_1 = pad_sequence(preference_list_1, batch_first=True, padding_value=PAD_IDX2)
    preference_list_2 = pad_sequence(preference_list_2, batch_first=True, padding_value=PAD_IDX2)

    if uf != 0:
        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2), cuda_(new_neg_list), cuda_(
            new_neg_list2), cuda_(preference_list_1), cuda_(preference_list_2), index_none, cuda_(residual_feature), cuda_(neg_feature)
    else:
        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2), cuda_(new_neg_list), cuda_(
            new_neg_list2), cuda_(preference_list_1), cuda_(preference_list_2), index_none, residual_feature, neg_feature
    
    

        
        


# # featrue evaluation
def performance_eval_(data_name,client,items_emb,feature_emb):
    #print("output format: ({ER},{NDCG})")
    #get validation data
    #data_name="LAST_FM_STAR"
    pickle_file = load_fm_sample(dataset=data_name, mode='valid')
    I = pickle_file[0]
    II = pickle_file[1]
    III = pickle_file[2]
    IV = pickle_file[3]


    def get_user_details(user_id, users, items, preferences):
        user_items = []
        user_preferences = []

        # Iterate through the users list to find matching user_ids
        for idx, uid in enumerate(users):
            if uid == user_id:
                user_items.append(items[idx])
                user_preferences.append(preferences[idx])
        return user_items, user_preferences


    #全部user都需要统一rating的items
    unique_items = []
    unique_p = []

    for it, p in zip(II, IV):
        if it not in unique_items:
            unique_items.append(it)
            unique_p.append(p)

    target_cnt = 0
    target_results = np.array([0., 0.])

    with torch.no_grad():
        for uid in list(set(I)):
            #ensure the uid now in the client_dic
            #if uid in self.pos_list[:,0]:

            predictions_dic={}
            for  i,p in zip(unique_items,unique_p):

                pos=[]
                pos.append(torch.LongTensor([uid, i + user_length]))        
                pos = pad_sequence(pos, batch_first=True, padding_value=PAD_IDX1)

                pre=[]
                pre.append(torch.LongTensor(p))
                pre = pad_sequence(pre, batch_first=True, padding_value=PAD_IDX2)

                pos2=[]
                f = kg.G[ITEM][i][ITEM_FEATURE]
                #if rd == 1:
                    #f = list(set(f) - set(preference_list))
                pos2.append(torch.LongTensor(f))
                pos2 = pad_sequence(pos2, batch_first=True, padding_value=PAD_IDX2)


                predictions, _, _ = client[uid].model(items_emb,feature_emb,cuda_(pos), cuda_(pos2), cuda_(pre))
                predictions_dic[i]=predictions

            ground_truth,_ = get_user_details(uid, I, II, IV)

            # Evaluate recall and NDCG
            er = evaluate_recall(predictions_dic, ground_truth, len(ground_truth))

            ndcg = evaluate_ndcg(predictions_dic, ground_truth,len(ground_truth))
            target_result = np.array([er, ndcg])
            target_cnt += 1
            target_results += target_result
    print("(%.4f, %.4f) average ER for a batch of clients." % (target_results[0]/target_cnt, target_results[1]/target_cnt))

    PATH = TMP_DIR[data_name] + '/FED-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[data_name] + '/FED-log-merge/'):
        os.makedirs(TMP_DIR[data_name] + '/FED-log-merge/')
    with open(PATH, 'a') as f:
        f.write("/n(%.4f, %.4f) average ER for a batch of clients./n" % (target_results[0]/target_cnt, target_results[1]/target_cnt))
        f.flush()


    return target_results/target_cnt



def save_client_model(dataset, model, filename, epoch,uid):
    model_file = TMP_DIR[dataset] + '/FED-model-merge/' + filename + '-epoch-{}.pt'.format(epoch) +'-user:{}-model'.format(uid)
    if not os.path.isdir(TMP_DIR[dataset] + '/FED-model-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/FED-model-merge/')
    print("client model dic",model.state_dict())
    torch.save(model.state_dict(), model_file)
    print('Model saved at {}'.format(model_file))
    
def save_server_model(dataset, model, filename, epoch):
    model_file = TMP_DIR[dataset] + '/FED-model-merge/' + filename + '-epoch-{}.pt'.format(epoch)+"-server_model"
    if not os.path.isdir(TMP_DIR[dataset] + '/FED-model-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/FED-model-merge/')
    torch.save(model.state_dict(), model_file)
    print('Model saved at {}'.format(model_file))
    
def save_s_embed(dataset, embeds, epoch):
    path = TMP_DIR[dataset] + '/FED-model-embeds/' + 'embeds-epoch-{}.pkl'.format(epoch)+"-server_emb"
    if not os.path.isdir(TMP_DIR[dataset] + '/FED-model-embeds/'):
        os.makedirs(TMP_DIR[dataset] + '/FED-model-embeds/')
    with open(path, 'wb') as f:
        pickle.dump(embeds, f)
        print('Embedding saved successfully!')
        
def save_c_embed(dataset, embeds, epoch,uid):
    path = TMP_DIR[dataset] + '/FED-model-embeds/' + 'embeds-epoch-{}.pkl'.format(epoch)+'-user:{}-emb'.format(uid)
    if not os.path.isdir(TMP_DIR[dataset] + '/FED-model-embeds/'):
        os.makedirs(TMP_DIR[dataset] + '/FED-model-embeds/')
    with open(path, 'wb') as f:
        pickle.dump(embeds, f)
        print('Embedding saved successfully!')
        
def load_server_model(dataset, model, filename, epoch):
    model_file = TMP_DIR[dataset] + '/FED-model-merge/' + filename + '-epoch-{}.pt'.format(epoch)+"-server_model"
    model_dict = torch.load(model_file)
    print('Model load at {}'.format(model_file))
    return model_dict

def load_client_model(dataset, model, filename, epoch,uid):
    model_file = TMP_DIR[dataset] + '/FED-model-merge/' + filename + '-epoch-{}.pt'.format(epoch)+'-user:{}-model'.format(uid)
    model_dict = torch.load(model_file)
    #print(model_dict)
    print('Model load at {}'.format(model_file))
    return model_dict
        
def save_server_embedding(model, filename, epoch):
    #model_dict = load_server_model(data_name, model, filename, epoch)
    #model.load_state_dict(model_dict)
    #print('Model loaded successfully!')
    items_emb = model.items_emb.weight.data.cpu().numpy()
    feature_emb = model.feature_emb.weight.data.cpu().numpy()
    print('items_size:{}'.format(items_emb.shape[0]))
    print('fea_size:{}'.format(feature_emb.shape[0]))
    embeds = {
        'items_emb': items_emb,
        'feature_emb': feature_emb
    }
    save_s_embed(data_name, embeds, epoch)
    
def save_client_embedding(model, filename, epoch,uid):
    #model_dict = load_client_model(data_name, model, filename, epoch,uid)
    #model.load_state_dict(model_dict)
    #print('Model loaded successfully!')
    user_emb = model.user_emb.weight.data.cpu().numpy()
    
    print('user_size:{}'.format(user_emb.shape[0]))
    embeds = {
        'user_emb': user_emb,
    }
    save_c_embed(data_name, embeds, epoch,uid)

def load_client_embed(dataset,model, epoch,uid):
    path = TMP_DIR[dataset] + '/FED-model-embeds/' + 'embeds-epoch-{}.pkl'.format(epoch)+'-user:{}-emb'.format(uid)
    with open(path, 'rb') as f:
        embeds = pickle.load(f)
        print('FM Epoch：{} client Embedding load successfully!'.format(epoch))
        #return embeds    
        if 'user_emb' in embeds:
            user_emb_tensor = torch.tensor(embeds['user_emb'])
            if model.user_emb.weight.shape == user_emb_tensor.shape:
                model.user_emb.weight.data = user_emb_tensor.to(model.user_emb.weight.device)
            else:
                print("Error: Loaded user embedding size does not match model's user embedding size.")
    
        print('Embeddings loaded into model successfully!')
        return model
    
def load_server_embed(dataset,model, epoch):
    path = TMP_DIR[dataset] + '/FED-model-embeds/' + 'embeds-epoch-{}.pkl'.format(epoch)+"-server_emb"
    with open(path, 'rb') as f:
        embeds = pickle.load(f)
        print('FM Epoch：{} server Embedding load successfully!'.format(epoch))
        #return embeds
        #print(embeds)
        if 'items_emb' and 'feature_emb' in embeds:
            #print(embeds['items_emb'])
            items_emb_tensor = torch.tensor(embeds['items_emb'])
            feature_emb_tensor = torch.tensor(embeds['feature_emb'])
            if model.items_emb.weight.shape == items_emb_tensor.shape and model.feature_emb.weight.shape == feature_emb_tensor.shape :
                model.feature_emb.weight.data = feature_emb_tensor.to(model.feature_emb.weight.device)
                model.items_emb.weight.data = items_emb_tensor.to(model.items_emb.weight.device)
            else:
                print("Error: Loaded items and feature embedding size does not match model's user embedding size.")

        print('Embeddings loaded into model successfully!')
        return model
    

    
    
pretrain = 0
# In[ ]:
file_name = 'v1-data-{}-lr-{}-flr-{}-reg-{}-bs-{}-command-{}-uf-{}-seed-{}'.format(
            data_name, lr, flr, reg, bs, command, uf, seed)


client_dic={}
pickle_file = load_fm_sample(dataset=data_name, mode='train', epoch=1)
unique_uid=set(pickle_file[0])

if pretrain == 0:  # means no pretrain
    server = FedRecServer(hs, user_length, item_length, feature_length, qonly, hs, ip, dr).to(device)
    for i in unique_uid:
        user_id = i  # Extract user id and convert tensor to python scalar
        if user_id not in client_dic:
            client_dic[user_id] = FedRecClient(hs, user_length, item_length, feature_length, qonly, hs, ip, dr).to(device)
        else:
            continue
            
elif pretrain == 1:
    server = FedRecServer(hs, user_length, item_length, feature_length, qonly, hs, ip, dr)  
    server = load_server_embed(data_name, server,0)
    cuda_(server)
    for i in unique_uid:
        user_id = i
        if user_id not in client_dic:
            client_dic[user_id] = FedRecClient(hs, user_length, item_length, feature_length, qonly, hs, ip, dr)
            client_dic[user_id].model = load_client_embed(data_name, client_dic[user_id].model, 0,user_id)
            cuda_(client_dic[user_id].model)





lsigmoid = nn.LogSigmoid()
reg_float = float(reg)
for epoch in range(0, max_epoch+1):
        target_result = np.array([0., 0.])
       
        # _______ Do the evaluation _______
        #if epoch % observe == 0 and epoch > -1:#0,25,50...
        #if epoch %10 ==0 and epoch>1:
        #if epoch ==1:
            #print('Evaluating on feature similarity')      #要改这里
            #evaluate_feature(kg, model, epoch, filename, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)#call the function from FM_feature_evaluate.py
            #print('Evaluating on item similarity')
            #auc=evaluate_item(kg, server.items_emb,server.feature_emb,client_dic, epoch, filename, 0, PAD_IDX1, PAD_IDX2, user_length, feature_length, data_name, ITEM, ITEM_FEATURE)
            #avg_auc+=auc
            #print("avg_auc_per_10epoch:",avg_auc)
            
        tt = time.time()#record the operation time
        pickle_file = load_fm_sample(dataset=data_name, mode='train', epoch=epoch % 50)# pick up the train set from utils.py  pickle file 里面有五个list

        print('Open pickle file: train_fm_data takes {} seconds'.format(time.time() - tt))
        pickle_file_length = len(pickle_file[0])
        
        mix = list(zip(pickle_file[0], pickle_file[1], pickle_file[2], pickle_file[3], pickle_file[4]))
        random.shuffle(mix)
        I, II, III, IV, V = zip(*mix)
        new_pk_file = [I, II, III, IV, V]
        start = time.time()
        print('Starting {} epoch'.format(epoch))
        epoch_loss = 0
        epoch_loss_2 = 0
        max_iter = int(pickle_file_length / float(bs))
        

        
        #分batch
        for iter_ in range(max_iter):#逐批逐批的将batch取出来
            if iter_ > 1 and iter_ % 1000 == 0:
                print('--')
                print('Takes {} seconds to finish {}% of this epoch'.format(str(time.time() - start),
                                                                            float(iter_) * 100 / max_iter))
                print('loss is: {}'.format(float(epoch_loss) / (bs * iter_)))
                print('iter_:{} Bias grad norm: {}, Static grad norm: {}, Preference grad norm: {}'.format(iter_, torch.norm(model.Bias.grad), torch.norm(model.ui_emb.weight.grad), torch.norm(model.feature_emb.weight.grad)))

            pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_new, index_none, residual_feature, neg_feature  = translate_pickle_to_data(dataset, kg, new_pk_file, iter_, bs, pickle_file_length, uf)
          
            batch_loss=0
            batch_loss_2=0
            
            #user_id = pos_list[i][0].item()
            batch_loss,batch_loss_2=server.train_(epoch,client_dic,pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_new, index_none, residual_feature, neg_feature)
                
            epoch_loss += batch_loss
            epoch_loss_2 +=batch_loss_2
            
            #if target_results is not None:
                #target_result+=target_results
            

            
                #break#!!!!!!!!
            #break#!!!!!!!!!!!!!
 
     
        print('epoch loss: {}'.format(epoch_loss / pickle_file_length))
        print('epoch loss 2: {}'.format(epoch_loss_2 / pickle_file_length))
        
        PATH = TMP_DIR[data_name] + '/FED-log-merge/' + filename + '.txt'
        if not os.path.isdir(TMP_DIR[data_name] + '/FED-log-merge/'):
            os.makedirs(TMP_DIR[data_name] + '/FED-log-merge/')
        with open(PATH, 'a') as f:

                f.write('/n epoch loss: {}/n'.format(epoch_loss / pickle_file_length))
                f.write('/n epoch loss 2: {}/n'.format(epoch_loss_2 / pickle_file_length))
                f.flush()
        
        if epoch % 5 == 0 and epoch > 0:
        #print(' Epoch：{} ; start saving clients model.'.format(epoch))
        #for uid in client_dic:
            #save_client_model(dataset=data_name, model=client_dic[uid].model, filename=filename, epoch=epoch,uid=uid)
        #print(' Epoch：{} ; start saving server model.'.format(epoch))
        #save_server_model(dataset=data_name, model=server, filename=filename, epoch=epoch)

            print(' Epoch：{} ; start saving server model item and feature embedding.'.format(epoch))
            save_server_embedding(server, filename, epoch=epoch)

            for uid in client_dic:
                print(' Epoch：{} ; start saving client user embedding.'.format(epoch))
                save_client_embedding(client_dic[uid].model, filename, epoch=epoch,uid = uid)
            
            
        
        #if epoch % 10 == 0 and epoch > 1:
        #target_result=performance_eval_(data_name,client_dic,server.items_emb,server.feature_emb)
        #if target_result[0] != 0 and target_result[1] != 0:
            #print("output format: ({ER},{NDCG})")
            #print("(%.4f, %.4f) average ER on a epoch." % (target_result[0], target_result[1]))




