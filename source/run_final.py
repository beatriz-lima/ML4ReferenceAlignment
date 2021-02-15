# All experiements

import os
import random
import itertools
import sys
import pickle

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

import utils as u

np.random.seed(42)
random.seed(42)

lb_measures = [
    'measure_agm',
    'measure_aml',
    'measure_dome',
    'measure_fcamap',
    'measure_logmap',
    'measure_logmapbio',
    'measure_logmaplt',
    'measure_pomap++',
    'measure_wiktionary'
]
cf_measures = [
    'measure_alin',
    'measure_aml',
    'measure_dome',
    'measure_lily',
    'measure_logmap',
    'measure_logmaplt',
    'measure_ontmat1',
    'measure_sanom',
    'measure_wiktionary'
]
cf_ontologies = [
    'cmt',
    'conference',
    'confof',
    'edas',
    'ekaw',
    'iasted',
    'sigkdd',
]
conf_lb_measures = [
    'measure_aml',
    'measure_dome',
    'measure_logmap',
    'measure_logmaplt',
    'measure_wiktionary'
    ]

## LOAD LB
lb_data_path = os.path.join("data", "largebio-results-2019")
lb_ref_path =os.path.join("data", "oaei2019_umls_flagged_reference")

df_lb1, df_lb1_ref = u.read_rdf(ont1='fma', 
                                ont2='nci', 
                                measures=lb_measures, 
                                track='largebio', 
                                data_path=lb_data_path, 
                                ref_path= os.path.join(lb_ref_path, "oaei_fma_nci_mappings_with_flagged_repairs.rdf"),
                                data_processed_path= os.path.join(lb_data_path,"data_lb1.csv"), 
                                ref_processed_path= os.path.join(lb_data_path,"ref_lb1.csv.csv"))

df_lb2, df_lb2_ref = u.read_rdf(ont1='fma', 
                                ont2='snomed', 
                                measures=lb_measures, 
                                track='largebio', 
                                data_path=lb_data_path, 
                                ref_path=os.path.join(lb_ref_path, "oaei_fma_snomed_mappings_with_flagged_repairs.rdf"),
                                data_processed_path= os.path.join(lb_data_path,"data_lb2.csv"), 
                                ref_processed_path= os.path.join(lb_data_path,"ref_lb2.csv"))

df_lb3, df_lb3_ref = u.read_rdf(ont1='snomed', 
                                ont2='nci', 
                                measures=lb_measures, 
                                track='largebio', 
                                data_path=lb_data_path, 
                                ref_path=os.path.join(lb_ref_path, "oaei_snomed_nci_UMLS_mappings_with_flagged_repairs.rdf"),
                                data_processed_path= os.path.join(lb_data_path,"data_lb3.csv"), 
                                ref_processed_path= os.path.join(lb_data_path,"ref_lb3.csv"))

## LOAD ANATOMY
df_an_path = os.path.join("data", "df_an.csv")
df_an_ref_path = os.path.join("data", "df_an_ref.csv")

if not os.path.isfile(df_an_path) or not os.path.isfile(df_an_ref_path):

    #load reference
    anatomy_reference_path = os.path.join(
        "data",
        "anatomy-2019-results",
        "reference.rdf"
    )
    df_an_ref = u.extract_mappings(anatomy_reference_path)
    df_an_ref.shape

    #load ontology matching algorithms outputs
    an_res_dir = os.path.join("data", "anatomy-2019")
    an_agm = u.extract_mappings(os.path.join(an_res_dir, "AGM.rdf"))
    an_aml = u.extract_mappings(os.path.join(an_res_dir, "AML.rdf"))
    an_dome = u.extract_mappings(os.path.join(an_res_dir, "DOME.rdf"))
    an_fcamap = u.extract_mappings(os.path.join(an_res_dir, "FCAMap-KG.rdf"))
    an_logmap = u.extract_mappings(os.path.join(an_res_dir, "LogMap.rdf"))
    an_logmapbio = u.extract_mappings(os.path.join(an_res_dir, "LogMapBio.rdf"))
    an_logmaplt = u.extract_mappings(os.path.join(an_res_dir, "LogMapLt.rdf"))
    an_pomappp = u.extract_mappings(os.path.join(an_res_dir, "POMAP++.rdf"))
    an_wiktionary = u.extract_mappings(os.path.join(an_res_dir, "Wiktionary.rdf"))

    an_tool_mappings = {
        "agm": an_agm,
        "aml": an_aml,
        "dome": an_dome,
        "fcamap": an_fcamap,
        "logmap": an_logmap,
        "logmapbio": an_logmapbio,
        "logmaplt": an_logmaplt,
        "pomap++": an_pomappp,
        "wiktionary": an_wiktionary,
    }

    #merge them all in a dataframe
    df_an = u.merge_mappings(an_tool_mappings)
    df_an.shape

    #export data
    df_an_ref.to_csv(df_an_ref_path, index=False)

#read files
else:
    print('read preprocessed files')
    df_an = pd.read_csv(df_an_path)
    df_an_ref = pd.read_csv(df_an_ref_path)

# Merge datasets
df_an = df_an.merge(df_an_ref, how='outer',on=["entity1", "entity2"])
df_an.rename(columns={"measure": "label"}, inplace=True)
df_lb1 = df_lb1.merge(df_lb1_ref, how='outer',on=["entity1", "entity2"])
df_lb1.rename(columns={"measure": "label"}, inplace=True)
df_lb2 = df_lb2.merge(df_lb2_ref, how='outer',on=["entity1", "entity2"])
df_lb2.rename(columns={"measure": "label"}, inplace=True)
df_lb3 = df_lb3.merge(df_lb3_ref, how='outer',on=["entity1", "entity2"])
df_lb3.rename(columns={"measure": "label"}, inplace=True)

# Missing values
df_lb1.fillna(0)
df_lb2.fillna(0)
df_lb3.fillna(0)
df_an.fillna(0)

#binary data
Xy_bins_lb1 = u.bin_features(df_lb1.copy(), 0,1)
Xy_bins_lb2 = u.bin_features(df_lb2.copy(), 0,1)
Xy_bins_lb3 = u.bin_features(df_lb3.copy(), 0,1)
Xy_bins_an = u.bin_features(df_an.copy(), 0,1)

#prepare dfs
X_bins_lb1 = Xy_bins_lb1.copy()
X_bins_lb2 = Xy_bins_lb2.copy()
X_bins_lb3 = Xy_bins_lb3.copy()
X_bins_an = Xy_bins_an.copy()

X_bins_an.drop('label', axis=1)
X_bins_lb1.drop('label', axis=1)
X_bins_lb2.drop('label', axis=1)
X_bins_lb3.drop('label', axis=1)

#READ CONFERENCE

conference_data_processed_path = 'data/df_conference.csv'
res_dir = os.path.join('data','conference-data')

if not os.path.isfile(conference_data_processed_path):
    dfs_data, dfs_refs = [],[]
    for ont1, ont2 in itertools.combinations(cf_ontologies,2):
        ref_path = os.path.join(
            "data",
            "conference-ref-data",
            "{}-{}.rdf".format(ont1,ont2),
        )
        df_data, df_ref = u.load_rdf('conference', res_dir,ref_path,ont1,ont2)
        df_data["ontologies"] = f"{ont1}-{ont2}"
        dfs_data.append(df_data)
        dfs_refs.append(df_ref)

    df_conf = pd.concat(dfs_data, ignore_index = True)
    df_ref = pd.concat(dfs_refs, ignore_index = True)
    df_conf = df_conf.merge(df_ref, how='outer',on=["entity1", "entity2"])
    df_conf.rename(columns={"measure": "label"}, inplace=True)
    df_conf.to_csv(conference_data_processed_path, index = False)
else:
    df_conf = pd.read_csv(conference_data_processed_path)

#fill missing values with 0
df_conf = df_conf.fillna(0)

#binary features
Xy_cf_bins = u.bin_features(df_conf.copy().drop('ontologies', axis=1),0,1)
X_cf, y_cf = Xy_cf_bins[cf_measures], Xy_cf_bins['label']

#DEFINE CLASSIFIERS & ARGS
classifiers = [
    RandomForestClassifier,
    KNeighborsClassifier,
    DecisionTreeClassifier,
    MLPClassifier,
    GaussianNB,
    GradientBoostingClassifier,
    LogisticRegression,
    AdaBoostClassifier
]

classifier_kwargs = [
    {"param_grid": {'n_estimators': list(range(50,250,50)) , 'criterion': ['gini', 'entropy']}},
    {"param_grid": {'n_neighbors': list(range(1,7)), 'p': [1,2]}},
    {"param_grid": {'criterion': ['gini', 'entropy']}},
    {"param_grid": {'hidden_layer_sizes':[(10,), (40,), (100,), (10, 10), (40, 40), (100, 100)], 'learning_rate_init': [0.01, 0.05, 0.1,]}},
    {"param_grid": {}},
    {"param_grid": {'n_estimators':list(range(50,250,50)),'learning_rate':[0.01, 0.1, 0.2]}},
    {"param_grid": {'C':[0.1,0.5,1,10], 'tol': [1e-2,1e-3,1e-4]}},
    {"param_grid": {'base_estimator': [LogisticRegression()], 'n_estimators': [50,100,150,200]}}
]


def get_conference_data(measures,ont_comb_train,df_data):
    lst_ont_comb = []
    for ont1, ont2 in itertools.combinations(cf_ontologies,2): lst_ont_comb.append(f"{ont1}-{ont2}")
    random.shuffle(lst_ont_comb)

    train_combs = np.array(lst_ont_comb)[:ont_comb_train]
    df_train = df_data[np.isin(df_data['ontologies'].values, train_combs)]
    df_test =  df_data[np.isin(df_data['ontologies'].values, np.array(lst_ont_comb)[-(len(lst_ont_comb)-ont_comb_train):])]

    #get just needed columns: measures and label
    columns_take = list(measures).copy()
    columns_take.append('label')

    df_train = df_train[columns_take]
    df_test = df_test[columns_take]

    return ([df_train], [df_test], train_combs)

#get just needed columns: measures and label
columns_inter = list(conf_lb_measures).copy()
columns_inter.append('label')

cross_tuples=[
    ([Xy_bins_an], [Xy_bins_lb1, Xy_bins_lb2, Xy_bins_lb3], "anatomy-lb"),
    ([Xy_bins_lb1, Xy_bins_lb2, Xy_bins_lb3], [Xy_bins_an], "lb-anatomy"),
]

#conference train in x test in y (x3)
ont_comb_train = 18
for _ in range(3): 
    cross_tuples.append(get_conference_data(cf_measures, ont_comb_train, df_conf))

 #
 #   ([Xy_bins_lb1], [Xy_bins_lb2,Xy_bins_lb3], "lb1-lb23"),
 #   ([Xy_bins_lb2], [Xy_bins_lb1,Xy_bins_lb3], "lb2-lb13"),
 #   ([Xy_bins_lb3], [Xy_bins_lb1,Xy_bins_lb2], "lb3-lb12"),
 #   ([Xy_cf_bins], [Xy_cf_bins.iloc[:100]], "cf"),
 #   ([Xy_cf_bins[columns_inter]], [Xy_bins_lb1[columns_inter], Xy_bins_lb2[columns_inter], Xy_bins_lb3[columns_inter]], "cf-lb"),
 #   ([Xy_bins_lb1[columns_inter], Xy_bins_lb2[columns_inter], Xy_bins_lb3[columns_inter]], [Xy_cf_bins[columns_inter]], "lb-cf")

u.train_and_eval(cross_tuples, classifiers, classifier_kwargs, undersample=True, save='data/anatomy-lb.pkl')


