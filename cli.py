import fire
import os
import pandas as pd
import yaml
import shutil
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn.feature_selection import chi2
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, classification_report
import random

import numpy as np

def get_rules(final_combined,algorithm='fpgrowth', min_support=0.8, min_conf=0.9):
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.preprocessing import TransactionEncoder
    import mlxtend.frequent_patterns as patterns
    te = TransactionEncoder()
    print(final_combined)
    te_ary = te.fit(final_combined).transform(final_combined)
    df_feq = pd.DataFrame(te_ary, columns=te.columns_)

    if algorithm=='fpgrowth':
        print('Using fpgrowth algorithm')

        frequent_itemsets = patterns.fpgrowth(df_feq, min_support=min_support, use_colnames=True,verbose=1)
    elif algorithm == 'fpmax':
        print('Using fpmax algorithm')
        frequent_itemsets = patterns.fpmax(df_feq, min_support=min_support, use_colnames=True,verbose=1)
   
    elif algorithm == 'apriori':
        print('Using apriori algorithm')
        print(df_feq)
        frequent_itemsets = patterns.apriori(df_feq, min_support=min_support, use_colnames=True,verbose=1)
   
    #frequent_itemsets_ap = patterns.apriori(df_feq, min_support=min_support, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    #rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=min_conf)
    return rules

def get_combined_item_list(df):
    
    positive_items = df.apply(lambda row: row[row == 1].index, axis=1)
    negative_items = df.apply(lambda row: row[row == 0].index, axis=1)

    final_items = [ i.to_list() for i in positive_items.to_list()]
    final_negative_items = []
    for i,row in negative_items.items():
        final_negative_items.append([ f'not_{i}' for i in row.to_list()])

    final_combined = []
    for i, items in enumerate(final_items):
        items.extend( final_negative_items[i])
        final_combined.append(items)
    return final_combined
def select_features(X_train, y_train, X_test,score,k='all'):
    fs = SelectKBest(score_func=score, k=k)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def get_feature_scores(X_train, y_train, X_test,raw_data,top_k=5,target_name=None,method=mutual_info_classif):
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test,method)
    # what are scores for the features
    feature_names = raw_data.drop(['Cluster',target_name,'animal'],axis=1).columns
    features_score = []
    for i in range(len(fs.scores_)):
     #print(f'Feature: {feature_names[i]} Score: {fs.scores_[i]}')
     features_score.append({'feature': feature_names[i], 'score': fs.scores_[i]})
    # plot the scores
    # pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    # pyplot.show()
    features_score_df = pd.DataFrame(features_score)
    top_feature = features_score_df.sort_values('score',ascending=False)[:top_k]
    fs_columns = [target_name] + list(top_feature['feature'].values)
    fs_raw_data = raw_data[fs_columns]

    return  fs_raw_data

def get_clusters_train_data_selected_features(train_df: pd.DataFrame, target='prededator',top_k_features=5):
    cluster_train_data_map = {}
    for name, group in train_df.groupby(['Cluster']):
        print(f'{name=} Size: {len(group)}')
        X_df = group.drop(['Cluster',target,'animal'],axis=1)
        if len(group) > 1:
            X = X_df.values

            y = group[target].values



            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
            # summarize
            print('Train', X_train.shape, y_train.shape)
            print('Test', X_test.shape, y_test.shape)
            if  X_train.shape[0] > 1 and X_test.shape[0] >= 1:
                df_best_features = get_feature_scores(X_train,y_train,X_test,group,target_name=target,top_k=top_k_features)
                print(f'Top {top_k_features} features: {df_best_features.columns}')
            else:

                fs_columns = [target] + random.sample(list(X_df.columns),top_k_features)
                df_best_features = group[fs_columns]
                print(f' Xtest and X_Train Sample to small. Select all features: {df_best_features.columns}')

        else:
         # Select all features if group is to small
            fs_columns = [target] + random.sample(list(X_df.columns),top_k_features)
            df_best_features = group[fs_columns]
            print(f'Sample to small. Select all features: {df_best_features.columns}')

        cluster_train_data_map[name[0]] = get_combined_item_list(df_best_features)
    
    for cluster, train_data in cluster_train_data_map.items():
        print(f'{cluster=}\nSize:{len(train_data)}')
    return cluster_train_data_map

def get_clusters_train_data_map(train_df):
    grouped_train_df = train_df.groupby('Cluster')
    keys = grouped_train_df.groups.keys()

    cluster_train_data_map = {}
    for i in keys:
        cluster_train_data_map[i] = get_combined_item_list(grouped_train_df.get_group(i).drop('Cluster',axis=1))
    for cluster, train_data in cluster_train_data_map.items():
        print(f'{cluster=}\nSize:{len(train_data)}')

    return cluster_train_data_map


def filter_rules_klein(rules: pd.DataFrame, key=None):
    filtered_thimm = filter_rules_thimm(rules)
    filter_klein = []
    for index, row in filtered_thimm.iterrows():
        found = False
        for word in row['antecedents']:
            if key in word:
                filter_klein.append(row)
                found = True
                break
        
        if not found:
            for word in row['consequents']:
                if 'lean' in word:
                    filter_klein.append(row)

        # if 'lean' in row['antecedents'] or 'lean' in row["consequents"]:
        #     filter_klein.append(row)
        # else:
        #     print(row["antecedents"].str.contains('lean'))
        #     print(f'lean not found in{row["antecedents"]} {row["consequents"]}')
   
    if len(filter_klein) == 0:
        print('No rules for lean found in cluster')
        return filtered_thimm
    f_klein = pd.DataFrame(filter_klein)
    return f_klein


def filter_rules_thimm(rules) -> pd.DataFrame:
    # Filtering as described by Thimm 
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))


    filtered_rules_fp = rules[ (rules['antecedent_len'] <= 3) &
       (rules['consequents_len'] == 1) ]
    return filtered_rules_fp

filter_function_dict = {'thimm': filter_rules_thimm,'klein': filter_rules_klein}

def is_valid_ruleset(rule_set,target):
    # Check if target is at least in one conclusion
    for index, row in rule_set.iterrows():
        #print(row)
        consequents = tuple(row['consequents'])[0]
        if target in consequents:
            return True
    return False

def extract_target_rules(cluster_filtered_rules: pd.DataFrame,target: str) -> pd.DataFrame:
    target_rules = []
    for index, row in cluster_filtered_rules.iterrows():
       
        consequents = tuple(row['consequents'])[0]
        if target in consequents:
            target_rules.append(row)
    return pd.DataFrame(target_rules)

def mine_rules(file,save_to,algorithm='fpgrowth',min_support=0.8,min_confidence=0.9,clustered=False,filter=True, filter_function='thimm',mine_rules_for=None,select_features=None,top_k_features=None,target=None):
    """_summary_

    Args:
        file (_type_): _description_
        algo (str, optional): _description_. Defaults to 'fpgrowth'.
        min_support (float, optional): _description_. Defaults to 0.8.
        min_conf (float, optional): _description_. Defaults to 0.9.
        save_to (_type_, optional): _description_. Defaults to None.
        clustered (bool, optional): _description_. Defaults to False.
        filter (bool, optional): _description_. Defaults to True.
    """
    import pickle
    if not os.path.exists(file):
       print(f'File {file} does not exists.')
       return
    df = pd.read_csv(file)


    if clustered:
        if not 'Cluster' in df.columns:
            print('No Clusters found.')
            return
        if mine_rules_for:
            mine_rules_for.append('Cluster')
            df = df[mine_rules_for]
        if select_features:
             cluster_data_map = get_clusters_train_data_selected_features(df,target=target,top_k_features=top_k_features)
        else:
            cluster_data_map = get_clusters_train_data_map(df)

        cluster_path_map = {}
        raw_cluster_path_map = {}
        
        for cluster_index, data in cluster_data_map.items():
            print(f'Mining rules for cluster {cluster_index} Size: {len(data)}')
            rule_set_is_valid = False
            temp_min_support = min_support
            temp_min_confidence = min_confidence
            run_number = 1
            while rule_set_is_valid != True:
                print(f'{run_number=}')
                cluster_rules = get_rules(data,min_support=temp_min_support,min_conf=temp_min_confidence,algorithm=algorithm)
                print(f'Total rules mined for cluster {cluster_index}:{cluster_rules.shape}')
                if filter:
                    cluster_filtered_rules = filter_function_dict[filter_function](cluster_rules)
                    print(f'Filtered rules mined for cluster {cluster_index}:{cluster_filtered_rules.shape}')
                else:
                    cluster_filtered_rules = cluster_rules
                
                if run_number == 1:
                    inital_ruleset = cluster_filtered_rules

                if is_valid_ruleset(cluster_filtered_rules,target):
                    if run_number > 1:
                        print('Extracting target rules!')
                        target_rules = extract_target_rules(cluster_filtered_rules,target)
                        cluster_filtered_rules = pd.concat([inital_ruleset, target_rules], ignore_index=True)
                    rule_set_is_valid = True
                else:
                    if run_number % 2 == 0:
                        print(f'Rule set not valid.\n Adusting min_confidence. Old:{temp_min_confidence}')
                        temp_min_confidence = temp_min_confidence - 0.05
                        print(f'New: {temp_min_confidence}')
                    else:
                        print(f'Rule set not valid.\n Adusting min_support. Old:{temp_min_support}')
                        temp_min_support = temp_min_support - 0.05
                        print(f'New: {temp_min_support}')
                    
                    if temp_min_confidence <= 0 or temp_min_support <= 0:
                        print(f"No rules for {target=} in cluster {cluster_index=} mined!")
                        break
                run_number += 1
            
            name = os.path.basename(file).split('.', 1)[0]
            raw_rules_directory = os.path.join(save_to,'raw_rules')
            filtered_rules_directory = os.path.join(save_to,'filtered_rules')
            os.makedirs(raw_rules_directory,exist_ok=True)
            os.makedirs(filtered_rules_directory,exist_ok=True)

            cluster_rules.to_pickle(os.path.join(raw_rules_directory,f'c_{cluster_index}_raw_{name}_rules.pkl'))
            filtered_cluster_rules_path = os.path.join(filtered_rules_directory,f'c_{cluster_index}_filtered_{name}_rules.pkl')
            cluster_filtered_rules.to_pickle(filtered_cluster_rules_path)
            cluster_path_map[cluster_index] = filtered_cluster_rules_path
            raw_cluster_path_map[cluster_index] = os.path.join(raw_rules_directory,f'c_{cluster_index}_raw_{name}_rules.pkl')
            # with open(os.path.join(save_to,f'items_combined_{name}.pkl'), 'wb') as fp:
            #     pickle.dump(items_combined, fp)
        return cluster_path_map, raw_cluster_path_map
            
    else:
        items_combined = get_combined_item_list(df)
        print(f'Size combined item list:{len(items_combined)}')
        with open('test.txt','w') as f:
            for i in items_combined:
                f.write(f'{i}\n')

        rules = get_rules(items_combined,min_support=min_support,min_conf=min_confidence,algorithm=algorithm)

        print(f'Total rules mined:{rules.shape}')
        if filter:
            filtered_rules = filter_rules_thimm(rules)
            print(f'Filtered rules mined:{filtered_rules.shape}')

        name = os.path.basename(file).split('.', 1)[0]
        rules.to_pickle(os.path.join(save_to,f'raw_{name}_rules.pkl'))
        filtered_rules.to_pickle(os.path.join(save_to,f'filtered_{name}_rules.pkl'))
        with open(os.path.join(save_to,f'items_combined_{name}.pkl'), 'wb') as fp:
            pickle.dump(items_combined, fp)
        
        return os.path.join(save_to,f'filtered_{name}_rules.pkl')


   

def split(data,test_size=0.1,save_to=None,seed=42):
    """Split data in train and test data.

    Args:
        data (_type_): Raw data in csv format.
        test_size (float, optional): Fraction of test data. Defaults to 0.1.
        save_to (_type_, optional): Path to save data to. Defaults to None.
    """
    from sklearn.model_selection import train_test_split
    if not os.path.exists(data):
       print(f'File {data} does not exists.')
       return
    if save_to is None:
       save_to = os.getcwd()
    
    df = pd.read_csv(data)
    print(test_size)
    train_df, test_df = train_test_split(df, test_size=test_size,random_state=seed,shuffle=True)
    path, name = os.path.split(data)
    train_df.to_csv(os.path.join(save_to,f'train_{name}'), index=False)
    test_df.to_csv(os.path.join(save_to,f'test_{name}'), index=False)


def kModes(data,save_to,n_clusters=5, init='random', n_init=10,exclude_column=None,verbose=1,save_model=True,merge_df=True):
    """K-Mode clustering for categorical data.

    Args:
        data (_type_): Data to cluster 
        n_clusters (int, optional): Number of clusters to generate. Defaults to 5.
        init (str, optional): _description_. Defaults to 'random'.
        n_init (int, optional): _description_. Defaults to 10.
        exclude_column (_type_, optional): Exclude column from clustering. Defaults to None.
        verbose (int, optional): _description_. Defaults to 1.
        save_model (bool, optional): Save the model. Defaults to True.
        merge_df (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    
    from kmodes.kmodes import KModes

    if not os.path.exists(data):
       print(f'File {data} does not exists.')
       return
    
    train_df = pd.read_csv(data)
    
    excluded = None
    if exclude_column is not None:
        excluded = train_df[exclude_column]
        train_df = train_df.loc[:, train_df.columns != exclude_column]
    
    km = KModes(n_clusters=n_clusters, init=init, n_init=n_init, verbose=verbose)
    
    clusters = km.fit_predict(train_df)
    file_name = os.path.basename(data).split('.', 1)[0]
    save_model_path = os.path.join(save_to, f'{file_name}_kmodes.pickle')
    if save_model:
        import pickle
        
        with open(save_model_path, 'wb') as f:
            pickle.dump(km, f)
    
   
    final_df = train_df.assign(Cluster=clusters )
    if excluded is not None:
        final_df[exclude_column] = excluded
    path, name = os.path.split(data)
    cluster_file_path = os.path.join(save_to,f'clusters_{name}')
    final_df.to_csv(cluster_file_path,index=False)
       
    return cluster_file_path,save_model_path

def filter_strict_rules(rules: pd.DataFrame, threshold=1.0):
    
    strict_rules = rules[rules.confidence]

def to_delp_format(rules,type='strict'):
    seperator = {'strict': '<-', 'defi': '-<'}
    combined_strings = []
    for index, row in rules.iterrows():
        #print(row)
        antecedents = tuple(row['antecedents'])
        consequents = tuple(row['consequents'])[0]
        
        # If antecedents contain multiple values, join them with a comma
        if len(antecedents) > 1 :
            antecedents_str = ''
            for a in antecedents:
                antecedents_str += f'{a}(X),'
            antecedents_str = antecedents_str[:-1] + '.'
        else:
            antecedents_str = f'{antecedents[0]}(X).'
        
        # Construct the combined string in the desired form
        combined_string = f"{consequents}(X) {seperator[type]} {antecedents_str}"
        combined_strings.append(combined_string)
    
    final_str = '\n'.join(combined_strings).replace('not_','~')
    return final_str


def format_and_filter_rules(file,threshold=1.0):
    """_summary_

    Args:
        file (_type_): _description_
        threshold (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    if not os.path.exists(file):
       print(f'File {file} does not exists.')
       return
    
    rules = pd.read_pickle(file)
    strict_rules = rules[rules.confidence >= threshold]
    delp_strict_rules = to_delp_format(strict_rules,type='strict')

    defi_rules = rules[rules.confidence < threshold]
    delp_defi_rules = to_delp_format(defi_rules,type='defi')
    return delp_strict_rules, delp_defi_rules


def att_to_delp(data_file,save_to):
    if save_to is None:
        save_to = os.getcwd()

    if data_file is not None:
        atts = format_attributes(data_file)
        with open(os.path.join(save_to,'atts.pl'),'w') as f:
            f.write(atts)
        

def format_attributes(file):
    if not os.path.exists(file):
       print(f'File {file} does not exists.')
       return
    df = pd.read_csv(file)
    formatted_attributes = attributes_to_delp(df)
    return formatted_attributes

def create_rule_file_delp(cluster_rules_map,save_to,threshold=1.0):
    delp_cluster_rules_path_map = {}
    delp_rule_file_dir = os.path.join(save_to,'delp_cluster_rules')
    os.makedirs(delp_rule_file_dir)
    for cluster_index, rule_file  in cluster_rules_map.items():
        print(f'{cluster_index=}')
        print(f'{rule_file=}')
        strict_rules, def_rules = format_and_filter_rules(rule_file,threshold=threshold)
        final_cluster_delp_file_content = f'{strict_rules}\n{def_rules}'
        name = os.path.basename(rule_file).split('.')[0]
        final_path = os.path.join(delp_rule_file_dir,f'{name}.pl')
        with open(final_path,'w') as f:
            f.write(final_cluster_delp_file_content)
        delp_cluster_rules_path_map[cluster_index] = final_path
    
    return delp_cluster_rules_path_map

        
def generate_delp(rule_file,data_file=None,save_to=None,name='rules',threshold=1.0):
    if data_file is not None:
        atts = format_attributes(data_file)
    else:
        atts = ''
    
    if save_to is None:
        save_to = os.getcwd()

    if os.path.isdir(rule_file):
        for cluster_file in os.listdir(rule_file):

            full = os.path.join(rule_file,cluster_file)
            strict_rules, def_rules = format_and_filter_rules(full,threshold=threshold)
            final_delp_file = f'{strict_rules}\n{def_rules}'
            name = cluster_file.split('.')[0]
            full_path = os.path.join(save_to,f'{name}.pl')
            with open(full_path,'w') as f:
                f.write(final_delp_file)

            

    else:

    
        strict_rules, def_rules = format_and_filter_rules(rule_file,threshold=threshold)

        final_delp_file = f'{atts}\n{strict_rules}\n{def_rules}'
        if save_to is None:
            save_to = os.getcwd()
        full_path = os.path.join(save_to,f'{name}.pl')
        with open(full_path,'w') as f:
            f.write(final_delp_file)
        return full_path


import json
def attributes_to_delp(df, exclude_attribute_to_predict=None,save_to=None):
    formatted_strings = []
    animal_attribute_map = {}
    print(df)
    if exclude_attribute_to_predict is not None:
        df = df.loc[:, df.columns != exclude_attribute_to_predict]
    for index, row in df.iterrows():
        animal = row['animal']
        attributes = row.drop('animal')
        
        formatted_attributes = []
        for attribute, value in attributes.items():
            if value == 1:
                formatted_attributes.append(f"{attribute}({animal}).")
            elif value == 0:
                formatted_attributes.append(f"~{attribute}({animal}).")
            else:
                print(value)
                raise ValueError("Invalid value found. Only 0 or 1 are allowed for one-hot encoding.")
        
        animal_attribute_map[animal]  = formatted_attributes

    if save_to is not None:
        with open(os.path.join(save_to,'attributes.yaml'),'w') as f:
            yaml.dump(animal_attribute_map,f)
        
        #formatted_string = "\n".join(formatted_attributes)
        #formatted_strings.append(formatted_string)
    
    return animal_attribute_map

def predict_clusters(csv_data_path,model_path,save_to,exclude_column=None, attribute_to_predict=None):
    import pickle
   

    model = pickle.load(open(model_path, 'rb'))
    test_data = pd.read_csv(csv_data_path)

    excluded = None
    if exclude_column is not None:
        excluded = test_data[exclude_column]
        test_data = test_data.loc[:, test_data.columns != exclude_column]
    
    test_data.astype({attribute_to_predict: 'bool'})
    #test_data[f'n_{attribute_to_predict}'] = ~test_data[attribute_to_predict].astype(bool)

    n_test_data = test_data.copy()
    n_test_data[attribute_to_predict] = n_test_data[attribute_to_predict].replace({0:1, 1:0})

    
    clusters = model.predict(test_data)
    n_cluster = model.predict(n_test_data)
    final_df = test_data.assign(cluster=clusters, n_cluster=n_cluster)
    if excluded is not None:
        final_df[exclude_column] = excluded
    path, name = os.path.split(csv_data_path)
    cluster_file_path = os.path.join(save_to,f'clusters_{name}')
    final_df.to_csv(cluster_file_path,index=False)
    
    return cluster_file_path

import tweety_delp_handler as td


def run_naiv(config_path):
    import config
    cfg = config.load_config_yaml(config_path)
    # print(type(cfg))
    # print(type(cfg.kmodes_configs))
    if cfg.save_to is None:
        cfg.save_to = os.getcwd()
    experiment_root_dir_parent = create_directory_with_suffix(cfg.save_to,cfg.experiment_name)

    results_combined = []
    shutil.copy2(config_path, experiment_root_dir_parent)
    for i in range(1,cfg.num_runs +1):
        experiment_root_dir = os.path.join(experiment_root_dir_parent,f'run_{i}')
        os.makedirs(experiment_root_dir)
        filtered_rule_file_path = mine_rules(file=cfg.train_csv_data_path,save_to=experiment_root_dir,target=None,**cfg.rule_mining_configs)
        test_data_df = pd.read_csv(cfg.test_csv_data_path)
        animal_attributes_map = attributes_to_delp(test_data_df,cfg.classification_configs
                                                   ['attribute_to_predict'],save_to=experiment_root_dir)
        
        delp_filtered_rules_path = generate_delp(filtered_rule_file_path,data_file=None,save_to=experiment_root_dir,name='rules',threshold=1.0)
        with open(delp_filtered_rules_path,'r') as f:
            rules = f.read()
        results = []
        for animal, attributes in animal_attributes_map.items():

            kb = '\n'.join(attributes) +'\n' + rules

            print(f'{animal=}')

            print('Calling tweety api...',end='')
            answer,code = td.call_tweety_delp(kb,f'{cfg.classification_configs["attribute_to_predict"]}({animal})')
                
            
                
            if answer['answer'] is not None:
                answer['answer'] = answer['answer'].strip().split(':')[1].strip()
            else:
                answer['answer'] = 'UNDECIDED'
            results.append({'animal':animal,'answer':answer['answer']})

        res = pd.DataFrame(results)
        res.to_csv(os.path.join(experiment_root_dir,'raw_classification_results.csv'))
        res['predicted'] = res['answer'].map({'YES': 1,'NO':0,'UNDECIDED':2})
        res['actual'] = test_data_df[cfg.classification_configs["attribute_to_predict"]]
        acc = accuracy_score(res['actual'], res['predicted'])
        report = classification_report(res['actual'], res['predicted'],output_dict=True)
        res.to_csv(os.path.join(experiment_root_dir,'clean_classification_results.csv'))
        prediction_counts = res['answer'].value_counts()
        false_predicted = res[res['predicted'] != res['actual']]

        print(f'+++++ RESULTS SUMMARY {cfg.experiment_name} RUN #{i} +++++')
        print(f'Predictions Counts:\n{prediction_counts}')
        print(f'Report:\n{report}')
        print(f'False predicted:\n{false_predicted}')

        print(report)

        num_false_predictions = len(false_predicted[false_predicted['answer'] != 'UNDECIDED'])

        run_summary = {'run': id, 'accuracy': report['accuracy'],'num_undecided':len(res[res['answer'] == 'UNDECIDED']),'num_false_prediction': num_false_predictions}

        results_combined.append(run_summary)

        with open(os.path.join(experiment_root_dir,'summary.txt'),'w') as f:
            f.write(f'{prediction_counts}\n Report:\n{report}\nFalse predicted:\n{false_predicted}')

    df_results_combined = pd.DataFrame(results_combined)
    
    stats_results_combined = df_results_combined.describe()
    stats_results_combined.to_csv(os.path.join(experiment_root_dir_parent,'summary_all_runs.csv'))
    stats_results_combined.to_latex(os.path.join(experiment_root_dir_parent,'summary_all_runs.tex'))
    print(stats_results_combined.to_string())
        

        






def run(config_path):
    import config
    cfg = config.load_config_yaml(config_path)
    # print(type(cfg))
    # print(type(cfg.kmodes_configs))
    if cfg.save_to is None:
        cfg.save_to = os.getcwd()
    experiment_root_dir_parent = create_directory_with_suffix(cfg.save_to,cfg.experiment_name)

    results_combined = []
    shutil.copy2(config_path, experiment_root_dir_parent)
    for i in range(1,cfg.num_runs +1):
        experiment_root_dir = os.path.join(experiment_root_dir_parent,f'run_{i}')
        os.makedirs(experiment_root_dir)
        cluster_file_path, save_model_path = kModes(cfg.train_csv_data_path,experiment_root_dir,**cfg.kmodes_configs)
        clusters_test_data_path = predict_clusters(cfg.test_csv_data_path,save_model_path,experiment_root_dir,cfg.kmodes_configs['exclude_column'],cfg.classification_configs
                                                   ['attribute_to_predict'])

        filtered_cluster_rules_path,raw_id_rules_map  = mine_rules(cluster_file_path,experiment_root_dir,target=cfg.classification_configs['attribute_to_predict'],**cfg.rule_mining_configs)
    
        test_data_df = pd.read_csv(cfg.test_csv_data_path)
        animal_attributes_map = attributes_to_delp(test_data_df,cfg.classification_configs
                                                   ['attribute_to_predict'],save_to=experiment_root_dir)

        delp_cluster_rules_map = create_rule_file_delp(filtered_cluster_rules_path,experiment_root_dir,cfg.threshold_strict_rules)

        clustered_test_data_df = pd.read_csv(clusters_test_data_path)

        results = []

        for index, row in clustered_test_data_df.iterrows():
            if row['cluster'] == row['n_cluster']:
                id_cluster = row['cluster']
                n_id_cluster = row['n_cluster']
                animal = row['animal']
                with open(delp_cluster_rules_map[id_cluster],'r') as f:
                    rules = f.read()

                kb = '\n'.join(animal_attributes_map[animal]) +'\n' + rules

                print(f'{id_cluster=} {animal=}')

                print('Calling tweety api...',end='')
                answer,code = td.call_tweety_delp(kb,f'{cfg.classification_configs["attribute_to_predict"]}({animal})')
                
            
                
                if answer['answer'] is not None:
                    answer['answer'] = answer['answer'].strip().split(':')[1].strip()
                else:
                    answer['answer'] = 'UNDECIDED'
                results.append({'animal':animal,'answer':answer['answer']})
            else:
                id_cluster = row['cluster']
                n_id_cluster = row['n_cluster']

                animal = row['animal']
                with open(delp_cluster_rules_map[id_cluster],'r') as f:
                    rules = f.read()
                kb = '\n'.join(animal_attributes_map[animal]) +'\n' + rules

                print(f'{id_cluster=} {animal=}')

                print('Calling tweety api...',end='')
                answer,code = td.call_tweety_delp(kb,f'{cfg.classification_configs["attribute_to_predict"]}({animal})')
                if answer['answer'] is not None:
                    answer['answer'] = answer['answer'].strip().split(':')[1].strip()
                else:
                    answer['answer'] = 'UNDECIDED'
                # results.append({'animal':f'{animal}_{id_cluster}','answer':answer['answer']})

                with open(delp_cluster_rules_map[n_id_cluster],'r') as f:
                    n_rules = f.read()

                n_kb = '\n'.join(animal_attributes_map[animal]) +'\n' + n_rules

                print(f'{n_id_cluster=} {animal=}')
                print('Calling tweety api...',end='')
                n_answer,code = td.call_tweety_delp(n_kb,f'{cfg.classification_configs["attribute_to_predict"]}({animal})')

                if n_answer['answer'] is not None :
                    n_answer['answer'] = n_answer['answer'].strip().split(':')[1].strip()
                else:
                    n_answer['answer'] = 'UNDECIDED'

                id_answer_map = [(id_cluster,answer['answer']), (n_id_cluster,n_answer['answer'])]
                final_answer = check_answer(id_answer_map,raw_id_rules_map)
                results.append({'animal':f'{animal}','answer':final_answer})


        res = pd.DataFrame(results)
        res.to_csv(os.path.join(experiment_root_dir,'raw_classification_results.csv'))
        res['predicted'] = res['answer'].map({'YES': 1,'NO':0,'UNDECIDED':2})
        res['actual'] = test_data_df[cfg.classification_configs["attribute_to_predict"]]
        acc = accuracy_score(res['actual'], res['predicted'])
        report = classification_report(res['actual'], res['predicted'],output_dict=True)
        res.to_csv(os.path.join(experiment_root_dir,'clean_classification_results.csv'))
        prediction_counts = res['answer'].value_counts()
        false_predicted = res[res['predicted'] != res['actual']]

        print(f'+++++ RESULTS SUMMARY {cfg.experiment_name} RUN #{i} +++++')
        print(f'Predictions Counts:\n{prediction_counts}')
        print(f'Report:\n{report}')
        print(f'False predicted:\n{false_predicted}')

        print(report)

        num_false_predictions = len(false_predicted[false_predicted['answer'] != 'UNDECIDED'])

        run_summary = {'run': id, 'accuracy': report['accuracy'],'num_undecided':len(res[res['answer'] == 'UNDECIDED']),'num_false_prediction': num_false_predictions}

        results_combined.append(run_summary)

        with open(os.path.join(experiment_root_dir,'summary.txt'),'w') as f:
            f.write(f'{prediction_counts}\n Report:\n{report}\nFalse predicted:\n{false_predicted}')

    df_results_combined = pd.DataFrame(results_combined)
    
    stats_results_combined = df_results_combined.describe()
    stats_results_combined.to_csv(os.path.join(experiment_root_dir_parent,'summary_all_runs.csv'))
    stats_results_combined.to_latex(os.path.join(experiment_root_dir_parent,'summary_all_runs.tex'))
    print(stats_results_combined.to_string())
           
def check_answer(id_answers: list, rule_map: dict) -> str:
    id,answer = id_answers[0]
    n_id, n_answer = id_answers[1]
    if answer is None and n_answer is not None:
        return n_answer
    if answer is not None and n_answer is None:
        return answer
    
    if n_answer == answer:
        return answer
    
    if n_answer != answer:
        if (n_answer == 'YES' or n_answer == 'NO') and answer == 'UNDECIDED':
            return n_answer
        elif (answer== 'YES' or answer == 'NO') and n_answer == 'UNDECIDED':
            return answer

        if (n_answer == 'YES' and answer == 'NO') or (n_answer == 'NO' and answer == 'YES'):
            rules = pd.read_pickle(rule_map[id])
            n_rules = pd.read_pickle(rule_map[n_id])

            stats_rules = rules.describe()
            mean_support = stats_rules.describe()['support'].loc['mean']
            mean_confidence = stats_rules.describe()['confidence'].loc['mean']
            n_stats_rules = n_rules.describe()
            n_mean_support = n_stats_rules.describe()['support'].loc['mean']
            n_mean_confidence = n_stats_rules.describe()['confidence'].loc['mean']

            if mean_confidence == n_mean_confidence:
                if mean_support == n_mean_support:
                    return answer
                if mean_support > n_mean_support:
                    return answer
                else:
                    return n_answer
            else:
                if mean_confidence > n_mean_confidence:
                    return answer
                else:
                    return n_answer
                


            
    
    
    



def generate_test_data_attributes_json(test_csv_data_path):
    pass




    
    


import os

def create_directory_with_suffix(base_dir, folder_name):
    target_dir = os.path.join(base_dir, folder_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        return target_dir

    # If the directory already exists, add a numbering suffix
    count = 1
    while True:
        new_folder_name = f"{folder_name}_{count}"
        target_dir = os.path.join(base_dir, new_folder_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            return target_dir
        count += 1



function_dict = {'kModes':kModes,
                 'split': split,
                 'mine-rules': mine_rules,
                 'format-rules': format_and_filter_rules,
                 'atts-to-DeLP': att_to_delp,
                 'generate-DeLP': generate_delp,
                 'run': run,
                 'run-naiv': run_naiv}

if __name__ == '__main__':
  fire.Fire(function_dict)