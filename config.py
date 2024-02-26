from dataclasses import dataclass
import yaml
import os

@dataclass
class KModesConfig:
    n_clusters: int
    init: str
    n_init: int
    exclude_column: str
    verbose: int 
    

@dataclass
class RuleMiningConfigs:
    algorithm: str
    min_support: float
    min_confidence: float
    clustered: bool
    filter: bool
    filter_function: str
    mine_rules_for: list
    select_features: bool
    top_k_features: int

@dataclass
class ClassificationConfigs:
    attribute_to_predict: str

@dataclass
class TweetyConfigs:
    cmd: str
    email: str      # Replace with your email
    compcriterion: str         



@dataclass
class BaseConfig:
    experiment_name: str
    save_to: str
    num_runs: int
    train_csv_data_path: str
    test_csv_data_path: str
    threshold_strict_rules: float
    kmodes_configs: KModesConfig
    classification_configs: ClassificationConfigs
    rule_mining_configs: RuleMiningConfigs
    tweety_configs: TweetyConfigs
    cluster_model_path: str # If cluster model is already trained
    rule_file: str # if rules are already mined


def load_config_yaml(path: str, as_obj=True) -> BaseConfig:
    with open(path,'r') as config_file:
        if path.endswith('yaml'):
            configs = yaml.load(config_file, Loader=yaml.FullLoader)
            
    if not as_obj:
        return configs
    else:
        return BaseConfig(**configs)

