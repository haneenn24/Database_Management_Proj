
import yaml
import os

def load_config(config_path="settings.yaml"):
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def print_core_settings(config):
    print("\n🔧 Core Configuration Settings Loaded:")
    print("📌 Missing Data Type:", config['missing_data']['type'])
    print("📌 Imputation Mode:", config['imputation']['mode'], "| Methods:", config['imputation']['methods'])
    print("📌 Tuple Probabilities:", config['tuple_probabilities']['mode'])
    print("📌 Possible Worlds Strategies:", len(config['possible_worlds']['strategies']), "defined")
    print("📌 Using Bayesian Network:", config['tuple_probabilities'].get('bayesian_net_file', 'None'))
    print("📌 Queries Available:", len(config['queries']['examples']))
    print("📌 Query Evaluation Strategy:", "FIND-PLAN Enabled" if config['query_evaluation']['model_decision']['use_find_plan'] else "Manual")

if __name__ == "__main__":
    cfg = load_config()
    print_core_settings(cfg)