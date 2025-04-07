
import os
import pandas as pd
from config.load_config import load_config
from preprocessing.custom_imputer import knn_impute_custom, rf_impute_custom
from preprocessing.custom_imputer_topk import knn_impute_topk, rf_impute_topk
from preprocessing.assign_probability import assign_uniform_probabilities, assign_bayesian_probabilities, bayesian_update_probabilities
from query_engine.safety_checker import find_plan
from models.independent_case import (
    evaluate_independent_model,
    query_target_probability as query_target_independent,
    probability_of_any_match,
)
from real_bn_evaluator import infer_bayesian_structure
from query_engine.optimize_performance import evaluate_query_optimized


def parse_and_run_query(df, query_string):
    # Basic SELECT-WHERE parsing
    query_string = query_string.strip().replace(";", "")
    if "WHERE" in query_string.upper():
        select_part, where_part = query_string.upper().split("WHERE")
        pandas_query = where_part.strip().replace("=", "==")
    else:
        select_part = query_string.upper()
        pandas_query = None

    select_cols = select_part.replace("SELECT", "").replace("FROM", "").split()[0]
    select_cols = [col.strip() for col in select_cols.split(",") if col.strip() != "*"]

    # Apply WHERE
    filtered = df.query(pandas_query) if pandas_query else df
    return filtered[select_cols + ["probability"]] if select_cols else filtered


def extract_evidence_from_query(query_string):
    """
    Very basic parser to extract WHERE condition into a dictionary
    Only supports a single condition for now (e.g., SalePrice < 200000)
    """
    if "WHERE" not in query_string.upper():
        return {}
    where_part = query_string.upper().split("WHERE")[1].strip().replace(";", "")
    if "AND" in where_part or "OR" in where_part:
        print("⚠️ Multiple conditions not fully supported in evidence parsing.")
    cond = where_part.replace("==", "=").split("=")
    key = cond[0].strip().split()[0]
    value = cond[1].strip().split()[0]
    return {key: value}


def run_pipeline(config_path="config/settings.yaml"):
    config = load_config(config_path)
    print("\n✅ Pipeline started with config:", config_path)

    # Step 1: Load data
    df = pd.read_csv(config["data"]["input_path"])
    print(f"📥 Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Step 2: Simulate missing data pattern
    from preprocessing.remove_cells import remove_missingness_pattern
    df = remove_missingness_pattern(df, pattern=config["missing_data"]["type"], percent=config["missing_data"]["percentage"])

    # Step 3: Impute
    target_col = config["imputation"]["target_column"]
    if config["imputation"]["mode"] == "fixed":
        if "KNN" in config["imputation"]["methods"]:
            df = knn_impute_custom(df, column=target_col)
        elif "RF" in config["imputation"]["methods"]:
            df = rf_impute_custom(df, column=target_col)

    # Step 4: Assign or Update Probabilities
    if "probability" in df.columns:
        print("🔄 Updating tuple probabilities via Bayesian update...")
        df = bayesian_update_probabilities(df, config["tuple_probabilities"]["bayesian_net_file"])
    else:
        if config["tuple_probabilities"]["mode"] == "uniform":
            df = assign_uniform_probabilities(df)
        elif config["tuple_probabilities"]["mode"] == "bayesian":
            df = assign_bayesian_probabilities(df, config["tuple_probabilities"]["bayesian_net_file"])

    # Step 5: Query Execution Loop
    queries = config["queries"]["examples"]
    for name, query in queries.items():
        print(f"\n▶ Executing Query: {name}")
        safe, plan_type = find_plan(query, config["query_evaluation"]["find_plan_logic"]["primary_keys"],
                                    config["query_evaluation"]["find_plan_logic"]["dependent_pairs"])
        print(f"  🔐 Safety Check → {'SAFE' if safe else '#P-Complete'}")
        print(f"  ⚙️  Model to Use: {plan_type}")

        if plan_type == "Independent":
            filtered_df = parse_and_run_query(df, query)
            results_with_probs = [(row.drop("probability").tolist(), row["probability"]) for _, row in filtered_df.iterrows()]
            result_table = evaluate_independent_model(results_with_probs)
            print("🔎 Probability of Any Match (OR logic):", probability_of_any_match(result_table))
            print(result_table.head())
        else:
            evidence = extract_evidence_from_query(query)
            output = infer_bayesian_structure(df, evidence)
            print("🎯 Posterior Distribution:", output)

    print("\n🏁 Pipeline complete.")

    # Optional Performance Evaluation (per query)
    print("\n⚡ Post-query performance check:")
    for name, query in queries.items():
        evidence = extract_evidence_from_query(query)
        filters = [f"{k} = {v}" for k, v in evidence.items()]
        evaluate_query_optimized(name, projection=list(evidence.keys()), filters=filters)



if __name__ == "__main__":
    run_pipeline()

