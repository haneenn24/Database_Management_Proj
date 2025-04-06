# Simulated query evaluation pipeline with optimizations applied
import time
import random

# Cache for intermediate inference results
inference_cache = {}

# Example Bayesian result store (pretend we precomputed them)
precomputed_world_stats = {
    "SalePrice < 200000": {
        "LotArea_mean": 8500,
        "LotArea_var": 1200000,
        "cached_posterior": 0.72
    }
}

# Known dependent attribute pairs for this example
dependent_pairs = [('SalePrice', 'LotArea'), ('Neighborhood', 'SalePrice')]

def find_plan(projection=None, join=False, filters=None, relation_keys=['Id']):
    """Determines query safety and if dependent model is needed."""
    if projection is None and not join:
        return "SAFE", "Independent"
    if isinstance(projection, str) and projection in relation_keys:
        return "SAFE", "Independent"
    if isinstance(projection, list) and all(attr in relation_keys for attr in projection):
        return "SAFE", "Independent"
    if filters and projection:
        for attr in filters:
            if isinstance(projection, list):
                for p in projection:
                    if (p, attr) in dependent_pairs or (attr, p) in dependent_pairs:
                        return "#P-complete", "Dependent"
            elif (projection, attr) in dependent_pairs or (attr, projection) in dependent_pairs:
                return "#P-complete", "Dependent"
    if join:
        for attr in filters:
            if attr in [p[0] for p in dependent_pairs]:
                return "#P-complete", "Dependent"
    return "SAFE", "Independent"

def evaluate_query(query_id, projection, filters, join=False, use_cached=True, use_sampling=True):
    """Smart evaluation engine for queries."""
    print(f"\n🔍 Evaluating Query {query_id}...")
    plan, model = find_plan(projection=projection, join=join, filters=filters)
    print(f"  ▶ FIND-PLAN: {plan} — Using {model} model")

    # Simulated time delay
    if plan == "SAFE":
        time.sleep(0.2)
        result = {"runtime_sec": 0.2, "used_model": "Independent", "posterior": round(random.uniform(0.6, 0.9), 3)}
    else:
        condition = " & ".join(filters)
        if use_cached and condition in inference_cache:
            print("  ✅ Using cached inference.")
            time.sleep(0.1)
            result = inference_cache[condition]
            result["cached"] = True
        elif use_cached and condition in precomputed_world_stats:
            print("  ✅ Using precomputed world stats.")
            time.sleep(0.2)
            result = {"runtime_sec": 0.2, "used_model": "Dependent", "posterior": precomputed_world_stats[condition]["cached_posterior"], "cached": True}
        elif use_sampling:
            print("  ⚡ Using Monte Carlo sampling for posterior estimation...")
            time.sleep(0.5)
            approx_post = round(random.uniform(0.55, 0.85), 3)
            result = {"runtime_sec": 0.5, "used_model": "Dependent (Sampled)", "posterior": approx_post, "cached": False}
            inference_cache[condition] = result
        else:
            print("  🧠 Full Bayesian inference (expensive)...")
            time.sleep(1.5)
            exact_post = round(random.uniform(0.65, 0.9), 3)
            result = {"runtime_sec": 1.5, "used_model": "Dependent (Exact)", "posterior": exact_post, "cached": False}

    print(f"  ⏱ Runtime: {result['runtime_sec']} sec | Posterior Prob: {result['posterior']} | Model: {result['used_model']}")
    return result

# Simulated query evaluations
evaluate_query("Q1", projection="LotArea", filters=["SalePrice < 200000"])
evaluate_query("Q2", projection="Neighborhood", filters=["GrLivArea > 1500"])
evaluate_query("Q3", projection="LotArea", filters=["SalePrice < 200000"])  # Will use cache
evaluate_query("Q4", projection=["GrLivArea", "SalePrice"], filters=["Neighborhood = 'NridgHt'", "SalePrice > 250000"], join=True)
