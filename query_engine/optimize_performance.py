import time
import random

inference_cache = {}

precomputed_world_stats = {
    "SalePrice < 200000": {
        "LotArea_mean": 8500,
        "LotArea_var": 1200000,
        "cached_posterior": 0.72
    }
}

dependent_pairs = [('SalePrice', 'LotArea'), ('Neighborhood', 'SalePrice')]

def evaluate_query_optimized(query_id, projection, filters, join=False, use_cached=True, use_sampling=True):
    plan, model = find_plan(projection=projection, join=join, filters=filters)
    print(f"  ▶ FIND-PLAN: {plan} — Using {model} model")

    condition = " & ".join(filters) if filters else "ALL"
    result = {}

    if plan == "SAFE":
        time.sleep(0.2)
        result = {"runtime_sec": 0.2, "used_model": "Independent", "posterior": round(random.uniform(0.6, 0.9), 3)}
    else:
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





