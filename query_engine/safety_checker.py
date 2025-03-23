"""
## What We're Implementing

### Goal of `FIND-PLAN(q)`
Given a conjunctive query `q`, decide:
- If it’s **safe**, return a valid **query plan** (composed of safe relational operators like σ, πᴰ, πᴵ, ⨝ₚ).
- If not, mark the query as **#P-complete** (too hard to compute directly).

---

## 🧱 Inputs to Algorithm
- `q`: a **conjunctive query**, expressed as:
  ```python
  {
    "head": ["city"],
    "body": [
      {"rel": "Order", "vars": ["prod", "price", "cust"], "pred": "price == 300"},
      {"rel": "CustomerMale", "vars": ["cust", "city", "profession"]}
    ]
  }
  ```
- You’ll need metadata about:
  - Keys for each relation (what columns form the key)
  - Which variables appear where

---

## Safe Relational Operators to Use

| Operator       | Symbol   | Use When |
|----------------|----------|----------|
| Selection      | σ        | filters on base tables |
| Join           | ⨝ₚ       | combine tables on shared variables |
| Disjoint Project | πᴰ     | when values are disjoint (e.g. exclusive tuples) |
| Independent Project | πᴵ | when values are independent |
| FAIL           | ❌       | query is #P-complete |
"""


# query_engine/safety_checker.py

def find_plan(query, relation_keys):
    """
    Implements Algorithm 3.1 FIND-PLAN from the paper:
    - Determines whether the conjunctive query is safe (PTIME)
    - If so, returns a relational plan
    - If not, returns "#P-complete"

    Parameters:
    ----------
    query : dict
        {
          "head": ["city"],
          "body": [
              {"rel": "Order", "vars": ["prod", "price", "cust"], "pred": "price == 300"},
              {"rel": "CustomerMale", "vars": ["cust", "city", "profession"]}
          ]
        }

    relation_keys : dict
        Mapping from relation name to list of key variables
        {
            "Order": ["prod", "price"],
            "CustomerMale": ["cust"]
        }

    Returns:
    -------
    plan : str or dict
        A nested dict representing the plan if safe, or "#P-complete"
    """

    def get_free_vars(q):
        head_vars = set(q["head"])
        body_vars = set()
        for atom in q["body"]:
            body_vars.update(atom["vars"])
        return list(body_vars - head_vars)

    def in_key(var, rel):
        return var in relation_keys.get(rel, [])

    def in_nonkey(var, rel, atom_vars):
        return var in atom_vars and var not in relation_keys.get(rel, [])

    # Base Case: single relation, no free vars
    if len(query["body"]) == 1 and not get_free_vars(query):
        return {"op": "select", "from": query["body"][0], "on": query["body"][0].get("pred", None)}

    free_vars = get_free_vars(query)

    # Case 1: Independent projection πᴵ
    for x in free_vars:
        if all(in_key(x, atom["rel"]) for atom in query["body"]):
            # safe to project
            new_query = {
                "head": list(set(query["head"] + [x])),
                "body": query["body"]
            }
            return {"op": "πᴵ", "attrs": query["head"], "sub": find_plan(new_query, relation_keys)}

    # Case 2: Disjoint projection πᴰ
    for x in free_vars:
        for atom in query["body"]:
            rel = atom["rel"]
            if in_nonkey(x, rel, atom["vars"]):
                key = set(relation_keys.get(rel, []))
                if key.isdisjoint(free_vars):
                    new_query = {
                        "head": list(set(query["head"] + [x])),
                        "body": query["body"]
                    }
                    return {"op": "πᴰ", "attrs": query["head"], "sub": find_plan(new_query, relation_keys)}

    # Case 3: Independent join
    vars_in_atoms = [set(atom["vars"]) for atom in query["body"]]
    for i in range(len(vars_in_atoms)):
        for j in range(i + 1, len(vars_in_atoms)):
            if vars_in_atoms[i].isdisjoint(vars_in_atoms[j]):
                q1 = {"head": query["head"], "body": [query["body"][i]]}
                q2 = {"head": query["head"], "body": [query["body"][j]]}
                return {"op": "⨝ₚ", "left": find_plan(q1, relation_keys), "right": find_plan(q2, relation_keys)}

    # If none apply
    return "#P-complete"



#Example Usage:
#Input:
query = {
    "head": ["city"],
    "body": [
        {"rel": "Order", "vars": ["prod", "price", "cust"], "pred": "price == 300"},
        {"rel": "CustomerMale", "vars": ["cust", "city", "profession"]}
    ]
}

relation_keys = {
    "Order": ["prod", "price"],
    "CustomerMale": ["cust"]
}

#Call:
from query_engine.safety_checker import find_plan
plan = find_plan(query, relation_keys)
print(plan)

