# dependent_model.py
"""
This module implements the Dependent Probabilistic Model using a Bayesian Network.
In this model, tuples are not assumed to be independent.
Instead, dependencies are captured via a Bayesian Network (BN), and the probability
of a query result is computed using conditional probabilities.

Intensional approach: Dependencies are explicitly modeled using a BN.
Only a subset of meaningful dependencies is supported (not all possible dependencies),
as computing general dependencies is NP-hard.
"""

import pandas as pd
import networkx as nx
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node


def build_bayesian_network(nodes_info, edges_info, cpts):
    """
    Build a Bayesian Network given structure and CPTs.

    Parameters:
    - nodes_info: dict of {node_name: [values]} for discrete variables
    - edges_info: list of (parent, child) edges
    - cpts: dict of {node_name: CPT as list of lists}

    Returns:
    - bn: pomegranate BayesianNetwork object
    """
    from pomegranate import Node
    
    node_objs = {}
    for node, values in nodes_info.items():
        if node not in cpts:
            dd = DiscreteDistribution({v: 1/len(values) for v in values})
            node_objs[node] = Node(dd, name=node)

    for node, table in cpts.items():
        parents = [p for p, c in edges_info if c == node]
        dist = ConditionalProbabilityTable(table, [node_objs[p].distribution for p in parents])
        node_objs[node] = Node(dist, name=node)

    bn = BayesianNetwork("Dependent Query BN")
    for node in node_objs.values():
        bn.add_node(node)
    for p, c in edges_info:
        bn.add_edge(node_objs[p], node_objs[c])

    bn.bake()
    return bn


def evaluate_dependent_model(bn, evidence):
    """
    Given a Bayesian Network and evidence, compute the probability distribution
    over the query variables.

    Parameters:
    - bn: BayesianNetwork object
    - evidence: dict of observed variable values

    Returns:
    - dict of variable: probability_distribution
    """
    beliefs = bn.predict_proba(evidence)
    result = {}
    for var, dist in zip(bn.states, beliefs):
        if isinstance(dist, DiscreteDistribution):
            result[var.name] = dict(dist.items())
    return result
