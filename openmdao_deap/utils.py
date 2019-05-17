import numpy as np
import pandas as pd
import scipy as sp


def _is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A boolean array of pareto-efficient points.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    # Next index in the is_efficient array to search for
    next_point_index = 0
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Removes dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


def is_pareto_efficient(costs):
    ixs = np.argsort(
        ((costs - costs.mean(axis=0)) / (costs.std(axis=0) + 1e-7)).sum(axis=1)
    )
    costs = costs[ixs]
    is_efficient = _is_pareto_efficient(costs)
    is_efficient[ixs] = is_efficient.copy()
    return is_efficient


def hyperplane_coefficients(points):
    A = np.c_[points[:,:-1], np.ones(points.shape[0])]
    B = points[:,-1]
    coeff, _, _, _ = sp.linalg.lstsq(A, B)
    return coeff


def cases_to_dataframe(case_reader):
    return pd.DataFrame({
        case.iteration_coordinate: pd.Series({
            "counter": case.counter,
            "timestamp": pd.Timestamp.fromtimestamp(case.timestamp),
            **case.get_design_vars(scaled=False),
            **case.get_responses(scaled=False),
        })
        for case in case_reader.get_cases(recurse=True, flat=True)
    }).T
