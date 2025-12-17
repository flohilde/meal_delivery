from typing import List, Dict, Union
from templates import DemandPolicy, Observation


def restaurant_proximity_filter(proximity: Union[float, List[float]], customer_node: int, restaurant_nodes: List[int],
                                tt_matrix: Dict) -> List[str]:
    r"""
    Filters restaurants based on proximity (in seconds of travel time) to customer. Proximity can either be an integer,
    i.e., a single proximity threshold for all restaurants, or a list of proximity thresholds for each restaurant.
    """
    if isinstance(proximity, float) or isinstance(proximity, int):
        filtered_restaurants = ["r_{}".format(r) for r, n in enumerate(restaurant_nodes)
                                if tt_matrix[str(customer_node)][str(n)] < proximity]
    else:
        assert len(proximity) == len(restaurant_nodes)
        filtered_restaurants = ["r_{}".format(r) for r, n in enumerate(restaurant_nodes)
                                if tt_matrix[str(customer_node)][str(n)] < proximity[r]]
    return filtered_restaurants


class SimpleProximityDemandControl(DemandPolicy):

    def __init__(self, proximity: Union[float, List], restaurant_nodes: List[int], tt_matrix: Dict):
        DemandPolicy.__init__(self)
        self.proximity = proximity
        self.restaurant_nodes = restaurant_nodes
        self.tt_matrix = tt_matrix

    def act(self, obs: Observation) -> Dict:
        r"""
        Returns a demand control decision for every customer that has yet to choose.
        """
        customer_node = obs["new_customer_info"]["location"]
        displayed_restaurants = restaurant_proximity_filter(proximity=self.proximity,
                                                            customer_node=customer_node,
                                                            restaurant_nodes=self.restaurant_nodes,
                                                            tt_matrix=self.tt_matrix)
        demand_action = {r: None for r in displayed_restaurants}
        return demand_action
