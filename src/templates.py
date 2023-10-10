from typing import NewType, Dict, List, Union
from collections import namedtuple, defaultdict
from abc import ABC, abstractmethod


# define important type hints for the simulation
VehicleAction = namedtuple("VehicleAction",
                           "destination start_at insertion_index orders_to_pickup orders_to_deliver")
RestaurantAction = namedtuple("RestaurantAction", "customer_id start_at insertion_index")
Action = NewType("Action", Dict[str, defaultdict])
Observation = NewType("Observation", Dict[str, Union[int, List, Dict]])


class Policy(ABC):
    r"""
    Abstract template for a policy, i.e., custom policies should inherit from
    this template class. A policy must map an observation to an action in every state.
    """

    def __init__(self):
        pass

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        r"""
        Every policy must implement the 'act' method that maps an Observation to an Action.
        """
        pass
