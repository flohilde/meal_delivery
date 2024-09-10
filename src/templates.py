from typing import NewType, Dict, List, Union
from collections import namedtuple, defaultdict
from abc import ABC, abstractmethod


Observation = NewType("Observation", Dict[str, Union[int, List, Dict]])


class Policy(ABC):
    r"""
    Abstract template for a fleet control and restaurant scheduling policy, i.e., custom policies should inherit from
    this template class. A policy must map an observation to an action in every state.
    """

    def __init__(self):
        pass

    @abstractmethod
    def act(self, obs: Observation) -> Dict:
        r"""
        Every policy must implement the 'act' method that maps an Observation to an Action.
        """
        pass


class DemandPolicy(ABC):
    r"""
    Abstract template for a demand control policy, i.e., custom demand policies should inherit from
    this template class. A policy must map an observation to list of restaurants to display to the customer.
    """

    def __init__(self):
        pass

    @abstractmethod
    def act(self, obs: Observation) -> Dict:
        r"""
        Every demand control policy must implement the 'act' method that maps an Observation to a demand control
        decision which is a dict of restaurants to display (keys)
        and their potentially modified characteristics (values), e.g., delivery fee,
        estimated delivery time, promotions, etc.
        """
        pass
