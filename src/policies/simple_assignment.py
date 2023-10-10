from collections import defaultdict
from src.templates import RestaurantAction, VehicleAction, Action, Observation, Policy


class SimpleAssignmentPolicy(Policy):
    r"""
    Implementation of a simple assignment policy that assigns each
    unassigned order to the vehicle with the lowest busy time.
    New stops are appended to the end of the assigned vehicle's route.
    Orders are appended to the end of the restaurant's queue.
    No postponement of order assignments is considered.
    """

    def __init__(self):
        Policy.__init__(self)

    def act(self, obs: Observation) -> Action:
        r"""
        Creates an action based on an observation.
        """
        action = {"vehicle_action": defaultdict(lambda: []),
                  "restaurant_action": defaultdict(lambda: [])}
        for customer_id, restaurant_id in obs["unassigned_orders"]:
            vehicle_index = sorted(obs["vehicle_info"].keys(),
                                   key=lambda x: obs["vehicle_info"][x]["busy_time"])[0]
            pickup_action = VehicleAction(restaurant_id, -1, -1, [customer_id], None)
            delivery_action = VehicleAction(customer_id, -1, -1, None, [restaurant_id])
            restaurant_action = RestaurantAction(customer_id, -1, -1)
            action["vehicle_action"][vehicle_index].extend([pickup_action, delivery_action])
            action["restaurant_action"][restaurant_id].append(restaurant_action)
        return Action(action)
