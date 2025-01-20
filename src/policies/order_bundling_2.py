from templates import Policy, Action, Observation, VehicleAction, RestaurantAction
from collections import defaultdict

class RestaurantProximityBatchingPolicy(Policy):
    def __init__(self, tt_matrix, max_travel_time=15):
        super().__init__()
        self.tt_matrix = tt_matrix
        self.max_travel_time = max_travel_time

    def act(self, obs: Observation) -> Action:
        action = {"vehicle_action": defaultdict(list),
                  "restaurant_action": defaultdict(list)}

        # Group orders into batches based on restaurant proximity
        unassigned_orders = obs["unassigned_orders"]
        batches = self.create_batches_based_on_restaurant_proximity(unassigned_orders, obs)

        assigned_vehicles = set()  # Keep track of vehicles already assigned to a batch
        for batch in batches:
            vehicle_index = self.find_least_busy_vehicle_excluding_assigned(obs, assigned_vehicles)
            if vehicle_index is not None:
                assigned_vehicles.add(vehicle_index)

            for customer_id, restaurant_id in batch:
                pickup_action = VehicleAction(restaurant_id, -1, -1, [customer_id], None)
                delivery_action = VehicleAction(customer_id, -1, -1, None, [restaurant_id])
                restaurant_action = RestaurantAction(customer_id, -1, -1)
                action["vehicle_action"][vehicle_index].extend([pickup_action, delivery_action])
                action["restaurant_action"][restaurant_id].append(restaurant_action)

        return Action(action)

    def create_batches_based_on_restaurant_proximity(self, orders, obs):
        batches = []
        while orders:
            batch, orders = self.find_restaurants_close_to_each_other(orders, obs)
            batches.append(batch)
        return batches

    def find_restaurants_close_to_each_other(self, orders, obs):
        if not orders:
            return [], []

        batch = [orders[0]]
        base_restaurant_location = obs["restaurant_info"][orders[0][1]]["location"]
        remaining_orders = orders[1:]

        for order in remaining_orders:
            restaurant_location = obs["restaurant_info"][order[1]]["location"]
            if self.get_travel_time(base_restaurant_location, restaurant_location) <= self.max_travel_time:
                batch.append(order)

        remaining_orders = [order for order in remaining_orders if order not in batch]
        return batch, remaining_orders

    def get_travel_time(self, location1, location2):
        return self.tt_matrix[str(location1)][str(location2)]

    def find_least_busy_vehicle_excluding_assigned(self, obs, assigned_vehicles):
        for vehicle_id in sorted(obs["vehicle_info"].keys(), key=lambda x: obs["vehicle_info"][x]["busy_time"]):
            if vehicle_id not in assigned_vehicles:
                return vehicle_id
        return None















