from templates import Policy, Action, Observation, VehicleAction, RestaurantAction
from collections import defaultdict
import numpy as np
import copy

from vehicle import Stop

class DynamicInsertionPolicy(Policy):
    def __init__(self, tt_matrix):
        super().__init__()
        self.tt_matrix = tt_matrix

    def get_travel_time(self, location1, location2):
       
        return self.tt_matrix.get(str(location1), {}).get(str(location2), float('inf'))

    def act(self, obs: Observation) -> Action:
        action = {"vehicle_action": defaultdict(list),
                  "restaurant_action": defaultdict(list)}
        
        for customer_id, restaurant_id in obs["unassigned_orders"]:
            best_vehicle, best_insertion_point = self.find_best_insertion_for_order(customer_id, restaurant_id, obs)
            
            if best_vehicle is not None:
                pickup_action = VehicleAction(restaurant_id, -1, best_insertion_point, [customer_id], None)
                delivery_action = VehicleAction(customer_id, -1, best_insertion_point + 1, None, [restaurant_id])
                restaurant_action = RestaurantAction(customer_id, -1, -1)

                action["vehicle_action"][best_vehicle].extend([pickup_action, delivery_action])
                action["restaurant_action"][restaurant_id].append(restaurant_action)

        return Action(action)

    def find_best_insertion_for_order(self, customer_id, restaurant_id, obs):
        min_impact = float('inf')
        best_vehicle = None
        best_insertion_point = None

        for vehicle_id, vehicle_info in obs["vehicle_info"].items():
            for insertion_point in range(len(vehicle_info["sequence_of_actions"]) + 1):
                if insertion_point == 0 and vehicle_info["sequence_of_actions"]:
                    if vehicle_info["sequence_of_actions"][insertion_point]["started_at"] is not None:
                        continue
                if not vehicle_info["sequence_of_actions"]:
                    impact = -1
                else:
                    impact = self.evaluate_insertion_impact(vehicle_id, insertion_point, customer_id, restaurant_id, obs)
                if impact < min_impact:
                    min_impact = impact
                    best_vehicle = vehicle_id
                    best_insertion_point = insertion_point            

        return best_vehicle, best_insertion_point

    def evaluate_insertion_impact(self, vehicle_id, insertion_point, customer_id, restaurant_id, obs):
        vehicle_info = obs["vehicle_info"][vehicle_id]
        old_route = vehicle_info["sequence_of_actions"]

        # Calculate old delay
        old_delay = self.calculate_route_delay(vehicle_id, old_route, obs)

        # Creating a tentative route by deep copying and inserting new actions
        tentative_route = copy.deepcopy(old_route)

        # Use consolidated time estimates for pickup and delivery actions
        pickup_stop = self.create_stop("pickup", restaurant_id, None, obs, vehicle_id)
        delivery_stop = self.create_stop("delivery", customer_id, restaurant_id, obs, vehicle_id)



        pickup_action = {
            "type": "pickup",
            "destination": pickup_stop.destination,
            "restaurant_id": restaurant_id,
            "customer_id": None,
            "start_at": -1,
            "started_at": None,
            "estimated_total_time": pickup_stop.estimated_total_time,
            "actual_total_time": pickup_stop.actual_total_time,
            "orders_to_pickup": [customer_id]
        }

        delivery_action = {
            "type": "delivery",
            "destination": delivery_stop.destination,
            "restaurant_id": None,
            "customer_id": customer_id,
            "start_at": -1,
            "started_at": None,
            "estimated_total_time": delivery_stop.estimated_total_time,
            "actual_total_time": pickup_stop.actual_total_time,
            "orders_to_deliver": [restaurant_id]
        }

        # Insert actions into the tentative route
        tentative_route.insert(insertion_point, pickup_action)
        tentative_route.insert(insertion_point + 1, delivery_action)

        # Repair the tentative route
        tentative_route = self.repair_route(tentative_route, obs)

        # Calculate tentative delay
        tentative_delay = self.calculate_route_delay(vehicle_id, tentative_route, obs)

        # Return the impact as the difference in delay
        return tentative_delay - old_delay


    def create_stop(self, stop_type, destination_id, customer_id, obs, vehicle_id):
        vehicle_current_location = obs["vehicle_info"][vehicle_id]["next_location"]
        destination_location = None
        estimated_travel_time = None

        if stop_type == "pickup":
            destination_location = obs["restaurant_info"][destination_id]["location"]
            estimated_travel_time = self.get_travel_time(vehicle_current_location, destination_location)

        elif stop_type == "delivery":
            destination_location = obs["customer_info"][destination_id]["location"]
            estimated_travel_time = self.get_travel_time(vehicle_current_location, destination_location)

        stop = Stop(
            stop_type=stop_type,
            destination=destination_location,
            restaurant_id=destination_id if stop_type == "pickup" else None,
            customer_id=customer_id if stop_type == "delivery" else None,
            start_at=-1,
            estimated_travel_time=estimated_travel_time,
            actual_travel_time=estimated_travel_time,  
            estimated_park_time=0,  # Default to 0, as it is included in total time
            actual_park_time=0,    # Default to 0, as it is included in total time
            estimated_wait_time=0,  # Default to 0, as it is included in total time
            actual_wait_time=0,    # Default to 0, as it is included in total time
            orders_to_pickup=[customer_id] if stop_type == "pickup" else None
        )

        return stop




    

    def calculate_route_delay(self, vehicle_id, route, obs):
        delay = 0
        if not route:

            return delay

        current_time = route[0]["started_at"] if route[0]["started_at"] is not None else obs["current_time"]

        for i, stop in enumerate(route):
            travel_time = self.get_travel_time(obs["vehicle_info"][vehicle_id]["next_location"], stop["destination"]) if i == 0 else self.get_travel_time(route[i - 1]["destination"], stop["destination"])
            current_time += travel_time + stop.get("estimated_park_time", 0) + stop.get("estimated_wait_time", 0)

            if stop["type"] == "delivery":
                customer_id = stop["customer_id"]
                customer_expected_time = obs["customer_info"][customer_id]["expected_delivery_time"] if customer_id in obs["customer_info"] else float("inf")
                if current_time > customer_expected_time:
                    delay += current_time - customer_expected_time

        return delay


    def repair_route(self, route, obs):
        repaired_route = copy.deepcopy(route)
        estimated_time = obs["current_time"]
        actual_time = obs["current_time"]

        for stop_index, stop in enumerate(repaired_route):
            if isinstance(stop, dict):  # Check if stop is already a dictionary
                stop_dict = stop.copy()  # Do a shallow copy instead of _asdict()
            else:
                stop_dict = stop._asdict()  # Convert namedtuple to dictionary

            #if stop_dict["type"] == "pickup":
            if isinstance(stop_dict, Stop) and stop_dict.type == "pickup":
                restaurant_info = obs["restaurant_info"][stop_dict["destination"]]
                customer_ids = stop_dict["orders_to_pickup"]
                
                estimated_wait_time = self.get_estimated_waiting_time(restaurant_info, customer_ids, estimated_time)
                actual_wait_time = self.get_actual_waiting_time(restaurant_info, customer_ids, actual_time)

                stop_dict["estimated_wait_time"] = estimated_wait_time
                stop_dict["actual_wait_time"] = actual_wait_time
                estimated_time += estimated_wait_time
                actual_time += actual_wait_time

                repaired_route[stop_index] = Stop(**stop_dict)

        return repaired_route
        

    def get_estimated_waiting_time(self, restaurant_info, customer_ids, estimated_time):
        queue = restaurant_info["orders_in_queue"]
        estimated_time_queue = restaurant_info["estimated_finish_times"]
        relevant_orders = [order for order in queue if order["customer_id"] in customer_ids]
        if not relevant_orders:
            return 0
        max_index = max([queue.index(order) for order in relevant_orders])
        return max(0, estimated_time_queue[max_index] - estimated_time)

    def get_actual_waiting_time(self, restaurant_info, customer_ids, actual_time):
        queue = restaurant_info["orders_in_queue"]
        actual_time_queue = restaurant_info["actual_finish_times"]
        relevant_orders = [order for order in queue if order["customer_id"] in customer_ids]
        if not relevant_orders:
            return 0
        max_index = max([queue.index(order) for order in relevant_orders])
        return max(0, actual_time_queue[max_index] - actual_time)