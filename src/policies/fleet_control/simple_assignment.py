import copy
from typing import Dict
from src.templates import Observation, Policy
import itertools
import operator
import numpy as np
from collections import defaultdict


class SimpleAssignmentPolicy(Policy):
    r"""
    Implementation of a simple assignment policy that assigns each
    unassigned order to the vehicle with the lowest busy time.
    New stops are appended to the end of the assigned vehicle's route.
    Orders are appended to the end of the restaurant's queue.
    No postponement of order assignments is considered.
    """

    def __init__(self, tt_matrix, expected_parking_time, expected_cook_time):
        Policy.__init__(self)
        self.tt_matrix = tt_matrix
        self.expected_parking_time = expected_parking_time
        self.expected_cook_time = expected_cook_time

    def get_travel_time(self, location1, location2):
        return self.tt_matrix[str(location1)][str(location2)]

    def act(self, obs: Observation) -> Dict:
        r"""
        Creates an action based on an observation.
        """
        action = {
            "vehicle_action": {},
            "restaurant_action": {}
        }

        # cluster orders by multi orders
        clustered_orders = [list(group) for _, group in itertools.groupby(obs["unassigned_orders"],
                                                                          operator.itemgetter(0))]
        route_plan = copy.deepcopy({vehicle_id: vehicle_info["sequence_of_actions"]
                                    for vehicle_id, vehicle_info in obs["vehicle_info"].items()})
        restaurant_plan = copy.deepcopy({restaurant_id: restaurant_info["orders_in_queue"]
                                         for restaurant_id, restaurant_info in
                                         obs["restaurant_info"].items()})

        for multi_order in clustered_orders:

            best_rp_cost = np.inf
            best_rp = copy.deepcopy(route_plan)

            # append orders to restaurant queues
            for c_id, r_id in multi_order:
                restaurant_plan[r_id].append({"customer_id": c_id, "start_at": -1})

            # shuffle vehicles so that a random vehicle gets assigned if delay is the same for all
            vehicle_keys = list(route_plan.keys())
            np.random.shuffle(vehicle_keys)
            for v_key in vehicle_keys:

                len_old_route = len(route_plan[v_key])
                t_rp = copy.deepcopy(route_plan)

                # append one pickup stop
                self._insert_pickup_stop(obs, t_rp, multi_order[0][0], multi_order[0][1],
                                         v_key, len_old_route)
                self._insert_delivery_stop(obs, t_rp, multi_order[0][0],
                                           v_key, len_old_route + 1)

                # we greedily insert the other pickup stops, if there are any
                for c_id, r_id in multi_order[1:]:

                    # append order to the restaurant's queue
                    best_t_rp = copy.deepcopy(t_rp)
                    best_pickup_insertion_cost = np.inf

                    for p_ix in range(len_old_route, len(t_rp[v_key]) - 1):

                        tt_rp = copy.deepcopy(t_rp)
                        self._insert_pickup_stop(obs, tt_rp, c_id, r_id,
                                                 v_key, p_ix)
                        pickup_cost = self._evaluate_route_plan(obs, tt_rp)
                        if pickup_cost < best_pickup_insertion_cost:
                            best_t_rp = copy.deepcopy(tt_rp)
                            best_pickup_insertion_cost = pickup_cost

                    # adapt tentative delivery route to include new pickup
                    t_rp = copy.deepcopy(best_t_rp)

                delivery_cost = self._evaluate_route_plan(obs, t_rp)
                if delivery_cost < best_rp_cost:
                    best_rp = copy.deepcopy(t_rp)
                    best_rp_cost = delivery_cost

            route_plan = copy.deepcopy(best_rp)

        # update restaurant queues
        action["vehicle_action"] = route_plan
        action["restaurant_action"] = restaurant_plan
        return action

    def _insert_delivery_stop(self, obs, route_plan, customer_id, vehicle_id, insertion_index):
        delivery_stop = {"type": "delivery",
                         "origin": None,
                         "destination": obs["customer_info"][customer_id]["location"],
                         "restaurant_id": None,
                         "customer_id": customer_id,
                         "start_at": -1,
                         "started_at": None,
                         "estimated_time_required": 0,  # this is automatically adjusted when repairing the route
                         "orders_to_pickup": None}
        route_plan[vehicle_id].insert(insertion_index, delivery_stop)

        # repair timings of route
        route_plan[vehicle_id] = self._update_route_timing(obs, route_plan[vehicle_id], vehicle_id)
        return route_plan

    def _insert_pickup_stop(self, obs, route_plan, customer_id, restaurant_id, vehicle_id, insertion_index):
        pickup_stop = {"type": "pickup",
                       "origin": None,
                       "destination": obs["restaurant_info"][restaurant_id]["location"],
                       "restaurant_id": restaurant_id,
                       "customer_id": None,
                       "start_at": -1,
                       "started_at": None,
                       "estimated_time_required": 0,  # this is automatically adjusted when repairing the route
                       "orders_to_pickup": [customer_id]}
        route_plan[vehicle_id].insert(insertion_index, pickup_stop)

        # repair timings of route
        route_plan[vehicle_id] = self._update_route_timing(obs, route_plan[vehicle_id], vehicle_id)
        return route_plan

    def _update_route_timing(self, obs, route, vehicle_id):
        """
        Repairs the timing of stops and adds the ETA to each delivery stop.
        """
        repaired_route = copy.deepcopy(route)
        estimated_time = obs["current_time"]

        # iterate through the route and update the required time and eta of each stop
        for stop_index, stop in enumerate(repaired_route):

            # if the stop has started, we go back in time
            if stop_index == 0:
                if stop["started_at"] is None:
                    stop["origin"] = obs["vehicle_info"][vehicle_id]["next_location"]
                else:
                    estimated_time = route[0]["started_at"]

            # we must consider if the stop has a time at which it should be started earliest
            estimated_time = max(estimated_time, stop["start_at"])

            # we calculate the driving time and waiting time
            if stop["origin"] is None:
                stop["origin"] = repaired_route[stop_index - 1]["destination"]
            estimated_time += self.get_travel_time(stop["origin"], stop["destination"]) + self.expected_parking_time

            # for pickups, we have to consider the synchronization of restaurant and vehicle
            estimated_pickup_time = 0
            if stop["type"] == "pickup":

                # check if orders already in restaurant queue, else append them and update estimated time_queue
                restaurant_info = obs["restaurant_info"][stop["restaurant_id"]]
                customer_ids = stop["orders_to_pickup"]
                queue = restaurant_info["orders_in_queue"]
                estimated_time_queue = restaurant_info["estimated_finish_times"]
                orders_to_prepare = [order["customer_id"] for order in queue]
                for c_id in customer_ids:
                    if c_id not in orders_to_prepare:
                        order = {"customer_id": c_id,
                                 "start_at": -1,
                                 "finished_at": None,
                                 "estimated_preparation_time": self.expected_cook_time}
                        queue.append(order)
                        orders_to_prepare.append(c_id)
                        if estimated_time_queue:
                            estimated_time_queue.append(estimated_time_queue[-1] + self.expected_cook_time)
                        else:
                            estimated_time_queue.append(obs["current_time"] + self.expected_cook_time)

                relevant_orders = [c_id for c_id in orders_to_prepare if c_id in customer_ids]
                max_index = max([orders_to_prepare.index(c_id) for c_id in relevant_orders])
                estimated_time = max(estimated_time, estimated_time_queue[max_index])

            stop["eta"] = estimated_time
            repaired_route[stop_index] = stop

        return repaired_route

    @staticmethod
    def _evaluate_route_plan(obs, route_plan) -> float:
        """
        Calculates the convex sum of delay and synchronization delay for a given route plan

        """
        customers_with_eta = defaultdict(list)
        for _, v_route in route_plan.items():
            for stop in v_route:
                if stop["type"] == "delivery":
                    customers_with_eta[stop["customer_id"]].append(stop["eta"])
        delays = np.array([[np.mean([max(0, eta - obs["customer_info"][c_id]["expected_delivery_time"])
                                     for eta in etas])]
                           for c_id, etas in customers_with_eta.items()])
        return float(np.sum(delays))
