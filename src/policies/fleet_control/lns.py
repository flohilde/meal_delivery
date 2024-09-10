import numpy as np
from src.templates import Policy, Observation
from collections import defaultdict
from typing import Dict, List
import copy
import itertools
import operator
from src.utils import route_to_array


class LNS(Policy):
    # TODO: convert routes to type that is faster to copy

    def __init__(self, tt_matrix, expected_parking_time, expected_cook_time, alpha=1.0):
        super().__init__()
        self.tt_matrix = tt_matrix
        self.expected_parking_time = expected_parking_time
        self.expected_cook_time = expected_cook_time
        self.alpha = alpha
        self.n_steps = 10
        self.n_steps_wo_improvement = 3
        self.operator_weights = [0.0, 1.0]

    def get_travel_time(self, location1, location2):
        return self.tt_matrix[str(location1)][str(location2)]

    def act(self, obs: Observation) -> Dict:
        route_plan, restaurant_plan, route_plan_cost = self.get_initial_solution(obs)
        route_plan, restaurant_plan = self.search(obs, route_plan, restaurant_plan, route_plan_cost)

        return {"vehicle_action": route_plan, "restaurant_action": restaurant_plan}

    def get_initial_solution(self, obs):
        """
        Inserts the orders grouped by customer greedily into the route plan.
        """
        # cluster orders by customer
        clustered_orders = [list(group) for _, group in itertools.groupby(obs["unassigned_orders"],
                                                                          operator.itemgetter(0))]
        clustered_orders = [orders for orders in clustered_orders if orders]

        # create feasible restaurant plan
        restaurant_plan = copy.deepcopy({restaurant_id: restaurant_info["orders_in_queue"]
                                         for restaurant_id, restaurant_info in
                                         obs["restaurant_info"].items()})
        # create feasible route plan
        route_plan = copy.deepcopy({vehicle_id: vehicle_info["sequence_of_actions"]
                                    for vehicle_id, vehicle_info in obs["vehicle_info"].items()})

        # if we do not have any new orders, return the old plan
        if not clustered_orders:
            return route_plan, restaurant_plan, 0

        # else: integrate the new orders in the restaurant plan and route plan
        for order in clustered_orders:
            for c_id, r_id in order:
                restaurant_plan[r_id].append({"customer_id": c_id, "start_at": -1})

        for order in clustered_orders:

            best_rp_cost = np.inf
            best_rp = copy.deepcopy(route_plan)

            # shuffle vehicles so that a random vehicle gets assigned if delay is the same for all
            vehicle_keys = list(route_plan.keys())
            np.random.shuffle(vehicle_keys)
            for v_key in vehicle_keys:

                len_old_route = len(route_plan[v_key])
                t_rp = copy.deepcopy(route_plan)

                # append one pickup stop
                self._insert_pickup_stop(obs, t_rp, [order[0][0]], order[0][1], v_key, len_old_route)
                self._insert_delivery_stop(obs, t_rp, order[0][0], v_key, len_old_route + 1)

                # we greedily insert the other pickup stops, if there are any
                for c_id, r_id in order[1:]:

                    # append order to the restaurant's queue
                    best_t_rp = copy.deepcopy(t_rp)
                    best_pickup_insertion_cost = np.inf

                    for p_ix in range(len_old_route, len(t_rp[v_key])):

                        tt_rp = copy.deepcopy(t_rp)
                        self._insert_pickup_stop(obs, tt_rp, [c_id], r_id, v_key, p_ix)
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

        return route_plan, restaurant_plan, best_rp_cost

    def search(self, obs, route_plan, restaurant_plan, route_plan_cost):

        current_route_plan = copy.deepcopy(route_plan)
        current_restaurant_plan = copy.deepcopy(restaurant_plan)
        current_route_plan_cost = np.inf
        k = 0
        n = 0

        while k < self.n_steps:

            operator_id = np.random.choice(list(range(len(self.operator_weights))), p=self.operator_weights)

            if operator_id == 0:
                current_route_plan, current_restaurant_plan, current_route_plan_cost \
                    = self._bundle_random_orders(obs, current_route_plan, current_restaurant_plan)

            # remove n orders and then insert them back into the route
            if operator_id == 1:
                n = np.ceil(np.random.random() * len([c for c in obs["customer_info"]
                            if len(obs["customer_info"][c]["restaurant_choice"]) == 1])).astype(int)
                removed_orders = []
                for _ in range(max(n, 1)):
                    customer_to_remove = self._choose_random_order(obs, filter_orders="multi")
                    removed_order, current_route_plan, current_restaurant_plan \
                        = self._remove_order(obs, current_route_plan, current_restaurant_plan, customer_to_remove)
                    if removed_order:
                        removed_orders.append(removed_order)

                for removed_order in removed_orders:

                    if len(removed_order[1]) == 1:
                        current_route_plan, current_restaurant_plan, current_route_plan_cost \
                            = self._repair_singleorder(obs, removed_order, current_route_plan, current_restaurant_plan)
                    else:
                        current_route_plan, current_restaurant_plan, current_route_plan_cost \
                            = self._repair_multiorder(obs, removed_order, current_route_plan, current_restaurant_plan)

            k += 1
            # we found a new best route plan
            if current_route_plan_cost < route_plan_cost:
                n = 0
                route_plan_cost = current_route_plan_cost
                route_plan = copy.deepcopy(current_route_plan)
                restaurant_plan = copy.deepcopy(current_restaurant_plan)
            else:
                n += 1
                if n >= self.n_steps_wo_improvement:
                    return route_plan, restaurant_plan

        return route_plan, restaurant_plan

    def _evaluate_route_plan(self, obs, route_plan) -> float:
        """
        Calculates the convex sum of delay and synchronization delay for a given route plan

        """
        # check if route plan is empty
        if np.all([len(route) == 0 for route in route_plan.values()]):
            return 0

        customers_with_eta = defaultdict(list)
        for _, v_route in route_plan.items():
            for stop in v_route:
                if stop["type"] == "delivery":
                    customers_with_eta[stop["customer_id"]].append(stop["eta"])
        delays_and_synchros = np.array([[np.mean([max(0, eta - obs["customer_info"][c_id]["expected_delivery_time"])
                                                  for eta in etas]),
                                         np.max(etas) - np.min(etas)]
                                        for c_id, etas in customers_with_eta.items()])
        return self.alpha * np.mean(delays_and_synchros[:, 0]) + (1 - self.alpha) * np.mean(delays_and_synchros[:, 1])

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

    def _insert_pickup_stop(self, obs: Observation, route_plan: Dict, customer_ids: List[str],
                            restaurant_id: str, vehicle_id: str, insertion_index: int):
        pickup_stop = {"type": "pickup",
                       "origin": None,
                       "destination": obs["restaurant_info"][restaurant_id]["location"],
                       "restaurant_id": restaurant_id,
                       "customer_id": None,
                       "start_at": -1,
                       "started_at": None,
                       "estimated_time_required": 0,  # this is automatically adjusted when repairing the route
                       "orders_to_pickup": [c_id for c_id in customer_ids]}
        route_plan[vehicle_id].insert(insertion_index, pickup_stop)

        # repair timings of route
        route_plan[vehicle_id] = self._update_route_timing(obs, route_plan[vehicle_id], vehicle_id)

    @staticmethod
    def _remove_order(obs, route_plan, restaurant_plan, customer_to_remove):

        if customer_to_remove is None:
                    return [], route_plan, restaurant_plan

        # remove the order corresponding to the customer
        order_ineligible = True
        destroyed_route_plan = {}
        for v_key, v_route in route_plan.items():
            destroyed_route_plan[v_key] = []

            for stop in v_route:
                new_stop = copy.deepcopy(stop)

                # check if vehicle stops at the common restaurant:
                if new_stop["type"] == "pickup":
                    if customer_to_remove in new_stop["orders_to_pickup"]:
                        new_stop["orders_to_pickup"].remove(customer_to_remove)

                        # make sure there is a pickup stop for the customer that has not yet been started
                        # otherwise the order is ineligible for removal
                        if stop["started_at"] is not None:
                            break
                        else:
                            order_ineligible = False

                    # only append pickup stop to the new route if orders are left
                    if new_stop["orders_to_pickup"]:
                        destroyed_route_plan[v_key].append(new_stop)

                # check for delivery stops
                else:
                    if new_stop["customer_id"] != customer_to_remove:
                        destroyed_route_plan[v_key].append(new_stop)

        if order_ineligible:
            return [], route_plan, restaurant_plan

        removed_order = (customer_to_remove, obs["customer_info"][customer_to_remove]["restaurant_choice"])
        return removed_order, destroyed_route_plan, restaurant_plan

    def _choose_random_order(self, obs, filter_orders=None):
        """Randomly choose among single orders (filter="multi"), among all multiorders (filter="single),
        or all orders (filter=None)."""

        # create a list of all single orders
        if filter_orders == "multi":
            orders = [c_id for c_id in obs["customer_info"].keys()
                      if len(obs["customer_info"][c_id]["restaurant_choice"]) == 1]
        # create a list of all multi orders
        elif filter_orders == "single":
            orders = [c_id for c_id in obs["customer_info"].keys()
                      if len(obs["customer_info"][c_id]["restaurant_choice"]) > 1]
        # create a list of all orders
        else:
            orders = [c_id for c_id in obs["customer_info"].keys()]

        # chose a customer randomly from the remaining list
        if orders:
            customer_to_remove = np.random.choice(orders)
            return customer_to_remove
        return None

    def _choose_most_time_consuming_order(self):
        # remove the order that adds the most time to a route
        pass

    def _choose_shaw(self):
        pass

    def _bundle_random_orders(self, obs, route_plan, restaurant_plan):
        """
        Destroy + Repair Operator.
        Selects a random pair of orders that can be bundled but are not.
        It removes these orders and then re-inserts them as bundle.
        Also optimizes the restaurant plan accordingly.
        """

        # first step: create a list of bundling candidates for each restaurant
        bundling_candidates = [[(order["customer_id"], r_id) for order in rest_plan if len(rest_plan) > 1]
                               for r_id, rest_plan in restaurant_plan.items()]
        bundling_candidates = [candidate for candidate in bundling_candidates if candidate]
        if not bundling_candidates:
            return route_plan, restaurant_plan, self._evaluate_route_plan(obs, route_plan)

        # sample a random pair of orders to bundle
        random_ix = np.random.randint(low=0, high=len(bundling_candidates), size=None)
        ix_rc = np.random.choice(list(range(len(bundling_candidates[random_ix]))), size=2, replace=False)
        rc0 = bundling_candidates[random_ix][ix_rc[0]]
        rc1 = bundling_candidates[random_ix][ix_rc[1]]

        # attempt to remove them while also checking if it is a multi-order
        destroyed_route_plan = {}
        for v_key, v_route in route_plan.items():
            destroyed_route_plan[v_key] = []

            for stop in v_route:
                new_stop = copy.deepcopy(stop)

                # check if vehicle stops at the common restaurant:
                if new_stop["type"] == "pickup":
                    if new_stop["restaurant_id"] == rc0[1]:
                        # check if vehicle picks up an order from exactly one of the two customers
                        if rc0[0] in new_stop["orders_to_pickup"]:
                            new_stop["orders_to_pickup"].remove(rc0[0])
                        if rc1[0] in new_stop["orders_to_pickup"]:
                            new_stop["orders_to_pickup"].remove(rc1[0])
                        # only append pickup stop to the new route if orders are left
                        if new_stop["orders_to_pickup"]:
                            destroyed_route_plan[v_key].append(new_stop)
                    else:
                        destroyed_route_plan[v_key].append(new_stop)

                # check for delivery stops
                else:
                    if new_stop["customer_id"] == rc0[0]:
                        if len(obs["customer_info"][rc0[0]]["restaurant_choice"]) > 1:
                            destroyed_route_plan[v_key].append(new_stop)
                    elif new_stop["customer_id"] == rc1[0]:
                        if len(obs["customer_info"][rc1[0]]["restaurant_choice"]) > 1:
                            destroyed_route_plan[v_key].append(new_stop)
                    else:
                        destroyed_route_plan[v_key].append(new_stop)

        # reinsert the bundled pickup stop and two delivery stops
        best_rp_cost = np.inf
        best_rp = copy.deepcopy(destroyed_route_plan)

        # shuffle vehicles so that a random vehicle gets assigned if delay is the same for all
        vehicle_keys = list(route_plan.keys())
        np.random.shuffle(vehicle_keys)
        for v_key in vehicle_keys:

            earliest_insertion_point = 0
            if destroyed_route_plan[v_key]:
                if destroyed_route_plan[v_key][0]["started_at"] is not None:
                    earliest_insertion_point = 1

            for insertion_index_pickup in range(earliest_insertion_point, len(destroyed_route_plan[v_key]) + 1):
                t_rp = copy.deepcopy(destroyed_route_plan)

                # modifier required if we merge instead of append the pickup
                delivery_insertion_modifier = 1

                # append one pickup stop or merge with previous pickup stop
                if insertion_index_pickup != 0:
                    if t_rp[v_key][insertion_index_pickup - 1]["restaurant_id"] == rc0[1]:
                        t_rp[v_key][insertion_index_pickup - 1]["orders_to_pickup"].extend([rc0[0], rc1[0]])
                        delivery_insertion_modifier = 0
                    else:
                        self._insert_pickup_stop(obs, t_rp, [rc0[0], rc1[0]], rc0[1], v_key, insertion_index_pickup)
                else:
                    self._insert_pickup_stop(obs, t_rp, [rc0[0], rc1[0]], rc0[1], v_key, insertion_index_pickup)

                # remove existing delivery stop, because we will insert a new one
                t_rp[v_key] = [stop for stop in t_rp[v_key] if not stop["customer_id"] in [rc0[0], rc1[0]]]

                # next, we need to insert both deliveries
                best_tt_rp = copy.deepcopy(t_rp)
                best_tt_rp_cost = np.inf

                # calculate earliest delivery index to make sure there is no pickup without subsequent delivery
                lowest_delivery_index = insertion_index_pickup + delivery_insertion_modifier
                # in case of a multi-order, we also have to check for other pickups
                if len(obs["customer_info"][rc0[0]]["restaurant_choice"]) > 1:
                    for d_ix in range(insertion_index_pickup + 1, len(t_rp[v_key])):
                        if t_rp[v_key][d_ix]["type"] == "pickup":
                            if rc0[0] in t_rp[v_key][d_ix]["orders_to_pickup"]:
                                lowest_delivery_index = d_ix + 1

                # iterate over possible insertion points
                for insertion_index_delivery_rc0 in range(lowest_delivery_index,
                                                          max(len(t_rp[v_key]) + 1, lowest_delivery_index + 1)):
                    tt_rp = copy.deepcopy(t_rp)

                    self._insert_delivery_stop(obs, tt_rp, rc0[0], v_key, insertion_index_delivery_rc0)

                    best_ttt_rp = copy.deepcopy(tt_rp)
                    best_ttt_rp_cost = np.inf

                    # calculate earliest delivery index to make sure there is no pickup without subsequent delivery
                    lowest_delivery_index = insertion_index_pickup + delivery_insertion_modifier
                    # in case of a multi-order, we also have to check for other pickups
                    if len(obs["customer_info"][rc1[0]]["restaurant_choice"]) > 1:
                        for d_ix in range(lowest_delivery_index, len(tt_rp[v_key])):
                            if tt_rp[v_key][d_ix]["type"] == "pickup":
                                if rc1[0] in tt_rp[v_key][d_ix]["orders_to_pickup"]:
                                    lowest_delivery_index = d_ix + 1

                    # iterate over possible insertion points
                    for insertion_index_delivery_rc1 in range(lowest_delivery_index,
                                                              max(len(tt_rp[v_key]) + 1, lowest_delivery_index + 1)):
                        ttt_rp = copy.deepcopy(tt_rp)

                        self._insert_delivery_stop(obs, ttt_rp, rc1[0], v_key, insertion_index_delivery_rc1)

                        delivery_cost = self._evaluate_route_plan(obs, ttt_rp)
                        if delivery_cost < best_ttt_rp_cost:
                            best_ttt_rp = copy.deepcopy(ttt_rp)
                            best_ttt_rp_cost = delivery_cost

                    tt_rp = best_ttt_rp
                    delivery_cost = self._evaluate_route_plan(obs, tt_rp)
                    if delivery_cost < best_tt_rp_cost:
                        best_tt_rp = copy.deepcopy(tt_rp)
                        best_tt_rp_cost = delivery_cost

                t_rp = best_tt_rp
                delivery_cost = self._evaluate_route_plan(obs, t_rp)
                if delivery_cost < best_rp_cost:
                    best_rp = copy.deepcopy(t_rp)
                    best_rp_cost = delivery_cost

        # TODO: optimize restaurant plan
        return best_rp, restaurant_plan, best_rp_cost

    def _repair_singleorder(self, obs, order, route_plan, restaurant_plan):
        """
        Inserts a single-order into the route plan.
        """

        if not order:
            return route_plan, restaurant_plan, np.inf

        # reinsert the bundled pickup stop and two delivery stops
        best_rp_cost = np.inf
        best_rp = copy.deepcopy(route_plan)

        # shuffle vehicles so that a random vehicle gets assigned if delay is the same for all
        vehicle_keys = list(route_plan.keys())
        np.random.shuffle(vehicle_keys)
        for v_key in vehicle_keys:

            # make sure to only insert after started actions
            earliest_insertion_point = 0
            if route_plan[v_key]:
                if route_plan[v_key][0]["started_at"] is not None:
                    earliest_insertion_point = 1

            for insertion_index_pickup in range(earliest_insertion_point, len(route_plan[v_key]) + 1):
                t_rp = copy.deepcopy(route_plan)

                # modifier required if we merge instead of append a pickup
                delivery_insertion_modifier = 1

                # append one pickup stop or merge with previous
                if insertion_index_pickup != 0:
                    if t_rp[v_key][insertion_index_pickup - 1]["restaurant_id"] == order[1][0]:
                        t_rp[v_key][insertion_index_pickup - 1]["orders_to_pickup"].append(order[0])
                        delivery_insertion_modifier = 0
                    else:
                        self._insert_pickup_stop(obs, t_rp, [order[0]], order[1][0], v_key, insertion_index_pickup)
                else:
                    self._insert_pickup_stop(obs, t_rp, [order[0]], order[1][0], v_key, insertion_index_pickup)

                # iterate over possible delivery insertion points
                for insertion_index_delivery in range(insertion_index_pickup + delivery_insertion_modifier,
                                                      len(t_rp[v_key]) + 1):
                    tt_rp = copy.deepcopy(t_rp)

                    self._insert_delivery_stop(obs, tt_rp, order[0], v_key, insertion_index_delivery)

                    delivery_cost = self._evaluate_route_plan(obs, tt_rp)
                    if delivery_cost < best_rp_cost:
                        best_rp = copy.deepcopy(tt_rp)
                        best_rp_cost = delivery_cost

        # TODO: optimize restaurant plan
        return best_rp, restaurant_plan, best_rp_cost

    def _repair_multiorder(self, obs, order, route_plan, restaurant_plan):

        best_rp_cost = np.inf
        best_rp = copy.deepcopy(route_plan)

        for v_key in route_plan.keys():

            # special case if route is empty:
            if not route_plan[v_key]:

                t_rp = copy.deepcopy(route_plan)
                self._insert_pickup_stop(obs, t_rp, order[0], order[1][0], v_key, 0)
                self._insert_delivery_stop(obs, t_rp, order[0], v_key, 1)
                best_t_rp = copy.deepcopy(t_rp)

                for r_id in order[1][1:]:

                    best_pickup_insertion_cost = np.inf

                    for p_ix in range(len(t_rp) - 1):

                        tt_rp = copy.deepcopy(t_rp)
                        self._insert_pickup_stop(obs, tt_rp, order[0], r_id, v_key, p_ix)
                        pickup_cost = self._evaluate_route_plan(obs, tt_rp)
                        if pickup_cost < best_pickup_insertion_cost:
                            best_t_rp = tt_rp
                            best_pickup_insertion_cost = pickup_cost

                    # adapt tentative delivery route to include new pickup
                    t_rp = copy.deepcopy(best_t_rp)

                delivery_cost = self._evaluate_route_plan(obs, t_rp)
                if delivery_cost < best_rp_cost:
                    best_rp = copy.deepcopy(t_rp)
                    best_rp_cost = delivery_cost
                continue

            # in the case that the route is non-empty
            # first try every delivery insertion
            for d_ix in range(len(route_plan[v_key])):

                # create a new route candidate for the given vehicle
                t_rp = copy.deepcopy(route_plan)

                # do not insert before actions that were already started
                if t_rp[v_key][d_ix]["started_at"] is not None:
                    continue

                self._insert_delivery_stop(obs, t_rp, order[0], v_key, d_ix)

                # before we can evaluate this candidate, we must find the best insertions for all the pickups
                for i, r_id in enumerate(order[1]):

                    best_pickup_insertion_cost = np.inf
                    best_t_rp = copy.deepcopy(t_rp)

                    # make sure to only insert after started actions
                    earliest_insertion_point = 0
                    if t_rp[v_key]:
                        if t_rp[v_key][0]["started_at"] is not None:
                            earliest_insertion_point = 1

                    for p_ix in range(earliest_insertion_point, d_ix + i):

                        tt_rp = copy.deepcopy(t_rp)
                        # always consolidate if previous stop is the same restaurant
                        if tt_rp[v_key][p_ix - 1]["restaurant_id"] == r_id:
                            tt_rp[v_key][p_ix - 1]["orders_to_pickup"].append((order[0], r_id))
                            tt_rp[v_key] = self._update_route_timing(obs, tt_rp[v_key], v_key)
                            pickup_cost = self._evaluate_route_plan(obs, tt_rp)
                            if pickup_cost < best_pickup_insertion_cost:
                                best_t_rp = tt_rp
                                best_pickup_insertion_cost = pickup_cost

                        # else insert a new stop
                        else:
                            self._insert_pickup_stop(obs, tt_rp, order[0], r_id,
                                                     v_key, p_ix)
                            pickup_cost = self._evaluate_route_plan(obs, tt_rp)
                            if pickup_cost < best_pickup_insertion_cost:
                                best_t_rp = tt_rp
                                best_pickup_insertion_cost = pickup_cost

                    # adapt tentative delivery route to include new pickup
                    t_rp = copy.deepcopy(best_t_rp)

                delivery_cost = self._evaluate_route_plan(obs, t_rp)
                if delivery_cost < best_rp_cost:
                    best_rp = t_rp

        # adapt best route plan
        route_plan = copy.deepcopy(best_rp)

        return route_plan, restaurant_plan, best_rp_cost

    def _repair_regret_insertion(self):
        pass

    def _create_action_from_route_plan(self):
        pass

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
            if stop_index != 0:
                stop["origin"] = repaired_route[stop_index - 1]["destination"]
            estimated_time += self.get_travel_time(stop["origin"], stop["destination"]) + self.expected_parking_time

            # for pickups, we have to consider the synchronization of restaurant and vehicle
            if stop["type"] == "pickup":

                # check if orders already in restaurant queue, else append them and update estimated time_queue
                restaurant_info = obs["restaurant_info"][stop["restaurant_id"]]
                customer_ids = stop["orders_to_pickup"]
                queue = restaurant_info["orders_in_queue"]
                estimated_time_queue = restaurant_info["estimated_finish_times"]
                orders_to_prepare = [order["customer_id"] for order in queue]

                for c_id in customer_ids:

                    # add customer to restaurant queue if necessary
                    if c_id not in orders_to_prepare:
                        order = {"customer_id": c_id,
                                 "start_at": -1,
                                 "finished_at": None,
                                 "estimated_preparation_time": self.expected_cook_time}
                        queue.append(order)
                        orders_to_prepare.append(c_id)

                        # updated time queue
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


if __name__ == "__main__":
    import configparser
    from src.state import MealDeliveryMDP
    from src.policies.fleet_control.simple_assignment import SimpleAssignmentPolicy
    import os

    os.chdir("../..")

    config = configparser.ConfigParser(allow_no_value=True)
    config.read('../data/instances/multi_order/iowa_110_5_55_80.ini')
    env = MealDeliveryMDP(config, seed=42)
    policy = SimpleAssignmentPolicy(env.tt_matrix, env.expected_parking_time, env.expected_cook_time)

    obs = env.reset()
    k = 0
    while True:

        # fleet control action and state transition
        action = policy.act(obs)
        obs, cost, done, info = env.step(action)
        k += 1

        if len(obs["new_customer_info"]["restaurant_choice"]) > 1 and k > 30:
            break

    # after making some steps, we use the state as a test case for the lns
    print(obs["unassigned_orders"])
    print({v_key: v_info["sequence_of_actions"] for v_key, v_info in obs["vehicle_info"].items()})
    lns = LNS(env.tt_matrix, env.expected_parking_time, env.expected_cook_time, alpha=0.5)
    route_plan, restaurant_plan = lns.get_initial_solution(obs)
    print()
    print(route_plan)
