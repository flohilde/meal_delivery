import numpy as np
from templates import Policy, Observation
from typing import Dict, List
import copy
from utils import route_to_array, array_to_route
# from scipy.stats import norm
from scipy.special import ndtr

N = ndtr


# N = norm(0, 1).cdf


class LNS(Policy):
    # Done: convert routes to type that is faster to copy

    def __init__(self, tt_matrix, expected_parking_time, expected_cook_time, var_cook_time,
                 alpha=1.0, buffer=0, synchro_tol=0):
        super().__init__()
        self.tt_matrix = tt_matrix
        self.expected_parking_time = expected_parking_time
        self.expected_cook_time = expected_cook_time
        self.var_cook_time = var_cook_time

        self.alpha = alpha
        self.buffer = buffer
        self.synchro_tol = synchro_tol
        self.vehicle_capacity = 8
        self.n_lns_steps = 4
        self.reset_after_n_steps_wo_improvement = 2

        self.stop_struct = [
            ('type', 'i4'),
            ('origin', 'i4'),
            ('destination', 'i4'),
            ('restaurant_id', 'i4'),
            ('customer_id', 'i4'),
            ('start_at', 'i4'),
            ('started_at', 'i4'),
            ('estimated_time_required', 'i4'),
            ('order_estimated_ready_time', 'i4'),
            ('order_ready_time_sigma', 'f4'),
            ('eta', 'i4'),
            ('eta_lb', 'i4'),
            ('eta_ub', 'i4'),

        ]
        for i in range(self.vehicle_capacity):
            self.stop_struct.append(('orders_to_pickup_{}'.format(i), 'i4'))

    def get_travel_time(self, location1, location2):
        return self.tt_matrix[str(location1)][str(location2)]

    def act(self, obs: Observation) -> Dict:

        obs = self._update_restaurants(obs)
        route_plan, restaurant_plan, route_plan_etas = self._get_initial_solution(obs)
        # route_plan, restaurant_plan = self._search(obs, route_plan, restaurant_plan, route_plan_etas)  # TODO: does not currently work
        route_plan = array_to_route(route_plan, vehicle_capacity=self.vehicle_capacity)

        return {"vehicle_action": route_plan, "restaurant_action": restaurant_plan}

    def _update_restaurants(self, obs):
        if obs["new_customer_info"] is None:
            return obs
        # else we update the restaurant queues corresponding to the new order
        new_customer = obs["new_customer_info"]["name"]
        for r_id in obs["new_customer_info"]["restaurant_choice"]:
            obs["restaurant_info"][r_id]["orders_in_queue"].append({"customer_id": new_customer, "start_at": -1})
            finish_times = obs["restaurant_info"][r_id]["estimated_finish_times"]
            if finish_times:
                finish_times.append(finish_times[-1] + self.expected_cook_time)
            else:
                finish_times.append(obs["current_time"] + self.expected_cook_time)
        return obs

    def _get_initial_solution(self, obs):

        # create feasible restaurant plan and route plan
        restaurant_plan = {restaurant_id: copy.copy(restaurant_info["orders_in_queue"])
                           for restaurant_id, restaurant_info in
                           obs["restaurant_info"].items()}
        route_plan = {vehicle_id: vehicle_info["sequence_of_actions"]
                      for vehicle_id, vehicle_info in obs["vehicle_info"].items()}

        route_plan_etas = {c_id: copy.copy(c_info["estimated_delivery_time"])
                           for c_id, c_info in obs["customer_info"].items()}

        # recast routeplan as dict of arrays to speedup insertion
        route_plan = route_to_array(route_plan, vehicle_capacity=self.vehicle_capacity)

        # if we do not have any new orders, return the old plan
        if not obs["unassigned_orders"]:
            return route_plan, restaurant_plan, 0

        # else: integrate the new orders in the restaurant plan and route plan
        for order in obs["unassigned_orders"]:
            route_plan, restaurant_plan, route_plan_etas = self._insert_order(obs, order, route_plan, restaurant_plan,
                                                                              route_plan_etas)

        return route_plan, restaurant_plan, route_plan_etas

    def _update_route_timing(self, obs, route, vehicle_id):
        """
        Repairs the timing of stops and adds the ETA to each delivery stop.
        """
        estimated_time = obs["current_time"]
        underestimated_time = obs["current_time"]
        overestimated_time = obs["current_time"]
        updated_etas = {}

        # iterate through the route and update the required time and eta of each stop
        for stop_index, stop in enumerate(route):

            # if the stop has started, we go back in time
            if stop_index == 0:
                if stop["started_at"] == -1:
                    stop["origin"] = obs["vehicle_info"][vehicle_id]["next_location"]
                else:
                    estimated_time = route[0]["started_at"]
                    underestimated_time = route[0]["started_at"]
                    overestimated_time = route[0]["started_at"]

            # we must consider if the stop has a time at which it should be started earliest
            estimated_time = max(estimated_time, stop["start_at"])
            underestimated_time = max(underestimated_time, stop["start_at"])
            overestimated_time = max(overestimated_time, stop["start_at"])

            # we calculate the driving time and waiting time
            if stop_index != 0:
                stop["origin"] = route[stop_index - 1]["destination"]
            estimated_time += self.get_travel_time(stop["origin"], stop["destination"]) + self.expected_parking_time
            underestimated_time += self.get_travel_time(stop["origin"], stop["destination"]) + self.expected_parking_time
            overestimated_time += self.get_travel_time(stop["origin"], stop["destination"]) + self.expected_parking_time

            # for pickups, we have to consider the synchronization of restaurant and vehicle
            if stop["type"] == 0:
                # underestimating synchronization:
                #estimated_time = max(estimated_time, stop["order_estimated_ready_time"]
                #                     + self.synchro_tol * stop["order_ready_time_sigma"])
                estimated_time = max(estimated_time, stop["order_estimated_ready_time"])
                underestimated_time = max(underestimated_time, stop["order_estimated_ready_time"])
                overestimated_time = max(overestimated_time, stop["order_estimated_ready_time"])

                # overestimating synchronization:
                # calculate moments of the truncated ready time dist and use them in a jensen-type inequality
                # https://math.stackexchange.com/questions/3305117/expectation-of-maximum-of-n-i-i-d-random-variables
                # https://stats.stackexchange.com/questions/511271/moments-of-limited-lognormal-distribution
                queue_info = obs["restaurant_info"]["r_{}".format(stop["restaurant_id"])]["orders_in_queue"]
                if stop["order_estimated_ready_time"] != 0 and queue_info:
                    first_customer_in_queue = queue_info[0]["customer_id"]
                    queue_start_time = obs["all_customers_info"][first_customer_in_queue]["order_time"]
                    mean_ready_time = (stop["order_estimated_ready_time"] - queue_start_time) / 60
                    sigma_ready_time = stop["order_ready_time_sigma"]
                    mu_underlying_normal = np.log(
                        mean_ready_time ** 2 / np.sqrt((sigma_ready_time + mean_ready_time ** 2)))
                    sigma_sqrd_underlying_normal = np.log((sigma_ready_time ** 2 / (mean_ready_time ** 2)) + 1)
                    sigma_underlying_normal = np.sqrt(sigma_sqrd_underlying_normal)

                    if not estimated_time <= queue_start_time:
                        trunc_alpha = (np.log(
                            (estimated_time - queue_start_time) / 60) - mu_underlying_normal) / sigma_underlying_normal
                        # only calculate the truncated lognormal moments
                        # if arrival time is reasonably close to prep time (from above)
                        if N(trunc_alpha) < 1 - 1e-3:
                            #expectation_truncated_lognormal = np.exp((2 * mu_underlying_normal
                            #                                          + sigma_sqrd_underlying_normal)
                            #                                         / 2) * ((1 - N(trunc_alpha -
                            #                                                        sigma_underlying_normal))
                            #                                                 / (1 - N(trunc_alpha)))
                            #overestimated_time = expectation_truncated_lognormal * 60 + queue_start_time
                            k_moments_ready_time = [np.exp(k * (2 * mu_underlying_normal + k * sigma_sqrd_underlying_normal) / 2)
                                                    * ((1 - N(trunc_alpha - sigma_underlying_normal * k)) / (1 - N(trunc_alpha)))
                                                    for k in range(1, 10)]
                            overestimated_time = np.min([np.power(2, 1 / k) * np.power(k_moments_ready_time[k - 1], 1 / k)
                                                     for k in range(1, 10)]) * 60 + queue_start_time
                            estimated_time = (overestimated_time + underestimated_time) / 2

            stop["eta"] = int(estimated_time)
            stop["eta_lb"] = int(underestimated_time)
            stop["eta_ub"] = int(overestimated_time)

            if stop["type"] == 1:
                c_id = "c_{}".format(stop["customer_id"])
                updated_etas[c_id] = [stop["eta_lb"], stop["eta"], stop["eta_ub"]]
        return updated_etas

    def _evaluate_route_plan(self, obs, route_plan_etas, updated_etas, v_key) -> tuple[float, float]:
        # Calculates the convex sum of delay and synchronization delay for a given route plan
        delay = 0
        synchro = 0
        slack = 0
        n = 0
        k = 0
        for c_id, etas in route_plan_etas.items():
            etas = [eta if v != v_key or c_id not in updated_etas.keys()
                    else updated_etas[c_id] for v, eta in etas.items()]
            if c_id in updated_etas.keys() and v_key not in route_plan_etas[c_id].keys():
                etas.append(updated_etas[c_id])
            if etas:
                etas_lb = [i[0] for i in etas]
                etas_ub = [i[2] for i in etas]
                etas = [i[1] for i in etas]
                delay += max(0, max(etas_lb) + self.buffer - obs["customer_info"][c_id]["expected_delivery_time"])  # Ulmer and ours
                #slack += max(0, obs["customer_info"][c_id]["expected_delivery_time"] - max(etas_lb))  # Ulmer
                slack += max(0, max(etas_ub) + self.buffer - obs["customer_info"][c_id]["expected_delivery_time"]) # ours
                n += 1
                if len(obs["customer_info"][c_id]["restaurant_choice"]) > 1:
                    synchro += (max(0, max(etas_lb) - min(etas_lb)) + max(0, max(etas_ub) - min(etas_ub)))/2  # ours (wrong)=
                    #synchro += max(0, max(etas_lb) - min(etas_lb))  # ulmer + ours
                    k += 1
        mean_delay = delay / n
        if k > 0:
            mean_synchro = synchro / k
        else:
            mean_synchro = 0
        return self.alpha * mean_delay + (1 - self.alpha) * mean_synchro, slack

    def _insert_delivery_stop(self, obs: Observation, route_plan: Dict, customer_id: str,
                              vehicle_id: str, insertion_index: int):
        delivery_stop = [1, -1, obs["customer_info"][customer_id]["location"], -1,
                         int(customer_id[2:]), -1, -1, 0, -1, -1, -1, -1, -1]
        for _ in range(self.vehicle_capacity):
            delivery_stop.append(-1)
        if route_plan[vehicle_id].shape[0] == 0:
            route_plan[vehicle_id] = np.array([tuple(delivery_stop)], dtype=self.stop_struct)
        else:
            trip = route_plan[vehicle_id].tolist()
            trip.insert(insertion_index, tuple(delivery_stop))
            route_plan[vehicle_id] = np.array(trip, dtype=self.stop_struct)

    def _insert_pickup_stop(self, obs: Observation, route_plan: Dict, customer_ids: List[str],
                            restaurant_id: str, vehicle_id: str, insertion_index: int):
        # first, we must calculate the preparation time of the restaurant
        queue = obs["restaurant_info"][restaurant_id]["orders_in_queue"]
        estimated_time_queue = obs["restaurant_info"][restaurant_id]["estimated_finish_times"]
        orders_to_prepare = [order["customer_id"] for order in queue]
        relevant_orders = [c_id for c_id in customer_ids if c_id in orders_to_prepare]
        if relevant_orders:
            max_index = max([orders_to_prepare.index(c_id) for c_id in relevant_orders])
            order_estimated_ready_time = estimated_time_queue[max_index]
            order_ready_time_sigma = np.sqrt(self.var_cook_time * (max_index + 1))

        else:
            order_estimated_ready_time = obs["current_time"]
            order_ready_time_sigma = -1

        # construct stop
        pickup_stop = [0, -1, obs["restaurant_info"][restaurant_id]["location"], int(restaurant_id[2:]),
                       -1, -1, -1, 0, order_estimated_ready_time, order_ready_time_sigma, -1, -1, -1]
        n = len(customer_ids)
        for i in range(self.vehicle_capacity):
            if i < n:
                pickup_stop.append(int(customer_ids[i][2:]))
            else:
                pickup_stop.append(-1)
        if route_plan[vehicle_id].shape[0] == 0:
            route_plan[vehicle_id] = np.array([tuple(pickup_stop)], dtype=self.stop_struct)
        else:
            trip = route_plan[vehicle_id].tolist()
            trip.insert(insertion_index, tuple(pickup_stop))
            route_plan[vehicle_id] = np.array(trip, dtype=self.stop_struct)

    def _insert_order(self, obs, order, route_plan, restaurant_plan, route_plan_etas):
        """
        Inserts a single-order into the route plan.
        """
        if not order:
            return route_plan, restaurant_plan, np.inf

        # reinsert the bundled pickup stop and two delivery stops
        best_rp_cost = np.inf
        best_tie_breaker = np.inf
        best_rp = {v_key: np.copy(route_plan[v_key]) for v_key in route_plan.keys()}
        best_rp_etas = {c_id: copy.copy(c_info) for c_id, c_info in route_plan_etas.items()}

        vehicle_keys = list(route_plan.keys())
        np.random.shuffle(vehicle_keys)
        for v_key in vehicle_keys:
            # make sure to only insert after started actions
            earliest_insertion_point = 0
            if len(route_plan[v_key]) > 0:
                if route_plan[v_key][0]["started_at"] != -1 or route_plan[v_key][0]["start_at"] == -1:
                    earliest_insertion_point = 1

            # insert pickups before a potentially already existing delivery (only in case of multi-order)
            # also test if there is already a delivery
            latest_delivery_point = len(route_plan[v_key]) + 1
            delivery_already_exists = False
            for stop_ix, stop in enumerate(route_plan[v_key]):
                if stop["customer_id"] == int(order[0][2:]):
                    latest_delivery_point = stop_ix + 1
                    delivery_already_exists = True

            for insertion_index_pickup in range(earliest_insertion_point, latest_delivery_point):
                t_rp = {v_key: np.copy(route_plan[v_key]) for v_key in route_plan.keys()}

                # modifier required if we merge instead of append a pickup
                delivery_insertion_modifier = 1

                # append one pickup stop or merge with previous
                if insertion_index_pickup != 0:
                    if t_rp[v_key][insertion_index_pickup - 1]["restaurant_id"] == int(order[1][2:]):
                        for i in range(self.vehicle_capacity):
                            if t_rp[v_key][insertion_index_pickup - 1]["orders_to_pickup_{}".format(i)] == -1:
                                t_rp[v_key][insertion_index_pickup - 1]["orders_to_pickup_{}".format(i)] = int(
                                    order[0][2:])
                                delivery_insertion_modifier = 0
                                # update the estimated order preparation time of the pickup stop
                                queue = obs["restaurant_info"][order[1]]["orders_in_queue"]
                                estimated_time_queue = obs["restaurant_info"][order[1]]["estimated_finish_times"]
                                orders_to_prepare = [order["customer_id"] for order in queue]
                                queue_index = orders_to_prepare.index(order[0])
                                t_rp[v_key][insertion_index_pickup - 1]["order_estimated_ready_time"] \
                                    = estimated_time_queue[queue_index]
                                t_rp[v_key][insertion_index_pickup - 1]["order_ready_time_sigma"] \
                                    = np.sqrt(self.var_cook_time * (queue_index + 1))
                                break
                            if i == self.vehicle_capacity - 1:
                                raise Warning("Vehicle capacity is full. Edge case not implemented.")
                    else:
                        self._insert_pickup_stop(obs, t_rp, [order[0]], order[1], v_key, insertion_index_pickup)
                else:
                    self._insert_pickup_stop(obs, t_rp, [order[0]], order[1], v_key, insertion_index_pickup)

                # iterate over possible delivery insertion points
                # case 1: its a multi-order and a delivery stop is already in the route
                if delivery_already_exists:
                    updated_etas = self._update_route_timing(obs, t_rp[v_key], v_key)
                    delivery_cost, tie_breaker = self._evaluate_route_plan(obs, route_plan_etas, updated_etas, v_key)
                    if delivery_cost < best_rp_cost or (delivery_cost == best_rp_cost
                                                        and tie_breaker < best_tie_breaker):
                        best_rp = {v_key: np.copy(t_rp[v_key]) for v_key in t_rp.keys()}
                        best_rp_cost = delivery_cost
                        best_tie_breaker = tie_breaker
                        best_rp_etas = {c_id: copy.copy(c_info) for c_id, c_info in route_plan_etas.items()}
                        for c_id, new_eta in updated_etas.items():
                            best_rp_etas[c_id][v_key] = new_eta

                # case 2: no delivery stop is in the route
                else:
                    for insertion_index_delivery in range(insertion_index_pickup + delivery_insertion_modifier,
                                                          len(t_rp[v_key]) + 1):
                        tt_rp = {v_key: np.copy(t_rp[v_key]) for v_key in t_rp.keys()}

                        self._insert_delivery_stop(obs, tt_rp, order[0], v_key, insertion_index_delivery)

                        updated_etas = self._update_route_timing(obs, tt_rp[v_key], v_key)
                        delivery_cost, tie_breaker = self._evaluate_route_plan(obs, route_plan_etas, updated_etas,
                                                                               v_key)
                        if delivery_cost < best_rp_cost or (delivery_cost == best_rp_cost
                                                            and tie_breaker < best_tie_breaker):
                            best_rp = {v_key: np.copy(tt_rp[v_key]) for v_key in tt_rp.keys()}
                            best_rp_cost = delivery_cost
                            best_tie_breaker = tie_breaker
                            best_rp_etas = {c_id: copy.copy(c_info) for c_id, c_info in route_plan_etas.items()}
                            for c_id, new_eta in updated_etas.items():
                                best_rp_etas[c_id][v_key] = new_eta

        return best_rp, restaurant_plan, best_rp_etas

    """
    def _search(self, obs, route_plan, restaurant_plan, route_plan_cost):
        #LNS to improve starting solution.

        k = 0
        current_route_plan = {v_key: np.copy(route_plan[v_key]) for v_key in route_plan.keys()}
        current_route_plan_cost = np.inf
        for _ in range(self.n_lns_steps):

            # remove a random number of orders
            n_orders_to_remove = int(np.random.random() * len(obs["customer_info"]))
            orders_to_remove = self._choose_random_order(obs, n_orders_to_remove)

            removed_orders = []
            for order_to_remove in orders_to_remove:
                removed_order, current_route_plan, _ = self._remove_order(obs, current_route_plan, restaurant_plan,
                                                                          order_to_remove)
                removed_orders.extend(removed_order)
            # reinsert the removed orders
            for order in removed_orders:
                current_route_plan, _, current_route_plan_cost = self._insert_order(obs, order, current_route_plan,
                                                                                    restaurant_plan)

            # update the route_plan if we improved
            if current_route_plan_cost < route_plan_cost:
                k = 0
                route_plan = {v_key: np.copy(current_route_plan[v_key]) for v_key in current_route_plan.keys()}
                route_plan_cost = current_route_plan_cost
            else:
                k += 1
                # reset to last best solution if no improvement for a given number of steps
                if k == self.reset_after_n_steps_wo_improvement:
                    k = 0
                    current_route_plan = {v_key: np.copy(route_plan[v_key]) for v_key in route_plan.keys()}
                    current_route_plan_cost = route_plan_cost

        return route_plan, restaurant_plan
    

    def _stop_contains_order(self, stop, order):
        # delivery stop corresponding to customer
        if stop["customer_id"] == int(order[0][2:]):
            return True
        # pickup stop corresponding to customer
        elif stop["restaurant_id"] == int(order[1][2:]):
            for i in range(self.vehicle_capacity):
                if int(order[0][2:]) == stop["orders_to_pickup_{}".format(i)]:
                    return True
        return False

    def _remove_order(self, obs, route_plan, restaurant_plan, order_to_remove):

        # nothing to remove
        if order_to_remove is None:
            return [], route_plan, restaurant_plan

        # pickup has already been done
        if np.any([order_to_remove in obs["vehicle_info"][v_key]["orders_in_backpack"]
                   for v_key in route_plan.keys()]):
            return [], route_plan, restaurant_plan

        # pickup action related to order has already started
        for v_key, v_route in route_plan.items():
            if len(v_route) > 0:
                stop = v_route[0]
                if stop["started_at"] == -1 and stop["type"] == 0 and self._stop_contains_order(stop, order_to_remove):
                    return [], route_plan, restaurant_plan

        # remove the order corresponding to the customer
        destroyed_route_plan = {}
        for v_key, v_route in route_plan.items():
            destroyed_route_plan[v_key] = []
            remove_delivery = True
            for stop in v_route:
                # stop does not match order
                if not self._stop_contains_order(stop, order_to_remove):
                    destroyed_route_plan[v_key].append(np.copy(stop))
                    # if it is a pickup stop related to the customer but another restaurant, we do not want to remove
                    # the corresponding delivery
                    if stop["type"] == 0:
                        for i in range(self.vehicle_capacity):
                            if int(order_to_remove[0][2:]) == stop["orders_to_pickup_{}".format(i)]:
                                remove_delivery = False
                # stop matches order
                else:
                    if stop["type"] == 1:
                        if not remove_delivery:
                            destroyed_route_plan[v_key].append(np.copy(stop))
                        remove_delivery = True

        destroyed_route_plan = {v_key: np.array(trip, dtype=self.stop_struct)
                                for v_key, trip in destroyed_route_plan.items()}

        return [order_to_remove], destroyed_route_plan, restaurant_plan

    def _choose_random_order(self, obs, n, filter_orders=None):
        
        #Randomly choose among single orders (filter="multi"), among all multiorders (filter="single),
        #or all orders (filter=None).
        
        # create a list of all single orders
        if filter_orders == "multi":
            orders = [(c_id, obs["customer_info"][c_id]["restaurant_choice"][0]) for c_id in obs["customer_info"].keys()
                      if len(obs["customer_info"][c_id]["restaurant_choice"]) == 1]
        # create a list of all multi orders
        elif filter_orders == "single":
            orders = [[(c_id, r_id) for r_id in obs["customer_info"][c_id]["restaurant_choice"]]
                      for c_id in obs["customer_info"].keys()
                      if len(obs["customer_info"][c_id]["restaurant_choice"]) > 1]
            orders = [x for xs in orders for x in xs]

        # create a list of all orders
        else:
            orders = [[(c_id, r_id) for r_id in obs["customer_info"][c_id]["restaurant_choice"]]
                      for c_id in obs["customer_info"].keys()]
            orders = [x for xs in orders for x in xs]

        # chose a customer randomly from the remaining list
        if orders:
            indices_to_remove = np.random.choice(range(len(orders)), size=n, replace=False)
            return [orders[i] for i in indices_to_remove]
        return []

    def _choose_most_time_consuming_order(self):
        # remove the order that adds the most time to a route
        pass

    def _choose_shaw(self):
        pass
    """


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
