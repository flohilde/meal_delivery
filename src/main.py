from state import MealDeliveryMDP
from policies.fleet_control.simple_assignment import SimpleAssignmentPolicy
from policies.fleet_control.lns import LNS
from policies.demand_control.simple_proximity import SimpleProximityDemandControl
from policies.demand_control.customer_choice_models import simple_customer_choice
import configparser
import numpy as np


def run(config, n_episodes=1, weights=[1.0, 1.0, 1.0], buffer=0, mode="ulmer", force_synchro=False):
    env = MealDeliveryMDP(config, seed=42)
    env.endogenous_choice = False
    # policy = SimpleAssignmentPolicy(env.tt_matrix, env.expected_parking_time, env.expected_cook_time)
    policy = LNS(env.tt_matrix, env.expected_parking_time, env.expected_cook_time, env.var_cook_time,
                 weights=weights, buffer=buffer, mode=mode, force_synchro=force_synchro)
    demand_policy = SimpleProximityDemandControl(proximity=[10 * 60] * 110,
                                                 restaurant_nodes=env.restaurant_location_list,
                                                 tt_matrix=env.tt_matrix)

    #results = []
    vehicle_results = []

    for i in range(0, n_episodes):

        obs = env.reset()
        last_demand_control_update = 0

        while True:

            # demand control action
            if env.endogenous_choice and obs["new_customer_info"] is not None:

                # update proximity parameter in demand control policy
                if obs["current_time"] >= last_demand_control_update + 600:
                    # calculate new proximity parameters
                    demand_policy.proximity = [5 * 60] * 110

                demand_action = demand_policy.act(obs)
                obs = env.update_customers_choices(demand_action=demand_action, choice_model=simple_customer_choice)

            # fleet control action and state transition
            try:
                action = policy.act(obs)
                obs, cost, done, info = env.step(action)
            except Exception as e:
                #print(action)
                #print(obs)
                raise e

            if done:

                """
                # Customer KPIs
                for customer in env.served_requests:
                    restaurant_ids = [int(r_id[2:]) for r_id in customer.restaurant_choice]
                    restaurant_locations = [env.restaurants[r_id].location
                                            for r_id in customer.restaurant_choice]
                    restaurant_preparation_times = [customer.order_prepared_at[r_id]
                                                    for r_id in customer.restaurant_choice]
                    delivery_times = [customer.delivery_time[r_id]
                                      for r_id in customer.restaurant_choice]
                    driver_id = [int(customer.delivery_driver[r_id][2:]) for r_id in customer.restaurant_choice]
                    padding_required = env.multi_order_n - len(restaurant_ids)
                    if padding_required > 0:
                        restaurant_ids = restaurant_ids + [-1] * padding_required
                        restaurant_locations = restaurant_locations + [-1] * padding_required
                        restaurant_preparation_times = restaurant_preparation_times + [-1] * padding_required
                        delivery_times = delivery_times + [-1] * padding_required
                        driver_id = driver_id + [-1] * padding_required

                    row = [i, customer.order_time, customer.location, *restaurant_ids, *restaurant_locations,
                           *restaurant_preparation_times, *delivery_times, *driver_id]
                    results.append(row)
                    """

                # Vehicle KPIs
                for vehicle in env.vehicles:
                    row = [i, vehicle.id, vehicle.total_travel_time,
                           vehicle.total_busy_time, env.time - vehicle.total_busy_time]
                    vehicle_results.append(row)

                summary = [i, env.mean_delay, env.mean_freshness, env.mean_sync_delay]
                print("Episode; {}; Mean delay; {}; Mean freshness; {}; Mean Sync-Delay; {}".format(*summary))
                break

    """
    results = np.array(results, dtype=float)
    np.save("../results/vehicle_info/iowa_40_40_{}_{}_n_{}_p_{}_alpha_{}_beta_{}_gamma_{}_forcesync_{}_buffer_{}_mode_{}_sorted".format(
                                                                                                         int(env.n_lunch_mu),
                                                                                                         int(env.n_dinner_mu),
                                                                                                         env.multi_order_n,
                                                                                                         str(env.multi_order_p).replace(".", "_"),
                                                                                                         str(weights[0]).replace(".", "_"), 
                                                                                                         str(weights[1]).replace(".", "_"),
                                                                                                         str(weights[2]).replace(".", "_"),
                                                                                                         int(force_synchro),
                                                                                                         buffer, mode), 
                                                                                                         results)
    """
    vehicle_results = np.array(vehicle_results, dtype=float)
    np.save(
        "../results/vehicle_info/vehicles_iowa_40_40_{}_{}_n_{}_p_{}_alpha_{}_beta_{}_gamma_{}_forcesync_{}_buffer_{}_mode_{}_sorted".format(
            int(env.n_lunch_mu),
            int(env.n_dinner_mu),
            env.multi_order_n,
            str(env.multi_order_p).replace(".", "_"),
            str(weights[0]).replace(".", "_"),
            str(weights[1]).replace(".", "_"),
            str(weights[2]).replace(".", "_"),
            int(force_synchro),
            buffer, mode),
        vehicle_results)


if __name__ == "__main__":

    config = configparser.ConfigParser(allow_no_value=True)
    config.read('../data/instances/multi_order/iowa_40_40_240_240.ini')

    for p in [0.15, 0.2, 0.25]:
        for weights in [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.25],
                        [0.0, 1.0, 0.25], [1.0, 1.0, 0.25]]:
            #for demand in range(250, 370, 10):
                #for buffer in list(range(0, 12, 2)):
            for buffer in [0]:
                #for mode in ["ulmer"]:
                #for mode in ["ub"]:
                #for mode in ["triangular"]:
                #for mode in ["uniform"]:
                if weights == [1.0, 0.0, 0.0]:
                    force_synchro_cases = [False, True]
                else:
                    force_synchro_cases = [False]
                    for force_synchro in force_synchro_cases:
                        #for (mu, sigma, perc) in [(7.9400412461595264, 1.5274705710772243, 10),
                        #                    (7.881410739124781, 1.5538909474195441, 20),
                        #                    (7.824060151812124, 1.579378640155617, 30),
                        #                    (7.76794358351971, 1.6040294814535416, 40),
                        #                    (7.713017405512268, 1.6279229283249443, 50),
                        #                    (7.659240118449287, 1.6511257998517854, 60),
                        #                    (7.606572220597797, 1.6736949794094307, 70),
                        #                    (7.5549760858787165, 1.6956794125111876, 80),
                        #                    (7.504415850891277, 1.717121612830398, 90),
                        #                    (7.454857310144586, 1.7380588170291, 100)]:
                        #    config.set('RESTAURANTS', 'COOK_TIME_MU', str(mu))
                        #    config.set('RESTAURANTS', 'COOK_TIME_SIGMA', str(sigma))
                        config.set('CUSTOMERS', 'MULTI_ORDER_BINOM_P', str(p))
                        #config.set('CUSTOMERS', 'N_LUNCH_MU', str(demand))
                        #config.set('CUSTOMERS', 'N_LUNCH_SIGMA', str(np.sqrt(demand)))
                        #config.set('CUSTOMERS', 'N_DINNER_MU', str(demand))
                        #config.set('CUSTOMERS', 'N_DINNER_SIGMA', str(np.sqrt(demand)))

                        run(config, n_episodes=100, weights=weights, buffer=buffer * 60,
                            mode="ulmer", force_synchro=force_synchro)


    #import pstats
    #import cProfile
    #
    #cProfile.run("run(n_episodes=1, config=config)", "my_func_stats")

    #p = pstats.Stats("my_func_stats")
    #p.sort_stats("cumulative").print_stats()

    #print(p)
