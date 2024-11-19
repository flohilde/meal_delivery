from src.state import MealDeliveryMDP
from src.policies.fleet_control.simple_assignment import SimpleAssignmentPolicy
from src.policies.fleet_control.lns import LNS
from src.policies.demand_control.simple_proximity import SimpleProximityDemandControl
from src.policies.demand_control.customer_choice_models import simple_customer_choice
import configparser
import numpy as np


def run(config, n_episodes=1, alpha=1.0, buffer=0, synchro_tol=0):
    env = MealDeliveryMDP(config, seed=42)
    env.endogenous_choice = False
    # policy = SimpleAssignmentPolicy(env.tt_matrix, env.expected_parking_time, env.expected_cook_time)
    policy = LNS(env.tt_matrix, env.expected_parking_time, env.expected_cook_time, env.var_cook_time,
                 alpha=alpha, buffer=buffer, synchro_tol=synchro_tol)
    demand_policy = SimpleProximityDemandControl(proximity=[10 * 60] * 110,
                                                 restaurant_nodes=env.restaurant_location_list,
                                                 tt_matrix=env.tt_matrix)

    results = []

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
                print(action)
                print(obs)
                raise e

            if done:
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

                summary = [i, env.mean_delay, env.mean_freshness, env.mean_sync_delay]
                print("Episode; {}; Mean delay; {}; Mean freshness; {}; Mean Sync-Delay; {}".format(*summary))
                break
    results = np.array(results, dtype=float)
    np.save("../results/iowa_110_5_80_80_n_{}_p_{}_alpha_{}_buffer_{}_synchrotol_{}".format(env.multi_order_n,
                                                                                            str(env.multi_order_p).replace(
                                                                                                ".", "_"),
                                                                                            str(alpha).replace(".",
                                                                                                               "_"),
                                                                                            buffer,
                                                                                            str(synchro_tol).replace(
                                                                                                ".", "_")),
            results)


if __name__ == "__main__":

    config = configparser.ConfigParser(allow_no_value=True)
    config.read('../data/instances/multi_order/iowa_110_20_320_320.ini')

    for p in [0.1]:
        for alpha in [0.2]:
            #for buffer in list(range(2, 12, 2)):
            for buffer in [10]:
                #for synchro_tol in [0.5, 1, 1.5, 2.0]:
                for synchro_tol in [-2]:
                    #for (mu, sigma, perc) in [(7.9400412461595264, 1.5274705710772243, 10),
                    #                    (7.881410739124781, 1.5538909474195441, 20),
                    #                    (7.824060151812124, 1.579378640155617, 30),
                    #                    (7.76794358351971, 1.6040294814535416, 40),
                    #                    (7.713017405512268, 1.,6279229283249443 50)]:
                        #config.set('CUSTOMERS', 'MULTI_ORDER_BINOM_P', str(p))
                        #config.set('RESTAURANTS', 'COOK_TIME_MU', str(mu))
                        #config.set('RESTAURANTS', 'COOK_TIME_SIGMA', str(sigma))
                    config.set('CUSTOMERS', 'MULTI_ORDER_BINOM_P', str(p))
                    run(config, n_episodes=100, alpha=alpha, buffer=buffer * 60, synchro_tol=synchro_tol)


    #import pstats
    #import cProfile
    #
    #cProfile.run("run(n_episodes=1, config=config)", "my_func_stats")

    #p = pstats.Stats("my_func_stats")
    #p.sort_stats("cumulative").print_stats()

    #print(p)
