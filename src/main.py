from src.state import MealDeliveryMDP
from src.policies.fleet_control.simple_assignment import SimpleAssignmentPolicy
from src.policies.fleet_control.lns import LNS
from src.policies.demand_control.simple_proximity import SimpleProximityDemandControl
from src.policies.demand_control.customer_choice_models import simple_customer_choice
import configparser


def run(n_episodes=1):
    config = configparser.ConfigParser(allow_no_value=True)
    #config.read('../data/instances/multi_order/iowa_110_20_220_320.ini')
    config.read('../data/instances/multi_order/iowa_110_5_80_80.ini')
    env = MealDeliveryMDP(config, seed=42)
    env.endogenous_choice = False
    #policy = SimpleAssignmentPolicy(env.tt_matrix, env.expected_parking_time, env.expected_cook_time)
    policy = LNS(env.tt_matrix, env.expected_parking_time, env.expected_cook_time, alpha=1.0)
    demand_policy = SimpleProximityDemandControl(proximity=[10*60]*110, restaurant_nodes=env.restaurant_location_list,
                                                 tt_matrix=env.tt_matrix)

    for i in range(0, n_episodes):

        obs = env.reset()
        last_demand_control_update = 0

        while True:

            # demand control action
            if env.endogenous_choice and obs["new_customer_info"] is not None:

                # update proximity parameter in demand control policy
                if obs["current_time"] >= last_demand_control_update + 600:
                    # calculate new proximity parameters
                    demand_policy.proximity = [5*60] * 110

                demand_action = demand_policy.act(obs)
                obs = env.update_customers_choices(demand_action=demand_action, choice_model=simple_customer_choice)

            # fleet control action and state transition
            try:
                action = policy.act(obs)
                obs, cost, done, info = env.step(action)
            except Exception as e:
                print(action)
                print(obs)
                raise(e)
            #print({vehicle_id: vehicle_info["sequence_of_actions"] for vehicle_id, vehicle_info in obs["vehicle_info"].items()})

            if done:
                summary = [i, env.mean_delay, env.mean_freshness, env.mean_sync_delay]
                print("Episode {}. Mean delay {}. Mean freshness {}. Mean Sync-Delay {}.".format(*summary))
                break


if __name__ == "__main__":

    import pstats
    import cProfile

    cProfile.run("run(n_episodes=1)", "my_func_stats")

    p = pstats.Stats("my_func_stats")
    p.sort_stats("cumulative").print_stats()

    print(p)
