from state import MealDeliveryMDP
#from policies.simple_assignment import SimpleAssignmentPolicy
from policies.order_bundling_2 import RestaurantProximityBatchingPolicy
#from policies.DynamicInsertion import DynamicInsertionPolicy
#from policies.DynamicInsertion_updated import DynamicInsertionPolicy_combine
import logging
import configparser
import pandas as pd

# Configure logging
logging.basicConfig(filename='policy_debug.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    config = configparser.ConfigParser(allow_no_value=True)
    #config.read('./data/instances/iowa_110_10_55_80.ini')
    config.read('./data/instances/iowa_110_20_110_160.ini')
    env = MealDeliveryMDP(config, seed=42)
    #policy = SimpleAssignmentPolicy()  ## policy 1
    policy = RestaurantProximityBatchingPolicy(tt_matrix=env.tt_matrix)  ## Restaurant_Proximity_Batching  ## policy 2
    #policy = DynamicInsertionPolicy(tt_matrix=env.tt_matrix)  ## policy 3
    #policy = DynamicInsertionPolicy_combine(tt_matrix=env.tt_matrix) ## policy 4

    results = [] # store result

#   Restaurant Proximity Batching Policy use the code from line 129 and comment out from line 30 to 127

    for i in range(0, 100):
            obs = env.reset()
            while True:
                action = policy.act(obs)
                obs, cost, done, info = env.step(action)
                # if done:
                #     # for each served request state.served request
                #     #print("Episode {}. Mean delay {}.".format(i, env.mean_delay))

                #     print("Episode {}. Mean delay {}. Mean freshness {}. Mean Sync Delay {}".format(
                #         i, env.mean_delay, env.mean_freshness, env.mean_sync_delay))
                #     #print("Episode {}. Mean delay {}. Mean Sync Delay {}.".format(
                #     #    i, env.mean_delay, env.mean_sync_delay))

                #     results.append([i, env.mean_delay, env.mean_freshness, env.mean_sync_delay])
                    
                #     break

                if done:
                    for customer in env.served_requests:
                        # Calculate mean delay for customer
                        customer_delays = [max(0, delivery_time - customer.expected_delivery_time) 
                                        for delivery_time in customer.delivery_time.values()]
                        if customer_delays:
                            customer_mean_delay = sum(customer_delays) / len(customer_delays) / 60  # Convert to minutes
                        else:
                            customer_mean_delay = 0

                        # Calculate mean freshness for customer
                        customer_freshness_totals = []
                        for restaurant_name in customer.restaurant_choice:
                            delivery_time = customer.delivery_time.get(restaurant_name)
                            if delivery_time:
                                for order in env.placed_orders:
                                    if order.customer_id == customer.name and order.restaurant_name == restaurant_name:
                                        
                                        if order.finished_at is not None:
                                            freshness = delivery_time - order.finished_at
                                            customer_freshness_totals.append(max(freshness, 0))
                        
                        # Calculate mean freshness
                        if customer_freshness_totals:
                            customer_mean_freshness = sum(customer_freshness_totals) / len(customer_freshness_totals) / 60  # Convert to minutes
                        else:
                            customer_mean_freshness = 0

                        # Calculate mean synchronization delay for customer
                        if len(customer.delivery_time.values()) > 1:
                            customer_sync_delays = [max(customer.delivery_time.values()) - min(customer.delivery_time.values())]
                            customer_mean_sync_delay = sum(customer_sync_delays) / len(customer_sync_delays) / 60  # convert to minutes
                        else:
                            customer_mean_sync_delay = 0  # no sync delay for single order

                        # Calculate average travel time from customer to restaurants ordered
                        total_travel_time_to_restaurants = 0
                        for restaurant_name in customer.restaurant_choice:
                            restaurant_location = env.restaurants[restaurant_name].location
                            
                            total_travel_time_to_restaurants += env.tt_matrix[str(customer.location)][str(restaurant_location)]

                        average_travel_time_to_restaurants = total_travel_time_to_restaurants / len(customer.restaurant_choice) if customer.restaurant_choice else 0

                        # Calculate average travel time between restaurants if more than one restaurant is chosen
                        total_travel_time_between_restaurants = 0
                        travel_times_count = 0
                        if len(customer.restaurant_choice) > 1:
                            for j, restaurant_name_1 in enumerate(customer.restaurant_choice[:-1]):
                                for restaurant_name_2 in customer.restaurant_choice[j+1:]:
                                    restaurant_location_1 = env.restaurants[restaurant_name_1].location
                                    restaurant_location_2 = env.restaurants[restaurant_name_2].location
                                    total_travel_time_between_restaurants += env.tt_matrix[str(restaurant_location_1)][str(restaurant_location_2)]
                                    travel_times_count += 1
                            average_travel_time_between_restaurants = total_travel_time_between_restaurants / travel_times_count if travel_times_count > 0 else 0
                        else:
                            average_travel_time_between_restaurants = 0

                        customer_detail = {
                            "Episode": i,
                            "Customer ID": customer.name,
                            "Customer Location": customer.location,
                            "Restaurants Chosen": ", ".join(customer.restaurant_choice),
                            "Customer Mean Delay": customer_mean_delay,
                            "Customer Mean Freshness": customer_mean_freshness,
                            "Customer Mean Sync Delay": customer_mean_sync_delay,
                            "Average Travel Time to Restaurants": average_travel_time_to_restaurants,
                            "Average Travel Time Between Restaurants": average_travel_time_between_restaurants
                        }
                        results.append(customer_detail)

                    print(f"Episode {i}. Processed {len(env.served_requests)} customers.")
                    break


        # df = pd.DataFrame(results, columns=["Episode", "Mean Delay", "Mean Freshness", "Mean Sync Delay"])
        # df.to_csv('DynamicInsertion_results.csv', index=False)

    df = pd.DataFrame(results)
    df.to_csv('DynamicInsertion_V3.csv', index=False)

    ## For Restaurant Proximit Batching Policy ##

    # for max_travel_time in range(1, 16):
    #     
    #     policy = RestaurantProximityBatchingPolicy(tt_matrix=env.tt_matrix, max_travel_time=max_travel_time)
        
    #     for i in range(0, 100): 
    #         obs = env.reset()
    #         while True:
    #             action = policy.act(obs)
    #             obs, cost, done, info = env.step(action)
    #             if done:
    #                 # Append results with additional max_travel_time info
    #                 for customer in env.served_requests:
    #                     
    #                     #Calculate mean delay for customer
    #                     customer_delays = [max(0, delivery_time - customer.expected_delivery_time) 
    #                                     for delivery_time in customer.delivery_time.values()]
    #                     if customer_delays:
    #                         customer_mean_delay = sum(customer_delays) / len(customer_delays) / 60  # Convert to minutes
    #                     else:
    #                         customer_mean_delay = 0

    #                     # Calculate mean freshness for customer
    #                     customer_freshness_totals = []
    #                     for restaurant_name in customer.restaurant_choice:
    #                         delivery_time = customer.delivery_time.get(restaurant_name)
    #                         if delivery_time:
    #                             for order in env.placed_orders:
    #                                 if order.customer_id == customer.name and order.restaurant_name == restaurant_name:
    #                                     
    #                                     if order.finished_at is not None:
    #                                         freshness = delivery_time - order.finished_at
    #                                         customer_freshness_totals.append(max(freshness, 0))
                        
    #                     # Calculate mean freshness if there are freshness totals; else set to 0
    #                     if customer_freshness_totals:
    #                         customer_mean_freshness = sum(customer_freshness_totals) / len(customer_freshness_totals) / 60  # Convert to minutes
    #                     else:
    #                         customer_mean_freshness = 0

    #                     # Calculate average synchronization delay for customer
    #                     if len(customer.delivery_time.values()) > 1:
    #                         customer_sync_delays = [max(customer.delivery_time.values()) - min(customer.delivery_time.values())]
    #                         customer_mean_sync_delay = sum(customer_sync_delays) / len(customer_sync_delays) / 60  # convert to minutes
    #                     else:
    #                         customer_mean_sync_delay = 0  # no sync delay for single order

    #                     # Calculate average travel time from customer to restaurants ordered
    #                     total_travel_time_to_restaurants = 0
    #                     for restaurant_name in customer.restaurant_choice:
    #                         restaurant_location = env.restaurants[restaurant_name].location
    #                         
    #                         total_travel_time_to_restaurants += env.tt_matrix[str(customer.location)][str(restaurant_location)]

    #                     average_travel_time_to_restaurants = total_travel_time_to_restaurants / len(customer.restaurant_choice) if customer.restaurant_choice else 0

    #                     # Calculate average travel time between restaurants if more than one restaurant is chosen
    #                     total_travel_time_between_restaurants = 0
    #                     travel_times_count = 0
    #                     if len(customer.restaurant_choice) > 1:
    #                         for j, restaurant_name_1 in enumerate(customer.restaurant_choice[:-1]):
    #                             for restaurant_name_2 in customer.restaurant_choice[j+1:]:
    #                                 restaurant_location_1 = env.restaurants[restaurant_name_1].location
    #                                 restaurant_location_2 = env.restaurants[restaurant_name_2].location
    #                                 total_travel_time_between_restaurants += env.tt_matrix[str(restaurant_location_1)][str(restaurant_location_2)]
    #                                 travel_times_count += 1
    #                         average_travel_time_between_restaurants = total_travel_time_between_restaurants / travel_times_count if travel_times_count > 0 else 0
    #                     else:
    #                         average_travel_time_between_restaurants = 0
                        
    #                     results.append({
    #                         "Episode": i,
    #                         "MaxTravelTime": max_travel_time,  # Keep track of the current max_travel_time
    #                         "Customer ID": customer.name,
    #                         "Customer Location": customer.location,
    #                         "Restaurants Chosen": ", ".join(customer.restaurant_choice),
    #                         "Customer Mean Delay": customer_mean_delay,  
    #                         "Customer Mean Freshness": customer_mean_freshness,  
    #                         "Customer Mean Sync Delay": customer_mean_sync_delay,  
    #                         "Average Travel Time to Restaurants": average_travel_time_to_restaurants,  
    #                         "Average Travel Time Between Restaurants": average_travel_time_between_restaurants
    #                     })
    #                 break

    # # Convert results to DataFrame and save to CSV
    # df = pd.DataFrame(results)
    # df.to_csv('RestaurantProximityBatchingPolicy_V3.csv', index=False)            