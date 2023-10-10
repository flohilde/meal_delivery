from src.state import MealDeliveryMDP
from src.policies.simple_assignment import SimpleAssignmentPolicy
import configparser


if __name__ == "__main__":

    config = configparser.ConfigParser(allow_no_value=True)
    config.read('../data/instances/iowa_110_5_55_80.ini')
    env = MealDeliveryMDP(config, seed=42)
    policy = SimpleAssignmentPolicy()

    for i in range(0, 100):
        obs = env.reset()
        while True:
            action = policy.act(obs)
            obs, cost, done, info = env.step(action)
            if done:
                print("Episode {}. Mean delay {}.".format(i, env.mean_delay))
                break
