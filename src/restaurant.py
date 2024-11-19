import numpy as np
from typing import List


class Order:
    r"""
    An order is placed by a customer at the platform, which is then forwarded to the restaurant.

    Parameters
    ----------
    customer_id : str
        Name of the customer that placed the order.
    start_at : int
        Earliest time at which the preparation process should be started at.
    estimated_preparation_time : int
        Estimated time to prepare the order assuming the preparation starts now.
    actual_preparation_time : int
        Actual time to prepare the order assuming the preparation starts now.


    Attributes
    ----------
    customer_id : str
        Name of the customer that placed the order.
    start_at : int
        Earliest time at which the preparation process should be started at.
    restaurant_name : str
        Name of the restaurant at which the meal was ordered.
    finished_at : int
        Time at which preparation actually finished
    estimated_preparation_time : int
        Estimated time to prepare the order assuming the preparation starts now.
    actual_preparation_time : int
        Actual time to prepare the order assuming the preparation starts now.
    """

    def __init__(self, customer_id, start_at, restaurant_name, estimated_preparation_time, actual_preparation_time):
        self.customer_id = customer_id
        self.start_at = start_at
        self.restaurant_name = restaurant_name   # Storing the name of the restaurant
        self.finished_at = None
        self.estimated_preparation_time = estimated_preparation_time
        self.actual_preparation_time = actual_preparation_time

    def summary(self):
        r"""
        Returns a summary of the order as a dictionary.
        """
        return {"customer_id": self.customer_id,
                "start_at": self.start_at,
                "finished_at": self.finished_at,
                "estimated_preparation_time": self.estimated_preparation_time}


class Restaurant:
    r"""
    A restaurant is given by a queue of orders to prepare and a storage for prepared orders.
    We assume that orders are prepared sequentially and not in parallel.

    Parameters
    ----------
    id_number : int
        The id_number completes the restaurant name, given by "r_{id_number}".
    location : int
        Node of the street graph where the restaurant is located at.

    Attributes
    ----------
    name : str
        Name of the restaurant used as a key in the MealDeliveryMDP to access the restaurant.
    location : int
        Node of the street graph where the restaurant is located at.
    queue : List[Order]
        Sequence of orders to prepare.
    time_queue : List[int]
        Time (of day in seconds) at which each order in the queue will be prepared (subject to changes
        if queue changes).
    estimated_time_queue : List[int]
        Estimated time (of day in seconds) at which each order in the queue will be prepared (subject to changes
        if queue changes).
    prepared_orders : List[str]
        List containing customer names whos order has been prepared by the restaurant but not yet
        picked up by a vehicle.
    """

    def __init__(self, id_number: int, location: int,
                 basket_size_mean: float = 1.0, basket_size_std: float = 0.0) -> None:
        self.name = "r_{}".format(id_number)
        self.location = int(location)
        self.queue = []
        self.time_queue = []
        self.estimated_time_queue = []
        self.prepared_orders = []
        self.basket_size_mean = basket_size_mean
        self.basket_size_std = basket_size_std

    def update(self, time: int) -> None:
        r"""
        Updates the queue, time queue, estimated time queue, and prepared orders by forwarding the current time to
        the input time.
        """
        for cook_time in self.time_queue:
            if cook_time <= time:
                index = self.time_queue.index(cook_time)
                self.prepared_orders.append(self.queue.pop(index))
                self.time_queue.pop(index)
                self.estimated_time_queue.pop(index)
                self.prepared_orders[-1].finished_at = cook_time
            else:
                break

    def reorder_queue(self, order_sequence: List[str], time: int) -> None:
        r"""
        Integrates an order into the restaurant by updating queue and time queues.
        """

        estimated_time = time
        if self.time_queue and self.queue[0].start_at < time:
            time = self.time_queue[0] - self.queue[0].actual_preparation_time
            estimated_time = self.estimated_time_queue[0] - self.queue[0].estimated_preparation_time

        self.queue = sorted(self.queue, key=lambda d: order_sequence.index(d.customer_id))

        new_time_queue = []
        new_estimated_time_queue = []
        for order in self.queue:
            time = max(time, order.start_at) + order.actual_preparation_time
            estimated_time = max(estimated_time, order.start_at) + order.estimated_preparation_time
            new_time_queue.append(time)
            new_estimated_time_queue.append(estimated_time)

        self.time_queue = new_time_queue
        self.estimated_time_queue = new_estimated_time_queue

    def sample_random_basket_size(self):
        r"""
        Samples a random monetary value of an order from a normal distribution with given mean and std.
        """
        return np.random.normal(loc=self.basket_size_mean, scale=self.basket_size_std)

    def get_actual_waiting_time(self, orders: list, time: int) -> float:
        r"""
        Returns the exact (not estimated) waiting time (in seconds) until a given list of orders is finished.
        """
        orders = [order for order in self.queue if (order.customer_id in orders and order not in self.prepared_orders)]
        if len(orders) == 0:
            return 0
        max_index = max([self.queue.index(order) for order in orders])
        return max(0, self.time_queue[max_index] - time)

    def get_estimated_waiting_time(self, orders: list, time: int) -> float:
        r"""
        Returns the estimated waiting time (in seconds) until a given list of orders is finished.
        """
        orders = [order for order in self.queue if (order.customer_id in orders and order not in self.prepared_orders)]
        if len(orders) == 0:
            return 0
        max_index = max([self.queue.index(order) for order in orders])
        return max(0, self.estimated_time_queue[max_index] - time)

    def get_estimated_prep_time(self, orders: list):
        orders = [order for order in self.queue if (order.customer_id in orders and order not in self.prepared_orders)]
        if len(orders) == 0:
            return 0
        max_index = max([self.queue.index(order) for order in orders])
        return self.estimated_time_queue[max_index]

    def get_position_in_queue(self, orders: list):
        orders = [order for order in self.queue if
                  (order.customer_id in orders and order not in self.prepared_orders)]
        if len(orders) == 0:
            return 0
        max_index = max([self.queue.index(order) for order in orders])
        return max_index

    def summary(self) -> dict:
        r"""
        Returns a summary of the restaurant as a dictionary.
        """
        return {"location": self.location,
                "orders_in_queue": [order.summary() for order in self.queue],
                "estimated_finish_times": self.estimated_time_queue,
                "prepared_orders": [order.summary() for order in self.prepared_orders]}
