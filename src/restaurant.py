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
    estimated_preparation_time : int
        Estimated time to prepare the order assuming the preparation starts now.
    actual_preparation_time : int
        Actual time to prepare the order assuming the preparation starts now.
    """

    def __init__(self, customer_id, start_at, estimated_preparation_time,
                 actual_preparation_time):
        self.customer_id = customer_id
        self.start_at = start_at
        self.estimated_preparation_time = estimated_preparation_time
        self.actual_preparation_time = actual_preparation_time

    def summary(self):
        r"""
        Returns a summary of the order as a dictionary.
        """
        return {"customer_id": self.customer_id,
                "start_at": self.start_at,
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

    def __init__(self, id_number: int, location: int) -> None:
        self.name = "r_{}".format(id_number)
        self.location = int(location)
        self.queue = []
        self.time_queue = []
        self.estimated_time_queue = []
        self.prepared_orders = []

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
            else:
                break

    def take_order(self, insertion_index: int, order: Order, time: int) -> None:
        r"""
        Integrates an order into the restaurant by updating queue and time queues.
        """
        # insert order into queue
        if insertion_index == -1:
            self.queue.append(order)
        else:
            self.queue.insert(insertion_index, order)
        # update time queues
        if len(self.queue) == 1:
            self.estimated_time_queue.append(time + order.estimated_preparation_time)
            self.time_queue.append(time + order.actual_preparation_time)
        else:
            if insertion_index == -1:
                self.estimated_time_queue.append(self.estimated_time_queue[-1] + order.estimated_preparation_time)
                self.time_queue.append(self.time_queue[-1] + order.actual_preparation_time)
            else:
                self.estimated_time_queue.insert(insertion_index, self.estimated_time_queue[insertion_index - 1]
                                                 + order.estimated_preparation_time)
                self.estimated_time_queue.insert(insertion_index, self.time_queue[insertion_index - 1]
                                                 + order.actual_preparation_time)
                for i in range(insertion_index + 1, len(self.queue)):
                    self.estimated_time_queue[i] += order.estimated_preparation_time
                    self.time_queue[i] += order.actual_preparation_time

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
        Returns the exact (not estimated) waiting time (in seconds) until a given list of orders is finished.
        """
        orders = [order for order in self.queue if (order.customer_id in orders and order not in self.prepared_orders)]
        if len(orders) == 0:
            return 0
        max_index = max([self.queue.index(order) for order in orders])
        return max(0, self.estimated_time_queue[max_index] - time)

    def summary(self) -> dict:
        r"""
        Returns a summary of the restaurant as a dictionary.
        """
        return {"location": self.location,
                "orders_in_queue": [order.summary() for order in self.queue],
                "estimated_finish_times": self.estimated_time_queue,
                "prepared_orders": [order.summary() for order in self.prepared_orders]}
