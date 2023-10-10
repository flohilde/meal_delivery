class Customer:
    r"""
    A class to describe and track customers.

    Parameters
    ----------
    id_number : int
        The id_number completes the customer name, given by "c_{id_number}".
    location : int
        Node id of the customer location in the street graph.
    order_time : int
        time of order (in seconds of day)
    expected_delivery_time : int
        Time at which customer expects the delivery. This value is used when calculating delay.
    restaurant_choice : list[str]
        List of chosen restaurant names, i.e., restaurants contained in the customer's order.

    Attributes
    ----------
    name : str
        Name of the customer and key for the customer in the MealDeliveryMDP class.
    location : int
        Node id of the customer location in the street graph.
    status : int
        -1 if not revealed yet, 0 if waiting to be served, 1 if served.
    order_time : int
        time of order (in seconds of day)
    expected_delivery_time : int
        Time at which customer expects the delivery. This value is used when calculating delay.
    restaurant_choice : list[str]
        List of chosen restaurant names, i.e., restaurants contained in the customer's order.
    delivery_time : dict[str, int or None]
        Dict of restaurant id (key) and realized delivery times (value) for each of the customer's orders.
    """

    def __init__(self, id_number: int, location: int, order_time: int,
                 expected_delivery_time: int, restaurant_choice: list) -> None:
        self.name = "c_{}".format(id_number)
        self.location = int(location)
        self.status = -1
        self.order_time = order_time
        self.expected_delivery_time = expected_delivery_time
        self.restaurant_choice = restaurant_choice
        self.delivery_time = {r_choice: None for r_choice in self.restaurant_choice}

    def summary(self) -> dict:
        r"""
        Returns a summary of the customer as a dictionary.
        """
        return {"location": self.location,
                "order_time": self.order_time,
                "expected_delivery_time": self.expected_delivery_time,
                "restaurant_choice": self.restaurant_choice}
