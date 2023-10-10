from typing import Tuple


class Stop:
    r"""
    Stop in the route of a vehicle. A stop may describe the pickup of orders at a restaurant,
    the deliver of orders to a customer, or a relocation of the vehicle to a new location in the service area.

    Attributes
    ----------
    type : str
        String indicating the type of stop. Feasible types are 'pickup', 'delivery', 'relocation'.
    destination : int
        Node at which the stop is located.
    customer_id : str or None
        Name of the customer where the delivery takes place, if type == 'delivery'. ELse, None.
    restaurant_id : str or None
        Name of the restaurant where the pickup takes place, if type == 'pickup'. ELse, None.
    start_at : int
        Earliest time at which the vehicle should leave its last location towards the stop.
    estimated_travel_time : int
        Estimated time (in seconds) required to drive from the last location to the stop.
    actual_travel_time : int
        Actual time (in seconds) required to drive from the last location to the stop.
    estimated_park_time : int
        Estimated time (in seconds) to park at the stop.
    actual_park_time : int
        Actual time (in seconds) to park at the stop.
    estimated_wait_time : int
        Estimated time to wait occurring in the synchronization process with the restaurant, i.e., wait time of the
        driver if the driver arrives before all orders to pickup are prepared by the restaurant.
    actual_wait_time : int
        Actual time to wait occurring in the synchronization process with the restaurant, i.e., wait time of the
        driver if the driver arrives before all orders to pickup are prepared by the restaurant.
    orders_to_pickup : list or None
        List of orders to pick up at the restaurant if the stop type is 'pickup'. Else, None.
    """

    def __init__(self, stop_type: str, destination: int, restaurant_id: str or None, customer_id: str or None,
                 start_at: int, estimated_travel_time: int, actual_travel_time: int, estimated_park_time: int,
                 actual_park_time: int, estimated_wait_time: int, actual_wait_time: int,
                 orders_to_pickup: list or None):
        assert stop_type in ["pickup", "delivery", "relocation"]
        if stop_type == "pickup":
            assert orders_to_pickup is not None and orders_to_pickup
            assert restaurant_id is not None
            assert customer_id is None
        elif stop_type == "delivery":
            assert estimated_wait_time == actual_wait_time == 0
            assert restaurant_id is None
            assert customer_id is not None
        else:
            assert estimated_wait_time == actual_wait_time == 0
            assert restaurant_id is None
            assert customer_id is None

        self.type = stop_type
        self.destination = destination
        self.customer_id = customer_id
        self.restaurant_id = restaurant_id
        self.start_at = start_at
        self.started_at = None
        self.estimated_travel_time = estimated_travel_time
        self.actual_travel_time = actual_travel_time
        self.estimated_park_time = estimated_park_time
        self.actual_park_time = actual_park_time
        self.estimated_wait_time = estimated_wait_time
        self.actual_wait_time = actual_wait_time
        self.orders_to_pickup = orders_to_pickup

    @property
    def estimated_total_time(self):
        r"""
        Returns the estimated time (in seconds) it takes to drive to the spot, find parking, and synchronize with the
        restaurant (if applicable).
        """
        return self.estimated_travel_time + self.estimated_park_time + self.estimated_wait_time

    @property
    def actual_total_time(self):
        r"""
        Returns the actual time (in seconds) it takes to drive to the spot, find parking, and synchronize with the
        restaurant (if applicable).
        """
        return self.estimated_travel_time + self.estimated_park_time + self.estimated_wait_time

    def summary(self):
        r"""
        Returns a summary of the vehicle as a dictionary. Only information that is known by the platform is contained.
        """
        return {"type": self.type,
                "destination": self.destination,
                "restaurant_id": self.restaurant_id,
                "customer_id": self.customer_id,
                "start_at": self.start_at,
                "estimated_time_required": self.estimated_total_time,
                "orders_to_pickup": self.orders_to_pickup}


class Vehicle:
    """
    A vehicle picks up orders at the restaurants and delivers them to the corresponding customers.

    Parameters
    ----------
    id_number : int
            The id_number completes the vehicle name, given by "v_{id_number}".
    location : int
        Node where the vehicle is currently idle or will be idle at the end of the route.

    Attributes
    ----------
    name : str
        Name of the vehicle. The name is used as a key to access the vehicle in the MealDeliveryMDP environment.
    location : int
        Node where the vehicle is currently idle or will be idle at the end of the route.
    sequence_of_stops : list[Stop]
        List of stops to sequentially traverse.
    orders_in_backpack : List[Tuple[str, str]]
        List of tuples containing restaurant name and customer name to identify orders that the vehicle is
        currently carrying.
    """

    def __init__(self, id_number: int, location: int) -> None:
        self.name = "v_{}".format(id_number)
        self.location = int(location)  # current (or next) idle location of vehicle given by node in the graph
        self.sequence_of_stops = []  # list of stops to visit
        self.orders_in_backpack = []  # list of tuples (customer_id, restaurant_id)

    def update(self, time: int) -> Tuple[dict, dict]:
        r"""
        Update the vehicle's route, i.e., sequence of stops by forwarding to the input time.

        Returns
        -------
        Returns a tuple containing a dict of orders that have been been picked up between the last time the vehicle's
        route was updated and the current time as well as a dict of orders that have been delivered to customers
        during this time period.
        """
        picked_up = {}
        delivered = {}
        _time = time
        while self.sequence_of_stops:
            if self.sequence_of_stops[0].started_at is None:
                if -1 != self.sequence_of_stops[0].start_at <= _time:
                    self.sequence_of_stops[0].started_at = self.sequence_of_stops[0].start_at
                else:
                    self.sequence_of_stops[0].started_at = _time
            _time = self.sequence_of_stops[0].started_at
            _time += self.sequence_of_stops[0].actual_total_time
            if _time > time:
                break
            else:
                # stop has been visited and is removed
                stop = self.sequence_of_stops.pop(0)
                # if pickup stop, we remove the orders from the restaurant's prepared meals
                if stop.type == "pickup":
                    picked_up[stop.restaurant_id] = stop.orders_to_pickup
                    self.orders_in_backpack.extend([(stop.restaurant_id, c_id) for c_id in stop.orders_to_pickup])
                # if delivery stop, we update the customer
                if stop.type == "delivery":
                    delivered[stop.customer_id] = [(r_id, _time) for (r_id, c_id) in self.orders_in_backpack
                                                   if c_id == stop.customer_id]
        return picked_up, delivered

    def estimated_busy_time(self, current_time: int) -> int:
        r"""
        Returns an estimated time required to perform all stops in the current route.
        """
        if self.sequence_of_stops:
            t_diff = 0
            if self.sequence_of_stops[0].started_at is not None:
                t_diff = self.sequence_of_stops[0].started_at - current_time
            return sum([stop.estimated_total_time for stop in self.sequence_of_stops]) + t_diff
        else:
            return 0

    def summary(self, time) -> dict:
        r"""
        Returns a summary of the vehicle as a dictionary. Only information that is known by the platform is contained.
        """
        return {"next_location": self.location,
                "orders_in_backpack": self.orders_in_backpack,
                "sequence_of_actions": [action.summary() for action in self.sequence_of_stops],
                "busy_time": self.estimated_busy_time(time)}
