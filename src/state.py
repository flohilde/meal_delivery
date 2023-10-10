from src.customer import Customer
from src.restaurant import Restaurant, Order
from src.vehicle import Vehicle, Stop
from src.templates import VehicleAction, Action, Observation
import numpy as np
import simplejson as json
from typing import Tuple


class MealDeliveryMDP:
    """
    Implements all elements of the Markov decision process describing an on-demand restaurant meal delivery platform
    located in Iowa City.

    Parameters
    ----------
    config : Config
        Config containing the instance information to construct the simulation.
    seed : int
        Random seed.

    Attributes
    ----------
    tt_matrix : dict[int, dict[int, float]]
        Travel times (in seconds) between to nodes, e.g., tt_matrix[10][91].
    restaurant_location_list : list[int]
        List of nodes corresponding to the locations of restaurants in the instance. Only used to initialize restaurants
        after resetting the instance.
    n_restaurants : int
        Number of restaurants in the instance.
    cook_mu : float
        Mu parameter (in minutes) of the log-normal distribution of preparation times.
    cook_sigma : float
        Sigma parameter (in minutes) of the log-normal distribution of preparation times.
    expected_cook_time : float
        Expected preparation time (in seconds) calculated from cook_mu and cook_sigma.
    vehicle_location_list : list[int]
        List of the initial vehicle locations. Only used to initialize vehicles after resetting the instance.
    n_vehicles : int
        Number of vehicles in the instance.
    park_mu : float
        Mu parameter (in minutes) of the log-normal park time distribution.
    park_sigma : float
        Sigma parameter (in minutes) of the log-normal park time distribution.
    expected_parking_time : float
        Expected parking time (in seconds) calculated from park_mu and park_sigma.
    locations : list[int]
        List of possible customer locations. Only used to initialize vehicles after resetting the instance.
    t_lunch_mu : float
        Mu parameter (daytime in minutes) of the normal distribution of lunch order times.
    t_lunch_sigma : float
        Sigma parameter (daytime in minutes) of the normal distribution of lunch order times.
    t_dinner_mu : float
        Mu parameter (daytime in minutes) of the normal distribution of dinner order times.
    t_dinner_sigma : float
        Sigma parameter (daytime in minutes) of the normal distribution of dinner order times.
    n_lunch_mu : float
        Mu parameter of the normal distribution of number of lunch orders.
    n_lunch_sigma : float
        Sigma parameter of the normal distribution of number of lunch orders.
    n_dinner_mu : float
        Mu parameter of the normal distribution of number of dinner orders.
    n_dinner_sigma : float
        Sigma parameter of the normal distribution of number of dinner orders.
    service_promise : int
        Allowed time span (in minutes) between order placement and order delivery. If a delivery takes longer, it is
        counted as a delay.
    vehicles : dict[str, Vehicle] or None
        Dictionary of vehicle names (key) and corresponding vehicles (value).
    restaurants : dict[str, Restaurant] or None
        Dictionary of restaurant names (key) and corresponding restaurants (value).
    customers : dict[str, Customer] or None
        Dictionary of customer names (key) and corresponding customers (value).
    time : int or None
        Current time of day in seconds.
    day : int or None
        Current day of the simulation.
    order_times : list[int] or None
        List of times (daytimes in seconds) at which customers place an order.
    unknown_requests : list[Customer] or None
        List of customers that have yet to order this very day.
    known_requests : list[Customer] or None
        List of customers that have ordered but were not delivered all orders yet.
    served_requests : list[Customer] or None
        List of customers that have ordered and were delivered all orders.
    unassigned_orders : list[tuple[str, str]] or None
        List of tuples of restaurant name and customer name corresponding to all orders that have been placed
        but were not assigned to a vehicle yet.
    """

    def __init__(self, config, seed):

        # set rng seed
        np.random.seed(seed)

        # load travel time matrix
        with open(config.get("GRAPH", "TT_MATRIX"), 'r') as f:
            self.tt_matrix = json.load(f)  # travel time dict: travel time between pair of nodes

        # read restaurant parameters
        with open(config.get("RESTAURANTS", "RESTAURANT_LOCATION_FILE"), 'r') as f:
            self.restaurant_location_list = json.load(f)  # locations of restaurants
        self.n_restaurants = config.getint("RESTAURANTS", "N_RESTAURANTS")  # number of restaurants
        self.cook_mu = config.getfloat("RESTAURANTS", "COOK_TIME_MU")  # mean cook time of a dish
        self.cook_sigma = config.getfloat("RESTAURANTS", "COOK_TIME_SIGMA")  # variance in cook time of a dish
        self.expected_cook_time = int(np.exp(np.log(self.cook_mu) + (np.log(self.cook_sigma) ** 2) / 2) * 60)

        # read vehicle parameters
        with open(config.get("VEHICLES", "VEHICLE_LOCATION_FILE"), 'r') as f:
            self.vehicle_location_list = json.load(f)  # location of vehicles
        self.n_vehicles = config.getint("VEHICLES", "N_VEHICLES")  # number of vehicles
        self.park_mu = config.getfloat("VEHICLES", "PARK_TIME_MU")  # mean time to park/finalize delivery
        self.park_sigma = config.getfloat("VEHICLES", "PARK_TIME_SIGMA")  # variance in time to  park/finalize delivery
        self.expected_parking_time = int(np.exp(np.log(self.park_mu) + (np.log(self.park_sigma) ** 2) / 2) * 60)

        # read demand/customer parameters
        with open(config.get("CUSTOMERS", "CUSTOMER_LOCATION_FILE"), 'r') as f:
            self.locations = json.load(f)  # customer locations
        self.t_lunch_mu = config.getfloat("CUSTOMERS", "T_LUNCH_MU")  # mean time of lunch time peak
        self.t_lunch_sigma = config.getfloat("CUSTOMERS", "T_LUNCH_SIGMA")  # variance in time of lunch time peak
        self.t_dinner_mu = config.getfloat("CUSTOMERS", "T_DINNER_MU")  # mean time of dinner time peak
        self.t_dinner_sigma = config.getfloat("CUSTOMERS", "T_DINNER_SIGMA")  # variance in time of dinner time peak
        self.n_lunch_mu = config.getfloat("CUSTOMERS", "N_LUNCH_MU")  # mean number of lunch orders
        self.n_lunch_sigma = config.getfloat("CUSTOMERS", "N_LUNCH_SIGMA")  # variance in number of lunch orders
        self.n_dinner_mu = config.getfloat("CUSTOMERS", "N_DINNER_MU")  # mean number of dinner orders
        self.n_dinner_sigma = config.getfloat("CUSTOMERS", "N_DINNER_SIGMA")  # variance in number of dinner orders
        self.service_promise = config.getfloat("CUSTOMERS", "SERVICE_PROMISE")  # allowed delivery time

        # prepare empty class attributes
        self.vehicles = None  # dict of vehicles in the fleet
        self.restaurants = None  # dict of restaurants in the service area
        self.customers = None  # dict of customers of the day
        self.time = None  # int representing current time
        self.day = 0  # int representing current day
        self.order_times = None  # list of ints representing order times of all customers on that day
        self.unknown_requests = None  # list of customers that will order this day but have not done so yet
        self.known_requests = None  # list of revealed but not served customers
        self.served_requests = None  # list of revealed and served requests
        self.unassigned_orders = None

    @property
    def observation(self) -> Observation:
        r"""
        Returns the current state observation as a dict.
        """
        obs = {"current_time": self.time,
               "unassigned_orders": self.unassigned_orders,
               "vehicle_info": {name: vehicle.summary(self.time) for name, vehicle in self.vehicles.items()},
               "restaurant_info": {name: restaurant.summary() for name, restaurant in self.restaurants.items()},
               "customer_info": {customer.name: customer.summary() for customer in self.known_requests}}
        return Observation(obs)

    @property
    def done(self) -> bool:
        r"""
        Returns true if all customers are served and the time horizon is reached and false, otherwise.
        """
        all_customers = self.unknown_requests + self.known_requests + self.served_requests
        return np.all([customer.status == 1 for customer in all_customers])

    @property
    def mean_delay(self) -> float:
        r"""
        Returns the mean delay (in minutes) of all customers that have been served this day. Delay per customer is
        calculated as the maximum delay over all orders of the customer.
        """
        return sum([max(0, max(customer.delivery_time.values()) - customer.expected_delivery_time)
                    for customer in self.served_requests]) / len(self.served_requests) / 60

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        r"""
        Transition from one state of the MDP to the next according to the action taken and
        the revealed stochastic information.

        Parameters
        ----------
            action : Action
                An action is a dictionary that contains a restaurant action and a vehicle action.
                The restaurant action is a dictionary with restaurant names as keys and a list of RestaurantAction
                tuples as value, i.e., tuples of the form:
                (customer_id: str, start_at: int, insertion_index: int) as value.
                    -'customer_id' specifies which customer placed the order.
                    -'start_at' is an int that specifies when the action should be started earliest.
                        If the value is -1, the action is started as soon as possible.
                    -'insertion_index' specifies at which position the action is inserted into the
                        queue of the restaurant. If the value is -1, the action is appended to the queue.
                The vehicle action is a dictionary with restaurant names as keys and a list of VehicleAction tuples,
                i.e., tuples of the form:
                (destination: str or int, start_at: int, insertion_index: int, load: list[str]) as values.
                    -'destination' is a customer or restaurant name if the action is a pickup or delivery, or an int
                        specifying a node if the action is a relocation.
                    -'start_at' is an int that specifies when the action should be started earliest.
                        If the value is -1, the action is started as soon as possible.
                    -'insertion_index' specifies at which position the action is inserted into the
                        route of the vehicle. If the value is -1, the action is appended to the route.
                    -'orders_to_pickup' list of customer names corresponding to orders to pickup at restaurant if the
                        action is a pickup action. Otherwise, None.
                    -'orders_to_deliver' list of restaurant names corresponding to orders to deliver at customer if the
                        action is a delivery action. Otherwise, None.

                Note that each list of restaurant action tuples and vehicle action tuples will be inserted ordered by
                insertion_index in a descending fashion.

        Returns
        -------
        Returns
            - the observation after the transition to the next state (induced by the next customer request,
            - the cost of the action in the given state,
            - a boolean indicating if the simulation is done,
            - and a dictionary containing additional info.

        Note
        ----
        The stops/orders corresponding to VehicleActions/RestaurantActions for each vehicle/restaurant are inserted in
        the respective route/queue in the order of their occurence in the list. Please choose the insertion indices
        accordingly.
        """

        vehicle_action = action["vehicle_action"]
        restaurant_action = action["restaurant_action"]

        # integrate action into restaurants (restaurants before vehicles as preparation times influence routes)
        for restaurant in [r for r in self.restaurants.values() if r.name in restaurant_action.keys()]:
            for r_action in restaurant_action[restaurant.name]:
                # construct Order
                customer_id, start_at, insertion_index = r_action
                estimated_preparation_time = self.expected_cook_time
                actual_preparation_time = self._sample_cook_time()
                order = Order(customer_id, start_at, estimated_preparation_time, actual_preparation_time)
                # insert Order
                restaurant.take_order(insertion_index, order, self.time)

        # integrate action into vehicles
        for vehicle in [v for v in self.vehicles.values() if v.name in vehicle_action.keys()]:
            # for each vehicle action construct a Stop and insert into vehicle
            for v_action in vehicle_action[vehicle.name]:
                # construct stop from VehicleAction
                stop = self._construct_stop_from_vehicle_action(vehicle, v_action)
                # track that this order has been assigned to a vehicle
                if stop.type == "pickup":
                    for customer_id in stop.orders_to_pickup:
                        self.unassigned_orders.remove((customer_id, stop.restaurant_id))
                # insert stop
                if v_action.insertion_index == -1:
                    vehicle.sequence_of_stops.append(stop)
                else:
                    vehicle.sequence_of_stops.insert(v_action.insertion_index, stop)
                # if it is the first stop in the sequence, start it right away
                if len(vehicle.sequence_of_stops) == 1:
                    if v_action.start_at <= self.time:
                        vehicle.sequence_of_stops[0].started_at = self.time
            # repair: adjust travel times and waiting times for all stops in the route after insertion
            if vehicle.sequence_of_stops:
                self._repair_vehicle_route(vehicle)

        # calculate route cost
        cost = 0  # For now, we set the immediate cost in a state to zero as there is no meaningful definition of cost

        # sample a new customer and update current time
        if self.unknown_requests:
            new_customer = self.unknown_requests.pop(0)
            new_customer.status = 0
            self.known_requests.append(new_customer)
            self.unassigned_orders.extend([(new_customer.name, restaurant_id)
                                           for restaurant_id in new_customer.restaurant_choice])
            self.time = new_customer.order_time
        else:
            self.time += 360  # forward one hour

        # update all restaurant status
        for restaurant in self.restaurants.values():
            restaurant.update(self.time)

        # update all vehicle status and their interactions with customers and restaurants
        for vehicle in self.vehicles.values():
            picked_up, delivered = vehicle.update(self.time)
            for restaurant_id, picked_up_orders in picked_up.items():
                restaurant = self.restaurants[restaurant_id]
                restaurant.prepared_orders = [order for order in restaurant.prepared_orders
                                              if order.customer_id not in picked_up_orders]
            for customer_id, delivered_orders in delivered.items():
                customer = self.customers[customer_id]
                for restaurant_id, time in delivered_orders:
                    customer.delivery_time[restaurant_id] = time
                    vehicle.orders_in_backpack.remove((restaurant_id, customer_id))
                if None not in customer.delivery_time.values():
                    customer.status = 1
                    self.served_requests.append(customer)
                    self.known_requests.remove(customer)

        return self.observation, cost, self.done, {}

    def reset(self) -> Observation:
        r"""
        Set the environment to a new initial state and return the initial observation.
        """
        # keep track of day
        self.day += 1

        # reset everything else
        self.vehicles = {}
        self.restaurants = {}
        self.customers = {}
        self.time = 0
        self.known_requests = []
        self.unknown_requests = []
        self.served_requests = []
        self.unassigned_orders = []

        # initialize restaurants
        self._init_restaurants()
        # initialize vehicles
        self._init_vehicles()

        # init demand
        self._init_demand()

        return self.observation

    def _init_restaurants(self) -> None:
        r"""
        Reset restaurants to have empty queues again. If less than 110 restaurants are considered, i.e., n < 110,
        n restaurants are uniformly random sampled from the list of 110 restaurants.
        instance are
        """
        if self.n_restaurants == 110:
            restaurant_iterator = range(110)
        elif self.n_restaurants < 110:
            restaurant_iterator = np.random.permutation(110)[:self.n_restaurants]
        else:
            raise Warning("Number of restaurant exceeds number of available restaurant for the given instance.")
        for i in restaurant_iterator:
            restaurant_location = int(self.restaurant_location_list[i])
            self.restaurants["r_{}".format(i)] = Restaurant(location=restaurant_location,
                                                            id_number=i)

    def _init_vehicles(self) -> None:
        r"""
        Reset vehicles to their initial position and empty routes.
        """
        for i in range(self.n_vehicles):
            vehicle_location = int(self.vehicle_location_list[i])
            self.vehicles["v_{}".format(i)] = Vehicle(location=vehicle_location,
                                                      id_number=i)

    def _init_demand(self) -> None:
        r"""
        Samples demand for lunch and dinner orders from two independent normal distributions describing the number of
        requests, two independent normal distributions describing the time of requests, and a uniform distribution
        describing the location of the request.
        """
        n_lunch = int(max(0, np.random.normal(loc=self.n_lunch_mu, scale=self.n_lunch_sigma)))
        n_dinner = int(max(0, np.random.normal(loc=self.n_dinner_mu, scale=self.n_dinner_sigma)))
        t_lunch = np.random.normal(loc=self.t_lunch_mu, scale=self.t_lunch_sigma, size=n_lunch) * 60
        t_dinner = np.random.normal(loc=self.t_dinner_mu, scale=self.t_dinner_sigma, size=n_dinner) * 60
        order_times = np.sort(np.hstack((t_lunch, t_dinner))).astype(int)
        self.order_times = order_times[(0 <= order_times) & (order_times <= 86399)].tolist()

        # initialize customer requests
        for i, order_time in enumerate(order_times):
            customer = Customer(id_number=i,
                                location=np.random.choice(self.locations),
                                order_time=order_times[i],
                                expected_delivery_time=order_times[i] + self.service_promise * 60,
                                restaurant_choice=["r_{}".format(i) for i in np.random.choice(a=self.n_restaurants,
                                                                                              size=1)])
            self.customers[customer.name] = customer
            self.unknown_requests.append(customer)

    def _sample_travel_time(self, origin: int, destination: int) -> int:
        r"""
        Returns the (deterministic) travel time (in seconds) between origin and destination.
        """
        return int(self.tt_matrix[str(origin)][str(destination)])

    def _sample_parking_time(self) -> int:
        r"""
        Sample a (random) parking time (in seconds) from the log-normal parking time distribution.
        """
        return int(np.clip(np.random.lognormal(mean=np.log(self.park_mu),
                                               sigma=np.log(self.park_sigma),
                                               size=None), a_min=0, a_max=None) * 60)

    def _sample_cook_time(self) -> int:
        r"""
        Samples a (random) cook time (in seconds) from the log-normal distribution.
        """
        return int(np.clip(np.random.lognormal(mean=np.log(self.cook_mu),
                                               sigma=np.log(self.cook_sigma),
                                               size=None), a_min=0, a_max=None) * 60)

    def _construct_stop_from_vehicle_action(self, vehicle: Vehicle, vehicle_action: VehicleAction) -> Stop:
        r"""
        Takes a vehicle and a VehicleAction and returns a Stop.
        """
        destination, start_at, insertion_index, orders_to_pickup, orders_to_deliver = vehicle_action
        # we must differentiate between relocations, pickups, and deliveries
        # relocation stop
        if type(destination) is int:
            stop_type = "relocation"
            restaurant_id = None
            customer_id = None
            estimated_wait_time = 0
            actual_wait_time = 0
            estimated_park_time = self.expected_parking_time
            actual_park_time = self._sample_parking_time()
            if not vehicle.sequence_of_stops:
                origin = vehicle.location
            else:
                if insertion_index != -1:
                    origin = vehicle.sequence_of_stops[insertion_index - 1].destination
                else:
                    origin = vehicle.sequence_of_stops[-1].destination
            actual_travel_time = self._sample_travel_time(origin, destination)
            estimated_travel_time = actual_travel_time
        # pickup stop
        elif destination[0] == "r":
            stop_type = "pickup"
            restaurant_id = destination
            customer_id = None
            _restaurant = self.restaurants[restaurant_id]
            destination = _restaurant.location
            estimated_park_time = self.expected_parking_time
            actual_park_time = self._sample_parking_time()
            estimated_wait_time = 0  # we will update this when we update the wait time of all stops
            actual_wait_time = 0  # we will update this when we update the wait time of all stops
            if not vehicle.sequence_of_stops:
                origin = vehicle.location
            else:
                if insertion_index != -1:
                    origin = vehicle.sequence_of_stops[insertion_index - 1].destination
                else:
                    origin = vehicle.sequence_of_stops[-1].destination
            actual_travel_time = self._sample_travel_time(origin, destination)
            estimated_travel_time = actual_travel_time
        # delivery stop
        elif destination[0] == "c":
            stop_type = "delivery"
            customer_id = destination
            restaurant_id = None
            destination = self.customers[destination].location
            estimated_wait_time = 0
            actual_wait_time = 0
            estimated_park_time = self.expected_parking_time
            actual_park_time = self._sample_parking_time()
            if not vehicle.sequence_of_stops:
                origin = vehicle.location
            else:
                if insertion_index != -1:
                    origin = vehicle.sequence_of_stops[insertion_index - 1].destination
                else:
                    origin = vehicle.sequence_of_stops[-1].destination
            actual_travel_time = self._sample_travel_time(origin, destination)
            estimated_travel_time = actual_travel_time
        else:
            raise Warning("Vehicle action contains invalid destination {}".format(destination))
        stop = Stop(stop_type, destination, restaurant_id,
                    customer_id, start_at,
                    estimated_travel_time, actual_travel_time,
                    estimated_park_time, actual_park_time,
                    estimated_wait_time, actual_wait_time,
                    orders_to_pickup)
        return stop

    def _repair_vehicle_route(self, vehicle: Vehicle) -> None:
        r"""
        Updates travel times, waiting times, and start at times of stops after new stops have been inserted
        into the route.
        """
        estimated_time = self.time  # estimated arrival at stop
        actual_time = self.time  # actual arrival at stop
        if vehicle.sequence_of_stops[0].started_at is not None:
            estimated_time = vehicle.sequence_of_stops[0].started_at
            actual_time = vehicle.sequence_of_stops[0].started_at
        for stop_index, stop in enumerate(vehicle.sequence_of_stops):
            # delay planned start of stop if necessary
            if -1 != stop.start_at <= actual_time:
                stop.start_at = actual_time
            # adjust travel time
            if stop_index != 0:
                prev_stop = vehicle.sequence_of_stops[stop_index - 1]
                stop.actual_travel_time = self._sample_travel_time(prev_stop.destination, stop.destination)
                stop.estimated_travel_time = stop.actual_travel_time
            estimated_time = max(stop.start_at, estimated_time) + stop.estimated_travel_time + stop.estimated_park_time
            actual_time = max(stop.start_at, actual_time) + stop.actual_travel_time + stop.actual_park_time
            # for each pickup stop we adjust the wait time
            if stop.type == "pickup":
                restaurant = self.restaurants[stop.restaurant_id]
                estimated_wait_time = restaurant.get_estimated_waiting_time(stop.orders_to_pickup, estimated_time)
                actual_wait_time = restaurant.get_actual_waiting_time(stop.orders_to_pickup, actual_time)
                stop.estimated_wait_time = estimated_wait_time
                stop.actual_wait_time = actual_wait_time
                estimated_time += estimated_wait_time
                actual_time += actual_wait_time
