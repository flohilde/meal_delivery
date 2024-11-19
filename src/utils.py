import numpy as np

type_dict = {"pickup": 0, "delivery": 1, "relocation": 2}
rev_type_dict = {0: "pickup", 1: "delivery", 2: "relocation"}


def stop_to_array(stop, vehicle_capacity=5):
    arr_stop = [type_dict[stop["type"]],
                stop["origin"],
                stop["destination"],
                int(stop["restaurant_id"][2:]) if stop["restaurant_id"] is not None else -1,
                int(stop["customer_id"][2:]) if stop["customer_id"] is not None else -1,
                stop["start_at"],
                stop["started_at"] if stop["started_at"] is not None else -1,
                stop["estimated_time_required"] if stop["estimated_time_required"] is not None else -1,
                stop["order_estimated_ready_time"] if stop["order_estimated_ready_time"] is not None else -1,
                stop["order_ready_time_sigma"] if stop["order_ready_time_sigma"] is not None else -1,
                stop["eta"],
                stop["eta_lb"],
                stop["eta_ub"],
                ]

    if stop["orders_to_pickup"] is None:
        n = 0
    else:
        n = len(stop["orders_to_pickup"])
        if n > vehicle_capacity:
            raise Warning("Vehicle capacity is full. Edge case not implemented.")
    for i in range(vehicle_capacity):
        if i < n:
            arr_stop.append(int(stop["orders_to_pickup"][i][2:]))
        else:
            arr_stop.append(-1)
    return tuple(arr_stop)


def array_to_stop(stop_arr, vehicle_capacity=5):
    stop = {
            "type": rev_type_dict[stop_arr["type"]],
            "origin": stop_arr["origin"],
            "destination": stop_arr["destination"],
            "restaurant_id": "r_{}".format(stop_arr["restaurant_id"]) if stop_arr["restaurant_id"] != -1 else None,
            "customer_id": "c_{}".format(stop_arr["customer_id"]) if stop_arr["customer_id"] != -1 else None,
            "start_at": stop_arr["start_at"],
            "started_at": stop_arr["started_at"] if stop_arr["started_at"] != -1 else None,
            "estimated_time_required": stop_arr["estimated_time_required"],
            "orders_to_pickup": [],
            "order_estimated_ready_time": stop_arr["order_estimated_ready_time"]
            if stop_arr["order_estimated_ready_time"] != -1 else None,
            "order_ready_time_sigma": stop_arr["order_ready_time_sigma"]
            if stop_arr["order_ready_time_sigma"] != -1 else None,
            "eta": stop_arr["eta"],
            "eta_lb": stop_arr["eta_lb"],
            "eta_ub": stop_arr["eta_ub"],

    }
    for i in range(vehicle_capacity):
        if stop_arr["orders_to_pickup_{}".format(i)] != -1:
            stop["orders_to_pickup"].append("c_{}".format(stop_arr["orders_to_pickup_{}".format(i)]))
    if not stop["orders_to_pickup"]:
        stop["orders_to_pickup"] = None

    if stop["type"] == "pickup" and stop["orders_to_pickup"] is None:
        raise Warning("Pickup-stop but nothing to pickup.")

    return stop


def route_to_array(route, vehicle_capacity=5):
    stop_struct = [
        ('type', 'i4'),
        ('origin', 'i4'),
        ('destination', 'i4'),
        ('restaurant_id', 'i4'),
        ('customer_id', 'i4'),
        ('start_at', 'i4'),
        ('started_at', 'i4'),
        ('estimated_time_required', 'i4'),
        ('order_estimated_ready_time', 'i4'),
        ('order_ready_time_sigma', 'f4'),
        ('eta', 'i4'),
        ('eta_lb', 'i4'),
        ('eta_ub', 'i4'),
    ]
    for i in range(vehicle_capacity):
        stop_struct.append(('orders_to_pickup_{}'.format(i), 'i4'))

    return {v_key: np.array([stop_to_array(stop, vehicle_capacity=vehicle_capacity) for stop in trip], dtype=stop_struct)
            for v_key, trip in route.items()}


def array_to_route(route_arr, vehicle_capacity=5):
    return {v_key: [array_to_stop(stop_arr, vehicle_capacity=vehicle_capacity) for stop_arr in trip_arr]
            for v_key, trip_arr in route_arr.items()}

