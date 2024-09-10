import numpy as np

type_dict = {"pickup": 0, "delivery": 1, "relocation": 2}


def stop_to_array(stop, max_order_size=3):
    arr_stop = [type_dict[stop["type"]],
                stop["origin"],
                stop["destination"],
                stop["restaurant_id"][2:] if stop["restaurant_id"] is not None else -1,
                stop["customer_id"][2:] if stop["customer_id"] is not None else -1,
                stop["start_at"],
                stop["started_at"] if stop["started_at"] is not None else -1,
                stop["estimated_time_required"],
                stop["estimated_time_required"] if stop["estimated_time_required"] is not None else -1,
                ]
    if stop["orders_to_pickup"] is None:
        n = 0
    else:
        n = len(stop["orders_to_pickup"])
    for i in range(max_order_size):
        if i < n:
            arr_stop.append(stop["orders_to_pickup"][i])
        else:
            arr_stop.append(-1)
    return arr_stop


def route_to_array(route):
    return {v_key: np.array([stop_to_array(stop) for stop in trip]) for v_key, trip in route.items()}
