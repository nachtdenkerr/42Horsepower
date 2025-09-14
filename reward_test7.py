import math
import numpy as np

ideal_data = np.array([[ 3.4882,  1.424 ,  2.1187],
				[ 3.2859,  1.4498,  2.196 ],
				[ 3.0778,  1.4676,  2.2593],
				[ 2.8556,  1.4806,  2.3101],
				[ 2.5985,  1.4918,  2.3499],
				[ 2.3217,  1.5002,  2.3797],
				[ 2.0395,  1.5056,  2.4   ],
				[ 1.7537,  1.5085,  2.4   ],
				[ 1.4654,  1.5092,  2.4   ],
				[ 1.175 ,  1.5082,  2.4   ],
				[ 0.8828,  1.5057,  2.4   ],
				[ 0.589 ,  1.5022,  2.4   ],
				[ 0.2938,  1.4978,  2.4   ],
				[-0.0025,  1.4928,  2.4   ],
				[-0.3053,  1.488 ,  2.4   ],
				[-0.6081,  1.4839,  2.4   ],
				[-0.911 ,  1.4804,  2.4   ],
				[-1.2139,  1.4779,  2.4   ],
				[-1.5169,  1.4765,  2.4   ],
				[-1.8199,  1.4766,  2.3943],
				[-2.1228,  1.4784,  2.3681],
				[-2.4255,  1.4825,  2.3203],
				[-2.7278,  1.4894,  2.2491],
				[-3.0296,  1.4998,  2.1537],
				[-3.3307,  1.515 ,  2.0659],
				[-3.6086,  1.5199,  1.9841],
				[-3.8717,  1.5027,  1.8739],
				[-4.1157,  1.4519,  1.6169],
				[-4.3378,  1.3589,  1.3504],
				[-4.5278,  1.2091,  1.4347],
				[-4.6815,  1.0101,  1.5005],
				[-4.7891,  0.7725,  1.5623],
				[-4.8433,  0.5167,  1.5391],
				[-4.8442,  0.2608,  1.5544],
				[-4.7976,  0.0156,  1.6068],
				[-4.7057, -0.2119,  1.54  ],
				[-4.5682, -0.4149,  1.4421],
				[-4.381 , -0.5828,  1.6196],
				[-4.1621, -0.724 ,  1.7848],
				[-3.9171, -0.8406,  1.942 ],
				[-3.6514, -0.9364,  2.0221],
				[-3.3663, -1.0146,  2.1003],
				[-3.0718, -1.0759,  2.1874],
				[-2.7759, -1.122 ,  2.2584],
				[-2.4795, -1.1551,  2.3143],
				[-2.1835, -1.1776,  2.3559],
				[-1.8892, -1.1908,  2.3838],
				[-1.5978, -1.1961,  2.3983],
				[-1.3093, -1.1947,  2.4   ],
				[-1.0235, -1.1876,  2.4   ],
				[-0.7402, -1.1756,  2.4   ],
				[-0.459 , -1.1591,  2.4   ],
				[-0.1799, -1.1384,  2.4   ],
				[ 0.0973, -1.1133,  2.4   ],
				[ 0.3724, -1.0834,  2.4   ],
				[ 0.6464, -1.0505,  2.3965],
				[ 0.9141, -1.0154,  2.3857],
				[ 1.1816, -0.9848,  2.3857],
				[ 1.4483, -0.9634,  2.3857],
				[ 1.7137, -0.9558,  2.3857],
				[ 1.9771, -0.9671,  2.3613],
				[ 2.2372, -1.0034,  2.2814],
				[ 2.493 , -1.0666,  2.3201],
				[ 2.7462, -1.1441,  2.2556],
				[ 2.9973, -1.2325,  2.1699],
				[ 3.2481, -1.3307,  2.0682],
				[ 3.4995, -1.4135,  1.9562],
				[ 3.7507, -1.4718,  1.8706],
				[ 3.9996, -1.4961,  1.6905],
				[ 4.2411, -1.477 ,  1.4575],
				[ 4.4662, -1.4051,  1.2819],
				[ 4.6593, -1.2723,  1.1287],
				[ 4.7926, -1.0708,  1.3401],
				[ 4.8845, -0.8359,  1.5299],
				[ 4.936 , -0.574 ,  1.6572],
				[ 4.9483, -0.2908,  1.6788],
				[ 4.9216,  0.0057,  1.7031],
				[ 4.8568,  0.2847,  1.7555],
				[ 4.7602,  0.5381,  1.801 ],
				[ 4.6357,  0.7654,  1.8268],
				[ 4.4837,  0.9627,  1.6951],
				[ 4.3055,  1.1254,  1.5492],
				[ 4.105 ,  1.245 ,  1.7042],
				[ 3.8971,  1.3285,  1.8407],
				[ 3.6909,  1.3856,  1.9622],
				[ 3.4882,  1.424 ,  2.0546],
				[ 3.4882,  1.424 ,  2.1187]])


def compute_direction(next_point, prev_point):
	x_prev, y_prev = prev_point[0], prev_point[1]
	x_next, y_next = next_point[0], next_point[1]
	return (math.degrees(math.atan2(y_next - y_prev, x_next - x_prev)))

def get_distance_from_ideal_line(x, y, ideal_point_1, ideal_point_2):
	"""Calculates the distance from the ideal line defined by ideal_line."""
	vector_ideal = np.array(ideal_point_2) - np.array(ideal_point_1)
	vector_car = np.array([x, y]) - np.array(ideal_point_1)
	if np.linalg.norm(vector_ideal) < 1e-6:
		distance = np.linalg.norm(vector_car)
	else:
		dot_product = np.dot(vector_car, vector_ideal)
		parallel_point = np.array(ideal_point_1) + (dot_product / np.dot(vector_ideal, vector_ideal)) * vector_ideal
		distance = np.linalg.norm(np.array([x, y]) - parallel_point)
	return distance

def	get_angle_diff_lookahead(ideal_line, params, n_lookahead):
	closest_idx = params['closest_waypoints'][1]
	ahead_idx = (closest_idx + n_lookahead) % len(params["waypoints"])

	angle_ahead = compute_direction(ideal_line[ahead_idx], ideal_line[closest_idx])
	angle_ahead = (angle_ahead + 180.0) % 360.0 - 180.0

	angle_adj = compute_direction(ideal_line[closest_idx + 1], ideal_line[closest_idx])

	angle_adj = (angle_adj + 180.0) % 360.0 - 180.0
	angle_diff = angle_ahead - angle_adj

	return ((angle_diff + 180.0) % 360.0 - 180.0)

def reward_function(params):
	# --- Read parameters ---
	speed = params['speed']
	heading = params['heading']
	steering = params['steering_angle']
	progress = params['progress']
	steps = max(params['steps'], 1)  # avoid division by zero
	waypoints = params['waypoints']
	closest_waypoints = params['closest_waypoints']
	track_width = params['track_width']
	is_offtrack = params['is_offtrack']

	# --- Base reward ---
	if is_offtrack:
		return 1e-3  # tiny reward if off track

	reward = 1e-3
	ideal_line = ideal_data[:, :2]
	ideal_vel = ideal_data[:, 2]
	prev_point = ideal_line[closest_waypoints[0]]
	next_point = ideal_line[closest_waypoints[1]]
	# --- Distance from center factor (smooth penalty) ---
	distance_from_ideal = get_distance_from_ideal_line(params['x'], params['y'], prev_point, next_point)
	distance_factor = max(1e-3, 1 - 2 * distance_from_ideal / track_width)

	reward += distance_factor

	angle_diff_lookahead = get_angle_diff_lookahead(ideal_line, params, 2)
	segment_type = 1
	if (abs(angle_diff_lookahead) >= 1): # will enter a turn
		segment_type = 2

	# --- Speed factor ---
	speed_0 = ideal_vel[closest_waypoints[0]]
	speed_1 = ideal_vel[closest_waypoints[1]]
	max_speed = (speed_0 + speed_1) / 2
	if max_speed > 1e-3:
		speed_factor = 1.0 - abs(speed - max_speed) / max_speed
	else:
		speed_factor = 0.1
	speed_factor = max(speed_factor, 1e-3)
	reward += speed_factor

	# --- Steering bonus ---
	steering_factor = 1.0
	if segment_type == 1:
		steering_factor = max(1 - abs(steering) / 10.0, 1e-3)
	else:
		wrong_turn = steering * angle_diff_lookahead
		if wrong_turn < 0.0:
			steering_factor -= abs(steering) / 30.0
	reward += steering_factor

	# --- Heading should be aligned with the track_direction
	track_direction = compute_direction(next_point, prev_point)
	direction_diff = abs(track_direction - heading)
	direction_diff = (direction_diff + 180.0) % 360.0 - 180.0
	if direction_diff > 10.0:
	    reward *= 0.5

	# --- Progress bonus ---
	progress_factor = (progress / steps)  # scales by efficiency
	reward += progress_factor

	return float(reward)