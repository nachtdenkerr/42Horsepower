import math
import numpy as np

ideal_data = np.array([[ 3.4823,  1.44  ,  2.119 ],
       [ 3.291 ,  1.4582,  2.4826],
       [ 3.0952,  1.4703,  2.8463],
       [ 2.8556,  1.4806,  3.2099],
       [ 2.5835,  1.4889,  3.5735],
       [ 2.3052,  1.4946,  3.9372],
       [ 2.0229,  1.4981,  4.    ],
       [ 1.7376,  1.4998,  4.    ],
       [ 1.4501,  1.5003,  4.    ],
       [ 1.1609,  1.4999,  4.    ],
       [ 0.8707,  1.4988,  4.    ],
       [ 0.58  ,  1.4974,  4.    ],
       [ 0.2889,  1.4958,  4.    ],
       [-0.0026,  1.4939,  4.    ],
       [-0.3054,  1.492 ,  4.    ],
       [-0.6082,  1.4903,  4.    ],
       [-0.9109,  1.4889,  4.    ],
       [-1.2137,  1.4878,  4.    ],
       [-1.5165,  1.487 ,  3.9995],
       [-1.8192,  1.4869,  3.9629],
       [-2.1219,  1.488 ,  3.8881],
       [-2.4245,  1.4907,  3.772 ],
       [-2.7269,  1.4957,  3.6096],
       [-3.0291,  1.5035,  3.4147],
       [-3.3307,  1.515 ,  3.2484],
       [-3.6153,  1.5304,  3.0895],
       [-3.8828,  1.524 ,  2.9372],
       [-4.1308,  1.4821,  2.7881],
       [-4.3562,  1.3915,  2.6418],
       [-4.546 ,  1.2365,  2.4876],
       [-4.7026,  1.0347,  2.362 ],
       [-4.8174,  0.7905,  2.2585],
       [-4.8787,  0.5218,  2.2277],
       [-4.8831,  0.2536,  2.2717],
       [-4.8367, -0.0008,  2.3745],
       [-4.7422, -0.2338,  2.4736],
       [-4.6007, -0.4392,  2.5895],
       [-4.4068, -0.6054,  2.7212],
       [-4.1801, -0.7416,  2.8703],
       [-3.928 , -0.8519,  3.0166],
       [-3.6563, -0.9407,  3.1712],
       [-3.3663, -1.0146,  3.3335],
       [-3.0715, -1.0732,  3.5118],
       [-2.7757, -1.1188,  3.6587],
       [-2.4794, -1.1536,  3.7772],
       [-2.1829, -1.1794,  3.8692],
       [-1.8867, -1.1976,  3.9363],
       [-1.5909, -1.2098,  3.9794],
       [-1.2961, -1.2172,  4.    ],
       [-1.0046, -1.2196,  4.    ],
       [-0.7203, -1.2159,  4.    ],
       [-0.4403, -1.2051,  4.    ],
       [-0.1637, -1.1854,  4.    ],
       [ 0.1096, -1.1546,  3.9938],
       [ 0.3793, -1.1088,  3.9553],
       [ 0.6475, -1.0558,  3.8811],
       [ 0.9155, -1.0062,  3.8054],
       [ 1.1836, -0.9641,  3.7353],
       [ 1.4513, -0.9338,  3.687 ],
       [ 1.718 , -0.9202,  3.6592],
       [ 1.9826, -0.9285,  3.6281],
       [ 2.2435, -0.9659,  3.5617],
       [ 2.4996, -1.0324,  3.4561],
       [ 2.7497, -1.1274,  3.3113],
       [ 2.9973, -1.2325,  3.1433],
       [ 3.2491, -1.337 ,  2.984 ],
       [ 3.5022, -1.4284,  2.8402],
       [ 3.757 , -1.4955,  2.7352],
       [ 4.0113, -1.5262,  2.6566],
       [ 4.2595, -1.51  ,  2.5856],
       [ 4.4906, -1.4369,  2.5215],
       [ 4.6863, -1.2985,  2.4847],
       [ 4.815 , -1.0884,  2.4642],
       [ 4.8988, -0.8451,  2.4638],
       [ 4.941 , -0.5766,  2.4866],
       [ 4.9483, -0.2908,  2.5147],
       [ 4.923 ,  0.0065,  2.54  ],
       [ 4.8644,  0.2909,  2.6146],
       [ 4.7742,  0.5514,  2.6769],
       [ 4.6537,  0.7853,  2.7281],
       [ 4.5039,  0.99  ,  2.7684],
       [ 4.3234,  1.159 ,  2.4966],
       [ 4.1133,  1.2785,  2.2008],
       [ 3.895 ,  1.3593,  1.9241],
       [ 3.6829,  1.4098,  1.6665],
       [ 3.4823,  1.44  ,  1.4248]])

def compute_direction(next_point, prev_point):
	x_prev, y_prev = prev_point[0], prev_point[1]
	x_next, y_next = next_point[0], next_point[1]
	angle = math.degrees(math.atan2(y_next - y_prev, x_next - x_prev))
	angle = (angle + 180.0) % 360.0 - 180.0
	return (angle)

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

	max_speed_on_straight = 2.0
	# --- Base reward ---
	if is_offtrack:
		return 1e-3  # tiny reward if off track

	reward = 1e-3
	ideal_line = ideal_data[:, :2]
	ideal_vel = ideal_data[:, 2]
	prev_point = ideal_line[closest_waypoints[0]]
	next_point = ideal_line[closest_waypoints[1]]
	index_ahead = closest_waypoints[1] + 1
	if (closest_waypoints[1] == 85):
		index_ahead = 1
	# --- Distance from center factor (smooth penalty) ---
	distance_from_ideal = get_distance_from_ideal_line(params['x'], params['y'], prev_point, next_point)
	distance_factor = max(1e-3, 1 - 2 * distance_from_ideal / track_width)
	reward += distance_factor

	# --- Speed factor ---
	speed_0 = ideal_vel[closest_waypoints[0]]
	speed_1 = ideal_vel[closest_waypoints[1]]
	max_speed = (speed_0 + speed_1) / 2 * (max_speed_on_straight / 4.0)
	if max_speed > 1e-3 and speed > max_speed:
		reward -= abs(speed - max_speed) / max_speed

	track_direction = compute_direction(next_point, prev_point)
	track_angle_ahead = compute_direction(ideal_line[index_ahead], next_point) 
	track_angle_diff = track_angle_ahead - track_direction
	track_angle_diff = (track_angle_diff + 180.0) % 360.0 - 180.0

	# --- Steering bonus ---
	steering_factor = 1.0
	if abs(track_angle_diff) <= 5.0:
		steering_factor = max(1 - abs(steering) / 15.0, 1e-3)
	else:
		if abs(steering) > 20.0:
			steering_factor = 0.8
	reward += steering_factor

	# --- Heading should be aligned with the track_direction
	direction_diff = abs(track_direction - heading)
	direction_diff = (direction_diff + 180.0) % 360.0 - 180.0
	
	heading_factor = 1.0
	if abs(track_angle_diff) <= 5.0:
		heading_factor = max(1 - (abs(direction_diff) / 10), 1e-3)
	else:
		heading_factor = max(1 - (abs(direction_diff) / 15), 1e-3)
	reward *= heading_factor

	# --- Progress bonus ---
	progress_factor = (progress / steps)  # scales by efficiency
	reward += progress_factor

	return float(reward)