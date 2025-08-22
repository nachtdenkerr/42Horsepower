import math
import numpy as np

ideal_line =	[[ 3.48527602e+00,  1.39515945e+00],
				[ 3.27787335e+00,  1.43259221e+00],
				[ 3.06481037e+00,  1.45931525e+00],
				[ 2.84149704e+00,  1.47826027e+00],
				[ 2.59854854e+00,  1.49176047e+00],
				[ 2.32381710e+00,  1.50117582e+00],
				[ 2.04167648e+00,  1.50702217e+00],
				[ 1.75563933e+00,  1.50995011e+00],
				[ 1.46692807e+00,  1.51050151e+00],
				[ 1.17626000e+00,  1.50921124e+00],
				[ 8.84139253e-01,  1.50658572e+00],
				[ 5.91033246e-01,  1.50309876e+00],
				[ 2.97378160e-01,  1.49915037e+00],
				[-4.57222222e-03,  1.49476358e+00],
				[-3.06674254e-01,  1.49080696e+00],
				[-6.08939300e-01,  1.48738555e+00],
				[-9.11369557e-01,  1.48462140e+00],
				[-1.21395244e+00,  1.48265416e+00],
				[-1.51665706e+00,  1.48164319e+00],
				[-1.81943488e+00,  1.48177282e+00],
				[-2.12222156e+00,  1.48331589e+00],
				[-2.42493029e+00,  1.48660224e+00],
				[-2.72744319e+00,  1.49198466e+00],
				[-3.02954714e+00,  1.50017982e+00],
				[-3.31646119e+00,  1.50299952e+00],
				[-3.58885710e+00,  1.49062695e+00],
				[-3.84451054e+00,  1.45549207e+00],
				[-4.08210264e+00,  1.39147932e+00],
				[-4.29960938e+00,  1.29345689e+00],
				[-4.47924176e+00,  1.13937645e+00],
				[-4.61745108e+00,  9.45433885e-01],
				[-4.70972045e+00,  7.26566570e-01],
				[-4.75698822e+00,  4.96128661e-01],
				[-4.76106751e+00,  2.63354080e-01],
				[-4.72084304e+00,  3.57809299e-02],
				[-4.63487406e+00, -1.78971731e-01],
				[-4.49872649e+00, -3.71087172e-01],
				[-4.32328834e+00, -5.39311987e-01],
				[-4.11628928e+00, -6.84668208e-01],
				[-3.88260824e+00, -8.08038120e-01],
				[-3.62635487e+00, -9.10401268e-01],
				[-3.35220228e+00, -9.93051240e-01],
				[-3.06586010e+00, -1.05716412e+00],
				[-2.77345612e+00, -1.10475993e+00],
				[-2.47920377e+00, -1.13836614e+00],
				[-2.18553592e+00, -1.16033515e+00],
				[-1.89373526e+00, -1.17266815e+00],
				[-1.60427929e+00, -1.17704947e+00],
				[-1.31717800e+00, -1.17489013e+00],
				[-1.03215574e+00, -1.16752163e+00],
				[-7.48900062e-01, -1.15596366e+00],
				[-4.67089920e-01, -1.14108282e+00],
				[-1.86402997e-01, -1.12366585e+00],
				[ 9.34193103e-02, -1.10431752e+00],
				[ 3.72628822e-01, -1.08360978e+00],
				[ 6.40740512e-01, -1.06285216e+00],
				[ 9.08384224e-01, -1.04434528e+00],
				[ 1.17512682e+00, -1.03039394e+00],
				[ 1.44056569e+00, -1.02365269e+00],
				[ 1.70443119e+00, -1.02673936e+00],
				[ 1.96647494e+00, -1.04263491e+00],
				[ 2.22614406e+00, -1.07586639e+00],
				[ 2.48417327e+00, -1.12019719e+00],
				[ 2.74047943e+00, -1.17481172e+00],
				[ 2.99477576e+00, -1.23911579e+00],
				[ 3.24631472e+00, -1.31279435e+00],
				[ 3.49329333e+00, -1.37486356e+00],
				[ 3.73734230e+00, -1.41686730e+00],
				[ 3.97613814e+00, -1.43072356e+00],
				[ 4.20579337e+00, -1.40788363e+00],
				[ 4.42004759e+00, -1.34025474e+00],
				[ 4.60705911e+00, -1.21734112e+00],
				[ 4.74211968e+00, -1.02739593e+00],
				[ 4.83961328e+00, -8.03633333e-01],
				[ 4.89730178e+00, -5.51485849e-01],
				[ 4.91305533e+00, -2.79591146e-01],
				[ 4.88523372e+00, -3.77982031e-03],
				[ 4.81765419e+00,  2.58094778e-01],
				[ 4.71704401e+00,  4.97222754e-01],
				[ 4.58966242e+00,  7.10875317e-01],
				[ 4.44004865e+00,  8.97906242e-01],
				[ 4.27144022e+00,  1.05665102e+00],
				[ 4.08517267e+00,  1.17935515e+00],
				[ 3.88967835e+00,  1.27312952e+00],
				[ 3.68912672e+00,  1.34337537e+00],
				[ 3.48527602e+00,  1.39515945e+00]]

prev_track_dir = []

def get_distance_from_ideal_line(x, y, ideal_point_1, ideal_point_2):
	"""Calculates the distance from the ideal line defined by ideal_line."""
	vector_ideal = np.array(ideal_point_2) - np.array(ideal_point_1)
	vector_car = np.array([x, y]) - np.array(ideal_point_1)
	dot_product = np.dot(vector_car, vector_ideal)
	parallel_point = np.array(ideal_point_1) + (dot_product / np.dot(vector_ideal, vector_ideal)) * vector_ideal
	distance = np.linalg.norm(np.array([x, y]) - parallel_point)
	return distance

def reward_function(params):
	# --- Read parameters ---
	speed = params['speed']
	heading = params['heading']
	steering = params['steering_angle']
	distance_from_center = params['distance_from_center']
	progress = params['progress']
	steps = max(params['steps'], 1)  # avoid division by zero
	waypoints = params['waypoints']
	closest_waypoints = params['closest_waypoints']
	track_width = params['track_width']
	is_offtrack = params['is_offtrack']

	# --- Max speed based on action space ---
	# should be changed according to the updated action space
	max_speed_straight = 2.5
	max_speed_soft_corner = 1.8
	max_speed_sharp_corner = 1.2

	# --- Base reward ---
	if is_offtrack:
		return 1e-3  # tiny reward if off track

	reward = 1.0 

	prev_point = ideal_line[closest_waypoints[0]]
	next_point = ideal_line[closest_waypoints[1]]
	# --- Distance from center factor (smooth penalty) ---
	distance_from_ideal = get_distance_from_ideal_line(params['x'], params['y'], prev_point, next_point)
	distance_factor = max(0.0, 1 - 2 * distance_from_ideal / track_width)

	reward *= (distance_factor + 0.5)

	track_direction = math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))
	if steps == 1:
		prev_track_dir.clear()
		prev_track_dir.append(heading)
	direction = 1 # Straight 
	track_dir_diff = track_direction - prev_track_dir[-1]
	track_dir_diff = (track_dir_diff + 180) % 360 - 180

	if (track_dir_diff >= 5.0):
		direction = 2 # Left 
	elif (track_dir_diff <= -5.0):
		direction = 3 #Right

	# --- Heading should be aligned with the track_direction
	direction_diff = abs(track_direction - heading)
	if direction_diff > 180:
		direction_diff = 360 - direction_diff
	heading_factor = 1.0
	if direction == 1:
		heading_factor = max((1 - (direction_diff / 20))**1.2, 0.2)
	else:
		heading_factor = max((1 - (direction_diff / 30))**1.2, 0.2)
	heading_factor *= 1.2
	reward *= heading_factor 

	# --- Speed factor ---
	if direction == 1:
		speed_factor = (speed / max_speed_straight) ** 2.0
	elif abs(direction_diff) < 20.0:  # Gentle curve
		speed_factor = (speed / max_speed_soft_corner) ** 1.5
	else:  # Sharp curve
		speed_factor = (speed / max_speed_sharp_corner) ** 1.5
	reward *= speed_factor * 2.0

	# --- Steering bonus ---
	steering_factor = 0.1
	if direction == 1:
		if steering <= 5.0:
			steering_factor = max((1 - steering / 5.0), 0.5)
	elif direction == 2:
		if track_dir_diff <= 15.0 and steering >= 5.0:
			steering_factor = 1.0
		if track_dir_diff > 15.0 and steering >= 12.0:
			steering_factor = 1.0
	elif direction == 3:
		if track_dir_diff >= -15.0 and steering <= -5.0:
			steering_factor = 1.0
		if track_dir_diff < -15.0 and steering <= -12.0:
			steering_factor = 1.0
	reward *= steering_factor * 1.2

	# --- Progress bonus ---
	progress_factor = (progress / steps) * 2.0  # scales by efficiency
	reward += progress_factor

	prev_track_dir.append(track_direction)
	return float(reward)