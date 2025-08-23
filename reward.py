import math
import numpy as np

ideal_line =	[[ 3.48816359e+00,  1.42395903e+00],
				[ 3.28587337e+00,  1.44977937e+00],
				[ 3.07777896e+00,  1.46759960e+00],
				[ 2.85560085e+00,  1.48058186e+00],
				[ 2.59854854e+00,  1.49176047e+00],
				[ 2.32173152e+00,  1.50016875e+00],
				[ 2.03947493e+00,  1.50557821e+00],
				[ 1.75371477e+00,  1.50845579e+00],
				[ 1.46537601e+00,  1.50919586e+00],
				[ 1.17496638e+00,  1.50817474e+00],
				[ 8.82770265e-01,  1.50573766e+00],
				[ 5.88992631e-01,  1.50218960e+00],
				[ 2.93837257e-01,  1.49779074e+00],
				[-2.47452957e-03,  1.49275711e+00],
				[-3.05251770e-01,  1.48803851e+00],
				[-6.08076941e-01,  1.48387722e+00],
				[-9.10957831e-01,  1.48043188e+00],
				[-1.21389817e+00,  1.47789871e+00],
				[-1.51688999e+00,  1.47651031e+00],
				[-1.81989907e+00,  1.47656341e+00],
				[-2.12283774e+00,  1.47842281e+00],
				[-2.42554462e+00,  1.48254293e+00],
				[-2.72784680e+00,  1.48941442e+00],
				[-3.02959544e+00,  1.49984481e+00],
				[-3.33065842e+00,  1.51495465e+00],
				[-3.60861788e+00,  1.51993536e+00],
				[-3.87166146e+00,  1.50266787e+00],
				[-4.11570237e+00,  1.45191090e+00],
				[-4.33778523e+00,  1.35888649e+00],
				[-4.52775133e+00,  1.20910898e+00],
				[-4.68153504e+00,  1.01008606e+00],
				[-4.78908787e+00,  7.72517935e-01],
				[-4.84325484e+00,  5.16679091e-01],
				[-4.84420716e+00,  2.60848924e-01],
				[-4.79759665e+00,  1.56410125e-02],
				[-4.70569282e+00, -2.11910797e-01],
				[-4.56823148e+00, -4.14855503e-01],
				[-4.38097570e+00, -5.82802859e-01],
				[-4.16206873e+00, -7.23960632e-01],
				[-3.91712774e+00, -8.40575540e-01],
				[-3.65136893e+00, -9.36399126e-01],
				[-3.36627812e+00, -1.01462628e+00],
				[-3.07175975e+00, -1.07589826e+00],
				[-2.77588033e+00, -1.12196601e+00],
				[-2.47948952e+00, -1.15512054e+00],
				[-2.18349357e+00, -1.17759764e+00],
				[-1.88922739e+00, -1.19084253e+00],
				[-1.59775905e+00, -1.19612871e+00],
				[-1.30928350e+00, -1.19469098e+00],
				[-1.02351668e+00, -1.18763271e+00],
				[-7.40179466e-01, -1.17562039e+00],
				[-4.59019524e-01, -1.15912452e+00],
				[-1.79866373e-01, -1.13838887e+00],
				[ 9.73013990e-02, -1.11328917e+00],
				[ 3.72417561e-01, -1.08337244e+00],
				[ 6.46367237e-01, -1.05051026e+00],
				[ 9.14119051e-01, -1.01537627e+00],
				[ 1.18155002e+00, -9.84802999e-01],
				[ 1.44826104e+00, -9.63440528e-01],
				[ 1.71372009e+00, -9.55807117e-01],
				[ 1.97712017e+00, -9.67050172e-01],
				[ 2.23719775e+00, -1.00344146e+00],
				[ 2.49299695e+00, -1.06664619e+00],
				[ 2.74620841e+00, -1.14409729e+00],
				[ 2.99730481e+00, -1.23246737e+00],
				[ 3.24812170e+00, -1.33067421e+00],
				[ 3.49950348e+00, -1.41350329e+00],
				[ 3.75074470e+00, -1.47175082e+00],
				[ 3.99958092e+00, -1.49605082e+00],
				[ 4.24112916e+00, -1.47700091e+00],
				[ 4.46616539e+00, -1.40512866e+00],
				[ 4.65927295e+00, -1.27231651e+00],
				[ 4.79261710e+00, -1.07082102e+00],
				[ 4.88454755e+00, -8.35910797e-01],
				[ 4.93597969e+00, -5.73975669e-01],
				[ 4.94831813e+00, -2.90757340e-01],
				[ 4.92157409e+00,  5.71567223e-03],
				[ 4.85684136e+00,  2.84735060e-01],
				[ 4.76016615e+00,  5.38136730e-01],
				[ 4.63574874e+00,  7.65405903e-01],
				[ 4.48374073e+00,  9.62706619e-01],
				[ 4.30548450e+00,  1.12544397e+00],
				[ 4.10496449e+00,  1.24500029e+00],
				[ 3.89713962e+00,  1.32851247e+00],
				[ 3.69090567e+00,  1.38557296e+00],
				[ 3.48816359e+00,  1.42395903e+00]]

prev_track_dir = []

def compute_direction(next_point, prev_point):
	x_prev, y_prev = prev_point[0], prev_point[1]
	x_next, y_next = next_point[0], next_point[1]
	return (math.degrees(math.atan2(x_next - x_prev, y_next - y_prev)))

def get_distance_from_ideal_line(x, y, ideal_point_1, ideal_point_2):
	"""Calculates the distance from the ideal line defined by ideal_line."""
	vector_ideal = np.array(ideal_point_2) - np.array(ideal_point_1)
	vector_car = np.array([x, y]) - np.array(ideal_point_1)
	dot_product = np.dot(vector_car, vector_ideal)
	parallel_point = np.array(ideal_point_1) + (dot_product / np.dot(vector_ideal, vector_ideal)) * vector_ideal
	distance = np.linalg.norm(np.array([x, y]) - parallel_point)
	return distance

def	angle_diff_lookahead(params, heading, n_lookahead=7):
	x, y = params["x"], params["y"]
	closest_idx = params['closest_waypoints'][1]
	target_idx = (closest_idx + n_lookahead) % len(params["waypoints"])
	x_ahead, y_ahead = ideal_line[target_idx]
	angle_ahead = math.degrees(math.atan2(y_ahead - y, x_ahead - x))
	angle_ahead = (angle_ahead + 180.0) % 360.0 - 180.0
	angle_diff = angle_ahead - heading
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

	# Reward in range [0.5, 1.5]
	reward *= (distance_factor + 0.5)

	track_direction = compute_direction(next_point, prev_point)
	if steps == 1:
		prev_track_dir.clear()
		prev_track_dir.append(heading)
	track_dir_diff = track_direction - prev_track_dir[-1]
	track_dir_diff = (track_dir_diff + 180.0) % 360.0 - 180.0

	angle_diff_lookahead = angle_diff_lookahead(params, heading)
	segment_type = 1
	if (abs(angle_diff_lookahead) >= 5.0): #will enter a turn, second tier of velocity
		segment_type = 2
	elif (abs(angle_diff_lookahead) >= 33.0): # sharper turn, third tier of velocity
		segment_type = 3
	
	# --- Heading should be aligned with the track_direction
	direction_diff = abs(track_direction - heading)
	direction_diff = (direction_diff + 180.0) % 360.0 - 180.0
	heading_factor = 1.0
	if segment_type == 1:
		heading_factor = max((1 - (direction_diff / 20))**1.2, 0.2)
	else:
		heading_factor = max((1 - (direction_diff / 30))**1.2, 0.2)
	heading_factor *= 1.2
	reward *= heading_factor 

	# --- Speed factor ---
	if segment_type == 1:
		speed_factor = (speed / max_speed_straight) ** 2.0
	elif segment_type == 2:  # Gentle curve
		speed_factor = (speed / max_speed_soft_corner) ** 1.5
	else:  # Sharp curve
		speed_factor = (speed / max_speed_sharp_corner) ** 1.5
	reward *= speed_factor * 2.0

	# --- Steering bonus ---
	steering_factor = 0.1
	if segment_type == 1:
		steering_factor = max((1 - abs(steering) / 30.0), 0.1)
	elif segment_type == 2:
		#check on rate of steering
	reward *= steering_factor * 1.2

	# --- Progress bonus ---
	progress_factor = (progress / steps) * 2.0  # scales by efficiency
	reward += progress_factor

	prev_track_dir.append(track_direction)
	return float(reward)