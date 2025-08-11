import math

def reward_function(params):

	# Read input parameters
	steps = params['steps']
	speed = params['speed']
	heading = params['heading']
	progress = params['progress']
	waypoints = params['waypoints']
	track_width = params['track_width']
	closest_waypoints = params['closest_waypoints']
	distance_from_center = params['distance_from_center']

	# Initialize the reward with typical value
	reward = 1.0

	# Calculate the distance from each border
	distance_from_border = 0.5 * track_width - distance_from_center
	# Reward higher if the car stays inside the track borders
	if distance_from_border <= 0.05:
		reward = 1e-3 # Low reward if too close to the border or goes off the track
		return float(reward)
	
	# Calculate the direction of the center line based on the closest waypoints
	next_point = waypoints[closest_waypoints[1]]
	prev_point = waypoints[closest_waypoints[0]]

	# Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
	track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
	# Convert to degree
	track_direction = math.degrees(track_direction)
	# Calculate the difference between the track direction and the heading direction of the car
	direction_diff = abs(track_direction - heading)
	if direction_diff > 180:
		direction_diff = 360 - direction_diff

	# Penalize the reward if the difference is too large
	DIRECTION_THRESHOLD = 10.0
	if (abs(track_direction) <= 0.1 and abs(heading) <= 0.2 and speed >= 1.8):
		reward += 1.0
	if direction_diff > DIRECTION_THRESHOLD:
		reward *= 0.5
	if (abs(track_direction) >= 15):
		abs_steering = abs(params['steering_angle']) 
		if abs_steering < 10 and speed > 2.0:
			reward += 2.0

	TOTAL_NUM_STEPS = 300
	if (steps % 100) == 0 and progress > (steps / TOTAL_NUM_STEPS) * 100 :
		reward += 10.0

	if progress == 1.0:
		return float(reward * 100.0)

	return float(reward)