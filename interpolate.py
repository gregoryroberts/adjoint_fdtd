import math
import numpy as np
import sys

class Interpolate():

	# Can we infer the number of dimensions from the size of the block?
	# Do we only allow the third dimension to be zero (shouldn't need to restrict like this)?
	# Yes, we should be able to beef up the significance of this
	def __init__(self, block_center, block_size, resolution, num_dim):
		self.block_center = block_center
		self.block_size = block_size
		self.resolution = resolution
		self.spacing = 1.0 / resolution
		self.num_dim = num_dim

		self.num_points = [0]*num_dim

		for n in range(0, num_dim):
			coord_left = block_center[n] - 0.5 * block_size[n]
			coord_right = block_center[n] + 0.5 * block_size[n]

			n_left = 1 + math.floor(coord_left * self.resolution - 0.5)
			n_right = 1 + math.ceil(coord_right * self.resolution - 0.5)

			self.num_points[n] = n_right - n_left + 1

	def interpolate_value(self, coordinate, values):

		loc_low = [0]*self.num_dim
		loc_high = [0]*self.num_dim
		weight_high = [0]*self.num_dim

		for n in range(0, self.num_dim):
			box_left = self.block_center[n] - 0.5 * self.block_size[n]
			box_left_low = self.spacing * (0.5 + math.floor(box_left * self.resolution - 0.5))

			offset_coord = coordinate[n] - box_left_low

			n_low = math.floor(offset_coord * self.resolution)
			n_high = 1 + n_low

			loc_low[n] = n_low
			loc_high[n] = n_high

			low_coord = n_low * self.spacing
			high_coord = n_high * self.spacing

			weight_high[n] = (offset_coord - low_coord) / self.spacing

		weight_x_high = weight_high[0]
		weight_x_low = 1 - weight_x_high
		loc_low_x = loc_low[0]
		loc_high_x = loc_high[0]

		weight_y_high = weight_high[1]
		weight_y_low = 1 - weight_y_high
		loc_low_y = loc_low[1]
		loc_high_y = loc_high[1]

		if (self.num_dim == 3):
			weight_z_high = weight_high[2]
			weight_z_low = 1 - weight_z_high
			loc_low_z = loc_low[2]
			loc_high_z = loc_high[2]

			x_interpolate_front_plane_low = weight_x_low * values[loc_low_x][loc_low_y][loc_low_z] + weight_x_high * values[loc_high_x][loc_low_y][loc_low_z]
			x_interpolate_front_plane_high = weight_x_low * values[loc_low_x][loc_high_y][loc_low_z] + weight_x_high * values[loc_high_x][loc_high_y][loc_low_z]

			x_interpolate_back_plane_low = weight_x_low * values[loc_low_x][loc_low_y][loc_high_z] + weight_x_high * values[loc_high_x][loc_low_y][loc_high_z]
			x_interpolate_back_plane_high = weight_x_low * values[loc_low_x][loc_high_y][loc_high_z] + weight_x_high * values[loc_high_x][loc_high_y][loc_high_z]

			y_interpolate_front_plane = weight_y_low * x_interpolate_front_plane_low + weight_y_high * x_interpolate_front_plane_high
			y_interpolate_back_plane = weight_y_low * x_interpolate_back_plane_low + weight_y_high * x_interpolate_back_plane_high

			z_interpolate = weight_z_low * y_interpolate_front_plane + weight_z_high * y_interpolate_back_plane

			return z_interpolate

		elif (self.num_dim == 2):
			x_interpolate_low = weight_x_low * values[loc_low_x][loc_low_y] + weight_x_high * values[loc_high_x][loc_low_y]
			x_interpolate_high = weight_x_low * values[loc_low_x][loc_high_y] + weight_x_high * values[loc_high_x][loc_high_y]

			y_interpolate = weight_y_low * x_interpolate_low + weight_y_high * x_interpolate_high

			return y_interpolate
