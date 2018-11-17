import numpy as np
import interpolate as interp
import math
import time
import sys
import topology

class Device:

	def __init__(self, location, resolution, size, permittivity_bounds, length_scale_meters):
		self.location = location
		self.resolution = resolution
		self.spacing = 1 / resolution
		self.size = size
		self.permittivity_bounds = permittivity_bounds
		self.length_scale_meters = length_scale_meters
		self.interpolation = interp.Interpolate(location, size, resolution, 3)
		self.num_filters = 0
		self.num_variables = self.num_filters + 1

		self.mid_permittivity = 0.5 * (self.permittivity_bounds[0] + self.permittivity_bounds[1])

		self.filters = []
		num_points = self.interpolation.num_points
		self.w = [np.zeros(num_points)] * self.num_variables
		self.w[0] = np.multiply(0.5, np.ones(num_points))

	def get_design_variable(self):
		return self.w[0]

	# def set_design_variable(self, design_variable):
	# 	# todo(groberts): copy versus reference assignment here
	# 	self.w[0] = design_variable.copy()
	# 	self.update_permittivity()

	def get_permittivity(self):
		return self.w[-1]

	# We can do this more cleanly by having the interpolator take different types of interpolation schemes
	# (i.e. - nearest or linear).  That will save some re-implementation here
	def get_permittivity_value(self, location):
		loc = [0]*3

		for n in range(0, 3):
			box_left = self.location[n] - 0.5 * self.size[n]
			box_left_below = self.spacing * (0.5 + math.floor(box_left * self.resolution - 0.5))

			loc[n] = int(np.floor((location[n] - box_left_below) * self.resolution))

		permittivity = self.get_permittivity()
		return permittivity[loc[0]][loc[1]][loc[2]]

	def update_permittivity(self):
		for filter_idx in range(0, self.num_filters):
			variable_in = self.w[filter_idx]
			variable_out = (self.filters[filter_idx]).forward(variable_in)
			self.w[filter_idx + 1] = variable_out

	def apply_binary_filter_pipeline(self, variable_in):
		# Hmm.. not sure how to make this explicit, but the filters run here do not need
		# to do any padding and only need to produce a partial output of the same size

		# Need a copy or else the original variable in will be modified, which is not what
		# is likely expected
		variable_out = np.copy(variable_in)
		for filter_idx in range(0, self.num_filters):
			if np.sum(np.isnan(variable_out)) > 0:
				print(filter_idx)
				# print(np.isnan(variable_out))
				sys.exit(1)

			variable_out = (self.filters[filter_idx]).fabricate(variable_out)

		return variable_out

	def proposed_design_step(self, gradient, step_size):
		for f_idx in range(0, self.num_filters):
			get_filter = self.filters[self.num_filters - f_idx - 1]
			variable_out = self.w[self.num_filters - f_idx]
			variable_in = self.w[self.num_filters - f_idx - 1]

			gradient = get_filter.chain_rule(gradient, variable_out, variable_in)

		proposed_design_variable = self.w[0] - np.multiply(step_size, gradient)
		proposed_design_variable = np.maximum(
									np.minimum(
										proposed_design_variable,
										self.maximum_design_value),
									self.minimum_design_value)

		return proposed_design_variable

	def fabrication_version(self):
		padded_design = np.pad(
			self.w[0],
			(
				(2 * self.pipeline_half_width[0], 2 * self.pipeline_half_width[0]),
				(2 * self.pipeline_half_width[1], 2 * self.pipeline_half_width[1]),
				(2 * self.pipeline_half_width[2], 2 * self.pipeline_half_width[2])
			),
			'constant'
		)

		# Step 4: We need here the current fabrication target before we start the stepping
		design_shape = (self.w[0]).shape

		current_device = self.convert_to_binary_map(self.apply_binary_filter_pipeline(
			padded_design[
				self.pipeline_half_width[0] : (self.pipeline_half_width[0] + design_shape[0]),
				self.pipeline_half_width[1] : (self.pipeline_half_width[1] + design_shape[1]),
				self.pipeline_half_width[2] : (self.pipeline_half_width[2] + design_shape[2])
			]
		))

		return current_device


	# In the step function, we should update the permittivity with update_permittivity
	# def step(self, gradient, step_size):
	# 	self.w[0] = self.proposed_design_step(gradient, step_size)
	# 	# Update the variable stack including getting the permittivity at the w[-1] position
	# 	self.update_permittivity()

	# Let's apply the topology on the device level!
	def step(self, gradient, step_size):
		topology_checker = topology.Topology()

		# Step 1: See which values cross the center line in the design variable
		proposed_design_variable = self.proposed_design_step(gradient, step_size)

		# Here, we have already made an assumption that you don't make a difference from your standpoint
		# unless you cross over the threshold
		proposed_changes = np.logical_xor(
			np.greater(proposed_design_variable, self.flip_threshold),
			np.greater(self.w[0], self.flip_threshold))

		print("We are trying to change " + str(np.sum(np.sum(proposed_changes))) + " voxels")

		# We have some we think we can safely move, so let's move those!
		self.w[0] = np.add(
			np.multiply(np.logical_not(proposed_changes), proposed_design_variable),
			np.multiply(proposed_changes, self.w[0]))

		# Step 2: The design we currently think we are manufacturing (i.e. - before we take this step)
		# is located in self.w[-1], so let's pull that out
		# pre_step_device = self.w[-1]

		# Step 3: We will need ultimately a double padded w[0] w.r.t. the pipeline half width in each dimension.  This will also serve
		# as the stepped design as we slowly figure out which pieces to change
		padded_design = np.pad(
			self.w[0],
			(
				(2 * self.pipeline_half_width[0], 2 * self.pipeline_half_width[0]),
				(2 * self.pipeline_half_width[1], 2 * self.pipeline_half_width[1]),
				(2 * self.pipeline_half_width[2], 2 * self.pipeline_half_width[2])
			),
			'constant'
		)

		# Step 4: We need here the current fabrication target before we start the stepping
		design_shape = (self.w[0]).shape

		current_device = self.convert_to_binary_map(self.apply_binary_filter_pipeline(
			padded_design[
				self.pipeline_half_width[0] : (self.pipeline_half_width[0] + design_shape[0]),
				self.pipeline_half_width[1] : (self.pipeline_half_width[1] + design_shape[1]),
				self.pipeline_half_width[2] : (self.pipeline_half_width[2] + design_shape[2])
			]
		))
		padded_current_device = np.pad(
			current_device,
			(
				(2 * self.pipeline_half_width[0], 2 * self.pipeline_half_width[0]),
				(2 * self.pipeline_half_width[1], 2 * self.pipeline_half_width[1]),
				(2 * self.pipeline_half_width[2], 2 * self.pipeline_half_width[2])
			),
			'constant'
		)

		# Step 5: We are going to need to look at each of the changes one at a time to ensure we move safely through the space
		design_shape = (self.w[0]).shape
		padded_design_shape = padded_design.shape
		# As is usually the case, these are inclusive start positions
		start_positions = np.array([2 * self.pipeline_half_width[0], 2 * self.pipeline_half_width[1], 2 * self.pipeline_half_width[2]])
		# May not be as obvious as start positions, but still usual case of exclusive end positions
		end_positions = padded_design_shape - start_positions

		# todo(groberts): only really need to calculate this once, so this can be factored out somewhere better
		x_coords = np.zeros((
			2 * self.pipeline_half_width[0] + 1,
			2 * self.pipeline_half_width[1] + 1,
			2 * self.pipeline_half_width[2] + 1),
			dtype=np.int32)
		y_coords = np.zeros((
			2 * self.pipeline_half_width[0] + 1,
			2 * self.pipeline_half_width[1] + 1,
			2 * self.pipeline_half_width[2] + 1),
			dtype=np.int32)
		z_coords = np.zeros((
			2 * self.pipeline_half_width[0] + 1,
			2 * self.pipeline_half_width[1] + 1,
			2 * self.pipeline_half_width[2] + 1),
			dtype=np.int32)


		for x_off in range(-self.pipeline_half_width[0], self.pipeline_half_width[0] + 1):
			for y_off in range(-self.pipeline_half_width[1], self.pipeline_half_width[1] + 1):
				for z_off in range(-self.pipeline_half_width[2], self.pipeline_half_width[2] + 1):
					x_coords[
						x_off + self.pipeline_half_width[0],
						y_off + self.pipeline_half_width[1],
						z_off + self.pipeline_half_width[2]] = x_off

					y_coords[
						x_off + self.pipeline_half_width[0],
						y_off + self.pipeline_half_width[1],
						z_off + self.pipeline_half_width[2]] = y_off

					z_coords[
						x_off + self.pipeline_half_width[0],
						y_off + self.pipeline_half_width[1],
						z_off + self.pipeline_half_width[2]] = z_off

		num_topology_evals = 0
		total_topo_time = 0

		for x in range(start_positions[0], end_positions[0]):
			for y in range(start_positions[1], end_positions[1]):
				for z in range(start_positions[2], end_positions[2]):

					# Do we want to change this voxel?
					if not proposed_changes[x - start_positions[0], y - start_positions[1], z - start_positions[2]]:
						continue

					# First, we extract the neighborhood we want to test the difference on
					stepped_design_neighborhood = padded_design[
						(x - 2 * self.pipeline_half_width[0]) : (x + 2 * self.pipeline_half_width[0]),
						(y - 2 * self.pipeline_half_width[1]) : (y + 2 * self.pipeline_half_width[1]),
						(z - 2 * self.pipeline_half_width[2]) : (z + 2 * self.pipeline_half_width[2]) ]
					stepped_design_neighborhood[
						2 * self.pipeline_half_width[0],
						2 * self.pipeline_half_width[1],
						2 * self.pipeline_half_width[2]] = proposed_design_variable[x - start_positions[0], y - start_positions[1], z - start_positions[2]]

					# Now, let's apply the fully binary version of our filters
					# print("before apply")
					stepped_device_neighborhood = self.convert_to_binary_map(self.apply_binary_filter_pipeline(stepped_design_neighborhood))

					# Ok, which ones are we trying to change
					desired_changes = np.logical_xor(
						stepped_device_neighborhood[
							self.pipeline_half_width[0] : (3 * self.pipeline_half_width[0] + 1),
							self.pipeline_half_width[1] : (3 * self.pipeline_half_width[1] + 1),
							self.pipeline_half_width[2] : (3 * self.pipeline_half_width[2] + 1) ],
						padded_current_device[
							(x - self.pipeline_half_width[0]) : (x + self.pipeline_half_width[0] + 1),
							(y - self.pipeline_half_width[1]) : (y + self.pipeline_half_width[1] + 1),
							(z - self.pipeline_half_width[2]) : (z + self.pipeline_half_width[2] + 1) ]
						)

					desired_x_coords = np.add(2 * self.pipeline_half_width[0], np.extract(desired_changes, x_coords))
					desired_y_coords = np.add(2 * self.pipeline_half_width[1], np.extract(desired_changes, y_coords))
					desired_z_coords = np.add(2 * self.pipeline_half_width[2], np.extract(desired_changes, z_coords))

					num_changes_needed = len(desired_x_coords)

					succesful_changes = np.zeros((num_changes_needed, 1))

					# points_to_address_stack = list(np.linspace(0, num_changes_needed - 1, num_changes_needed, dtype=np.int32))
					changed = True
					# print()
					start_topo = time.time()
					while changed and (np.sum(succesful_changes) < num_changes_needed):
						changed = False
						# print(str(len(points_to_address_stack)) + " " + str(num_changes_needed))

						for point_idx in range(0, num_changes_needed):

							# point_idx = points_to_address_stack.pop()
							if succesful_changes[point_idx]:
								continue

							x_coord = desired_x_coords[point_idx]
							y_coord = desired_y_coords[point_idx]
							z_coord = desired_z_coords[point_idx]	

							# todo(groberts): in case of pipeline_half_width = [0, 0, 0] or just 0 in some direction,
							# we may not have this amount of padding.  So it comes down to if you want to treat that
							# as a special case or not.  I say we treat it as a special case because it is much simpler!
							# Why is this the case?, Draw a picture!
							topology_check = topology_checker.d3_topology.topology_check(
								stepped_device_neighborhood[
									(x_coord - 1) : (x_coord + 2),
									(y_coord - 1) : (y_coord + 2),
									(z_coord - 1) : (z_coord + 2) ]
								)
							num_topology_evals += 1

							if topology_check:
								succesful_changes[point_idx] = 1
								changed = True
								stepped_device_neighborhood[x_coord, y_coord, z_coord] ^= True

					elapsed_topo = time.time() - start_topo
					total_topo_time += elapsed_topo

					# What do we do if we find out we actually can't move this one?
					if np.sum(succesful_changes) < num_changes_needed:
						# We actually don't know if we can move it at all... so we better not move it (in the case where we have no
						# spreading effect, then we know we can at least move it closer the border.  Here we don't know if we can move
						# it in either direction, so the possibility we get stuck is real!  If we are stuck, we can try to move things with
						# a smaller step size though, which will allow us to test a smaller step size.  Or we can try it with step directions
						# that move stuck pixels in the opposite direction
						# In some spreading cases, I believe we can employ the neutral zone technique as well given the nature of the filters
						# Actually, we have already made this assumption, so might as well play the neutral zone game
						# padded_design[
						# 	(x - 2 * self.pipeline_half_width[0]) : (x + 2 * self.pipeline_half_width[0]),
						# 	(y - 2 * self.pipeline_half_width[1]) : (y + 2 * self.pipeline_half_width[1]),
						# 	(z - 2 * self.pipeline_half_width[2]) : (z + 2 * self.pipeline_half_width[2]) ] = (
						# 		self.snap_design_value_to_neutral_zone_border(
						# 			padded_design[
						# 	(x - 2 * self.pipeline_half_width[0]) : (x + 2 * self.pipeline_half_width[0]),
						# 	(y - 2 * self.pipeline_half_width[1]) : (y + 2 * self.pipeline_half_width[1]),
						# 	(z - 2 * self.pipeline_half_width[2]) : (z + 2 * self.pipeline_half_width[2]) ]))

						padded_design[x, y, z] = self.snap_design_value_to_neutral_zone_border(padded_design[x, y, z])

						# Flip back the points we shouldn't have flipped
						for point_idx in range(0, num_changes_needed):
							if succesful_changes[point_idx]:
								x_coord = desired_x_coords[point_idx]
								y_coord = desired_y_coords[point_idx]
								z_coord = desired_z_coords[point_idx]

								stepped_device_neighborhood[x_coord, y_coord, z_coord] ^= True
					else:
						# print('Success!')
						# Success! So we can make this change!
						padded_design[
							(x - 2 * self.pipeline_half_width[0]) : (x + 2 * self.pipeline_half_width[0]),
							(y - 2 * self.pipeline_half_width[1]) : (y + 2 * self.pipeline_half_width[1]),
							(z - 2 * self.pipeline_half_width[2]) : (z + 2 * self.pipeline_half_width[2]) ] = (
								proposed_design_variable[x - start_positions[0], y - start_positions[1], z - start_positions[2]])



		print("Number of topology evals per point on average " + str(num_topology_evals / (.001 + np.sum(np.sum(proposed_changes)))))
		print("Number of total topo evals was " + str(num_topology_evals))
		print("The total time spent doing topology checks was " + str(total_topo_time) + " seconds")
		# print()
		# Now that we have made all the changes we think we can make, we need to set self.w[0] to the proper value
		# print(start_positions)
		# print(end_positions)
		# print(self.w[0].shape)
		self.w[0] = padded_design[start_positions[0] : end_positions[0], start_positions[1] : end_positions[1], start_positions[2] : end_positions[2]]
		# Also, make sure to update the whole chain!
		self.update_permittivity()





