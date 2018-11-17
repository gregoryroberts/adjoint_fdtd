import bayer_filter
import math
import meep as mp
import interpolate as interp
import time
import numpy as np
import sys

# This should be pulled out to a different file that can be used for normalizations
# So should the other type of normalization
def normalize_by_frequency_gaussian_pulse(freq, fwidth, center_freq, t_offset):
	width = 1.0 / fwidth
	omega = 2 * math.pi * freq
	center_omega = 2 * math.pi * center_freq


	return np.abs(width) * (-omega / center_omega) * np.exp(-0.5 * width * width * (omega - center_omega)**2) * np.exp(np.complex(0, 1) * omega * t_offset)

	# return np.abs(width) * (-omega / center_omega) * np.exp(-0.5 * width * width * (omega - center_omega)**2) * np.exp(-np.complex(0, 1) * omega * t_offset)



class BayerOptimization():

	# Let's set up constants so that we can keep track of these quadrants and ranges in a more readable way
	RANGE_RED = 0
	RANGE_GREEN = 1
	RANGE_BLUE = 2

	QUADRANT_RED = 2
	QUADRANT_GREEN_XPOL = 1
	QUADRANT_GREEN_YPOL = 3
	QUADRANT_BLUE = 0

	QUADRANT_TOP_RIGHT = 0
	QUADRANT_TOP_LEFT = 1
	QUADRANT_BOTTOM_LEFT = 2
	QUADRANT_BOTTOM_RIGHT = 3

	###################################
	#                #                #
	#                #                #
	#   P1 (G, X)    #      P0 (B)    #
	#                #                #
	#                #                #
	###################################
	#                #                #
	#                #                #
	#     P2 (R)     #    P3 (G, Y)   #
	#                #                #
	#                #                #
	###################################

	def __init__(self, resolution):
		self.block_index = 1.53
		self.block_permittivity = self.block_index**2
		self.permittivity_bounds = [1.0, self.block_permittivity]

		self.length_scale_meters = 1e-6
		self.length_scale_microns = self.length_scale_meters * 1e6

		self.num_wavelengths_per_range = 1
		self.num_ranges = 3
		self.num_total_wavelengths = self.num_ranges * self.num_wavelengths_per_range
		self.num_total_frequencies = self.num_total_wavelengths

		self.blue_lower_lambda_um = .4
		self.blue_upper_lambda_um = .5
		self.green_lower_lambda_um = self.blue_upper_lambda_um
		self.green_upper_lambda_um = .6
		self.red_lower_lambda_um = self.green_upper_lambda_um
		self.red_upper_lambda_um = .7

		self.source_lower_lambda_um = self.blue_lower_lambda_um
		self.source_upper_lambda_um = self.red_upper_lambda_um
		self.source_upper_freq_unitless = self.length_scale_microns / self.source_lower_lambda_um;
		self.source_lower_freq_unitless = self.length_scale_microns / self.source_upper_lambda_um;
		self.source_freq_spread_unitless = self.source_upper_freq_unitless - self.source_lower_freq_unitless;
		self.source_freq_center_unitless = 0.5 * (self.source_lower_freq_unitless + self.source_upper_freq_unitless);

		self.dft_freq_min = self.source_lower_freq_unitless
		self.dft_freq_max = self.source_upper_freq_unitless
		self.dft_freq_spread = self.dft_freq_max - self.dft_freq_min
		self.dft_freq_center = self.dft_freq_min + 0.5 * self.dft_freq_spread


		self.frequencies_in_ranges = [[], [], []]
		self.max_intensity = [0] * self.num_total_frequencies
		for freq_idx in range(0, self.num_total_frequencies):
			frequency = ((float(freq_idx) / float(self.num_total_frequencies - 1)) * self.dft_freq_spread) + self.dft_freq_min
			wavelength_um = self.length_scale_microns / frequency

			self.max_intensity[freq_idx] = 1.0 / (wavelength_um * wavelength_um)

			self.frequencies_in_ranges[self.check_frequency_range(frequency)].append(frequency)

		self.max_frequencies_in_a_range = max(list(map(lambda y: len(y), self.frequencies_in_ranges)))

		# Maybe... set this by a real length scaled by the length scale?
		# self.spacing = 0.025 / self.length_scale_microns
		# self.resolution = int(1 / self.spacing)
		self.resolution = resolution
		self.spacing = 1 / resolution

		self.focal_length = 1.5 / self.length_scale_microns
		self.side_gap = 0.5 / self.length_scale_microns
		self.top_bottom_gap = .2 / self.length_scale_microns


		self.block_dim_x = 2 / self.length_scale_microns;
		self.block_dim_y = 2 / self.length_scale_microns;
		self.block_dim_z = 2 / self.length_scale_microns;
		self.block_size = mp.Vector3(self.block_dim_x, self.block_dim_y, self.block_dim_z)

		self.source_loc_z_from_pml = 0.0 / self.length_scale_microns;

		self.t_pml_x = 1.0 / self.length_scale_microns
		self.t_pml_y = 1.0 / self.length_scale_microns
		self.t_pml_z = 1.0 / self.length_scale_microns

		self.cell_dim_x = 2 * self.t_pml_x + 2 * self.side_gap + self.block_dim_x;
		self.cell_dim_y = 2 * self.t_pml_y + 2 * self.side_gap + self.block_dim_y;
		self.cell_dim_z = 2 * self.t_pml_z + 2 * self.top_bottom_gap + self.block_dim_z + self.focal_length;

		self.cell = mp.Vector3(self.cell_dim_x, self.cell_dim_y, self.cell_dim_z)

		self.block_center = mp.Vector3(0, 0, 0.5 * self.cell_dim_z - self.t_pml_z - self.top_bottom_gap - 0.5 * self.block_dim_z)
		self.top_source_center = mp.Vector3(0, 0, 0.5 * self.cell_dim_z - self.t_pml_z)

		self.device = bayer_filter.BayerFilter(self.block_center, self.resolution, self.block_size, self.permittivity_bounds, self.length_scale_meters);

		self.pml_layers = [mp.PML(thickness = self.t_pml_x, direction = mp.X), mp.PML(thickness = self.t_pml_y, direction = mp.Y), mp.PML(thickness = self.t_pml_z, direction = mp.Z)]

		# Run all the way up to and through the PML boundary layers
		self.top_bottom_source_size_x = self.cell_dim_x
		self.top_bottom_source_size_y = self.cell_dim_y
		# Surface currenets
		self.top_bottom_source_size_z = 0;
		self.top_bottom_source_size = mp.Vector3(self.top_bottom_source_size_x, self.top_bottom_source_size_y, self.top_bottom_source_size_z)

		# This is the default value for it, but let's be explicit
		self.gaussian_cutoff = 5.0
		self.gaussian_width = 1.0 / self.source_freq_spread_unitless
		self.gaussian_peak_time = self.gaussian_cutoff * self.gaussian_width

		top_source_J = mp.Source(mp.GaussianSource(self.source_freq_center_unitless, fwidth=self.source_freq_spread_unitless, cutoff=self.gaussian_cutoff, start_time=0),
			component=mp.Ex, amplitude=1.0, center=self.top_source_center, size=self.top_bottom_source_size)
		self.forward_sources = [top_source_J]

		self.geometry = [mp.Block(size=self.block_size, center=self.block_center, epsilon_func=lambda loc: self.device.get_permittivity_value(loc))]

		self.z_measurement_point = self.block_center.z - 0.5 * self.block_dim_z - self.focal_length


		# Let's set up where we want to measure the figure of merit
		self.measure_blue_focal = mp.Vector3(self.block_dim_x / 4.0, self.block_dim_y / 4.0, self.z_measurement_point)
		self.measure_green_xpol_focal = mp.Vector3(-self.block_dim_x / 4.0, self.block_dim_y / 4.0, self.z_measurement_point)
		self.measure_red_focal = mp.Vector3(-self.block_dim_x / 4.0, -self.block_dim_y / 4.0, self.z_measurement_point)
		self.measure_green_ypol_focal = mp.Vector3(self.block_dim_x / 4.0, -self.block_dim_y / 4.0, self.z_measurement_point)

		self.dft_focal_plane_fields_center = mp.Vector3(0, 0, self.z_measurement_point)
		self.dft_focal_area_fields_size = mp.Vector3(self.block_dim_x, self.block_dim_y, 0);
		self.interpolate_focal_area_fields = interp.Interpolate(self.dft_focal_plane_fields_center, self.dft_focal_area_fields_size, self.resolution, 2)


		self.dft_block_fields_center = self.block_center
		self.dft_block_fields_size = self.block_size
		self.focal_plane_centers = [self.measure_blue_focal, self.measure_green_xpol_focal, self.measure_red_focal, self.measure_green_ypol_focal]


		self.xy_symmetry_transformation = [0] * 4
		self.xy_symmetry_transformation[BayerOptimization.QUADRANT_TOP_RIGHT] = BayerOptimization.QUADRANT_TOP_RIGHT
		self.xy_symmetry_transformation[BayerOptimization.QUADRANT_TOP_LEFT] = BayerOptimization.QUADRANT_BOTTOM_RIGHT
		self.xy_symmetry_transformation[BayerOptimization.QUADRANT_BOTTOM_LEFT] = BayerOptimization.QUADRANT_BOTTOM_LEFT
		self.xy_symmetry_transformation[BayerOptimization.QUADRANT_BOTTOM_RIGHT] = BayerOptimization.QUADRANT_TOP_LEFT

		self.range_to_quadrant = [0] * 4
		self.range_to_quadrant[BayerOptimization.RANGE_RED] = BayerOptimization.QUADRANT_RED
		self.range_to_quadrant[BayerOptimization.RANGE_GREEN] = [BayerOptimization.QUADRANT_GREEN_XPOL, BayerOptimization.QUADRANT_GREEN_YPOL]
		self.range_to_quadrant[BayerOptimization.RANGE_BLUE] = BayerOptimization.QUADRANT_BLUE

		self.quadrant_to_range = [0] * 4
		self.quadrant_to_range[BayerOptimization.QUADRANT_RED] = BayerOptimization.RANGE_RED
		self.quadrant_to_range[BayerOptimization.QUADRANT_GREEN_XPOL] = BayerOptimization.RANGE_GREEN
		self.quadrant_to_range[BayerOptimization.QUADRANT_GREEN_YPOL] = BayerOptimization.RANGE_GREEN
		self.quadrant_to_range[BayerOptimization.QUADRANT_BLUE] = BayerOptimization.RANGE_BLUE


		# Dipole sources have the following size in Meep
		dipole_source_size = mp.Vector3(0, 0, 0)

		# Adjoint source 0 corresponds to the blue quadrant
		adjoint_source_0 = mp.Source(mp.GaussianSource(self.source_freq_center_unitless, fwidth=self.source_freq_spread_unitless, cutoff=self.gaussian_cutoff, start_time=0), component=mp.Ex,
							 center=self.measure_blue_focal, amplitude=np.complex(0, 1), size=dipole_source_size);


		# Adjoint source 1 corresponds to the green x-polarized quadrant
		adjoint_source_1 = mp.Source(mp.GaussianSource(self.source_freq_center_unitless, fwidth=self.source_freq_spread_unitless, cutoff=self.gaussian_cutoff, start_time=0), component=mp.Ex,
							 center=self.measure_green_xpol_focal, amplitude=np.complex(0, 1), size=dipole_source_size);


		# Adjoint source 2 corresponds to the red quadrant
		adjoint_source_2 = mp.Source(mp.GaussianSource(self.source_freq_center_unitless, fwidth=self.source_freq_spread_unitless, cutoff=self.gaussian_cutoff, start_time=0), component=mp.Ex,
							 center=self.measure_red_focal, amplitude=np.complex(0, 1), size=dipole_source_size);


		# Adjoint source 3 corresponds to the green y-polarized quadrant
		adjoint_source_3 = mp.Source(mp.GaussianSource(self.source_freq_center_unitless, fwidth=self.source_freq_spread_unitless, cutoff=self.gaussian_cutoff, start_time=0), component=mp.Ex,
							 center=self.measure_green_ypol_focal, amplitude=np.complex(0, 1), size=dipole_source_size);

		self.adjoint_sources = [adjoint_source_0, adjoint_source_1, adjoint_source_2, adjoint_source_3]


		num_material_points = self.device.interpolation.num_points

		self.forward_fields_xpol = np.zeros((self.num_ranges, self.max_frequencies_in_a_range, 3, *num_material_points), dtype=np.complex)
		self.adjoint_fields_xpol = np.zeros((len(self.adjoint_sources), self.num_ranges, self.max_frequencies_in_a_range, 3, *num_material_points), dtype=np.complex)


	def gen_empty_phasor(self, num_frequencies):
		return np.zeros((num_frequencies,
			*(self.device.interpolation.num_points)),
			dtype=np.complex)

	def check_frequency_range(self, frequency):
		lambda_um = self.length_scale_microns / frequency
		if (lambda_um < self.blue_upper_lambda_um):
			return 2
		elif (lambda_um < self.green_upper_lambda_um):
			return 1
		else:
			return 0

	def collect_focal_plane_power(self, sim, Ex_xpol_focal, Ey_xpol_focal, Ex_ypol_focal, Ey_ypol_focal, focal_plane_powers):
		# First collect all the data for all the frequences at every quadrant's focal point for the x-polarized input
		for freq_idx in range(0, self.num_total_frequencies):
			frequency = ((float(freq_idx) / float(self.num_total_frequencies - 1)) * self.dft_freq_spread) + self.dft_freq_min

			normalization_for_frequency = normalize_by_frequency_gaussian_pulse(frequency, self.source_freq_spread_unitless, self.source_freq_center_unitless, self.gaussian_peak_time)

			Ex_dft_data_xpol = np.divide(sim.get_dft_array(self.dft_focal_area_fields, mp.Ex, freq_idx), normalization_for_frequency)
			Ey_dft_data_xpol = np.divide(sim.get_dft_array(self.dft_focal_area_fields, mp.Ey, freq_idx), normalization_for_frequency)

			for quadrant_idx in range(0, 4):
				measure_point_xpol = self.focal_plane_centers[quadrant_idx]

				ex_xpol_datapoint = self.interpolate_focal_area_fields.interpolate_value(measure_point_xpol, Ex_dft_data_xpol)
				ey_xpol_datapoint = self.interpolate_focal_area_fields.interpolate_value(measure_point_xpol, Ey_dft_data_xpol)

				Ex_xpol_focal[quadrant_idx].append(ex_xpol_datapoint)
				Ey_xpol_focal[quadrant_idx].append(ey_xpol_datapoint)


		# Now, use symmetry to fill in the y-polarized data
		for freq_idx in range(0, self.num_total_frequencies):
			for quadrant_idx in range(0, 4):
				ex_ypol_datapoint = Ey_xpol_focal[self.xy_symmetry_transformation[quadrant_idx]][freq_idx]
				ey_ypol_datapoint = Ex_xpol_focal[self.xy_symmetry_transformation[quadrant_idx]][freq_idx]

				Ex_ypol_focal[quadrant_idx].append(ex_ypol_datapoint)
				Ey_ypol_focal[quadrant_idx].append(ey_ypol_datapoint)


		# Now, we can compute the intensities at each focal point for our figures of merit
		for freq_idx in range(0, self.num_total_frequencies):
			frequency = ((float(freq_idx) / float(self.num_total_frequencies - 1)) * self.dft_freq_spread) + self.dft_freq_min
			get_bin = self.check_frequency_range(frequency)
			quadrant = self.range_to_quadrant[get_bin]

			if (get_bin == 1):
				focal_plane_powers[get_bin].append(
					(np.abs(Ex_xpol_focal[quadrant[0]][freq_idx])**2 + np.abs(Ey_xpol_focal[quadrant[0]][freq_idx])**2 +
					np.abs(Ex_ypol_focal[quadrant[1]][freq_idx])**2 + np.abs(Ey_ypol_focal[quadrant[1]][freq_idx])**2) / self.max_intensity[freq_idx])
			else:
				focal_plane_powers[get_bin].append(
					(np.abs(Ex_xpol_focal[quadrant][freq_idx])**2 + np.abs(Ey_xpol_focal[quadrant][freq_idx])**2 +
					np.abs(Ex_ypol_focal[quadrant][freq_idx])**2 + np.abs(Ey_ypol_focal[quadrant][freq_idx])**2) / self.max_intensity[freq_idx])


	def collect_e_fields(self, sim, e_fields):
		bin_lengths = [0] * self.num_ranges

		for freq_idx in range(0, self.num_total_frequencies):
			frequency = ((float(freq_idx) / float(self.num_total_frequencies - 1)) * self.dft_freq_spread) + self.dft_freq_min
			get_bin = self.check_frequency_range(frequency)

			normalization_for_frequency = normalize_by_frequency_gaussian_pulse(frequency, self.source_freq_spread_unitless, self.source_freq_center_unitless, self.gaussian_peak_time)

			Ex_dft_data = np.divide(sim.get_dft_array(self.dft_block_fields, mp.Ex, freq_idx), normalization_for_frequency)
			Ey_dft_data = np.divide(sim.get_dft_array(self.dft_block_fields, mp.Ey, freq_idx), normalization_for_frequency)
			Ez_dft_data = np.divide(sim.get_dft_array(self.dft_block_fields, mp.Ez, freq_idx), normalization_for_frequency)

			current_bin_location = bin_lengths[get_bin]
			bin_lengths[get_bin] += 1

			e_fields[get_bin][current_bin_location][0] = Ex_dft_data
			e_fields[get_bin][current_bin_location][1] = Ey_dft_data
			e_fields[get_bin][current_bin_location][2] = Ez_dft_data


	def collect_forward_volume_e_fields_xpol(self, sim):
		self.collect_e_fields(sim, self.forward_fields_xpol)

	def collect_adjoint_volume_e_fields_xpol(self, sim, adj_src_idx):
		self.collect_e_fields(sim, self.adjoint_fields_xpol[adj_src_idx])

	def run_optimization(self, num_epochs, num_iterations_per_epoch):
		num_frequencies_blue = len(self.frequencies_in_ranges[BayerOptimization.RANGE_BLUE])
		num_frequencies_green = len(self.frequencies_in_ranges[BayerOptimization.RANGE_GREEN])
		num_frequencies_red = len(self.frequencies_in_ranges[BayerOptimization.RANGE_RED])

		frequency_offset_blue = 0
		frequency_offset_green = 0
		frequency_offset_red = 0

		for range_idx in range(0, self.num_ranges):
			if (range_idx < BayerOptimization.RANGE_RED):
				frequency_offset_red += len(self.frequencies_in_ranges[range_idx])

			if (range_idx < BayerOptimization.RANGE_GREEN):
				frequency_offset_green += len(self.frequencies_in_ranges[range_idx])

			if (range_idx < BayerOptimization.RANGE_BLUE):
				frequency_offset_blue += len(self.frequencies_in_ranges[range_idx])


		phasor_blue_xx = self.gen_empty_phasor(num_frequencies_blue)
		phasor_blue_xy = self.gen_empty_phasor(num_frequencies_blue)
		phasor_blue_yx = self.gen_empty_phasor(num_frequencies_blue)
		phasor_blue_yy = self.gen_empty_phasor(num_frequencies_blue)

		phasor_green_xpol_xx = self.gen_empty_phasor(num_frequencies_green)
		phasor_green_xpol_xy = self.gen_empty_phasor(num_frequencies_green)
		phasor_green_xpol_yx = self.gen_empty_phasor(num_frequencies_green)
		phasor_green_xpol_yy = self.gen_empty_phasor(num_frequencies_green)

		phasor_red_xx = self.gen_empty_phasor(num_frequencies_red)
		phasor_red_xy = self.gen_empty_phasor(num_frequencies_red)
		phasor_red_yx = self.gen_empty_phasor(num_frequencies_red)
		phasor_red_yy = self.gen_empty_phasor(num_frequencies_red)

		phasor_green_ypol_xx = self.gen_empty_phasor(num_frequencies_green)
		phasor_green_ypol_xy = self.gen_empty_phasor(num_frequencies_green)
		phasor_green_ypol_yx = self.gen_empty_phasor(num_frequencies_green)
		phasor_green_ypol_yy = self.gen_empty_phasor(num_frequencies_green)

		self.figure_of_merit = np.zeros((num_epochs * num_iterations_per_epoch, self.num_ranges))
		self.figure_of_merit_average = np.zeros(num_epochs * num_iterations_per_epoch)
		self.iteration_time_sec = np.zeros(num_epochs * num_iterations_per_epoch)
		self.weights = np.zeros((num_epochs * num_iterations_per_epoch, self.num_ranges))

		self.sim = None
		max_step_size = 4 * (1 / (self.block_permittivity - 1)) * 0.0625 * self.block_dim_x * self.block_dim_y * self.block_dim_z
		min_step_size = 0.1 * max_step_size
		step_size_range = (max_step_size - min_step_size)
		step_size_step = 0
		if num_iterations_per_epoch > 1:
			step_size_step = (step_size_range / (num_iterations_per_epoch - 1))

		for epoch in range(0, num_epochs):
			self.device.update_filters(epoch)
			step_size = max_step_size
			data_offset = epoch * num_iterations_per_epoch

			for iteration in range(0, num_iterations_per_epoch):
				start_time = time.time()

				if self.sim is not None:
					self.sim.reset_meep()

				self.sim = mp.Simulation(cell_size=self.cell,
				                    boundary_layers=self.pml_layers,
				                    geometry=self.geometry,
				                    sources=self.forward_sources,
				                    resolution=self.resolution)
				# should you widen your pulse in frequency space to make sure it includes these frequence components at stronger levels?
				self.dft_block_fields = self.sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], self.dft_freq_min, self.dft_freq_max, self.num_total_frequencies,
					center=self.dft_block_fields_center, size=self.dft_block_fields_size)
				self.dft_focal_area_fields = self.sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], self.dft_freq_min, self.dft_freq_max, self.num_total_frequencies,
					center=self.dft_focal_plane_fields_center, size=self.dft_focal_area_fields_size)

				focal_plane_powers = [[], [], []]
				Ex_xpol_focal = [[], [], [], []]
				Ey_xpol_focal = [[], [], [], []]
				Ex_ypol_focal = [[], [], [], []]
				Ey_ypol_focal = [[], [], [], []]

				focal_plane_power_fn = lambda sim: self.collect_focal_plane_power(sim, Ex_xpol_focal, Ey_xpol_focal, Ex_ypol_focal, Ey_ypol_focal, focal_plane_powers)
				forward_volume_fields_fn = lambda sim: self.collect_forward_volume_e_fields_xpol(sim)

				self.sim.run(mp.at_end(focal_plane_power_fn, forward_volume_fields_fn), until=mp.stop_when_fields_decayed(8, mp.Ex, self.block_center, 1e-6))

				self.figure_of_merit[data_offset + iteration][BayerOptimization.RANGE_RED] = focal_plane_powers[BayerOptimization.RANGE_RED][0]
				self.figure_of_merit[data_offset + iteration][BayerOptimization.RANGE_GREEN] = focal_plane_powers[BayerOptimization.RANGE_GREEN][0]
				self.figure_of_merit[data_offset + iteration][BayerOptimization.RANGE_BLUE] = focal_plane_powers[BayerOptimization.RANGE_BLUE][0]

				# On the lookout that these weights can be negative in some extreme cases and this should probably be addressed or at
				# the very check for
				weights = np.subtract(2/3., self.figure_of_merit[data_offset + iteration, :]**2 / np.sum(self.figure_of_merit[data_offset + iteration, :]**2, 0))
				weights = np.subtract(weights, min(0, np.min(weights)))
				weights = np.divide(weights, np.sum(weights))

				self.weights[data_offset + iteration] = weights
				self.figure_of_merit_average[data_offset + iteration] = np.sum(np.multiply(weights, self.figure_of_merit[data_offset + iteration, :]))


				for adj_src_idx in range(0, len(self.adjoint_sources)):
					self.sim.reset_meep()
					self.sim = mp.Simulation(cell_size=self.cell,
				                    boundary_layers=self.pml_layers,
				                    geometry=self.geometry,
				                    sources=[self.adjoint_sources[adj_src_idx]],
				                    resolution=self.resolution)
					self.dft_block_fields = self.sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], self.dft_freq_min, self.dft_freq_max, self.num_total_frequencies,
						center=self.dft_block_fields_center, size=self.dft_block_fields_size)

					adjoint_fields_fn = lambda sim: self.collect_adjoint_volume_e_fields_xpol(sim, adj_src_idx)

					self.sim.run(mp.at_end(adjoint_fields_fn), until=mp.stop_when_fields_decayed(8, mp.Ex, self.block_center, 1e-6))


				self.forward_fields_ypol = np.zeros(self.forward_fields_xpol.shape, dtype=np.complex)
				extract_ex_xpol = self.forward_fields_xpol[:, :, 0, :, :, :]
				extract_ey_xpol = self.forward_fields_xpol[:, :, 1, :, :, :]
				extract_ez_xpol = self.forward_fields_xpol[:, :, 2, :, :, :]
				self.forward_fields_ypol[:, :, 1, :, :, :] = np.swapaxes(extract_ex_xpol, 2, 3)
				self.forward_fields_ypol[:, :, 0, :, :, :] = np.swapaxes(extract_ey_xpol, 2, 3)
				self.forward_fields_ypol[:, :, 2, :, :, :] = np.swapaxes(extract_ez_xpol, 2, 3)

				self.adjoint_fields_ypol = np.zeros(self.adjoint_fields_xpol.shape, dtype=np.complex)
				for quadrant in range(0, 4):
					extract_ex_xpol = self.adjoint_fields_xpol[self.xy_symmetry_transformation[quadrant], :, :, 0, :, :, :]
					extract_ey_xpol = self.adjoint_fields_xpol[self.xy_symmetry_transformation[quadrant], :, :, 1, :, :, :]
					extract_ez_xpol = self.adjoint_fields_xpol[self.xy_symmetry_transformation[quadrant], :, :, 2, :, :, :]

					self.adjoint_fields_ypol[quadrant, :, :, 1, :, :, :] = np.swapaxes(extract_ex_xpol, 2, 3)
					self.adjoint_fields_ypol[quadrant, :, :, 0, :, :, :] = np.swapaxes(extract_ey_xpol, 2, 3)
					self.adjoint_fields_ypol[quadrant, :, :, 2, :, :, :] = np.swapaxes(extract_ez_xpol, 2, 3)


				# todo: Do we want to dot the z-components in?  We are not phasing the adjoint source based on the z-component anyway?
				sensitivity_blue_xx = np.sum(
										np.multiply(
											self.adjoint_fields_xpol[BayerOptimization.QUADRANT_BLUE][BayerOptimization.RANGE_BLUE],
											self.forward_fields_xpol[BayerOptimization.RANGE_BLUE]),
										axis=0);

				sensitivity_blue_xy = np.sum(
										np.multiply(
											self.adjoint_fields_ypol[BayerOptimization.QUADRANT_BLUE][BayerOptimization.RANGE_BLUE],
											self.forward_fields_xpol[BayerOptimization.RANGE_BLUE]),
										axis=0);

				sensitivity_blue_yx = np.sum(
										np.multiply(
											self.adjoint_fields_xpol[BayerOptimization.QUADRANT_BLUE][BayerOptimization.RANGE_BLUE],
											self.forward_fields_ypol[BayerOptimization.RANGE_BLUE]),
										axis=0);

				sensitivity_blue_yy = np.sum(
										np.multiply(
											self.adjoint_fields_ypol[BayerOptimization.QUADRANT_BLUE][BayerOptimization.RANGE_BLUE],
											self.forward_fields_ypol[BayerOptimization.RANGE_BLUE]),
										axis=0);


				sensitivity_green_xpol_xx = np.sum(
												np.multiply(
													self.adjoint_fields_xpol[BayerOptimization.QUADRANT_GREEN_XPOL][BayerOptimization.RANGE_GREEN],
													self.forward_fields_xpol[BayerOptimization.RANGE_GREEN]),
												axis=0);

				sensitivity_green_xpol_xy = np.sum(
												np.multiply(
													self.adjoint_fields_ypol[BayerOptimization.QUADRANT_GREEN_XPOL][BayerOptimization.RANGE_GREEN],
													self.forward_fields_xpol[BayerOptimization.RANGE_GREEN]),
												axis=0);

				sensitivity_green_xpol_yx = np.sum(
												np.multiply(
													self.adjoint_fields_xpol[BayerOptimization.QUADRANT_GREEN_XPOL][BayerOptimization.RANGE_GREEN],
													self.forward_fields_ypol[BayerOptimization.RANGE_GREEN]),
												axis=0);

				sensitivity_green_xpol_yy = np.sum(
												np.multiply(
													self.adjoint_fields_ypol[BayerOptimization.QUADRANT_GREEN_XPOL][BayerOptimization.RANGE_GREEN],
													self.forward_fields_ypol[BayerOptimization.RANGE_GREEN]),
												axis=0);


				sensitivity_red_xx = np.sum(
										np.multiply(
											self.adjoint_fields_xpol[BayerOptimization.QUADRANT_RED][BayerOptimization.RANGE_RED],
											self.forward_fields_xpol[BayerOptimization.RANGE_RED]),
										axis=0);

				sensitivity_red_xy = np.sum(
										np.multiply(
											self.adjoint_fields_ypol[BayerOptimization.QUADRANT_RED][BayerOptimization.RANGE_RED],
											self.forward_fields_xpol[BayerOptimization.RANGE_RED]),
										axis=0);

				sensitivity_red_yx = np.sum(
										np.multiply(
											self.adjoint_fields_xpol[BayerOptimization.QUADRANT_RED][BayerOptimization.RANGE_RED],
											self.forward_fields_ypol[BayerOptimization.RANGE_RED]),
										axis=0);

				sensitivity_red_yy = np.sum(
										np.multiply(
											self.adjoint_fields_ypol[BayerOptimization.QUADRANT_RED][BayerOptimization.RANGE_RED],
											self.forward_fields_ypol[BayerOptimization.RANGE_RED]),
										axis=0);


				sensitivity_green_ypol_xx = np.sum(
												np.multiply(
													self.adjoint_fields_xpol[BayerOptimization.QUADRANT_GREEN_YPOL][BayerOptimization.RANGE_GREEN],
													self.forward_fields_xpol[BayerOptimization.RANGE_GREEN]),
												axis=0);

				sensitivity_green_ypol_xy = np.sum(
												np.multiply(
													self.adjoint_fields_ypol[BayerOptimization.QUADRANT_GREEN_YPOL][BayerOptimization.RANGE_GREEN],
													self.forward_fields_xpol[BayerOptimization.RANGE_GREEN]),
												axis=0);

				sensitivity_green_ypol_yx = np.sum(
												np.multiply(
													self.adjoint_fields_xpol[BayerOptimization.QUADRANT_GREEN_YPOL][BayerOptimization.RANGE_GREEN],
													self.forward_fields_ypol[BayerOptimization.RANGE_GREEN]),
												axis=0);

				sensitivity_green_ypol_yy = np.sum(
												np.multiply(
													self.adjoint_fields_ypol[BayerOptimization.QUADRANT_GREEN_YPOL][BayerOptimization.RANGE_GREEN],
													self.forward_fields_ypol[BayerOptimization.RANGE_GREEN]),
												axis=0);


				for f_idx in range(0, num_frequencies_blue):
					phasor_blue_xx[f_idx, :, :, :] = np.conj(Ex_xpol_focal[BayerOptimization.QUADRANT_BLUE][f_idx + frequency_offset_blue])
					phasor_blue_xy[f_idx, :, :, :] = np.conj(Ey_xpol_focal[BayerOptimization.QUADRANT_BLUE][f_idx + frequency_offset_blue])
					phasor_blue_yx[f_idx, :, :, :] = np.conj(Ex_ypol_focal[BayerOptimization.QUADRANT_BLUE][f_idx + frequency_offset_blue])
					phasor_blue_yy[f_idx, :, :, :] = np.conj(Ey_ypol_focal[BayerOptimization.QUADRANT_BLUE][f_idx + frequency_offset_blue])

				for f_idx in range(0, num_frequencies_green):
					phasor_green_xpol_xx[f_idx, :, :, :] = np.conj(Ex_xpol_focal[BayerOptimization.QUADRANT_GREEN_XPOL][f_idx + frequency_offset_green])
					phasor_green_xpol_xy[f_idx, :, :, :] = np.conj(Ey_xpol_focal[BayerOptimization.QUADRANT_GREEN_XPOL][f_idx + frequency_offset_green])
					phasor_green_xpol_yx[f_idx, :, :, :] = np.conj(Ex_ypol_focal[BayerOptimization.QUADRANT_GREEN_XPOL][f_idx + frequency_offset_green])
					phasor_green_xpol_yy[f_idx, :, :, :] = np.conj(Ey_ypol_focal[BayerOptimization.QUADRANT_GREEN_XPOL][f_idx + frequency_offset_green])

				for f_idx in range(0, num_frequencies_red):
					phasor_red_xx[f_idx, :, :, :] = np.conj(Ex_xpol_focal[BayerOptimization.QUADRANT_RED][f_idx + frequency_offset_red])
					phasor_red_xy[f_idx, :, :, :] = np.conj(Ey_xpol_focal[BayerOptimization.QUADRANT_RED][f_idx + frequency_offset_red])
					phasor_red_yx[f_idx, :, :, :] = np.conj(Ex_ypol_focal[BayerOptimization.QUADRANT_RED][f_idx + frequency_offset_red])
					phasor_red_yy[f_idx, :, :, :] = np.conj(Ey_ypol_focal[BayerOptimization.QUADRANT_RED][f_idx + frequency_offset_red])

				for f_idx in range(0, num_frequencies_green):
					phasor_green_ypol_xx[f_idx, :, :, :] = np.conj(Ex_xpol_focal[BayerOptimization.QUADRANT_GREEN_YPOL][f_idx + frequency_offset_green])
					phasor_green_ypol_xy[f_idx, :, :, :] = np.conj(Ey_xpol_focal[BayerOptimization.QUADRANT_GREEN_YPOL][f_idx + frequency_offset_green])
					phasor_green_ypol_yx[f_idx, :, :, :] = np.conj(Ex_ypol_focal[BayerOptimization.QUADRANT_GREEN_YPOL][f_idx + frequency_offset_green])
					phasor_green_ypol_yy[f_idx, :, :, :] = np.conj(Ey_ypol_focal[BayerOptimization.QUADRANT_GREEN_YPOL][f_idx + frequency_offset_green])


				sensitivity_blue_x = np.multiply(2, np.real(np.add(sensitivity_blue_xx * phasor_blue_xx, sensitivity_blue_xy * phasor_blue_xy)))
				sensitivity_blue_y = np.multiply(2, np.real(np.add(sensitivity_blue_yx * phasor_blue_yx, sensitivity_blue_yy * phasor_blue_yy)))

				sensitivity_green_xpol_x = np.multiply(2, np.real(np.add(sensitivity_green_xpol_xx * phasor_green_xpol_xx, sensitivity_green_xpol_xy * phasor_green_xpol_xy)))
				sensitivity_green_xpol_y = np.multiply(2, np.real(np.add(sensitivity_green_xpol_yx * phasor_green_xpol_yx, sensitivity_green_xpol_yy * phasor_green_xpol_yy)))

				sensitivity_red_x = np.multiply(2, np.real(np.add(sensitivity_red_xx * phasor_red_xx, sensitivity_red_xy * phasor_red_xy)))
				sensitivity_red_y = np.multiply(2, np.real(np.add(sensitivity_red_yx * phasor_red_yx, sensitivity_red_yy * phasor_red_yy)))

				sensitivity_green_ypol_x = np.multiply(2, np.real(np.add(sensitivity_green_ypol_xx * phasor_green_ypol_xx, sensitivity_green_ypol_xy * phasor_green_ypol_xy)))
				sensitivity_green_ypol_y = np.multiply(2, np.real(np.add(sensitivity_green_ypol_yx * phasor_green_ypol_yx, sensitivity_green_ypol_yy * phasor_green_ypol_yy)))

				sensitivity_red = np.multiply(0.5 / self.max_intensity[BayerOptimization.RANGE_RED], np.sum(np.add(sensitivity_red_x, sensitivity_red_y), axis=0))
				sensitivity_green = np.multiply(0.5 / self.max_intensity[BayerOptimization.RANGE_GREEN], np.sum(np.add(sensitivity_green_xpol_x, sensitivity_green_ypol_y), axis=0))
				sensitivity_blue = np.multiply(0.5 / self.max_intensity[BayerOptimization.RANGE_BLUE], np.sum(np.add(sensitivity_blue_x, sensitivity_blue_y), axis=0))

				weighted_sensitivity_red = np.multiply(weights[BayerOptimization.RANGE_RED], sensitivity_red)
				weighted_sensitivity_green = np.multiply(weights[BayerOptimization.RANGE_GREEN], sensitivity_green)
				weighted_sensitivity_blue = np.multiply(weights[BayerOptimization.RANGE_BLUE], sensitivity_blue)
				weighted_sensitivity = np.add(np.add(weighted_sensitivity_red, weighted_sensitivity_green), weighted_sensitivity_blue)

				self.device.step(-weighted_sensitivity, step_size)
				step_size -= step_size_step

				np.save('data/fom.npy', self.figure_of_merit)
				np.save('data/fom_avg.npy', self.figure_of_merit_average)
				np.save('data/weights.npy', self.weights)
				np.save('data/times.npy', self.iteration_time_sec)
				np.save('permittivity.npy', self.device.get_permittivity())
				np.save('optimization_progress.npy', np.array([epoch, iteration]))

				end_time = time.time()
				self.iteration_time_sec[data_offset + iteration] = end_time - start_time



