from filter import Filter
import numpy as np

#
# Blurs can be more general.  We just need to specify a mask and maximum approximation function (and its derivative)
#
class EllipsoidalBlur(Filter):

	#
	# alpha: strength of blur
	# minor_blur: half of blur width at center pixel in x and y directions in units of whole pixels
	# major_blur: half of blur width at center pixel in z direction in units of whole pixels
	#
	def __init__(self, alpha, minor_blur, major_blur, variable_bounds=[0, 1]):
		super(EllipsoidalBlur, self).__init__(variable_bounds)

		self.alpha = alpha
		self.minor_blur = int(minor_blur)
		self.major_blur = int(major_blur)

		# At this point, we can compute which points in a volume will be part of this blurring operation
		self.ellipse_mask_size_xy = 1 + 2 * self.minor_blur
		self.ellipse_mask_size_z = 1 + 2 * self.major_blur

		self.ellipse_mask = np.zeros((self.ellipse_mask_size_xy, self.ellipse_mask_size_xy, self.ellipse_mask_size_z))

		for mask_x in range(-self.minor_blur, self.minor_blur + 1):
			for mask_y in range(-self.minor_blur, self.minor_blur + 1):
				for mask_z in range(-self.major_blur, self.major_blur + 1):
					x_contribution = mask_x**2 / (self.minor_blur + 1e-6)**2
					y_contribution = mask_y**2 / (self.minor_blur + 1e-6)**2
					z_contribution = mask_z**2 / (self.major_blur + 1e-6)**2

					if (x_contribution + y_contribution + z_contribution) <= 1:
						self.ellipse_mask[self.minor_blur + mask_x, self.minor_blur + mask_y, self.major_blur + mask_z] = 1

		self.number_to_blur = sum((self.ellipse_mask).flatten())


	def forward(self, variable_in):
		pad_variable_in = np.pad(
			variable_in,
			((self.minor_blur, self.minor_blur), (self.minor_blur, self.minor_blur), (self.major_blur, self.major_blur)),
			'constant'
		)

		unpadded_shape = variable_in.shape
		padded_shape = pad_variable_in.shape

		start_xy = self.minor_blur
		start_z = self.major_blur

		x_length = unpadded_shape[0]
		y_length = unpadded_shape[1]
		z_length = unpadded_shape[2]

		blurred_variable = np.zeros((x_length, y_length, z_length))
		for mask_x in range(-self.minor_blur, self.minor_blur + 1):
			offset_x = start_xy + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.minor_blur, self.minor_blur + 1):
				offset_y = start_xy + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]
				for mask_z in range(-self.major_blur, self.major_blur + 1):
					offset_z = start_z + mask_z
					z_bounds = [offset_z, (offset_z + z_length)]

					check_mask = self.ellipse_mask[mask_x + self.minor_blur, mask_y + self.minor_blur, mask_z + self.major_blur]

					if check_mask == 1:
						blurred_variable = np.add(
							blurred_variable,
							np.exp(
								np.multiply(
									self.alpha,
									pad_variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]]
									)
								)
							)

		blurred_variable = np.divide(np.log(np.divide(blurred_variable, self.number_to_blur)), self.alpha)

		return blurred_variable

	def propagate_delta_backwards(self, delta_out, variable_out, variable_in):
		pad_variable_out = np.pad(
			variable_out,
			((self.minor_blur, self.minor_blur), (self.minor_blur, self.minor_blur), (self.major_blur, self.major_blur)),
			'constant'
		)

		pad_delta_out = np.pad(
			delta_out,
			((self.minor_blur, self.minor_blur), (self.minor_blur, self.minor_blur), (self.major_blur, self.major_blur)),
			'constant'
		)

		start_xy = self.minor_blur
		start_z = self.major_blur

		unpadded_shape = variable_in.shape
		x_length = unpadded_shape[0]
		y_length = unpadded_shape[1]
		z_length = unpadded_shape[2]

		delta_in = np.zeros(delta_out.shape)

		for mask_x in range(-self.minor_blur, self.minor_blur + 1):
			offset_x = start_xy + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.minor_blur, self.minor_blur + 1):
				offset_y = start_xy + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]
				for mask_z in range(-self.major_blur, self.major_blur + 1):
					offset_z = start_z + mask_z
					z_bounds = [offset_z, (offset_z + z_length)]

					check_mask = self.ellipse_mask[mask_x + self.minor_blur, mask_y + self.minor_blur, mask_z + self.major_blur]

					if check_mask == 1:
						delta_in = np.add(
							delta_in,
							np.multiply(
								np.exp(
									-np.multiply(
										self.alpha,
										np.subtract(
											variable_in,
											pad_variable_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]]
										)
									)
								),
							pad_delta_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]])
						)

		delta_in = np.divide(delta_in, self.number_to_blur)
		return delta_in


	def chain_rule(self, derivative_out, variable_out, variable_in):
		pad_variable_out = np.pad(
			variable_out,
			((self.minor_blur, self.minor_blur), (self.minor_blur, self.minor_blur), (self.major_blur, self.major_blur)),
			'constant'
		)

		pad_derivative_out = np.pad(
			derivative_out,
			((self.minor_blur, self.minor_blur), (self.minor_blur, self.minor_blur), (self.major_blur, self.major_blur)),
			'constant'
		)

		start_xy = self.minor_blur
		start_z = self.major_blur

		unpadded_shape = variable_in.shape
		x_length = unpadded_shape[0]
		y_length = unpadded_shape[1]
		z_length = unpadded_shape[2]

		derivative_in = np.zeros(derivative_out.shape)

		for mask_x in range(-self.minor_blur, self.minor_blur + 1):
			offset_x = start_xy + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.minor_blur, self.minor_blur + 1):
				offset_y = start_xy + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]
				for mask_z in range(-self.major_blur, self.major_blur + 1):
					offset_z = start_z + mask_z
					z_bounds = [offset_z, (offset_z + z_length)]

					check_mask = self.ellipse_mask[mask_x + self.minor_blur, mask_y + self.minor_blur, mask_z + self.major_blur]

					if check_mask == 1:
						derivative_in = np.add(
							derivative_in,
							np.multiply(
								np.exp(
									np.multiply(
										self.alpha,
										np.subtract(
											variable_in,
											pad_variable_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]]
										)
									)
								),
							pad_derivative_out[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]])
						)

		derivative_in = np.divide(derivative_in, self.number_to_blur)
		return derivative_in

	def fabricate(self, variable_in):
		variable_shape = variable_in.shape

		start_xy = self.minor_blur
		start_z = self.major_blur

		x_length = variable_shape[0]
		y_length = variable_shape[1]
		z_length = variable_shape[2]

		blurred_variable = np.zeros((x_length, y_length, z_length))
		for mask_x in range(-self.minor_blur, self.minor_blur + 1):
			offset_x = start_xy + mask_x
			x_bounds = [offset_x, (offset_x + x_length)]
			for mask_y in range(-self.minor_blur, self.minor_blur + 1):
				offset_y = start_xy + mask_y
				y_bounds = [offset_y, (offset_y + y_length)]
				for mask_z in range(-self.major_blur, self.major_blur + 1):
					offset_z = start_z + mask_z
					z_bounds = [offset_z, (offset_z + z_length)]

					check_mask = self.ellipse_mask[mask_x + self.minor_blur, mask_y + self.minor_blur, mask_z + self.major_blur]

					if check_mask == 1:
						blurred_variable[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]] = np.maximum(
							blurred_variable[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]],
							variable_in[x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], z_bounds[0]:z_bounds[1]])

		return blurred_variable



