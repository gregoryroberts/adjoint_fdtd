import numpy as np

class Topology:

	def __init__(self):
		self.d3_topology = self.D3()

	# class D2:
		# def NumberConnectedComponents(img, color, connectivity):


	class D3:
		MIDDLE_PIXEL = 13

		def __init__(self):
			self.connectivity_26_table = []
			self.connectivity_6_table = []

			self.build_connectivity_tables()

		def count_object_voxels(self, img):
			return np.sum(img)

		def count_void_voxels(self, img):
			return np.sum(np.subtract(1, img))

		def count_void_s_voxels(self, img):
			return np.sum(np.subtract(1, img[self.connectivity_6_table[Topology.D3.MIDDLE_PIXEL]]))
			# return np.sum(np.subtract(1, img[self.connectivity_6_table[13]]))

		def dfs(self, flat_img, connectivity, seed, counting_rule=lambda x: 1):
			visited = np.zeros(3**3)
			dfs_stack = [seed]
			visited[seed] = 1
			num = 0

			while len(dfs_stack) > 0:
				search = dfs_stack.pop()
				num += counting_rule(search)
				visited[search] = 1

				for c in connectivity[search]:
					if (visited[c] == 0) and (flat_img[c] == 1):
						dfs_stack.append(c)
						visited[c] = 1

			return num

		def idx_3d_to_flat_idx(self, x, y, z):
			return (9 * x + 3 * y + z)

		def build_connectivity_tables(self):
			for x in range(0, 3):
				for y in range(0, 3):
					for z in range(0, 3):
						my_idx = self.idx_3d_to_flat_idx(x, y, z)
						self.connectivity_26_table.append(self.connectivity(x, y, z, self.connectivity_rule_26))
						self.connectivity_6_table.append(self.connectivity(x, y, z, self.connectivity_rule_6))


		def connectivity(self, x, y, z, connectivity_rule):
			connected = []
			for x_off in range(-1, 2):
				for y_off in range(-1, 2):
					for z_off in range(-1, 2):
						x_neighbor = x + x_off
						y_neighbor = y + y_off
						z_neighbor = z + z_off

						if (
							(x_neighbor >= 0) and (x_neighbor < 3) and
							(y_neighbor >= 0) and (y_neighbor < 3) and
							(z_neighbor >= 0) and (z_neighbor < 3) and
							connectivity_rule(x_off, y_off, z_off) ):

							connected.append(self.idx_3d_to_flat_idx(x_neighbor, y_neighbor, z_neighbor))

			return connected

		def connectivity_rule_26(self, x_off, y_off, z_off):
			return (max([abs(x_off), abs(y_off), abs(z_off)]) == 1)

		def connectivity_rule_6(self, x_off, y_off, z_off):
			return ((abs(x_off) + abs(y_off) + abs(z_off)) == 1)


		def object_components(self, flat):
			# flat = img.flatten()
			# Need to find seed and need to do connectivity based on flattened array
			seed = -1
			for idx in range(0, len(flat)):
				if flat[idx] == 1:
					seed = idx
					break
			if seed == -1:
				return False
			else:
				get_dfs_num = self.dfs(flat, self.connectivity_26_table, seed)
				get_total_num = self.count_object_voxels(flat)

				return (get_dfs_num == get_total_num)

		def void_components(self, flat, num_s_points):
			s_point_connectivity = self.connectivity_6_table[Topology.D3.MIDDLE_PIXEL]
			# s_point_connectivity = self.connectivity_6_table[13]

			seed = -1
			for idx in s_point_connectivity:
				if flat[idx] == 1:
					seed = idx
					break

			if seed == -1:
				return False
			else:
				get_dfs_num = self.dfs(flat, self.connectivity_6_table, seed, lambda x: int(x in s_point_connectivity))

				return (get_dfs_num == num_s_points)

		def topology_check_with_object_center_point(self, flat):
			num_s_points = self.count_void_s_voxels(flat)

			if (num_s_points == 0):
				return False

			check_initial_void_components = self.void_components(np.subtract(1, flat), num_s_points)

			if not check_initial_void_components:
				return False

			flat[Topology.D3.MIDDLE_PIXEL] = 0
			# flat[13] = 0
			return self.object_components(flat)

		def topology_check_with_void_center_point(self, flat):
			num_object_points = self.count_object_voxels(flat)

			if (num_object_points == 0):
				return False

			check_initial_object_components = self.object_components(flat)

			if not check_initial_object_components:
				return False

			flat[Topology.D3.MIDDLE_PIXEL] = 1
			# flat[13] = 1
			num_void_points = self.count_void_voxels(flat)

			if (num_void_points == 0):
				return False

			num_s_points = self.count_void_s_voxels(flat)
			return self.void_components(np.subtract(1, flat), num_s_points)

		def topology_check(self, img):
			flat = img.flatten()

			if flat[Topology.D3.MIDDLE_PIXEL] == 1:
			# if flat[13] == 1:
				return self.topology_check_with_object_center_point(flat)
			else:
				return self.topology_check_with_void_center_point(flat)






						