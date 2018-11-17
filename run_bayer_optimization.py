import bayer_optimization
import sys
import numpy as np

resolution = 40
num_epochs = 4
num_iterations_per_epoch = 25
version = '_nanoscribe'
optimizer = bayer_optimization.BayerOptimization(resolution)

optimizer.run_optimization(num_epochs, num_iterations_per_epoch)

print(optimizer.figure_of_merit)
print(optimizer.figure_of_merit_average)
print(optimizer.iteration_time_sec)
print(optimizer.weights)

# It would be good to save out a file describing the permittivity and a good file naming scheme!
final_permittivity_name = 'permittivity_' + str(num_epochs) + '_' + str(num_iterations_per_epoch) + version + '.npy'

np.save(final_permittivity_name, optimizer.device.get_permittivity())

sys.stdout.flush()
