# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

######## Basic ########

#base_path = r'/media/shbe/data/share-to-docker/bop/'
base_path = r'/home/hampus/vision/AugmentedAutoencoder/pytorch3d/data/bop/'

# Folder with the BOP datasets.
datasets_path = r'{}bop-tless-dataset'.format(base_path)

# Folder with pose results to be evaluated.
results_path = r'{}bop_sample_results/bop_challenge_2019'.format(base_path)

# Folder for the calculated pose errors and performance scores.
eval_path = r'{}eval'.format(base_path)

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'{}output'.format(base_path)

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'

print(datasets_path)
