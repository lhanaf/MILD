# comp5421_sfm
Project for COMP5421


To compile the code:
$ mkdir build

$ cd build

$ cmake ..

$ make 



To run the code:

./sfm ../imagelist.txt ../configure.yaml

imagelist.txt: path to each image
configure.yaml: user configurations. 
save_image save image or not.
use_sparse_match: use fast match or brute-force match.

output:

global_points.ply:  sparse point cloud in the global frame.
