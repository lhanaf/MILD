<<<<<<< HEAD
# MILD
Project for MILD
=======
# comp5421_sfm
Project for COMP5421
>>>>>>> ddd32b4df956b8c222fa2ee4c4e926ca5397f1ec


To compile the code:
$ mkdir build

$ cd build

$ cmake ..

$ make 



To run the code:

<<<<<<< HEAD
./mild ../imagelist.txt ../settings.yaml

imageList.txt: indicats the path of each input RGB image per line
settings.yaml: indicats the parameters used in loop closure detection

output:

output/imagelist/lcd_shared_flag.bin: detected loop closure are set as 1. To be used in the run_scritp.m to check the accuracy of the detected loop closure.
=======
./sfm ../imagelist.txt ../configure.yaml

imagelist.txt: path to each image
configure.yaml: user configurations. 
save_image save image or not.
use_sparse_match: use fast match or brute-force match.

output:

global_points.ply:  sparse point cloud in the global frame.
>>>>>>> ddd32b4df956b8c222fa2ee4c4e926ca5397f1ec
