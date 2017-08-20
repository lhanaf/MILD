# MILD
Project for MILD: An efficient loop closure detection libary based on binary features.

Related papers:

1.  Multi-Index Hashing for Loop closure Detection. International Conference on Multimedia Expo, 2017. Best Student Paper Awards. 

2.  Beyond SIFT Using Binary features in Loop Closure Detection. IROS 2017.

# Prerequisites  ################################################################

Ubuntu 14.04

cmake 3.2.0

OpenCV 3.1 http://xfloyd.net/blog/?p=987

eigen3

octave (optional, only used for evaluation)

# Installation ################################################################
$ mkdir build

$ cd build

$ cmake ..

$ make 



# Usage ##############################################################


./mild imagelist.txt settings.yaml

input:

imageList.txt: indicats the path of each input RGB image per line
settings.yaml: indicats the parameters used in loop closure detection

output:

output/imagelist/lcd_shared_flag.bin: detected loop closure are set as 1. To be used in the run_scritp.m to check the accuracy of the detected loop closure.
output/imagelist/lcd_shared_score_mild.bin: the image similarity calculated using MILD.
output/imagelist/relocalization_time_per_frame.bin: lcd time of each frame.

evluation: (based on MATLAB/OCTAVE)

evaluation('build/output/imageList_NewCollege/lcd_shared_flag.bin','build/output/imageList_NewCollege/lcd_shared_probability.bin','data/truthNewCollege.mat');
