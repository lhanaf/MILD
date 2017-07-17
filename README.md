# MILD
Project for MILD

# Prerequisites  ################################################################

Ubuntu 14.04
OpenCV 3.0 or above http://xfloyd.net/blog/?p=987
eigen3



# Installation ################################################################
$ mkdir build

$ cd build

$ cmake ..

$ make 



# Usage ##############################################################


./mild ../imagelist.txt ../settings.yaml

imageList.txt: indicats the path of each input RGB image per line
settings.yaml: indicats the parameters used in loop closure detection

output:

output/imagelist/lcd_shared_flag.bin: detected loop closure are set as 1. To be used in the run_scritp.m to check the accuracy of the detected loop closure.
