# MILD
Project for MILD


To compile the code:
$ mkdir build

$ cd build

$ cmake ..

$ make 



To run the code:

./mild ../imagelist.txt ../settings.yaml

imageList.txt: indicats the path of each input RGB image per line
settings.yaml: indicats the parameters used in loop closure detection

output:

output/imagelist/lcd_shared_flag.bin: detected loop closure are set as 1. To be used in the run_scritp.m to check the accuracy of the detected loop closure.
