# improved_trajectory
Here is a sample code for extracting improved trajectories, which is forked from Limin Wang and originally adapted from Heng Wang's code with a few modification.


Some notes:

- The input could be video or image sequence

- example execution command: 
./DenseTrackStab -f <path_to_file>/test.avi -o <path_to_output>/idt.bin -r <path_to_output>/tra.bin

./DenseTrackStab -f <path_to_file>/img_%05d.png -o <path_to_output>/idt.bin -r <path_to_output>/tra.bin
(suppose that the image sequence would be like: img_00001.png)



