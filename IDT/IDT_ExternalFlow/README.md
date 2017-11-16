# improved_trajectory
This is a sample code for idt feature extraction, which is adapted from Limin Wang's code with some modification.


New features:

- Input: Image sequence and optical flow

- Output: idt.bin  tra.bin  

- Example Command: 
./DenseTrackStab -f "<path_to_flow>/flow*.png" -p "<path_to_image>/img*.png" -o <path_to_output>/idt.bin -r <path_to_output>/tra.bin

