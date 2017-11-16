#!/bin/bash
while read line
do
	FILE_VIDEO="$line"
	PREFIX="/home/.../idt_feature/" # path to output folder
	PATH_IMAGE="/home/.../imageSeq/" # path to image files
	FILE_NAME_VIDEO="${FILE_VIDEO:: -2}" # get the name of the video by wiping out the label and the blankspace, which take 2 space in total
	FILE_IMAGE_SUFFIX="/img_%05d.png"
	FILE_NAME_IDT_SUFFIX="_iDT_Features.bin"
	FILE_NAME_TRA_SUFFIX="_tra.bin"

	FILE_NAME_IDT=$PREFIX${FILE_NAME_VIDEO}${FILE_NAME_IDT_SUFFIX}
	FILE_NAME_TRA=$PREFIX${FILE_NAME_VIDEO}${FILE_NAME_TRA_SUFFIX}
	FILE_IMAGE_FULL_PATH=${PATH_IMAGE}${FILE_NAME_VIDEO}${FILE_IMAGE_SUFFIX}

	./DenseTrackStab -f $FILE_FLOW_FULL_PATH -o $FILE_NAME_IDT -r $FILE_NAME_TRA

	echo "finish ${FILE_NAME_VIDEO}"

done < ./train_test_list/trainlist.txt # path to list