cmake_minimum_required(VERSION 2.8)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
project( DenseTrackStab_SingleScale )
#find_package( OpenCV REQUIRED )
include_directories(  include/
/home/su/opencv/include/ /home/su/opencv/include/opencv/ )
add_executable( DenseTrackStab_SingleScale src/DenseTrackStab_SingleScale.cpp )

SET(exec_prefix /home/su/opencv/)
SET(OpenCV_LIBS -L${exec_prefix}/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu 
-lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching 
-lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab)

target_link_libraries( DenseTrackStab_SingleScale ${OpenCV_LIBS} )
