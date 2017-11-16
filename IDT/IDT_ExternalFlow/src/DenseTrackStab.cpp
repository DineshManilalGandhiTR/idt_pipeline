#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"
#include <time.h>

using namespace cv;

int main(int argc, char** argv)
{
	/***** IO operation *****/

	const char* keys =
		{
			"{ f  | flow_path    | <path_to_flow>/flow*.png | filename of flow sequence }"
			"{ p  | image_path   | <path_to_image>/img*.png | filename of image sequence }"
			"{ o  | idt_file   | test.bin | filename of idt features }"
			"{ r  | tra_file   | tra.bin  | filename of track files  }"
			"{ L  | track_length   | 15 | the length of trajectory }"
			"{ S  | start_frame     | 0 | start frame of tracking }"
			"{ E  | end_frame | 1000000 | end frame of tracking }"
			"{ W  | min_distance | 5 | min distance }"
			"{ N  | patch_size   | 32  | patch size }"
			"{ s  | nxy_cell  | 2 | descriptor parameter }"
			"{ t  | nt_cell  | 3 | discriptor parameter }"
			"{ I  | init_gap  | 1 | gap }"
			"{ T  | show_track | 0 | whether show tracks}"
		};

	CommandLineParser cmd(argc, argv, keys);
	string flowSeq = cmd.get<string>("flow_path");
	string imageSeq = cmd.get<string>("image_path");
	string out_file = cmd.get<string>("idt_file");
	string tra_file = cmd.get<string>("tra_file");
	track_length = cmd.get<int>("track_length");
	start_frame = cmd.get<int>("start_frame");
	end_frame = cmd.get<int>("end_frame");
	min_distance = cmd.get<int>("min_distance");
	patch_size = cmd.get<int>("patch_size");
	nxy_cell = cmd.get<int>("nxy_cell");
	nt_cell = cmd.get<int>("nt_cell");
	init_gap = cmd.get<int>("init_gap");

   	/******************************************************************/

	/*
	Input: Image Sequence, flowx, flowy
	Output: Track(points, hog, hof, mbh)
	*/

	// output files, which hold trajectory and feature info
	FILE* outfile = fopen(out_file.c_str(), "wb");
	FILE* trafile = fopen(tra_file.c_str(), "wb");


	// preparation: encode flow and image sequence into flowx, flowy, image
	std::vector<Mat> flowx(0), flowy(0), frame(0);
	SeqInfo seqInfo;
	int frameNum;
	InitSeqInfo(&seqInfo, flowSeq, imageSeq, flowx, flowy, frame);
	frameNum = seqInfo.length;
	printf( "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	// initialize track and feature info
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);


	if(show_track == 1)
		namedWindow("DenseTrackStab", 0);

	/*******************************************************************/

	/* The main part of IDT*/

	// image of previous and current frame
	Mat image, prev_grey, grey;

	/*
	list of track. which stores feature and track info
	each feature point gets a track
	*/
	std::list<Track> xyScaleTracks;

	int init_counter = 0; // indicate when to detect new feature points

	for(int frame_num = 0; frame_num < frameNum; frame_num++)
	{
		// only works in between start_frame and end_frame
		if(frame_num < start_frame || frame_num > end_frame)
		{
			frame_num++;
			continue;
		}

		if(frame_num == start_frame)
		{
			image.create((frame.at(frame_num)).size(), CV_8UC3);
			grey.create((frame.at(frame_num)).size(), CV_8UC1);
			prev_grey.create((frame.at(frame_num)).size(), CV_8UC1);


			(frame.at(frame_num)).copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);


			// dense sampling feature points
			std::vector<Point2f> points(0);
			DenseSample(prev_grey, points, quality, min_distance);


			// initialize the info of feature(sampled) points in the track buff
			std::list<Track>& tracks = xyScaleTracks;
			for(size_t i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));

			frame_num++;
			continue;
		}


		init_counter++; // count for TrackInfo.gap, if (count == TrackInfo.gap), resampling
		(frame.at(frame_num)).copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);
		int width = grey.cols;
		int height = grey.rows;

		// compute the integral histograms
		DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
		HogComp(prev_grey, hogMat->desc, hogInfo);

		DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
		HofComp(flowx.at(frame_num-1), flowy.at(frame_num-1), hofMat->desc, hofInfo);

		DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
		DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
		MbhComp(flowx.at(frame_num-1), flowy.at(frame_num-1), mbhMatX->desc, mbhMatY->desc, mbhInfo);


		std::list<Track>& tracks = xyScaleTracks;
		for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();)
		{
			int index = iTrack->index;
			Point2f prev_point = iTrack->point[index];
			int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
			int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

			Point2f point;

			point.x = prev_point.x + flowx.at(frame_num-1).ptr<float>(y)[x];
			point.y = prev_point.y + flowy.at(frame_num-1).ptr<float>(y)[x];


			if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height)
			{
				iTrack = tracks.erase(iTrack);
				continue;
			}

			iTrack->disp[index].x = flowx.at(frame_num-1).ptr<float>(y)[x];
			iTrack->disp[index].y = flowy.at(frame_num-1).ptr<float>(y)[x];
			
			// get the descriptors for the feature point
			RectInfo rect;
			GetRect(prev_point, rect, width, height, hogInfo);
			GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
			GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
			GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
			GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
			iTrack->addPoint(point);

			// onöy check out the track with the minimal valid length 
			if(iTrack->index >= trackInfo.length)
			{
				std::vector<Point2f> trajectory(trackInfo.length+1), trajectory1(trackInfo.length+1);
				for(int i = 0; i <= trackInfo.length; ++i)
				{
					trajectory[i] = iTrack->point[i];
					trajectory1[i] = iTrack->point[i];
				}

				std::vector<Point2f> displacement(trackInfo.length);
				for (int i = 0; i < trackInfo.length; ++i)
					displacement[i] = iTrack->disp[i];

				float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
				if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement))
				{
					if(show_track == 1)
						DrawTrack(iTrack->point, iTrack->index, image);

					// output the basic information
					fwrite(&frame_num,sizeof(frame_num),1,outfile);
					fwrite(&mean_x,sizeof(mean_x),1,outfile);
					fwrite(&mean_y,sizeof(mean_y),1,outfile);
					fwrite(&var_x,sizeof(var_x),1,outfile);
					fwrite(&var_y,sizeof(var_y),1,outfile);
					fwrite(&length,sizeof(var_y),1,outfile);

					// for spatio-temporal info
					float temp = std::min<float>(max<float>(mean_x/float(seqInfo.width), 0), 0.999);
					fwrite(&temp,sizeof(temp),1,outfile);
					temp = std::min<float>(max<float>(mean_y/float(seqInfo.height), 0), 0.999);
					fwrite(&temp,sizeof(temp),1,outfile);
					temp =  std::min<float>(max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999);
					fwrite(&temp,sizeof(temp),1,outfile);

					// output trajectory point coordinates
		       	 	for (int i=0; i< trackInfo.length; ++i)
		        	{
						temp = trajectory1[i].x;
						fwrite(&temp, sizeof(temp), 1, outfile);
						fwrite(&temp, sizeof(temp), 1, trafile);
						temp = trajectory1[i].y;
						fwrite(&temp, sizeof(temp), 1, outfile);
						fwrite(&temp, sizeof(temp), 1, trafile);
					}

					// output the trajectory features
					for (int i = 0; i < trackInfo.length; ++i)
					{
						temp = displacement[i].x;
						fwrite(&temp,sizeof(temp),1,outfile);
						temp = displacement[i].y;
						fwrite(&temp,sizeof(temp),1,outfile);
					}

					PrintDesc(iTrack->hog, hogInfo, trackInfo, outfile);
					PrintDesc(iTrack->hof, hofInfo, trackInfo, outfile);
					PrintDesc(iTrack->mbhX, mbhInfo, trackInfo, outfile);
					PrintDesc(iTrack->mbhY, mbhInfo, trackInfo, outfile);
				}

				iTrack = tracks.erase(iTrack);
				continue;
			}
			++iTrack;
		}

		ReleDescMat(hogMat);
		ReleDescMat(hofMat);
		ReleDescMat(mbhMatX);
		ReleDescMat(mbhMatY);

		if(init_counter != trackInfo.gap)
			continue;

		// detect new feature points every gap frames
		std::vector<Point2f> points(0);
		for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
			points.push_back(iTrack->point[iTrack->index]);

		DenseSample(grey, points, quality, min_distance);

		// save the new feature points
		for(size_t i = 0; i < points.size(); i++)
			tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));

		init_counter = 0;
		grey.copyTo(prev_grey);

		frame_num++;

		if( show_track == 1 )
		{
			imshow( "DenseTrackStab", image);
			int c = cvWaitKey(3);
			if((char)c == 27) break;
		}


	}

	if( show_track == 1 )
		destroyWindow("DenseTrackStab");


	fclose(outfile);
	fclose(trafile);

	return 0;
}
