#ifndef INITIALIZE_SEP_H_
#define INITIALIZE_SEP_H_

#include "DenseTrackStab.h"

using namespace cv;

void InitTrackInfo(TrackInfo* trackInfo, int track_length, int init_gap)
{
	trackInfo->length = track_length;
	trackInfo->gap = init_gap;
}

DescMat* InitDescMat(int height, int width, int nBins)
{
	DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));
	descMat->height = height;
	descMat->width = width;
	descMat->nBins = nBins;

	long size = height*width*nBins;
	descMat->desc = (float*)malloc(size*sizeof(float));
	memset(descMat->desc, 0, size*sizeof(float));
	return descMat;
}

void ReleDescMat(DescMat* descMat)
{
	free(descMat->desc);
	free(descMat);
}

void InitDescInfo(DescInfo* descInfo, int nBins, bool isHof, int size, int nxy_cell, int nt_cell)
{
	descInfo->nBins = nBins;
	descInfo->isHof = isHof;
	descInfo->nxCells = nxy_cell;
	descInfo->nyCells = nxy_cell;
	descInfo->ntCells = nt_cell;
	descInfo->dim = nBins*nxy_cell*nxy_cell;
	descInfo->height = size;
	descInfo->width = size;
}

void convertImageToFlow(const Mat& image, Mat& flow, double lowerBound, double higherBound)
{
	for(int i = 0; i < image.rows; ++i)
		for(int j = 0; j < image.cols; ++j)
			flow.at<float>(i,j) = image.at<float>(i,j) * (higherBound - lowerBound) / 255.0 + lowerBound;
}

void InitSeqInfo(SeqInfo* seqInfo, const string& flowSeq, const string& imageSeq, std::vector<Mat>& flowx,
				 std::vector<Mat>& flowy, std::vector<Mat>& image)
{
	std::vector<string> flowNames(0);
	std::vector<string> imageNames(0);

	glob(flowSeq, flowNames);
	glob(imageSeq, imageNames);

	int flowNum = flowNames.size();
	int imageNum = imageNames.size();
	int frame_num = flowNum/2;

	double lowerBound = -20.0;
	double higherBound = 20.0;

	string flowFoundX = "flow_x_";
	string flowFoundY = "flow_y_";
	string imageFound = "img_";

	for(int counter = 0; counter < flowNum; ++counter)
	{
		Mat frame;

		if(flowNames.at(counter).find(flowFoundX) != string::npos)
		{
			imread(flowNames.at(counter),0).convertTo(frame, CV_32FC1);

			if(!frame.data)
			{
				std::cerr << "Problem loading image ..." << std::endl;
				break;
			}

			Mat flowx_tmp = Mat::zeros(frame.size(), CV_32FC1);
			convertImageToFlow(frame, flowx_tmp, lowerBound, higherBound);
			flowx.push_back(flowx_tmp);
		}

		else if(flowNames.at(counter).find(flowFoundY) != string::npos)
		{
			imread(flowNames.at(counter),0).convertTo(frame, CV_32FC1);

			if(!frame.data)
			{
				std::cerr << "Problem loading image ..." << std::endl;
				break;
			}

			Mat flowy_tmp = Mat::zeros(frame.size(), CV_32FC1);
			convertImageToFlow(frame, flowy_tmp, lowerBound, higherBound);
			flowy.push_back(flowy_tmp);
		}


		if(counter == 0)
		{
			seqInfo->width = frame.cols;
			seqInfo->height = frame.rows;
		}

    }

    for(int counter = 0; counter < imageNum; ++counter)
    {
    	Mat frame;

    	if(imageNames.at(counter).find(imageFound) != string::npos)
    	{
    		imread(imageNames.at(counter)).convertTo(frame, CV_8UC3);

			if(!frame.data)
			{
				std::cerr << "Problem loading image ..." << std::endl;
				break;
			}

			image.push_back(frame);
    	}
    }

	seqInfo->length = frame_num;
}


void usage()
{
	fprintf(stderr, "Extract improved trajectories from a video\n\n");
	fprintf(stderr, "Usage: DenseTrackStab video_file [options]\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -h                        Display this message and exit\n");
	fprintf(stderr, "  -S [start frame]          The start frame to compute feature (default: S=0 frame)\n");
	fprintf(stderr, "  -E [end frame]            The end frame for feature computing (default: E=last frame)\n");
	fprintf(stderr, "  -L [trajectory length]    The length of the trajectory (default: L=15 frames)\n");
	fprintf(stderr, "  -W [sampling stride]      The stride for dense sampling feature points (default: W=5 pixels)\n");
	fprintf(stderr, "  -N [neighborhood size]    The neighborhood size for computing the descriptor (default: N=32 pixels)\n");
	fprintf(stderr, "  -s [spatial cells]        The number of cells in the nxy axis (default: nxy=2 cells)\n");
	fprintf(stderr, "  -t [temporal cells]       The number of cells in the nt axis (default: nt=3 cells)\n");
	fprintf(stderr, "  -A [scale number]         The number of maximal spatial scales (default: 8 scales)\n");
	fprintf(stderr, "  -I [initial gap]          The gap for re-sampling feature points (default: 1 frame)\n");
	fprintf(stderr, "  -H [human bounding box]   The human bounding box file to remove outlier matches (default: None)\n");
}

bool arg_parse(int argc, char** argv)
{
	// Example of idt cmd: idt -f test.avi -o test.bin -h
	int c = 5;
	bool flag = false;
	while (c+1 < argc){
		char ind = argv[c][1];
		switch(ind){
		case 'S':
			start_frame = atoi(argv[c+1]);
			flag = true;
			break;
		case 'E':
			end_frame = atoi(argv[c+1]);
			flag = true;
			break;
		case 'L':
			track_length = atoi(argv[c+1]);
			break;
		case 'W':
			min_distance = atoi(argv[c+1]);
			break;
		case 'N':
			patch_size = atoi(argv[c+1]);
			break;
		case 's':
			nxy_cell = atoi(argv[c+1]);
			break;
		case 't':
			nt_cell = atoi(argv[c+1]);
			break;
		case 'I':
			init_gap = atoi(argv[c+1]);
			break;
		case 'H':
			bb_file = argv[c+1];
			break;
		case 'T':
			show_track = atoi(argv[c+1]);
			break;
		case 'h':
			usage();
			exit(0);
			break;
		default:
			printf("error parsing arguments at -%c\n  Try '%s -h' for help.", c, argv[c] );
			abort();

		}
		c = c+2;

	}

	return flag;
}

#endif /*INITIALIZE_SEP_H_*/
