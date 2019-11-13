#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace cv;
using namespace std;
using namespace dnn;

Mat eval(Net net, Mat face) {
	Mat blob = blobFromImage(face, 1.0 / 255, Size(96, 96), Scalar(0, 0, 0, 0), true, false);
	net.setInput(blob);
	//Mat result = net.forward();
	Mat result = net.forward().clone();
	return result;
}

inline int write_embedded(Mat& face, bool header = true)
{
	int i;

	FILE* fp;
	errno_t err;

	if (header)
		err = fopen_s(&fp, "face_embedded_value.csv", "wt");
	else
		err = fopen_s(&fp, "face_embedded_value.csv", "at");

	if (err != 0)
		return err;

	if (header == true)
	{
		// csv header
		for (i = 0; i < 127; i++)
			fprintf(fp, "\"v_%d\",", i);
		fprintf(fp, "\"v_127\"\n");
	}

	//csv data
	for (i = 0; i < 127; i++)
	{
		fprintf(fp, "%f,", face.at<float>(i));
	}
	fprintf(fp, "%lf\n", face.at<double>(i));

	fclose(fp);

	return 0;
}



int main(int argc, char** argv)
{
	Net net = readNetFromTorch("nn4.small2.v1.t7");
	net.setPreferableTarget(DNN_TARGET_OPENCL);
	cout << "read model" << endl;

	Mat face1 = imread("face_1.jpg");
	Mat face1Vec = eval(net, face1);
	cout << "Face1Vec" << face1Vec << endl;

	Mat face2 = imread("face_2.jpg");
	Mat face2Vec = eval(net, face2);
	cout << "Face2Vec" << face2Vec << endl;

	auto s1 = face1Vec.dot(face1Vec);
	auto s2 = face1Vec.dot(face2Vec);

	cout << "s1: " << s1 << endl;
	cout << "s2: " << s2 << endl;

	write_embedded(face1Vec);
	write_embedded(face2Vec, false);
	// write_embedded(face3Vec, false);
	// ...

	return 0;
}
