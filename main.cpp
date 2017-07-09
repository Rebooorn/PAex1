
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

// functions for drawing
void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);
void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ);

// functions for dimensionality reduction
// first parameter: features saved in rows
// second parameter: desired # of dimensions ( here: 2)
Mat reducePCA(Mat &dataMatrix, unsigned int dim);
Mat reduceIsomap(Mat &dataMatrix, unsigned int dim);
Mat reduceLLE(Mat &dataMatrix, unsigned int dim);
template<typename T> T minLch(T a, T b);

int main(int argc, char** argv)
{
	// generate Data Matrix
	unsigned int nSamplesI = 10;
	unsigned int nSamplesJ = 10;
	Mat dataMatrix =  Mat(nSamplesI*nSamplesJ, 3, CV_64F);
	// noise in the data
	double noiseScaling = 1000.0;
	
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			dataMatrix.at<double>(i*nSamplesJ+j,0) = (i/(double)nSamplesI * 2.0 * 3.14 + 3.14) * cos(i/(double)nSamplesI * 2.0 * 3.14) + (rand() % 100)/noiseScaling;
			dataMatrix.at<double>(i*nSamplesJ+j,1) = (i/(double)nSamplesI * 2.0 * 3.14 + 3.14) * sin(i/(double)nSamplesI * 2.0 * 3.14) + (rand() % 100)/noiseScaling;
			dataMatrix.at<double>(i*nSamplesJ+j,2) = 10.0*j/(double)nSamplesJ + (rand() % 100)/noiseScaling;
		}
	}
	
	// Draw 3D Manifold
	Draw3DManifold(dataMatrix, "3D Points",nSamplesI,nSamplesJ);
	
	// PCA
	Mat dataPCA = reducePCA(dataMatrix,2);
	Draw2DManifold(dataPCA,"PCA",nSamplesI,nSamplesJ);
	
	// Isomap
	Mat dataIsomap = reduceIsomap(dataMatrix,2);
	Draw2DManifold(dataIsomap,"ISOMAP",nSamplesI,nSamplesJ);
	
	//LLE
	//Mat dataLLE = reduceLLE(dataMatrix,2);
	//Draw2DManifold(dataLLE,"LLE",nSamplesI,nSamplesJ);
	
	waitKey(0);


	return 0;
}

Mat reducePCA(Mat &dataMatrix, unsigned int dim)
{
	Mat PCAmat = dataMatrix.clone();
	// SALIENT: NORMALIZATION!!
	//cout << PCAmat << endl;
	//cout << mean(PCAmat) << endl;
	Mat PCAnorm;
	reduce(PCAmat, PCAnorm, 0, CV_REDUCE_AVG);
	//cout << PCAnorm << endl;
	Mat tmp(dataMatrix.rows, 1, CV_64F);
	tmp.setTo(1);	//set all to 1
	PCAnorm = tmp*PCAnorm;
	//cout << PCAnorm << endl;
	PCAmat = PCAmat - PCAnorm;
	//cout << PCAmat << endl;
	SVD Proc(PCAmat);	// SVD decomposition
	Mat umat = Proc.u;	// get U and W
	Mat wmat = Proc.w;
	//cout << "src size:" << dataMatrix.size() << endl;
	//cout << "u size:" << umat << endl;
	//cout << "w size:" << wmat << endl;
	Mat wprime(dim, dim, CV_64F);	// extract U' and W'
	Mat uprime(PCAmat.rows, dim, CV_64F);
	wprime.setTo(0);
	uprime.setTo(0);
	wprime = Mat::diag(wmat.rowRange(0,dim));
	uprime = umat.colRange(0, dim);
	//cout << "u':" << uprime << endl << "w':" << wprime << endl;
	PCAmat = uprime*wprime;
	//cout << "lower dim size:" << PCAmat.size() << endl;
	//cout << "PCAmat:" << PCAmat << endl;
	// scale mat to (-0.5,0.5) for display....that is why I love python more...
	normalize(PCAmat, PCAmat,0.4,-0.4,NORM_MINMAX);
	//cout << PCAmat << endl;
	return PCAmat;
}

Mat reduceLLE(Mat &dataMatrix, unsigned int dim)
{
	return dataMatrix;
}

Mat reduceIsomap(Mat &dataMatrix, unsigned int dim)
{
	// form distance matrix
	int KNN_win = 10;	//knn window size
	int r = dataMatrix.rows;
	Mat Dist = Mat(r, r, CV_64F);

	double tmp = 0;
	for (int i = 0;i < r;i++) {
		for (int j = i;j < r; j++) {
			tmp = norm(dataMatrix.row(i), dataMatrix.row(j));
			Dist.at<double>(i, j) = tmp;
			Dist.at<double>(j, i) = tmp;
		}
	}
	//cout << "Dist:" << endl;
	//cout << Dist << endl;
	// k nearest neighbour
	Mat KNN(r, r, CV_64F);
	Mat DistCopy = Dist.clone();
	KNN.setTo(10000);
	for (int i = 0;i < r;i++) {
		Mat row_temp = DistCopy.row(i);
		// find 10 smallest values in DistCopy and move to KNN
		for (int j = 0;j < KNN_win+1 ;j++) {
			double min, max;
			Point minloc, maxloc;
			minMaxLoc(row_temp, &min, &max, &minloc, &maxloc);
			KNN.at<double>(i, minloc.x) = min;
			row_temp.at<double>(minloc) = 10000;
		}
	}
	//cout << "KNN:" << endl;
	//cout << KNN << endl;
	// extract shortest path
	//TODO
	for (int i = 0;i < r;i++) {
		for (int j = 0;j < r;j++) {
			for (int k = 0;k < r;k++) {
				KNN.at<double>(i, j) = minLch(KNN.at<double>(i, j), KNN.at<double>(i, k)+KNN.at<double>(k, j));
			}
		}
	}
	//cout << "KNN shortest path" << endl;
	//cout << KNN << endl;
	// make sure dissimilarity
	
	for (int i = 0;i < r;i++) {
		for (int j = 0;j < i;j++) {
			KNN.at<double>(j, i) = KNN.at<double>(i, j);
		}
	}
	
	// MDS
	Mat distSquare;
	multiply(KNN, KNN, distSquare);
	distSquare *= -0.5;
	double sumAll = sum(distSquare)[0];
	sumAll /= r*r;
	Mat B(r, r, CV_64F);
	B.setTo(0.0);
	double sumCol = 0, sumRow = 0;
	for (int i = 0;i < r; i++) {
		sumRow = sum(distSquare.row(i))[0];
		sumRow /= r;
		for (int j = 0;j < r;j++) {
			sumCol = sum(distSquare.col(j))[0];
			sumCol /= r;
			B.at<double>(i, j) = distSquare.at<double>(i, j) + sumAll - sumRow - sumCol;
		}
	}

	//cout << "B:" << endl;
	//cout << B << endl;
	// eigen analysis
	Mat eVal, eVec;
	eigen(B, eVal, eVec);
	//cout << eVec << endl;
	eVec = eVec.rowRange(0, 2);
	eVal = eVal.diag(eVal.rowRange(0,2));
	//cout << eVec << endl;
	sqrt(eVal, eVal);
	Mat Y = eVal*eVec;
	//cout << "Y:" << endl;
	//cout << Y << endl;
	normalize(Y.t(), Y, 0.4, -0.4,NORM_MINMAX);
	return Y;
}

void Draw3DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
	Mat origImage = Mat(1000,1000,CV_8UC3);
	origImage.setTo(0.0);
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			Point p1;
			p1.x = dataMatrix.at<double>(i*nSamplesJ+j,0)*50.0 +500.0 - dataMatrix.at<double>(i*nSamplesJ+j,2)*10;
			p1.y = dataMatrix.at<double>(i*nSamplesJ+j,1) *50.0 + 500.0 - dataMatrix.at<double>(i*nSamplesJ+j,2)*10;
			circle(origImage,p1,3,Scalar( 255, 255, 255 ));
			
			Point p2;
			if(i < nSamplesI-1)
			{
				p2.x = dataMatrix.at<double>((i+1)*nSamplesJ+j,0)*50.0 +500.0 - dataMatrix.at<double>((i+1)*nSamplesJ+(j),2)*10;
				p2.y = dataMatrix.at<double>((i+1)*nSamplesJ+j,1) *50.0 + 500.0 - dataMatrix.at<double>((i+1)*nSamplesJ+(j),2)*10;
			
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
			if(j < nSamplesJ-1)
			{
				p2.x = dataMatrix.at<double>((i)*nSamplesJ+j+1,0)*50.0 +500.0 - dataMatrix.at<double>((i)*nSamplesJ+(j+1),2)*10;
				p2.y = dataMatrix.at<double>((i)*nSamplesJ+j+1,1) *50.0 + 500.0 - dataMatrix.at<double>((i)*nSamplesJ+(j+1),2)*10;
			
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
		}
	}
	

	namedWindow( name, WINDOW_AUTOSIZE );
	imshow( name, origImage ); 
}

void Draw2DManifold(Mat &dataMatrix, char const * name, int nSamplesI, int nSamplesJ)
{
	Mat origImage = Mat(1000,1000,CV_8UC3);
	origImage.setTo(0.0);
	for (int i = 0; i < nSamplesI; i++)
	{
		for (int j = 0; j < nSamplesJ; j++)
		{
			Point p1;
			p1.x = dataMatrix.at<double>(i*nSamplesJ+j,0)*1000.0 +500.0;
			p1.y = dataMatrix.at<double>(i*nSamplesJ+j,1) *1000.0 + 500.0;
			//circle(origImage,p1,3,Scalar( 255, 255, 255 ));
			
			Point p2;
			if(i < nSamplesI-1)
			{
				p2.x = dataMatrix.at<double>((i+1)*nSamplesJ+j,0)*1000.0 +500.0;
				p2.y = dataMatrix.at<double>((i+1)*nSamplesJ+j,1) *1000.0 + 500.0;
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
			if(j < nSamplesJ-1)
			{
				p2.x = dataMatrix.at<double>((i)*nSamplesJ+j+1,0)*1000.0 +500.0;
				p2.y = dataMatrix.at<double>((i)*nSamplesJ+j+1,1) *1000.0 + 500.0;
				line( origImage, p1, p2, Scalar( 255, 255, 255 ), 1, 8 );
			}
			
		}
	}
	

	namedWindow( name, WINDOW_NORMAL );
	imshow( name, origImage ); 
	imwrite( (String(name) + ".png").c_str(),origImage);
}

//helper function
template<typename T>
T minLch(T a, T b) {
	if (a < b) return a;
	else return b;
}