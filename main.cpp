#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

const char LENA[]              = "lena.jpg";
const char IDOJUN[]            = "compare.jpg";
const char defaultUnit[]       = "[ms]";
const char defaultWindowName[] = "hoge";
const char dumpTitleOfTime[]   = "CPU\t\tCuda\t\t(w/transfer)\tOpenCL\t\t(w/transfer)";
double coordinateLeft[]  = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0};
double coordinateRight[] = {0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
const int iterationBefore      = 10;
const int iterationAfter       = 50;

void dumpConsumedTimeForDiff(int64 countStart, int64 countStop, const char* unit = NULL)
{
	if(unit == NULL)
	{
		unit = defaultUnit;
	}
	int64 elapsedTime = countStop - countStart;
	int64 elapsedDiff = (elapsedTime * 1000) / (iterationAfter);
	std::cout << (elapsedDiff) / cv::getTickFrequency() << unit << '\t';


}
void dumpConsumedTime(int64 countStart, int64 countStop, int64 countStartAfter, int64 countStopAfter, const char* unit = NULL)
{
	if(unit == NULL)
	{
		unit = defaultUnit;
	}
	int64 elapsedBefore  = countStop - countStart;
	int64 elapsedAfter   = countStopAfter - countStartAfter;
	int64 elapsedDiff    = ((elapsedAfter - elapsedBefore) * 1000) / (iterationAfter - iterationBefore);
	std::cout << (elapsedDiff) / cv::getTickFrequency() << unit << '\t';
}

void comapareWarpAffine(const char* filename)
{
	using namespace cv;
	Mat input;

	// Load input image
	input = imread(filename);

	cv::Mat coordinatesLeft  = cv::Mat(1, 4, CV_64FC2, (void*)coordinateLeft);
	cv::Mat coordinatesRight = cv::Mat(1, 4, CV_64FC2, (void*)coordinateRight);
	cv::Mat outputMaskArray  = cv::Mat(1, 4, CV_8U);
	cv::Mat homography;
	coordinatesLeft  *= (double)input.size().width;
	coordinatesRight *= (double)input.size().width;

	homography = cv::findHomography(coordinatesLeft, coordinatesRight,outputMaskArray);
	//std::cout << homography << std::endl;

	Mat result;
	// Start measureing the Sobel filter
	int64 countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		warpPerspective(input, result, homography, input.size());
	}
	// Stop measuring
	int64 countStop = getTickCount();
	int64 countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		warpPerspective(input, result, homography, input.size());
	}
	// Stop measuring
	int64 countStopAfter = getTickCount();


	dumpConsumedTime(countStart, countStop, countStartAfter, countStopAfter, defaultUnit);

	// Convert the image to RGBA, so it can be uploaded to GPU
	cvtColor(input.clone(), input, COLOR_RGB2RGBA);
	// Start measureing including the upload time
	int64 countBeforeTransfer = getTickCount();
	// Upload the image to GPU (convert cv::Mat to cv::cuda::GpuMat)
	cuda::GpuMat gpuInput = cuda::GpuMat(input);
	cuda::GpuMat gpuResult;
	// Prepare the Sobel filter
	cuda::warpPerspective(gpuInput, gpuResult, homography, gpuInput.size());


	// Start measureing the Sobel filter
	countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		cuda::warpPerspective(gpuInput, gpuResult, homography, gpuInput.size());
	}
	// Stop measuring
	countStop = getTickCount();
	countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		cuda::warpPerspective(gpuInput, gpuResult, homography, input.size());
	}
	// Stop measuring
	countStopAfter = getTickCount();

	dumpConsumedTime(countStart, countStop, countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);
	//dumpConsumedTime(countBeforeTransfer, countStop, defaultUnit);

	// Input using OpenCL
	UMat umatInput = input.getUMat(ACCESS_READ);
	// Output using OpenCL
	UMat umatResult;
	// dummy call
	warpPerspective(umatInput, umatResult, homography, input.size());

	// To flush the cache, reload the image
	input = imread(filename);
	// Start measureing including the upload time
	countBeforeTransfer = getTickCount();
	// Upload the image to OpenCL device
	umatInput = input.getUMat(ACCESS_READ);
	// Start measureing the Sobel filter
	countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		// Apply Sobel filter
		warpPerspective(umatInput, umatResult, homography, input.size());
	}
	// Stop measuring
	countStop = getTickCount();
	countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		warpPerspective(umatInput, umatResult, homography, input.size());
	}
	// Stop measuring
	countStopAfter = getTickCount();


	dumpConsumedTime(countStart, countStop, countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);
	//dumpConsumedTime(countBeforeTransfer, countStop, defaultUnit);
	std::cout << std::endl;
	// Show the result from OpenCL
	imshow(defaultWindowName, umatResult);
	imshow("input", input);
	waitKey(0);
}


void comapareSobel(const char* filename)
{
	using namespace cv;
	Mat input;

	// Load input image
	input = imread(filename);

	Mat result;
	// Start measureing the Sobel filter
	int64 countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		Sobel(input, result, -1, 1, 1);
	}
	// Stop measuring
	int64 countStop = getTickCount();
	int64 countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		Sobel(input, result, -1, 1, 1);
	}
	int64 countStopAfter = getTickCount();

	dumpConsumedTime(countStart, countStop, countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);

	// Convert the image to RGBA, so it can be uploaded to GPU
	cvtColor(input.clone(), input, COLOR_RGB2RGBA);
	// Start measureing including the upload time
	int64 countBeforeTransfer = getTickCount();
	// Upload the image to GPU (convert cv::Mat to cv::cuda::GpuMat)
	cuda::GpuMat gpuInput = cuda::GpuMat(input);
	cuda::GpuMat gpuResult;
	// Prepare the Sobel filter
	Ptr<cuda::Filter> gpuSobel = cuda::createSobelFilter(gpuInput.type(), gpuInput.type(), 1, 1);

	// Start measureing the Sobel filter
	countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		gpuSobel->apply(gpuInput, gpuResult);
	}
	// Stop measuring
	countStop = getTickCount();
	countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		gpuSobel->apply(gpuInput, gpuResult);
	}
	countStopAfter = getTickCount();

	dumpConsumedTime(countStart, countStop, countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);
	//dumpConsumedTime(countBeforeTransfer, countStop, defaultUnit);

	// Input using OpenCL
	UMat umatInput = input.getUMat(ACCESS_READ);
	// Output using OpenCL
	UMat umatResult;
	// dummy call
	Sobel(umatInput, umatResult, -1, 1, 1);

	// To flush the cache, reload the image
	input = imread(filename);
	// Start measureing including the upload time
	countBeforeTransfer = getTickCount();
	// Upload the image to OpenCL device
	umatInput = input.getUMat(ACCESS_READ);
	// Start measureing the Sobel filter
	countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		// Apply Sobel filter
		Sobel(umatInput, umatResult, -1, 1, 1);
	}
	// Stop measuring
	countStop = getTickCount();


	dumpConsumedTime(countStart, countStop, countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);
	//dumpConsumedTime(countBeforeTransfer, countStop, defaultUnit);
	std::cout << std::endl;
	// Show the result from OpenCL
	imshow(defaultWindowName, umatResult);
	imshow("input", input);
	waitKey(0);
}

void compareDiff(const char* filenameBefore, const char* filenameAfter)
{
	using namespace cv;
	Mat before;
	Mat after;

	before = imread(filenameBefore);
	after = imread(filenameAfter);

	Mat diff;
	// Start measureing the subtraction
	int64 countStart = getTickCount();
	// Apply subtraction
	for(int i = 0;i < iterationBefore;i++)
	{
		diff = before - after;
	}
	// Stop measuring
	int64 countStop = getTickCount();
	int64 countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		diff = before - after;
	}
	int64 countStopAfter = getTickCount();

	dumpConsumedTimeForDiff(countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);

	// Start measureing including the upload time
	int64 countBeforeTransfer = getTickCount();
	// Upload the image to GPU (convert cv::Mat to cv::cuda::GpuMat)
	cuda::GpuMat gpuBefore = cuda::GpuMat(before);
	cuda::GpuMat gpuAfter  = cuda::GpuMat(after);
	cuda::GpuMat gpuDiff;

	// Start measureing the subtraction
	countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		// Apply Subtraction
		cuda::subtract(gpuBefore, gpuAfter, gpuDiff);
	}
	// Stop measuring
	countStop = getTickCount();
	countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		// Apply Subtraction
		cuda::subtract(gpuBefore, gpuAfter, gpuDiff);
	}
	// Stop measuring
	countStopAfter = getTickCount();

	dumpConsumedTimeForDiff(countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);
	//dumpConsumedTime(countBeforeTransfer, countStop, defaultUnit);


	UMat umatBefore = before.getUMat(ACCESS_READ);
	UMat umatAfter  = after.getUMat(ACCESS_READ);
	UMat umatDiff;
	// dummy call
	subtract(umatBefore, umatAfter, umatDiff);

	// To flush the cache, reload the image
	before = imread(filenameBefore);
	after = imread(filenameAfter);
	// Start measureing including the upload time
	countBeforeTransfer = getTickCount();
	umatBefore = before.getUMat(ACCESS_READ);
	umatAfter  = after.getUMat(ACCESS_READ);
	// Start measureing the subtraction
	countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		// Apply subtract on OpenCL device
		subtract(umatBefore, umatAfter, umatDiff);
	}
	// Convert the image to RGBA, so it can be uploaded to GPU
	countStop = getTickCount();
	countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		// Apply subtract on OpenCL device
		subtract(umatBefore, umatAfter, umatDiff);
	}
	countStopAfter = getTickCount();


	dumpConsumedTimeForDiff(countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);
	//dumpConsumedTime(countBeforeTransfer, countStop, defaultUnit);
	std::cout << std::endl;
	imshow(defaultWindowName, umatDiff);
	waitKey(0);
}

void compareAbsDiff(const char* filenameBefore, const char* filenameAfter)
{
	using namespace cv;
	Mat before;
	Mat after;

	before = imread(filenameBefore);
	after = imread(filenameAfter);

	Mat diff;
	// Start measureing the absdiff
	int64 countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		// Apply absdiff
		absdiff(before, after, diff);
	}
	// Stop measuring
	int64 countStop = getTickCount();
	int64 countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		// Apply absdiff
		absdiff(before, after, diff);
	}
	int64 countStopAfter = getTickCount();

	//std::cout << countStart << '\t' << countStop << '\t' << countStartAfter << '\t'  << countStopAfter << std::endl;
	dumpConsumedTimeForDiff(countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);

	// Start measureing including the upload time
	int64 countBeforeTransfer = getTickCount();
	// Upload the image to GPU (convert cv::Mat to cv::cuda::GpuMat)
	cuda::GpuMat gpuBefore = cuda::GpuMat(before);
	cuda::GpuMat gpuAfter  = cuda::GpuMat(after);
	cuda::GpuMat gpuDiff;

	// Start measureing the absdiff
	countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		// Apply abdiff on CUDA
		cuda::absdiff(gpuBefore, gpuAfter, gpuDiff);
	}
	// Stop measuring
	countStop = getTickCount();
	countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		// Apply abdiff on CUDA
		cuda::absdiff(gpuBefore, gpuAfter, gpuDiff);
	}
	countStopAfter = getTickCount();

	dumpConsumedTimeForDiff(countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);
	//dumpConsumedTime(countBeforeTransfer, countStop, defaultUnit);

	UMat umatBefore = before.getUMat(ACCESS_READ);
	UMat umatAfter  = after.getUMat(ACCESS_READ);
	UMat umatDiff;

	// dummy call
	absdiff(umatBefore, umatAfter, umatDiff);

	// To flush the cache, reload the image
	before = imread(filenameBefore);
	after = imread(filenameAfter);
	// Start measureing including the upload time
	countBeforeTransfer = getTickCount();
	// Upload the image (convert it to UMat)
	umatBefore = before.getUMat(ACCESS_READ);
	umatAfter  = after.getUMat(ACCESS_READ);
	// Start measureing the subtraction
	countStart = getTickCount();
	for(int i = 0;i < iterationBefore;i++)
	{
		// Apply absdiff on OpenCL device
		absdiff(umatBefore, umatAfter, umatDiff);
	}
	// Stop measuring
	countStop = getTickCount();
	countStartAfter = getTickCount();
	for(int i = 0;i < iterationAfter;i++)
	{
		// Apply absdiff on OpenCL device
		absdiff(umatBefore, umatAfter, umatDiff);
	}
	countStopAfter = getTickCount();

	dumpConsumedTimeForDiff(countStartAfter, countStopAfter, defaultUnit);
	//dumpConsumedTime(countStart, countStop, defaultUnit);
	//dumpConsumedTime(countBeforeTransfer, countStop, defaultUnit);
	std::cout << std::endl;
	imshow(defaultWindowName, umatDiff);
	waitKey(0);
}

int main(int argc, const char* argv[])
{
	// dump the build information
	//std::cout << cv::getBuildInformation() << std::endl;
#if 0
	// Check HW support of CPU vector
	std::cout << "CPU_MMX   : " << cv::checkHardwareSupport(CV_CPU_MMX   ) << std::endl;
	std::cout << "CPU_SSE   : " << cv::checkHardwareSupport(CV_CPU_SSE   ) << std::endl;
	std::cout << "CPU_SSE2  : " << cv::checkHardwareSupport(CV_CPU_SSE2  ) << std::endl;
	std::cout << "CPU_SSE3  : " << cv::checkHardwareSupport(CV_CPU_SSE3  ) << std::endl;
	std::cout << "CPU_SSSE3 : " << cv::checkHardwareSupport(CV_CPU_SSSE3 ) << std::endl;
	std::cout << "CPU_SSE4_1: " << cv::checkHardwareSupport(CV_CPU_SSE4_1) << std::endl;
	std::cout << "CPU_SSE4_2: " << cv::checkHardwareSupport(CV_CPU_SSE4_2) << std::endl;
	std::cout << "CPU_POPCNT: " << cv::checkHardwareSupport(CV_CPU_POPCNT) << std::endl;
	std::cout << "CPU_AVX   : " << cv::checkHardwareSupport(CV_CPU_AVX   ) << std::endl;
	// this will always return 0
	// Today (Jan 2015), runtime check of NEON optimization is not yet implemented in OpenCv
	std::cout << "CPU_NEON  : " << cv::checkHardwareSupport(CV_CPU_NEON  ) << std::endl;

	// Count the # of CUDA devices(i.e. GPU)
	int cudaDeviceCount;
	cudaDeviceCount = cv::cuda::getCudaEnabledDeviceCount();
	std::cout << "Device # of CUDA       : " << cudaDeviceCount << std::endl;

	// Dump the HW feature of each GPU
	for(int iDevice = 0;iDevice < cudaDeviceCount;iDevice++)
	{
		cv::cuda::DeviceInfo hoge(iDevice);
		std::cout << "name [" << cv::cuda::getDevice() << "]               : " << hoge.name() << std::endl;

		std::cout << "FEATURE_SET_COMPUTE_10 : " << cv::cuda::deviceSupports(cv::cuda::FEATURE_SET_COMPUTE_10) << std::endl;
		std::cout << "FEATURE_SET_COMPUTE_11 : " << cv::cuda::deviceSupports(cv::cuda::FEATURE_SET_COMPUTE_11) << std::endl;
		std::cout << "FEATURE_SET_COMPUTE_12 : " << cv::cuda::deviceSupports(cv::cuda::FEATURE_SET_COMPUTE_12) << std::endl;
		std::cout << "FEATURE_SET_COMPUTE_13 : " << cv::cuda::deviceSupports(cv::cuda::FEATURE_SET_COMPUTE_13) << std::endl;
		std::cout << "FEATURE_SET_COMPUTE_20 : " << cv::cuda::deviceSupports(cv::cuda::FEATURE_SET_COMPUTE_20) << std::endl;
		std::cout << "FEATURE_SET_COMPUTE_21 : " << cv::cuda::deviceSupports(cv::cuda::FEATURE_SET_COMPUTE_21) << std::endl;
		std::cout << "FEATURE_SET_COMPUTE_30 : " << cv::cuda::deviceSupports(cv::cuda::FEATURE_SET_COMPUTE_30) << std::endl;
		std::cout << "FEATURE_SET_COMPUTE_35 : " << cv::cuda::deviceSupports(cv::cuda::FEATURE_SET_COMPUTE_35) << std::endl;

		std::cout << "GLOBAL_ATOMICS         : " << cv::cuda::deviceSupports(cv::cuda::GLOBAL_ATOMICS) << std::endl;
		std::cout << "SHARED_ATOMICS         : " << cv::cuda::deviceSupports(cv::cuda::SHARED_ATOMICS) << std::endl;
		std::cout << "NATIVE_DOUBLE          : " << cv::cuda::deviceSupports(cv::cuda::NATIVE_DOUBLE) << std::endl;
		std::cout << "WARP_SHUFFLE_FUNCTIONS : " << cv::cuda::deviceSupports(cv::cuda::WARP_SHUFFLE_FUNCTIONS) << std::endl;
		std::cout << "DYNAMIC_PARALLELISM    : " << cv::cuda::deviceSupports(cv::cuda::DYNAMIC_PARALLELISM) << std::endl;
	}

	cv::ocl::setUseOpenCL(true);
	std::vector<cv::ocl::PlatformInfo> ocl_platform_info;
	// ocl::getPlatfomsInfo might not get correct number of OpenCL device on Jetson TK1
	cv::ocl::getPlatfomsInfo(ocl_platform_info);

	// Count the # of OpenCL capable device
	std::cout << "Device # of OpenCL     : " << ocl_platform_info.size() << std::endl;
	for(unsigned int iDevice = 0;iDevice < ocl_platform_info.size();iDevice++)
	{
		std::cout << "name   [" << iDevice << "]             : " << ocl_platform_info[iDevice].name() << std::endl;
		std::cout << "vendor [" << iDevice << "]             : " << ocl_platform_info[iDevice].vendor() << std::endl;
		std::cout << "version[" << iDevice << "]             : " << ocl_platform_info[iDevice].version() << std::endl;
		cv::ocl::Device hoge;
		ocl_platform_info[iDevice].getDevice(hoge, iDevice);
		std::cout << "name            [" << iDevice << "]    : " << hoge.name() << std::endl;
		std::cout << "extensions      [" << iDevice << "]    : " << hoge.extensions() <<  std::endl;
		std::cout << "version         [" << iDevice << "]    : " << hoge.version()    <<  std::endl;
		std::cout << "vendorName      [" << iDevice << "]    : " << hoge.vendorName() <<  std::endl;
		std::cout << "OpenCL_C_Version[" << iDevice << "]    : " << hoge.OpenCL_C_Version() << std::endl;
		std::cout << "OpenCL Version  [" << iDevice << "]    : " << hoge.OpenCLVersion() <<  std::endl;
		std::cout << "Version Major   [" << iDevice << "]    : " << hoge.deviceVersionMajor() <<  std::endl;
		std::cout << "Version Minor   [" << iDevice << "]    : " << hoge.deviceVersionMinor() <<  std::endl;
		std::cout << "Driver Version  [" << iDevice << "]    : " << hoge.driverVersion() << std::endl;
	}
	std::cout << "haveOpenCL()           : " << cv::ocl::haveOpenCL() << std::endl;
	std::cout << "useOpenCL()            : " << cv::ocl::useOpenCL()  << std::endl;
#endif
	std::cout << dumpTitleOfTime                                      << std::endl;

	cv::namedWindow(defaultWindowName);
	// Compare the performance of CPU, CUDA and OpenCL
	// Compare the performance on AbsDiff
	compareAbsDiff(LENA, IDOJUN);
	// Compare the performance on Subtraction
	compareDiff(LENA, IDOJUN);
	// Compare the performance on Sobel
	comapareSobel(IDOJUN);
	comapareWarpAffine(IDOJUN);

	cv::Mat coordinatesLeft  = cv::Mat(1, 4, CV_64FC2, (void*)coordinateLeft);
	cv::Mat coordinatesRight = cv::Mat(1, 4, CV_64FC2, (void*)coordinateRight);
	cv::Mat outputMaskArray  = cv::Mat(1, 4, CV_8U);
	cv::Mat homography;

	homography = cv::findHomography(coordinatesLeft, coordinatesRight,outputMaskArray);
	//std::cout << homography << std::endl;

	return 0;
}

