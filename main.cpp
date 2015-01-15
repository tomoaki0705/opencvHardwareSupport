#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>

const char LENA[] = "lena.jpg";
const char IDOJUN[] = "compare.jpg";
const char defaultUnit[] = "[ms]";
const char defaultWindowName[] = "hoge";

void dumpConsumedTime(int64 countStart, int64 countStop, const char* unit = NULL)
{
	if(unit == NULL)
	{
		unit = defaultUnit;
	}
	std::cout << ((countStop - countStart) * 1000) / cv::getTickFrequency() << unit << std::endl;
}

#if 0
void comapareSobel(const char* filename)
{
	using namespace cv;
	Mat input;

	input = imread(filename);

	Mat result;
	int64 countStart = getTickCount();
	cv::blur(input, result, 3);
	int64 countStop = getTickCount();

	dumpConsumedTime(countStart, countStop, defaultUnit);

	int64 countBeforeTransfer = getTickCount();
	cuda::GpuMat gpuInput = cuda::GpuMat(input);
	cuda::GpuMat gpuResult;
	cuda::createGaussianFilter(gpuInput.type, gpuInput.type, 3);

	countStart = getTickCount();
	cuda::Filter::apply(input, result, 
	cuda::subtract(gpuBefore, gpuAfter, gpuDiff);
	countStop = getTickCount();

	dumpConsumedTime(countStart, countStop, defaultUnit);
	dumpConsumedTime(countBeforeTransfer, countStop, "[ms] (including transfer)");
}
#endif

void compareDiff(const char* filenameBefore, const char* filenameAfter)
{
	using namespace cv;
	Mat before;
	Mat after;

	before = imread(filenameBefore);
	after = imread(filenameAfter);

	Mat diff;
	int64 countStart = getTickCount();
	diff = before - after;
	int64 countStop = getTickCount();

	dumpConsumedTime(countStart, countStop, defaultUnit);

	int64 countBeforeTransfer = getTickCount();
	cuda::GpuMat gpuBefore = cuda::GpuMat(before);
	cuda::GpuMat gpuAfter  = cuda::GpuMat(after);
	cuda::GpuMat gpuDiff;

	countStart = getTickCount();
	cuda::subtract(gpuBefore, gpuAfter, gpuDiff);
	countStop = getTickCount();

	dumpConsumedTime(countStart, countStop, defaultUnit);
	dumpConsumedTime(countBeforeTransfer, countStop, "[ms] (including transfer)");
}

void compareAbsDiff(const char* filenameBefore, const char* filenameAfter)
{
	using namespace cv;
	Mat before;
	Mat after;

	before = imread(filenameBefore);
	after = imread(filenameAfter);

	Mat diff;
	int64 countStart = getTickCount();
	for(int i = 0;i < 10;i++)
	{
		absdiff(before, after, diff);
	}
	int64 countStop = getTickCount();

	dumpConsumedTime(countStart, countStop, defaultUnit);

	int64 countBeforeTransfer = getTickCount();
	cuda::GpuMat gpuBefore = cuda::GpuMat(before);
	cuda::GpuMat gpuAfter  = cuda::GpuMat(after);
	cuda::GpuMat gpuDiff;

	countStart = getTickCount();
	for(int i = 0;i < 10;i++)
	{
		cuda::absdiff(gpuBefore, gpuAfter, gpuDiff);
	}
	countStop = getTickCount();

	dumpConsumedTime(countStart, countStop, defaultUnit);
	dumpConsumedTime(countBeforeTransfer, countStop, "[ms] (including transfer)");
	imshow(defaultWindowName, diff);
	waitKey(0);
}

int main(int argc, const char* argv[])
{
	// dump the build information
	std::cout << cv::getBuildInformation() << std::endl;

	std::cout << "CPU_MMX   : " << cv::checkHardwareSupport(CV_CPU_MMX   ) << std::endl;
	std::cout << "CPU_SSE   : " << cv::checkHardwareSupport(CV_CPU_SSE   ) << std::endl;
	std::cout << "CPU_SSE2  : " << cv::checkHardwareSupport(CV_CPU_SSE2  ) << std::endl;
	std::cout << "CPU_SSE3  : " << cv::checkHardwareSupport(CV_CPU_SSE3  ) << std::endl;
	std::cout << "CPU_SSSE3 : " << cv::checkHardwareSupport(CV_CPU_SSSE3 ) << std::endl;
	std::cout << "CPU_SSE4_1: " << cv::checkHardwareSupport(CV_CPU_SSE4_1) << std::endl;
	std::cout << "CPU_SSE4_2: " << cv::checkHardwareSupport(CV_CPU_SSE4_2) << std::endl;
	std::cout << "CPU_POPCNT: " << cv::checkHardwareSupport(CV_CPU_POPCNT) << std::endl;
	std::cout << "CPU_AVX   : " << cv::checkHardwareSupport(CV_CPU_AVX   ) << std::endl;
	std::cout << "CPU_NEON  : " << cv::checkHardwareSupport(CV_CPU_NEON  ) << std::endl;

	int cudaDeviceCount;
	cudaDeviceCount = cv::cuda::getCudaEnabledDeviceCount();
	std::cout << "Device # of CUDA       : " << cudaDeviceCount << std::endl;

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

	std::vector<cv::ocl::PlatformInfo> ocl_platform_info;
	cv::ocl::getPlatfomsInfo(ocl_platform_info);

	std::cout << "Device # of OpenCL     : " << cudaDeviceCount << std::endl;
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

	cv::namedWindow(defaultWindowName);
	compareAbsDiff(LENA, IDOJUN);
	compareDiff(LENA, IDOJUN);

	return 0;
}

