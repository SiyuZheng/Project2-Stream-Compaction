#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweep(int n, int d, int *odata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int step = 1 << (d + 1);
			if (index % step == 0) {
				odata[index + step - 1] += odata[index + (1 << d) - 1];
			}
		}

		__global__ void kernDownSweep(int n, int d, int *odata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int step = 1 << (d + 1);
			if (index % step == 0) {
				int t = odata[index + (1 << d) - 1];
				odata[index + (1 << d) - 1] = odata[index + step - 1];
				odata[index + step - 1] += t;
			}
		}



        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void Nonscan(int n, int *odata, const int *idata) {
            
            // TODO
			
			
			int upLimit = ilog2ceil(n);
			int len = 1 << upLimit;

			int blockSize = 1024;
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((len + blockSize - 1) / blockSize);

			int *dev_data;
			cudaMalloc((void**)&dev_data, len * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaMemcpy(dev_data, idata, len * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_data failed!");
			
			

			timer().startGpuTimer();
			for (int d = 0; d <= upLimit - 1; d++) {
				kernUpSweep << <blocksPerGrid, threadsPerBlock >> > (len, d, dev_data);
				checkCUDAError("kernUpSweep failed!");
			}

			cudaMemset(&dev_data[len - 1], 0, sizeof(int));
			checkCUDAError("cudaMemcpy set last one to be zero failed!");
			
			for (int d = upLimit - 1; d >= 0; d--) {
				kernDownSweep << <blocksPerGrid, threadsPerBlock >> > (len, d, dev_data);
				checkCUDAError("kernDownSweep failed!");
			}
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_data, len * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_data failed!");
			cudaFree(dev_data);
			cudaDeviceSynchronize();

            
        }

		__global__ void kernOptUpSweep(int n, int d, int *odata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int step = 1 << (d + 1);
			odata[index * step + step - 1] += odata[index * step + (1 << d) - 1];
		}

		__global__ void kernOptDownSweep(int n, int d, int *odata) {
			int index = (blockDim.x * blockIdx.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int step = 1 << (d + 1);
			int t = odata[index*step + (1 << d) - 1];
			odata[index * step + (1 << d) - 1] = odata[index * step + step - 1];
			odata[index * step + step - 1] += t;
		}

		void scan(int n, int *odata, const int *idata) {
			int upLimit = ilog2ceil(n);
			int len = 1 << upLimit;

			int blockSize = 1024;
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((len + blockSize - 1) / blockSize);

			int *dev_data;
			cudaMalloc((void**)&dev_data, len * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaMemcpy(dev_data, idata, len * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_data failed!");



			timer().startGpuTimer();
			for (int d = 0; d <= upLimit - 1; d++) {
				int step = 1 << (d + 1);
				int tempLen = len / step;
				blocksPerGrid = dim3((tempLen + blockSize) / blockSize);
				kernOptUpSweep << <blocksPerGrid, threadsPerBlock >> > (tempLen, d, dev_data);
				checkCUDAError("kernUpSweep failed!");
			}

			cudaMemset(&dev_data[len - 1], 0, sizeof(int));
			checkCUDAError("cudaMemcpy set last one to be zero failed!");

			for (int d = upLimit - 1; d >= 0; d--) {
				int step = 1 << (d + 1);
				int tempLen = len / step;
				blocksPerGrid = dim3((tempLen + blockSize) / blockSize);
				kernOptDownSweep << <blocksPerGrid, threadsPerBlock >> > (tempLen, d, dev_data);
				checkCUDAError("kernDownSweep failed!");
			}
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_data, len * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy dev_data failed!");
			cudaFree(dev_data);
			cudaDeviceSynchronize();
		}

		void gpuScan(int n, int *data) {
			int upLimit = ilog2ceil(n);
			int blockSize = 1024;
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

			for (int d = 0; d <= upLimit - 1; d++) {
				kernUpSweep << <blocksPerGrid, threadsPerBlock >> > (n, d, data);
				checkCUDAError("kernUpSweep failed!");
			}

			cudaMemset(&data[n - 1], 0, sizeof(int));
			checkCUDAError("cudaMemcpy set last one to be zero failed!");

			for (int d = upLimit - 1; d >= 0; d--) {
				kernDownSweep << <blocksPerGrid, threadsPerBlock >> > (n, d, data);
				checkCUDAError("kernDownSweep failed!");
			}
		}

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
			// Todo
			int upLimit = ilog2ceil(n);
			int len = 1 << upLimit;
			int blockSize = 1024;
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((len + blockSize - 1) / blockSize);
			int *dev_odata;
			int *dev_idata;
			int *dev_bools;
			int *dev_indices;

			cudaMalloc((void**)&dev_odata, sizeof(int) * len);
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_idata, sizeof(int) * len);
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_bools, sizeof(int) * len);
			checkCUDAError("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_indices, sizeof(int) * len);
			checkCUDAError("cudaMalloc dev_indices failed!");
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed!");
			cudaMemset(dev_odata, 0, sizeof(int) * len);
			checkCUDAError("cudaMemset dev_odata failed!");
			cudaMemset(dev_bools, 0, sizeof(int) * len);
			checkCUDAError("cudaMemset dev_bools failed!");

            timer().startGpuTimer();

			StreamCompaction::Common::kernMapToBoolean << <blocksPerGrid, threadsPerBlock >> > (len, dev_bools, dev_idata);
			checkCUDAError("kernMaptoBoolean failed!");

			cudaMemcpy(dev_indices, dev_bools, sizeof(int) * len, cudaMemcpyDeviceToDevice);
			cudaMemset(&dev_odata[n], 0, sizeof(int) * (len - n));
			gpuScan(len, dev_indices);
			
			StreamCompaction::Common::kernScatter << <blocksPerGrid, threadsPerBlock >> > (len, dev_odata, dev_idata, dev_bools, dev_indices);
			checkCUDAError("kernScatter failed!");

			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, sizeof(int) * len, cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy odata failed!");

			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_indices);
			cudaFree(dev_bools);
			int count = 0;
			while (odata[count] != 0) { 
				count++; 
			}
            return count;
        }
    }
}
