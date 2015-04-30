#ifndef CUDATEST_H
#define CUDATEST_H

// we don't need extern "C" here
int launchKernel ();
void cudaDevicesTest();
void cudaSquareTest();
void thrustVectorTest();
void thrustSortTest();

#endif // CUDATEST_H
