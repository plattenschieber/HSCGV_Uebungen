#ifndef LBMCU_H
#define LBMCU_H

// we don't need extern "C" here
int launchKernel ();
void cudaDevicesTest();
void cudaSquareTest();
void thrustVectorTest();
void thrustSortTest();

#endif // LBMCU_H
