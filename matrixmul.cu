#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cutensor.h>
#include<cuda.h>

#include <unordered_map>
#include <vector>

#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                 \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}

void print(std::vector <int> const &a) {
   //std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++)
   std::cout << a.at(i) << ' ';

   std::cout << "\n";
}

void print_64(std::vector <int64_t> const &a) {
   //std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++)
   std::cout << a.at(i) << ' ';

   std::cout << "\n";
}

int main(int argc, char** argv)
{
  // Host element type definition
  typedef float floatTypeA;
  typedef float floatTypeB;
  typedef float floatTypeC;
  typedef float floatTypeCompute;

  // CUDA types
  cudaDataType_t typeA = CUDA_R_64F;
  cudaDataType_t typeB = CUDA_R_64F;
  cudaDataType_t typeC = CUDA_R_64F;
  cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_64F;

  floatTypeCompute alpha = (floatTypeCompute)1.0f;
  floatTypeCompute beta  = (floatTypeCompute)1.0f;

  printf("Include headers and define data types\n");

  //=====================

  // Create vector of modes--ie, indices along each axis of tensor
  std::vector<int> modeC{'i','j'};
  std::cout << "modeC: "; 
  print(modeC);
  std::vector<int> modeA{'i','k'};
  std::cout << "modeA: "; 
  print(modeA);
  std::vector<int> modeB{'k','j'};
  std::cout << "modeB: ";
  print(modeB);
  int nmodeA = modeA.size();
  std::cout << "int nmodeA is: " << nmodeA << "\n";
  int nmodeB = modeB.size();
  std::cout << "int nmodeB is: " << nmodeB << "\n";
  int nmodeC = modeC.size();
  std::cout << "int nmodeC is: " << nmodeC << "\n";

  // Extents--size of each axis, ie the index runs from 0 to extent-1
  std::unordered_map<int, int64_t> extent;
  extent['i'] = 2;
  std::cout << "extent['i'] is: " << extent['i'] << "\n";
  extent['j'] = 2;
  std::cout << "extent['j'] is: " << extent['j'] << "\n";
  extent['k'] = 2;
  std::cout << "extent['k'] is: " << extent['k'] << "\n";
  /*extent['d'] = 2;
  std::cout << "extent['d'] is: " << extent['d'] << "\n";
  extent['e'] = 2;
  std::cout << "extent['e'] is: " << extent['e'] << "\n";
  extent['f'] = 2;
  std::cout << "extent['f'] is: " << extent['f'] << "\n";
  extent['g'] = 2;
  std::cout << "extent['g'] is: " << extent['g'] << "\n";
  extent['h'] = 2;
  std::cout << "extent['h'] is: " << extent['h'] << "\n";*/

  //print_map(extent);

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentC;
  for(auto mode : modeC)
      extentC.push_back(extent[mode]);
  std::vector<int64_t> extentA;
  for(auto mode : modeA)
      extentA.push_back(extent[mode]);
  std::vector<int64_t> extentB;
  for(auto mode : modeB)
      extentB.push_back(extent[mode]);

  std::cout << "extentC: "; 
  print_64(extentC);
  std::cout << "extentB: "; 
  print_64(extentB);
  std::cout << "extentA: "; 
  print_64(extentA);

  printf("Define modes and extents\n");

  // ============================

  // Number of elements of each tensor
  //size_t is unsigned integer type in C/C++
  //multiply the size of each dimension in the tensor
  size_t elementsA = 1;
  for(auto mode : modeA)
      elementsA *= extent[mode];
  size_t elementsB = 1;
  for(auto mode : modeB)
      elementsB *= extent[mode];
  size_t elementsC = 1;
  for(auto mode : modeC)
      elementsC *= extent[mode];

  std::cout << "int elementsA is: " << elementsA << "\n";
  std::cout << "int elementsB is: " << elementsB << "\n";
  std::cout << "int elementsC is: " << elementsC << "\n";

  // Size in bytes
  size_t sizeA = sizeof(floatTypeA) * elementsA;
  size_t sizeB = sizeof(floatTypeB) * elementsB;
  size_t sizeC = sizeof(floatTypeC) * elementsC;

  // Allocate on device
  void *A_d, *B_d, *C_d;
  cudaMalloc((void**)&A_d, sizeA);
  cudaMalloc((void**)&B_d, sizeB);
  cudaMalloc((void**)&C_d, sizeC);

  // Allocate on host
  //A is the old circuit
  //B is the gate
  //C is the circuit after applying the gate
  floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
  floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
  floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

  // Initialize data on host
  for(int64_t i = 0; i <= elementsA; i++)
      A[i] = floatTypeA(1); //A runs from 1 to 64
  for(int64_t i = 1; i <= elementsB; i++)
      B[i] = floatTypeB(1);  //B is all -1's, just to see which elements actually got manipulated
  for(int64_t i = 0; i < elementsC; i++)
      C[i] = floatTypeC(0);  //we initially put these elements to 0, the contraction will actually fill them up

      std::cout << "=============================printing initialized data \n";

  for(int64_t i = 0; i < 4; i++)
  {
    for(int64_t j = 0; j <4; j++)
    {
        std::cout << A[i*4 + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  for(int64_t i = 0; i < 4; i++)
  {
    for(int64_t j = 0; j <4; j++)
    {
        std::cout << B[i*4 + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n===============================done\n\n";

  // Copy to device
  cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);
  cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice);

  /*floatTypeA *A_check = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
  floatTypeB *B_check = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
  floatTypeC *C_check = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

  cudaMemcpy(C_check, C_d, sizeC, cudaMemcpyDeviceToHost);
  cudaMemcpy(A_check, A_d, sizeA, cudaMemcpyDeviceToHost);
  cudaMemcpy(B_check, B_d, sizeB, cudaMemcpyDeviceToHost);

  for(int64_t i = 0; i < 8; i++)
  {
    for(int64_t j = 0; j <8; j++)
    {
        std::cout << A_check[i*8 + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  for(int64_t i = 0; i < 4; i++)
  {
    for(int64_t j = 0; j <4; j++)
    {
        std::cout << B_check[i*4 + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n===============================done\n\n"; */

  printf("Allocate, initialize and transfer tensors\n");

  // ==============================

  // Initialize cuTENSOR library
  cutensorHandle_t handle;
  cutensorInit(&handle);

  // Create Tensor Descriptors
  cutensorTensorDescriptor_t descA;
  HANDLE_ERROR( cutensorInitTensorDescriptor( &handle,
              &descA,
              nmodeA,
              extentA.data(), //The std::vector::data() is an STL in C++ which returns a direct pointer to the memory array used internally by the vector to store its owned elements.
              NULL,// stride -- means that the tensor is packed, not sparse
              typeA, CUTENSOR_OP_IDENTITY ) );

  cutensorTensorDescriptor_t descB;
  HANDLE_ERROR( cutensorInitTensorDescriptor( &handle,
              &descB,
              nmodeB,
              extentB.data(),
              NULL,//stride
              typeB, CUTENSOR_OP_IDENTITY ) );

  cutensorTensorDescriptor_t descC;
  HANDLE_ERROR( cutensorInitTensorDescriptor( &handle,
              &descC,
              nmodeC,
              extentC.data(),
              NULL,//stride
              typeC, CUTENSOR_OP_IDENTITY ) );

  printf("Initialize cuTENSOR and tensor descriptors\n");

  // ==========================

   //Retrieve the memory alignment for each tensor
   uint32_t alignmentRequirementA;
   HANDLE_ERROR( cutensorGetAlignmentRequirement( &handle,
              A_d,
              &descA,
              &alignmentRequirementA) );

   uint32_t alignmentRequirementB;
   HANDLE_ERROR( cutensorGetAlignmentRequirement( &handle,
              B_d,
              &descB,
              &alignmentRequirementB) );

   uint32_t alignmentRequirementC;
   HANDLE_ERROR( cutensorGetAlignmentRequirement( &handle,
              C_d,
              &descC,
              &alignmentRequirementC) );

  printf("Query best alignment requirement for our pointers\n");

  // ====================================

  // Create the Contraction Descriptor
  cutensorContractionDescriptor_t desc;
  HANDLE_ERROR( cutensorInitContractionDescriptor( &handle,
              &desc,
              &descA, modeA.data(), alignmentRequirementA,
              &descB, modeB.data(), alignmentRequirementB,
              &descC, modeC.data(), alignmentRequirementC,
              &descC, modeC.data(), alignmentRequirementC,
              typeCompute) );

  printf("Initialize contraction descriptor\n");

  // ==================================
  // Set the algorithm to use
  cutensorContractionFind_t find;
  HANDLE_ERROR( cutensorInitContractionFind(
              &handle, &find,
              CUTENSOR_ALGO_DEFAULT) );

  printf("Initialize settings to find algorithm\n");

  // =================================

  // Query workspace
  size_t worksize = 0;
  HANDLE_ERROR( cutensorContractionGetWorkspace(&handle,
              &desc,
              &find,
              CUTENSOR_WORKSPACE_RECOMMENDED, &worksize ) );

  // Allocate workspace
  void *work = nullptr;
  if(worksize > 0)
  {
      if( cudaSuccess != cudaMalloc(&work, worksize) ) // This is optional!
      {
          work = nullptr;
          worksize = 0;
      }
  }

  printf("Query recommended workspace size and allocate it\n");

  // ===============================

  // Create Contraction Plan
  cutensorContractionPlan_t plan;
  HANDLE_ERROR( cutensorInitContractionPlan(&handle,
                                            &plan,
                                            &desc,
                                            &find,
                                            worksize) );

  printf("Create plan for contraction\n");

  // ================================

  cutensorStatus_t err;

  // Execute the tensor contraction
  err = cutensorContraction(&handle,
                            &plan,
                     (void*)&alpha, A_d,
                                    B_d,
                     (void*)&beta,  C_d,
                                    C_d,
                            work, worksize, 0); // stream );
  cudaDeviceSynchronize();

  // Check for errors
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
      printf("ERROR: %s\n", cutensorGetErrorString(err));
  }

  printf("Execute contraction from plan\n");

  // ============================

  floatTypeA *A_res = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
  floatTypeB *B_res = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
  floatTypeC *C_res = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

  cudaMemcpy(C_d, C_res, sizeC, cudaMemcpyDeviceToHost);
  cudaMemcpy(A_d, A_res, sizeA, cudaMemcpyDeviceToHost);
  cudaMemcpy(B_d, B_res, sizeB, cudaMemcpyDeviceToHost);

  /*for(int64_t i = 0; i < elementsC; i++)
    std::cout << C_res[i] << " ";

  std::cout << "\n";*/

  //for(int64_t i = 0; i < 4; i++)
  //{
    for(int64_t j = 0; j <4; j++)
    {
        std::cout << C_res[j] << " ";
    }
    std::cout << "\n";
  //}
  std::cout << "\n";


  if ( A ) free( A );
  if ( B ) free( B );
  if ( C ) free( C );
  if ( A_d ) cudaFree( A_d );
  if ( B_d ) cudaFree( B_d );
  if ( C_d ) cudaFree( C_d );
  if ( work ) cudaFree( work );

  printf("Successful completion\n");

  return 0; 
} 
