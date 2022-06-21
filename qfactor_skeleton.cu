#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cutensor.h>
#include<cuda.h>

#include <unordered_map>
#include <vector>

struct cudaTensorDescriptor{
    constcutensorHandle_t*handle, 
    cutensorContractionDescriptor_t*desc, 
    constcutensorTensorDescriptor_t*descA, constint32_tmodeA[], 
    constuint32_talignmentRequirementA, 
    constcutensorTensorDescriptor_t*descB, 
    constint32_tmodeB[], 
    constuint32_talignmentRequirementB, 
    constcutensorTensorDescriptor_t*descC, constint32_tmodeC[], 
    constuint32_talignmentRequirementC, 
    constcutensorTensorDescriptor_t*descD, 
    constint32_tmodeD[], 
    constuint32_talignmentRequirementD, 
    cutensorComputeType_ttypeCompute
} cudaTensorDescriptor;

struct gate{
    float *gate_unitary; //pointer to gate unitary
    int size_unitary; //size of the unitary
    int *location; //location of the gate in the circuit
} gate;

int* readDataFromFile(file *f, gate **ckt, float **unitary, int *unitary_size)
{
    //reads data from a .csv file that includes the following data:
    //1. number of circuits
    //2. number of gates in each circuit
    //3. unitary data for gates and circuits

    //This function reads in the data into their proper places into the unitary and ckt 2-D arrays
    //and a 1-D unitary_size array which contains the size of each unitary

    x = [num_ckts, num_gates];
    
    //and returns a pointer to an array containing num_ckts and num_gates
    return *x
}

__global__ createInvTargetUnitaryTensors( float** unitary, int num_ckts)
{
    for(int i=0; i<num_ckts; i++)
    {
        //take in unitary[i] and calculate it's inverse
        //create a tensor out of it based on it's size, by initializing a cudaTensorDescriptorInit 
        //and an internal cudaTensorDescriptor object
    }
}

__global__ createGateTensors(gate **ckt, float **unitary, int num_ckts, int num_gates)
{
    for(int i=0; i<num_ckts; i++)
    {
        //take in a gate unitary, and create a tensor out of it based on it's size and location, 
        //by initializing a cudaTensorDescriptorInit and an internal cudaTensorDescriptor object
    }
}

//first maybe write a CPU version for sanity check
__global__ createCombinedTensor (int num_ckts, int num_gates)
{
    //use threadIdx and blockIdx to compute the locations of the (U*)^T and gate tensors
    for(i=o; i<num_gates; i++)
    {
        err = cutensorContraction(&handle,
                            &plan,
                     (void*)&alpha, A_d,
                                    B_d,
                     (void*)&beta,  C_d,
                                    C_d,
                            work, worksize, 0 ); // stream );
    }
}

__global__ qfactor(cudaTensorDescriptor* combined_tensor, cudaTensorDescriptor **gate_tensor, cudaTensorDescriptor *result_tensor, int num_ckts, int num_gates)
{
    //perform iteration/error/threshold checks

    for(int i=0; i<num_gates; i++)
    {
        //sweep left
            //1. apply_right is not going to be a kernel itself, manipulate the descriptor if needed and then:
            err = cutensorContraction(&handle,
                                &plan,
                        (void*)&alpha, A_d,
                                        B_d,
                        (void*)&beta,  C_d,
                                        C_d,
                                work, worksize, 0 ); // stream );

            //2. calculate environment matrix--how will you calculate the trace using tensors--trace is a rank 2 tensor contraction
            err = cutensorContraction(...)

            //3. update gate, needs SVD--which can also be calculated using tensor contraction--manipulate the tensor if needed
            err = cutensorContraction(...)

            //4. apply_left will not be a kernel itself either, manipulate the descriptor if needed then:
            err = cutensorContraction(...)

        //sweep right -- basically same as sweep_left but in the opposite direction, so just manipulate
        //               the tensors in the appropraite manner and make the same 4 calls to the cutensorContraction function
            //1. apply_right is not going to be a kernel itself, manipulate the descriptor if needed and then:
            err = cutensorContraction(&handle,
                                &plan,
                        (void*)&alpha, A_d,
                                        B_d,
                        (void*)&beta,  C_d,
                                        C_d,
                                work, worksize, 0 ); // stream );

            //2. calculate environment matrix--how will you calculate the trace using tensors--trace is a rank 2 tensor contraction
            err = cutensorContraction(...)

            //3. update gate, needs SVD--which can also be calculated using tensor contraction--manipulate the tensor if needed
            err = cutensorContraction(...)

            //4. apply_left will not be a kernel itself either, manipulate the descriptor if needed then:
            err = cutensorContraction(...)
    }

}

int main()
{
    file *f; //file containing the data of the circuits and their target unitaries
    gate **ckt; 
    int unitary_size;
    
    //1.open the file

    //2.read data from the file into the appropriate structures--a unitary array and a circuit array
    readDataFromFile(gate **ckt, float **unitary, int *unitary_size);

    //3. create an array of target unitary tensors
    __global__ createInvTargetUnitaryTensors( float** unitary, int num_ckts);

    //4. create an array of the gate tensors inside each circuit
    __global__ createGateTensors(gate **ckt, float **unitary, int num_ckts, int num_gates)

    //5. copy the data arrays (column major, 1-D flattened) onto the GPU's
    for (int i=0; i<num_ckts; i++)
    {
        cudaMalloc();
        cudaMemCopy(A,A_d,cudaMemcpyHostToDevice);
    }

    //6. initialize cutensor library
    cutensorInit(); //and other relevant functions

    //7. create the combined circuit tensor
    __global__ createCombinedTensor (int num_ckts, int num_gates)

    //8. finally, make the call to qfactor
    __global__ qfactor(cudaTensorDescriptor* combined_tensor, cudaTensorDescriptor **gate_tensor, cudaTensorDescriptor *result_tensor, int num_ckts, int num_gates)

    //9. copy the result arrays back from GPU to CPU and free relevant arrays
    cudaMalloc();
    cudaMemCopy();
    cudaFree();

    //10. post-process these result arrays

    //11. write the post processed arrays to stdout or to a result file

    //12. free up any remaining memory, and close files
}




