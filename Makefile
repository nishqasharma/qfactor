ROOT = /global/homes/n/nishqa/.local/cori/3.9-anaconda-2021.11/lib/python3.9/site-packages/cutensor

default: contraction

contraction: tensor_contraction.cu
	nvcc tensor_contraction.cu -L$(ROOT)/lib/10.1/ -I$(ROOT)/include -std=c++11 -lcutensor -o contraction