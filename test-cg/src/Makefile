NVCC = /usr/local/cuda-12.1/bin/nvcc
CC_FLAGS = -Xcompiler -fPIC -shared --std=c++14
OBJ_DIR := objs
LL = -lcusparse
GPU_OPTS = -O3 --std=c++14 -gencode arch=compute_86,code=sm_86 --use_fast_math
INC = -I ../include -I /usr/local/cuda-12.1/lib64

all : $(TARGET)

$(TARGET): test.o Makefile
	$(NVCC) $(INC) $(LL) -o $@ $(OBJ_DIR)/$<

%.o: %.cu Makefile
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(CC_FLAGS) $(INC) -c $< -o $(OBJ_DIR)/$@

clean:
	rm -f *.o 
	rm -rf $(OBJ_DIR)
	rm ${TARGET}