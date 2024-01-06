NVCC = /usr/local/cuda-12.1/bin/nvcc
CC_FLAGS = -Xcompiler -fPIC -shared --std=c++14
BOOST_INC = -I /home/zgh23/tools/boost_1_83_0
OBJ_DIR := objs
GPU_OPTS = -O3 --std=c++14 -gencode arch=compute_86,code=sm_86 --use_fast_math
INC = -I ../include -I /usr/local/cuda-12.1/lib64

all : $(TARGET)

$(TARGET): main.o Makefile
	$(NVCC) $(INC) ${BOOST_INC} -o $@ $(OBJ_DIR)/$<

main.o: main.cu Makefile
	mkdir -p $(OBJ_DIR)
	$(NVCC) $(CC_FLAGS) $(INC) ${BOOST_INC} -c $< -o $(OBJ_DIR)/$@

clean:
	rm -f *.o 
	rm -rf $(OBJ_DIR)
	rm ${TARGET}