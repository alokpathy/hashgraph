CCXX = g++
NVCC = nvcc

GPU_TARGETS= edmonds-karp-gpu-naive  push-relabel-gpu-naive mpm-gpu-naive
GPU_TARGETS_GUNROCK= edmonds-karp-gpu-gunrock push-relabel-gpu-gunrock
CPU_TARGETS= edmonds-karp-cpu push-relabel-cpu boost-push-relabel
TARGETS= $(GPU_TARGETS) $(GPU_TARGETS_GUNROCK) $(CPU_TARGETS)

CXXFLAGS = -O3 -fno-optimize-sibling-calls --std=c++11
NVCC_FLAGS = -DUSE_GPU -O3 --std=c++11 --expt-extended-lambda -lnvToolsExt

# add support for more cuda architectures below
#NVCC_ARCH += -gencode arch=compute_35,code=sm_35
#NVCC_ARCH += -gencode arch=compute_52,code=sm_52
NVCC_ARCH += -gencode arch=compute_60,code=sm_60

# update gunrock location if necessary
GUNROCK_DIR = ./gunrock
GUNROCK_OPTS = $(GUNROCK_DIR)/gunrock/util/test_utils.cu $(GUNROCK_DIR)/gunrock/util/error_utils.cu $(GUNROCK_DIR)/externals/moderngpu/src/mgpucontext.cu $(GUNROCK_DIR)/externals/moderngpu/src/mgpuutil.cpp --std=c++11 -I$(GUNROCK_DIR)/externals/moderngpu/include -I$(GUNROCK_DIR)/externals/cub -Xcompiler -DMETIS_FOUND -isystem $(GUNROCK_DIR)/gunrock -isystem $(GUNROCK_DIR)
GUNROCK_LIBS = -L$(GUNROCK_DIR)/build/lib -lgunrock -Xlinker -lboost_system -Xlinker -lboost_chrono -Xlinker -lboost_timer -Xlinker -lboost_filesystem -Xlinker -lgomp -Xlinker -lmetis

CLEAN=rm -Rf *.o core bfs/*.o edmonds-karp/*.o push-relabel/*.o MPM/*.o boost_push_relabel/*.o galois-preflowpush

ifeq ($(LOG_LEVEL),)
  LOG_LEVEL = 0
endif

.PHONY: all_cpu all_gpu clean

all_gpu: clean $(GPU_TARGETS) $(GPU_TARGETS_GUNROCK) 
all_cpu: clean $(CPU_TARGETS) 

$(GPU_TARGETS) $(GPU_TARGETS_GUNROCK): C = $(NVCC)
$(GPU_TARGETS) $(GPU_TARGETS_GUNROCK): CFLAGS = $(NVCC_FLAGS) $(NVCC_ARCH)

$(CPU_TARGETS): C = $(CCXX)
$(CPU_TARGETS): CFLAGS = $(CXXFLAGS)


boost-push-relabel: boost_push_relabel/push-relabel.o
edmonds-karp-cpu:  bfs/bfs_cpu.o edmonds-karp/edmonds-karp.o maxflow.o
edmonds-karp-gpu-naive:   bfs/bfs_gpu_naive.o edmonds-karp/edmonds-karp.o maxflow.o
edmonds-karp-gpu-gunrock: edmonds-karp/edmonds-karp.o maxflow.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) $(GUNROCK_OPTS) $(GUNROCK_LIBS) -DLOG_LEVEL=$(LOG_LEVEL) -o $@ $^ bfs/bfs_gpu_gunrock.cu


mpm-gpu-naive:  bfs/bfs_gpu_naive.o MPM/MPM_implem.o maxflow.o MPM/MPM.o

push-relabel-cpu: push-relabel/push-relabel_operations_cpu_omp.o graph_tools_cpu.o bfs/bfs_cpu.o push-relabel/push-relabel.o maxflow.o
push-relabel-gpu-naive: push-relabel/push-relabel_operations_gpu.o graph_tools_gpu.o bfs/bfs_gpu_naive.o push-relabel/push-relabel.o maxflow.o
push-relabel-gpu-gunrock: graph_tools_gpu.o maxflow.o push-relabel/push-relabel_operations_gpu.o graph_tools_gpu.o push-relabel/push-relabel.o
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) $(GUNROCK_OPTS) $(GUNROCK_LIBS) -DLOG_LEVEL=$(LOG_LEVEL) -o $@ $^ bfs/bfs_gpu_gunrock.cu
mtx2dimacs:
	g++ --std=c++11 IO/mtx2dimacs.cpp -o mtx2dimacs

$(CPU_TARGETS):
	$(C) $(CFLAGS) -DLOG_LEVEL=$(LOG_LEVEL) -o $@ $^
$(GPU_TARGETS):
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -DLOG_LEVEL=$(LOG_LEVEL) -o $@ $^

galois-preflowpush:
	cd galois/build && cmake .. && make preflowpush && cp ../../galois-preflowpush.sh ../../galois-preflowpush


#Implicit rules
%.o: %.cpp
	$(C) $(CFLAGS) -DLOG_LEVEL=$(LOG_LEVEL) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -DLOG_LEVEL=$(LOG_LEVEL) -o $@ -c $<

clean: 
	$(CLEAN)

mrproper: clean 
	rm -f $(TARGETS)

