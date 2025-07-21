#pragma once

#include <cassert>
#include <iostream>
#include <queue>
#include <set>
#include <vector>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/count.h>
#include <thrust/remove.h>

#define NUM_BLOCKS(n, block_size) (((n) + (block_size) - 1) / (block_size))
#define THREAD_PER_BLOCK 128

// does not use uint64_t in cstdint since it's not supported by atomicCAS
using uint64 = unsigned long long int;
using uint32 = unsigned int;

// for static_assert false
template <class... T>
constexpr bool always_false = false;

inline int AigNodeID(int lit) {return lit >> 1;}
inline int AigNodeIsComplement(int lit) {return lit & 1;}
inline unsigned invertConstTrueFalse(unsigned lit) {
    // swap 0 and 1
    return lit < 2 ? 1 - lit : lit;
}


namespace dUtils {

__host__ __device__ __forceinline__ int AigNodeID(int lit) {return lit >> 1;}
__host__ __device__ __forceinline__ int AigNodeIsComplement(int lit) {return lit & 1;}
__host__ __device__ __forceinline__ int AigIsNode(int nodeId, int nPIs) {return nodeId > nPIs;} // considering const 1
__host__ __device__ __forceinline__ int AigIsPIConst(int nodeId, int nPIs) {return nodeId <= nPIs;}
__host__ __device__ __forceinline__ int AigNodeLitCond(int nodeId, int complement) {
    return (int)(((unsigned)nodeId << 1) | (unsigned)(complement != 0));
}
__host__ __device__ __forceinline__ int AigNodeNotCond(int lit, int complement) {
    return (int)((unsigned)lit ^ (unsigned)(complement != 0));
}
__host__ __device__ __forceinline__ int AigNodeNot(int lit) {return lit ^ 1;}
__host__ __device__ __forceinline__ int AigNodeIDDebug(int lit, int nPIs, int nPOs) {
    int id = dUtils::AigNodeID(lit);
    return dUtils::AigIsNode(id, nPIs) ? (id + nPOs) : id;
}
__host__ __device__ __forceinline__ int TruthWordNum(int nVars) {return nVars <= 5 ? 1 : (1 << (nVars - 5));}
__host__ __device__ __forceinline__ int Truth6WordNum(int nVars) {return nVars <= 6 ? 1 : (1 << (nVars - 6));}

__host__ __device__ __forceinline__
bool hasVisited(const int* addr, int target, int size){
    for(int i=0; i<size; i++)
        if(addr[i]==target) return true;
    return false;
}

__host__ __device__ __forceinline__
int invertConstTrueFalse(int lit) {
    // swap 0 and 1
    return lit < 2 ? 1 - lit : lit;
}
// unary thrust functor
template <typename ValueT, typename MaskT>
struct getValueFilteredByMask {
    const MaskT maskTrue;

    getValueFilteredByMask() : maskTrue(1) {}
    getValueFilteredByMask(MaskT _maskTrue) : maskTrue(_maskTrue) {}
    
    __host__ __device__
    ValueT operator()(const thrust::tuple<ValueT, MaskT> &elem) const { 
        return thrust::get<1>(elem) == maskTrue ? thrust::get<0>(elem) : 0;
    }
};

template <typename IdT, typename LevelT>
struct decreaseLevelIds {
    __host__ __device__
    bool operator()(const thrust::tuple<IdT, LevelT> &e1, const thrust::tuple<IdT, LevelT> &e2) const {
        return thrust::get<1>(e1) == thrust::get<1>(e2) ? 
            (thrust::get<0>(e1) > thrust::get<0>(e2)) : (thrust::get<1>(e1) > thrust::get<1>(e2));
        // return thrust::get<1>(e1) > thrust::get<1>(e2);
    }
};

template <typename IdT, typename LevelT>
struct decreaseLevels {
    __host__ __device__
    bool operator()(const thrust::tuple<IdT, LevelT> &e1, const thrust::tuple<IdT, LevelT> &e2) const {
        return thrust::get<1>(e1) > thrust::get<1>(e2);
    }
};

template <typename IdT, typename LevelT>
struct decreaseLevelsPerm {
    __host__ __device__
    bool operator()(const thrust::tuple<IdT, LevelT, int> &e1, const thrust::tuple<IdT, LevelT, int> &e2) const {
        return thrust::get<1>(e1) == thrust::get<1>(e2) ? 
            (thrust::get<2>(e1) > thrust::get<2>(e2)) : (thrust::get<1>(e1) > thrust::get<1>(e2));
    }
};

template <typename T>
struct isMinusOne { 
    __host__ __device__
    bool operator()(const T &elem) {
        return elem == -1;
    }
};

template <typename T>
struct isOne { 
    __host__ __device__
    bool operator()(const T &elem) {
        return elem == 1;
    }
};

template <typename T>
struct isNotOne { 
    __host__ __device__
    bool operator()(const T &elem) {
        return elem != 1;
    }
};

template <typename T, T val>
struct equalsVal {
    __host__ __device__
    bool operator()(const T &elem) {
        return elem == val;
    }
};

template <typename T>
struct equalsValNC {
    const T val;
    equalsValNC() = delete;
    equalsValNC(const T val): val(val) {}
    __host__ __device__
    bool operator()(const T &elem) {
        return elem == val;
    }
};

template <typename T>
struct maskEqualsVal {
    const T val;
    const T* mask;
    maskEqualsVal() = delete;
    maskEqualsVal(const T val, const T* mask):
        val(val), mask(mask) {}
    __host__ __device__
    bool operator()(const T &elem) {
        return mask[elem] == val;
    }
};

template <typename T, T val>
struct notEqualsVal {
    __host__ __device__
    bool operator()(const T &elem) {
        return elem != val;
    }
};

// for non-const value
template <typename T>
struct notEqualsValNC {
    const T val;
    notEqualsValNC() = delete;
    notEqualsValNC(const T val):
        val(val) {}
    __host__ __device__
    bool operator()(const T &elem) {
        return elem != val;
    }
};

template <typename T>
struct maskNotEqualsVal {
    const T val;
    const T* mask;
    maskNotEqualsVal() = delete;
    maskNotEqualsVal(const T val, const T* mask):
        val(val), mask(mask) {}
    __host__ __device__
    bool operator()(const T &elem) {
        return mask[elem] != val;
    }
};

template <typename T, T val>
struct dividesVal {
    __host__ __device__
    T operator()(const T &elem) {
        return elem / val;
    }
};

template <typename T, T val>
struct greaterThanVal {
    __host__ __device__
    bool operator()(const T &elem) {
        return elem > val;
    }
};

template <typename T>
struct maskGreaterThanVal {
    const T val;
    const T* mask;
    maskGreaterThanVal() = delete;
    maskGreaterThanVal(const T val, const T* mask):
        val(val), mask(mask) {}
    __host__ __device__
    bool operator()(const T &elem) {
        return mask[elem] > val;
    }
};

template <typename T, T val>
struct greaterThanEqualsValInt {
    __host__ __device__
    int operator()(const T &elem) {
        return (elem >= val ? 1 : 0);
    }
};

template <typename T, T val>
struct lessThanVal {
    __host__ __device__
    bool operator()(const T &elem) {
        return elem < val;
    }
};

struct sameNodeID {
    __host__ __device__
    bool operator()(const int &lhs, const int &rhs) const {
        return (lhs >> 1) == (rhs >> 1);
    }
};

struct equals {
    __host__ __device__
    bool operator()(const int &lhs, const int &rhs) const {
        return lhs == rhs ;
    }
};

template <typename T>
struct isNodeLit {
    const int nPIs;

    isNodeLit() = delete;
    isNodeLit(const int _nPIs) : nPIs(_nPIs) {}

    __host__ __device__
    bool operator()(const T &elem) {
        return (elem >> 1) > nPIs;
    }
};

template <typename T>
struct isPIConstLit {
    const int nPIs;

    isPIConstLit() = delete;
    isPIConstLit(const int _nPIs) : nPIs(_nPIs) {}

    __host__ __device__
    bool operator()(const T &elem) {
        return (elem >> 1) <= nPIs;
    }
};

struct getNodeID {
    __host__ __device__
    int operator()(const int elem) {
        return elem >> 1;
    }
};

const int AigConst1 = 0;
const int AigConst0 = 1;
} // namespace dUtils

// error checking helpers
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ __forceinline__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s,\nat %s, line %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s,\nat %s, line %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuChkStackOverflow(ans) { gpuAssertStack((ans), __FILE__, __LINE__); }
inline void gpuAssertStack(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s,\nat %s, line %d\n", cudaGetErrorString(code), file, line);
      fprintf(stderr, "This is most likely due to insufficient CUDA call stack size. Try to increase cudaLimitStackSize.\n");
      if (abort) exit(code);
   }
}

#define CHECK_CUDA(val) check_cuda((val), #val, __FILE__, __LINE__)
template <typename T>
inline void check_cuda(T err, const char* const func, const char* const file, const int line){
    if (err != cudaSuccess){
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void printCudaMem(){
    size_t f,t;
    cudaMemGetInfo(&f, &t);
    printf("CUDA memory    free %.2f MB, total %.2f MB\n", (double)f/1024/1024, (double)t/1024/1024);
}

// attention that rootId is not a PO
inline void printTFI(int rootId, int nPIs, int* pFanin0, int* pFanin1){
    std::queue<int> vis;
    std::set<int> visited, inputSupport;
    bool printTFI = true;
    if(printTFI)    printf("--------------------\nthe TFI of node %d:\n", rootId);
    vis.push(rootId);
    while(!vis.empty()){
        int node = vis.front();
        vis.pop();
        if(node<=nPIs){
            inputSupport.insert(node);
            continue;
        }
        if(printTFI)
            printf("%d: %s%d %s%d\n", node, (pFanin0[node]&1)?"!":" ", pFanin0[node]>>1, (pFanin1[node]&1)?"!":" ", pFanin1[node]>>1);
        int left = pFanin0[node]>>1, right = pFanin1[node]>>1;
        if(visited.find(left)==visited.end()){
            vis.push(left); visited.insert(left);
        }
        if(visited.find(right)==visited.end()){
            vis.push(right); visited.insert(right);
        }
    }
    printf("the input support of node %d:", rootId);
    for(auto i: inputSupport)
        printf(" %d", i);
    printf("\n--------------------\n");

}

inline void printCheckPoint(int phase, clock_t& preTime, std::string cap = ""){
    clock_t curTime = clock();
    if(cap.empty())
        printf("**************  phase %d:  **************\n", phase);
    else
        printf("************  phase %d: %s  ************\n", phase, cap.c_str());

    printf("  time: %.3f sec\n", (curTime - preTime) / (double) CLOCKS_PER_SEC);
    printf("  ");
    printCudaMem();
    preTime = curTime;
}
