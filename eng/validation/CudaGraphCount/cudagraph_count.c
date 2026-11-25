// LD_PRELOAD shim: counts the CUDA runtime's graph captures and launches, printed at exit (diagnostic only).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
static unsigned long begins, instantiates, updates, launches;
static void* real(const char* name)
{
    static void* cudart;
    if (!cudart) cudart = dlopen("libcudart.so.13", RTLD_NOLOAD | RTLD_LAZY);
    if (!cudart) cudart = dlopen("libcudart.so.13", RTLD_LAZY);
    void* f = cudart ? dlsym(cudart, name) : NULL;
    if (!f) { fprintf(stderr, "[cudagraph] %s not found\n", name); }
    return f;
}
typedef int (*f2)(void*, void*);
typedef int (*f3)(void*, void*, void*);
typedef int (*fbc)(void*, int);
int cudaStreamBeginCapture(void* s, int mode) { static fbc r; if (!r) r = (fbc)real("cudaStreamBeginCapture"); begins++; return r(s, mode); }
int cudaGraphInstantiate(void* e, void* g, unsigned long long f) { static f3 r; if (!r) r = (f3)real("cudaGraphInstantiate"); instantiates++; return r(e, g, (void*)f); }
int cudaGraphExecUpdate(void* e, void* g, void* info) { static f3 r; if (!r) r = (f3)real("cudaGraphExecUpdate"); updates++; return r(e, g, info); }
int cudaGraphLaunch(void* e, void* s) { static f2 r; if (!r) r = (f2)real("cudaGraphLaunch"); launches++; return r(e, s); }
__attribute__((destructor)) static void report(void) { fprintf(stderr, "[cudagraph] captures %lu instantiates %lu updates %lu launches %lu\n", begins, instantiates, updates, launches); }
