#include "e0-measure.h"
#include <sys/prctl.h>
#include <sys/mman.h>
#include <unistd.h>
int main() {
    prctl(PR_SET_NAME, "E0 ) comm (x)", 0, 0, 0);
    e0_measure::event("counter_before", -1, true);
    const size_t size=16*1024*1024;
    volatile char * p=(volatile char *)mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (p==MAP_FAILED) return 2;
    for (size_t i=0;i<size;i+=sysconf(_SC_PAGESIZE)) p[i]=1;
    e0_measure::event("counter_after", -1, true);
    munmap((void *)p,size);
}
