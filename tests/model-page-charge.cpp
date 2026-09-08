// Read-only file-cache attribution probe. Touch only pages reported resident.
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static long long value(const std::string & p) { std::ifstream f(p); long long n=-1; f>>n; return n; }
static void die(const char * p) { perror(p); std::exit(2); }
int main(int argc, char ** argv) {
    if(argc!=3) return 2;
    struct stat model{}, cg{};
    int fd=open(argv[1],O_RDONLY|O_CLOEXEC);
    if(fd<0 || fstat(fd,&model) || stat(argv[2],&cg)) die("open/stat");
    const size_t page=sysconf(_SC_PAGESIZE), bytes=model.st_size, n=(bytes+page-1)/page;
    auto base=static_cast<unsigned char *>(mmap(nullptr,bytes,PROT_READ,MAP_SHARED,fd,0));
    if(base==MAP_FAILED) die("mmap");
    // A page can be reclaimed between mincore and the load. Suppress read-ahead
    // so a raced fault does not populate a large new cache region in the observer.
    if(madvise(base,bytes,MADV_RANDOM)) die("madvise random");
    int pm=open("/proc/self/pagemap",O_RDONLY|O_CLOEXEC), km=open("/proc/kpagecgroup",O_RDONLY|O_CLOEXEC);
    if(pm<0 || km<0) die("pagemap/kpagecgroup");
    std::vector<unsigned char> resident(n), after(n);
    if(mincore(base,bytes,resident.data())) die("mincore");
    const std::string cgpath=argv[2];
    const long long before=value(cgpath+"/memory.usage_in_bytes");
    struct rusage ru0{},ru1{}; getrusage(RUSAGE_SELF,&ru0);
    std::map<uint64_t,uint64_t> owners;
    size_t cached=0, present=0, pfn_zero=0, gone=0;
    volatile unsigned char checksum=0;
    for(size_t i=0;i<n;++i) {
        if(!(resident[i]&1)) continue;
        ++cached;
        checksum=checksum^base[i*page];
        uint64_t entry=0, owner=0;
        off_t pos=(reinterpret_cast<uintptr_t>(base)/page+i)*sizeof(entry);
        if(pread(pm,&entry,sizeof(entry),pos)!=sizeof(entry)) die("read pagemap");
        if(!(entry&(1ULL<<63))) { ++gone; continue; }
        ++present;
        uint64_t pfn=entry&((1ULL<<55)-1);
        if(!pfn) { ++pfn_zero; continue; }
        if(pread(km,&owner,sizeof(owner),pfn*sizeof(owner))!=sizeof(owner)) die("read kpagecgroup");
        owners[owner]+=page;
    }
    getrusage(RUSAGE_SELF,&ru1);
    if(mincore(base,bytes,after.data())) die("mincore after");
    size_t after_cached=0; for(auto v:after) after_cached+=bool(v&1);
    printf("{\"file_bytes\":%zu,\"page_bytes\":%zu,\"self_memcg_inode\":%llu,\"cached_before_bytes\":%zu,\"cached_after_bytes\":%zu,\"present_pages\":%zu,\"pfn_zero_pages\":%zu,\"raced_nonpresent_pages\":%zu,\"major_faults\":%ld,\"minor_faults\":%ld,\"cgroup_before_bytes\":%lld,\"cgroup_after_bytes\":%lld,\"checksum\":%u,\"owners\":[",
        bytes,page,(unsigned long long)cg.st_ino,cached*page,after_cached*page,present,pfn_zero,gone,
        ru1.ru_majflt-ru0.ru_majflt,ru1.ru_minflt-ru0.ru_minflt,before,value(cgpath+"/memory.usage_in_bytes"),(unsigned)checksum);
    bool first=true;
    for(auto [owner,size]:owners) {
        printf("%s{\"memcg_inode\":%llu,\"bytes\":%llu}",first?"":",",(unsigned long long)owner,(unsigned long long)size);
        first=false;
    }
    printf("]}\n");
    munmap(base,bytes); close(pm);close(km);close(fd);
    return pfn_zero || gone ? 1 : 0;
}
