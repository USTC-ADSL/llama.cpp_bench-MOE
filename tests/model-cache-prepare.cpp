// Explicitly authorized, file-scoped eviction only. Never writes model bytes.
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void die(const char * s) { perror(s); std::exit(2); }
static size_t resident(int fd, size_t bytes, size_t page) {
    void * p=mmap(nullptr,bytes,PROT_READ,MAP_SHARED,fd,0);
    if(p==MAP_FAILED) die("mmap");
    std::vector<unsigned char> v((bytes+page-1)/page);
    if(mincore(p,bytes,v.data())) die("mincore");
    size_t n=0;for(auto b:v)n+=bool(b&1);
    munmap(p,bytes);return n*page;
}
int main(int argc,char ** argv) {
    if(argc!=3 || std::strcmp(argv[1],"--evict")) return 2;
    const std::string path=argv[2];
    if(path!="/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf") return 2;
    int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);
    struct stat st{};if(fd<0 || fstat(fd,&st))die("model stat");
    if(!S_ISREG(st.st_mode) || st.st_size!=8149865792LL) return 2;
    DIR * proc=opendir("/proc");if(!proc)die("proc");
    unsigned holders=0, unreadable=0;
    while(auto ent=readdir(proc)) {
        char * end=nullptr;long pid=std::strtol(ent->d_name,&end,10);
        if(!pid || *end || pid==getpid())continue;
        std::string root="/proc/"+std::to_string(pid);
        bool held=false;
        errno=0;std::ifstream maps(root+"/maps");
        if(!maps && errno!=ENOENT && errno!=ESRCH)++unreadable;
        std::string line;while(std::getline(maps,line)) if(line.find(path)!=std::string::npos)held=true;
        errno=0;DIR * fds=opendir((root+"/fd").c_str());
        if(!fds) { if(errno!=ENOENT && errno!=ESRCH)++unreadable; }
        else {
            while(auto f=readdir(fds)) {
                if(f->d_name[0]=='.')continue;
                struct stat fs{};
                if(!stat((root+"/fd/"+f->d_name).c_str(),&fs) && fs.st_ino==st.st_ino && fs.st_dev==st.st_dev)held=true;
            }
            closedir(fds);
        }
        if(held) { ++holders;std::fprintf(stderr,"model holder pid=%ld\n",pid); }
    }
    closedir(proc);
    if(holders || unreadable) {
        printf("{\"evicted\":false,\"holders\":%u,\"unreadable_proc_entries\":%u}\n",holders,unreadable);
        return 3;
    }
    size_t page=sysconf(_SC_PAGESIZE), before=resident(fd,st.st_size,page);
    int err=posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);
    size_t after=resident(fd,st.st_size,page);
    struct stat end_stat{};if(fstat(fd,&end_stat))die("final stat");
    bool unchanged=end_stat.st_ino==st.st_ino && end_stat.st_size==st.st_size &&
        end_stat.st_mtim.tv_sec==st.st_mtim.tv_sec && end_stat.st_mtim.tv_nsec==st.st_mtim.tv_nsec;
    printf("{\"evicted\":%s,\"fadvise_error\":%d,\"resident_before_bytes\":%zu,\"resident_after_bytes\":%zu,\"file_bytes\":%lld,\"inode\":%llu,\"metadata_unchanged\":%s,\"holders\":0,\"unreadable_proc_entries\":0}\n",
        err?"false":"true",err,before,after,(long long)st.st_size,(unsigned long long)st.st_ino,unchanged?"true":"false");
    close(fd);
    return err || !unchanged || after>16ULL*1024*1024 ? 1:0;
}
