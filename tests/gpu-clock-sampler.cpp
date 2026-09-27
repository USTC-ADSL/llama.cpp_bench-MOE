// Separate read-only privileged observer; inference must retain its own uid.
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using sample_clock = std::chrono::steady_clock;
static volatile sig_atomic_t stopped = 0;
static void stop_signal(int) { stopped = 1; }
static double now_us() {
    return std::chrono::duration<double,std::micro>(sample_clock::now().time_since_epoch()).count();
}
static double read_number(int fd, int & error) {
    char buffer[128]; error = 0;
    if(fd < 0) { error = ENOENT; return NAN; }
    ssize_t n = pread(fd,buffer,sizeof(buffer)-1,0);
    if(n <= 0) { error = n < 0 ? errno : EIO; return NAN; }
    buffer[n] = 0; char * end = nullptr; errno = 0;
    double value = strtod(buffer,&end);
    if(end == buffer || errno || !std::isfinite(value)) { error = EINVAL; return NAN; }
    return value;
}
int main(int argc, char ** argv) {
    if(argc != 6) { std::fprintf(stderr,"usage: gpu-clock-sampler TRACE PIDFILE STOPFILE PERIOD_US MAX_SECONDS\n"); return 2; }
    // argv[5] is a hard lifetime cap, including an interrupted/disconnected runner.
    const long period = std::strtol(argv[4],nullptr,10), seconds = std::strtol(argv[5],nullptr,10);
    if(period < 1000 || period > 1000000 || seconds < 1 || seconds > 300 || geteuid() != 0) { return 2; }
    int freq = open("/sys/class/kgsl/kgsl-3d0/gpuclk",O_RDONLY|O_CLOEXEC);
    int busy = open("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",O_RDONLY|O_CLOEXEC);
    int error = 0; double initial = read_number(freq,error);
    if(error || initial <= 0) { std::fprintf(stderr,"gpuclk read failed: %d\n",error); return 3; }
    int outfd = open(argv[1],O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(outfd < 0) { std::perror("trace open"); return 4; }
    FILE * out = fdopen(outfd,"w");
    if(!out) { close(outfd); return 4; }
    char write_buffer[65536]; setvbuf(out,write_buffer,_IOFBF,sizeof(write_buffer));
    signal(SIGTERM,stop_signal); signal(SIGINT,stop_signal);
    const double start = now_us();
    std::fprintf(out,"{\"type\":\"header\",\"pid\":%d,\"uid\":%d,\"euid\":%d,\"period_us\":%ld,\"start_monotonic_us\":%.3f,\"clock\":\"Android steady_clock CLOCK_MONOTONIC\"}\n",getpid(),getuid(),geteuid(),period,start);
    fflush(out);
    int pidfd = open(argv[2],O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    if(pidfd < 0) { fclose(out); return 4; }
    dprintf(pidfd,"%d\n",getpid()); close(pidfd);
    auto next = sample_clock::now();
    size_t count=0, missed=0, errors=0; double observation_sum=0, observation_max=0;
    while(!stopped && access(argv[3],F_OK) != 0 && now_us()-start < seconds*1e6) {
        double begin=now_us(); int fe=0,be=0;
        double hz=read_number(freq,fe), percent=read_number(busy,be), end=now_us();
        char hz_text[64], busy_text[64];
        if(fe) { std::strcpy(hz_text,"null"); } else { std::snprintf(hz_text,sizeof(hz_text),"%.0f",hz); }
        if(be) { std::strcpy(busy_text,"null"); } else { std::snprintf(busy_text,sizeof(busy_text),"%.3f",percent); }
        std::fprintf(out,"{\"type\":\"sample\",\"begin_us\":%.3f,\"end_us\":%.3f,\"gpu_hz\":%s,\"busy_percent\":%s,\"frequency_errno\":%d,\"busy_errno\":%d}\n",begin,end,hz_text,busy_text,fe,be);
        ++count; errors += bool(fe || be); observation_sum += end-begin; observation_max=std::fmax(observation_max,end-begin);
        next += std::chrono::microseconds(period);
        auto current=sample_clock::now();
        while(next < current) { next += std::chrono::microseconds(period); ++missed; }
        std::this_thread::sleep_until(next);
    }
    rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    double cpu_us=usage.ru_utime.tv_sec*1e6+usage.ru_utime.tv_usec+usage.ru_stime.tv_sec*1e6+usage.ru_stime.tv_usec;
    bool timed_out=now_us()-start >= seconds*1e6;
    std::fprintf(out,"{\"type\":\"footer\",\"end_monotonic_us\":%.3f,\"samples\":%zu,\"missed_periods\":%zu,\"read_errors\":%zu,\"read_total_us\":%.3f,\"read_max_us\":%.3f,\"cpu_us\":%.3f,\"timed_out\":%s}\n",now_us(),count,missed,errors,observation_sum,observation_max,cpu_us,timed_out?"true":"false");
    bool failed = ferror(out); failed |= fclose(out) != 0;
    if(freq >= 0) { close(freq); } if(busy >= 0) { close(busy); }
    return failed || timed_out ? 5 : 0;
}
