// Tiny loopback fixture server; not an App substitute or memory-pressure load.
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iterator>
#include <thread>

static long long micros(clockid_t clock) {
    timespec t{}; clock_gettime(clock,&t);
    return (long long)t.tv_sec*1000000+t.tv_nsec/1000;
}
int main(int argc,char **argv) {
    if(argc==2 && !strcmp(argv[1],"--clock")) {printf("MONOTONIC_US=%lld\n",micros(CLOCK_MONOTONIC));return 0;}
    if(argc!=3)return 2;
    std::ifstream file(argv[1]);
    std::string page((std::istreambuf_iterator<char>(file)),{});
    if(page.empty())return 3;
    int server=socket(AF_INET,SOCK_STREAM,0), reuse=1;
    setsockopt(server,SOL_SOCKET,SO_REUSEADDR,&reuse,sizeof(reuse));
    sockaddr_in addr{}; addr.sin_family=AF_INET;addr.sin_port=htons(18765);addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if(bind(server,(sockaddr*)&addr,sizeof(addr))||listen(server,8))return 4;
    for(;;) {
        int fd=accept(server,nullptr,nullptr);if(fd<0)continue;
        // Chromium may leave a speculative connection idle. It must not delay
        // the page's state/paint evidence on a different connection.
        std::thread([fd,&page,argv]() {
        timeval timeout{3,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        std::string request;char buf[4096];size_t end;
        while((end=request.find("\r\n\r\n"))==std::string::npos && request.size()<16384) {
            int n=recv(fd,buf,sizeof(buf),0);if(n<=0)break;request.append(buf,n);
        }
        std::string body,status="200 OK",type="text/html; charset=utf-8";
        if(request.rfind("GET /task",0)==0)body=page;
        else if(request.rfind("POST /event ",0)==0 && end!=std::string::npos) {
            size_t len=0;auto at=request.find("Content-Length:");
            if(at!=std::string::npos)len=strtoul(request.c_str()+at+15,nullptr,10);
            while(len<=4096 && request.size()<end+4+len) {int n=recv(fd,buf,sizeof(buf),0);if(n<=0)break;request.append(buf,n);}
            std::string event=request.substr(end+4,len);
            if(len<=4096 && event.size()==len && event.size()>1 && event.back()=='}') {
                event.pop_back();event+=",\"device_monotonic_us\":"+std::to_string(micros(CLOCK_MONOTONIC))+",\"device_realtime_us\":"+std::to_string(micros(CLOCK_REALTIME))+"}\n";
                FILE *out=fopen(argv[2],"a");if(out){fwrite(event.data(),1,event.size(),out);fclose(out);}
            }
            status="204 No Content";
        } else {status="404 Not Found";body="Not found";}
        std::string response="HTTP/1.1 "+status+"\r\nContent-Type: "+type+"\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: "+std::to_string(body.size())+"\r\n\r\n"+body;
        for(size_t off=0;off<response.size();) {int n=send(fd,response.data()+off,response.size()-off,MSG_NOSIGNAL);if(n<=0)break;off+=n;}
        close(fd);
        }).detach();
    }
}
