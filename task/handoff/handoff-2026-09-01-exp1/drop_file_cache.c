#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <file>\n", argv[0]);
        return 2;
    }

    const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "open(%s): %s\n", argv[1], strerror(errno));
        return 1;
    }

    sync();
    const int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);

    if (rc != 0) {
        fprintf(stderr, "posix_fadvise(%s): %s\n", argv[1], strerror(rc));
        return 1;
    }

    printf("dropped clean page-cache pages for %s\n", argv[1]);
    return 0;
}

