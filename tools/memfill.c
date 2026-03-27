/* memfill: Allocate and hold a specified amount of memory.
 * Usage: memfill <MB> [hold_seconds]
 * - Allocates <MB> megabytes using mmap(MAP_ANONYMOUS|MAP_PRIVATE).
 * - Continuously writes non-compressible patterns to defeat zram compression.
 * - Holds for hold_seconds (default: 300), then frees and exits.
 * - Responds to SIGTERM/SIGINT for graceful cleanup.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

static volatile int running = 1;

static void handler(int sig) {
    (void)sig;
    running = 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <MB> [hold_seconds]\n", argv[0]);
        return 1;
    }

    size_t mb = (size_t)atol(argv[1]);
    int hold = argc > 2 ? atoi(argv[2]) : 300;
    size_t bytes = mb * 1024UL * 1024UL;
    size_t pages = bytes / 4096;

    signal(SIGTERM, handler);
    signal(SIGINT, handler);

    fprintf(stderr, "[memfill] Allocating %zu MB (%zu pages)...\n", mb, pages);

    /* Use mmap instead of malloc — more predictable behavior with large allocs */
    char *buf = (char *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                              MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (buf == MAP_FAILED) {
        fprintf(stderr, "[memfill] mmap failed for %zu MB\n", mb);
        return 1;
    }

    /* Initial fill: write non-compressible data (pseudo-random via LCG) */
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)getpid();
    for (size_t i = 0; i < bytes; i += 4) {
        seed = seed * 1103515245 + 12345;
        *(unsigned int *)(buf + i) = seed;
    }
    fprintf(stderr, "[memfill] Initial fill complete. Holding %zu MB for %d s (PID=%d)\n",
            mb, hold, getpid());

    /* Continuously re-dirty pages to prevent zram from reclaiming them.
     * Each second: write 1 byte per page with varying data → pages stay dirty. */
    for (int t = 0; t < hold && running; t++) {
        unsigned char val = (unsigned char)(t & 0xFF);
        /* Touch 1/4 of pages each second (rotating) to limit CPU overhead */
        size_t offset = (t % 4) * 4096;
        for (size_t i = offset; i < bytes; i += 4096 * 4) {
            buf[i] = val;
        }
        sleep(1);
    }

    fprintf(stderr, "[memfill] Releasing %zu MB\n", mb);
    munmap(buf, bytes);
    return 0;
}
