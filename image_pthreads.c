#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>  
#include <pthread.h>
#include <unistd.h>
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static Matrix algorithms[] = {
    {{0,-1,0},{-1,4,-1},{0,-1,0}},                                      // EDGE
    {{0,-1,0},{-1,5,-1},{0,-1,0}},                                      // SHARPEN
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}},      // BLUR
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}},  // GAUSE_BLUR
    {{-2,-1,0},{-1,1,1},{0,1,2}},                                       // EMBOSS
    {{0,0,0},{0,1,0},{0,0,0}}                                           // IDENTITY
};

uint8_t getPixelValue(Image* srcImage,int x,int y,int bit,Matrix algorithm){
    int px=x+1, mx=x-1, py=y+1, my=y-1;
    if (mx<0) mx=0;
    if (my<0) my=0;
    if (px>=srcImage->width)  px=srcImage->width-1;
    if (py>=srcImage->height) py=srcImage->height-1;

    double sum =
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][1]*srcImage->data[Index(x ,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][0]*srcImage->data[Index(mx,y ,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][1]*srcImage->data[Index(x ,y ,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][2]*srcImage->data[Index(px,y ,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][1]*srcImage->data[Index(x ,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];

    if (sum < 0.0) sum = 0.0;
    if (sum > 255.0) sum = 255.0;
    return (uint8_t)sum;
}

// ---- Thread worker setup ----
typedef struct {
    Image *src;
    Image *dest;
    Matrix kernel;
    int start_row;  // inclusive
    int end_row;    // exclusive
} WorkerArgs;

static void* worker_fn(void* arg) {
    WorkerArgs* w = (WorkerArgs*)arg;
    Image* src = w->src;
    Image* dest = w->dest;
    Matrix K;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) K[i][j] = w->kernel[i][j];

    for (int row = w->start_row; row < w->end_row; row++) {
        for (int x = 0; x < src->width; x++) {
            for (int bit = 0; bit < src->bpp; bit++) {
                dest->data[Index(x,row,src->width,bit,src->bpp)] =
                    getPixelValue(src, x, row, bit, K);
            }
        }
    }
    return NULL;
}

// ---- Utilities / CLI ----
int Usage(){
    printf("Usage: image_pthreads <filename> <type> [threads]\n");
    printf("  type: edge | sharpen | blur | gauss | emboss | identity\n");
    printf("  threads (optional): number of worker threads (default = cores)\n");
    return -1;
}

enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

// Portable timer for Darwin (no -lrt needed)
static double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) return Usage();

    const char* fileName = argv[1];
    enum KernelTypes type = GetKernelType(argv[2]);

    // Optional threads argument; default to #cores
    int threads = 0;
    if (argc == 4) {
        threads = atoi(argv[3]);
        if (threads < 1) threads = 1;
    } else {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        threads = (n > 0) ? (int)n : 4;
    }

    if (!strcmp(argv[1],"pic4.jpg") && !strcmp(argv[2],"gauss")) {
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }

    Image src = {0}, dest = {0};
    stbi_set_flip_vertically_on_load(0);
    src.data = stbi_load(fileName, &src.width, &src.height, &src.bpp, 0);
    if (!src.data) {
        printf("Error loading file %s.\n", fileName);
        return -1;
    }

    dest.width = src.width;
    dest.height = src.height;
    dest.bpp = src.bpp;
    dest.data = (uint8_t*)malloc((size_t)dest.width * dest.height * dest.bpp);
    if (!dest.data) {
        fprintf(stderr, "Failed to allocate dest buffer\n");
        stbi_image_free(src.data);
        return -1;
    }

    // Prepare workers
    Matrix kernel;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) kernel[i][j] = algorithms[type][i][j];

    double t1 = now_seconds();

    pthread_t *tids = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)threads);
    WorkerArgs *args = (WorkerArgs*)malloc(sizeof(WorkerArgs) * (size_t)threads);
    if (!tids || !args) {
        fprintf(stderr, "Failed to allocate thread structures\n");
        free(tids); free(args);
        free(dest.data);
        stbi_image_free(src.data);
        return -1;
    }

    // Divide rows as evenly as possible
    int H = src.height;
    int base = H / threads;
    int rem  = H % threads;
    int start = 0;
    for (int t = 0; t < threads; t++) {
        int take = base + (t < rem ? 1 : 0);
        args[t].src = &src;
        args[t].dest = &dest;
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) args[t].kernel[i][j] = kernel[i][j];
        args[t].start_row = start;
        args[t].end_row = start + take;
        start += take;

        if (pthread_create(&tids[t], NULL, worker_fn, &args[t]) != 0) {
            fprintf(stderr, "pthread_create failed on thread %d\n", t);
            for (int k=0;k<t;k++) pthread_join(tids[k], NULL);
            free(tids); free(args);
            free(dest.data);
            stbi_image_free(src.data);
            return -1;
        }
    }

    for (int t = 0; t < threads; t++) {
        pthread_join(tids[t], NULL);
    }

    double t2 = now_seconds();

    stbi_write_png("output.png", dest.width, dest.height, dest.bpp, dest.data, dest.bpp*dest.width);
    printf("Threads: %d, Took %.3f seconds\n", threads, (t2 - t1));

    free(tids);
    free(args);
    free(dest.data);
    stbi_image_free(src.data);
    return 0;
}
