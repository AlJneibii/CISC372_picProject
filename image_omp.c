// image_omp.c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
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

int Usage(){
    printf("Usage: image_omp <filename> <type> [threads]\n");
    printf("  type: edge | sharpen | blur | gauss | emboss | identity\n");
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

int main(int argc, char** argv){
    if (argc < 3 || argc > 4) return Usage();

    const char* fileName = argv[1];
    enum KernelTypes type = GetKernelType(argv[2]);
    int threads = (argc==4) ? atoi(argv[3]) : 0;
    if (threads < 1) threads = omp_get_max_threads();

    if (!strcmp(argv[1],"pic4.jpg") && !strcmp(argv[2],"gauss")) {
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }

    Image src = {0}, dest = {0};
    stbi_set_flip_vertically_on_load(0);
    src.data = stbi_load(fileName, &src.width, &src.height, &src.bpp, 0);
    if (!src.data){
        printf("Error loading file %s.\n", fileName);
        return -1;
    }
    dest.width = src.width; dest.height = src.height; dest.bpp = src.bpp;
    dest.data = (uint8_t*)malloc((size_t)dest.width * dest.height * dest.bpp);
    if (!dest.data){ fprintf(stderr,"Alloc failed\n"); stbi_image_free(src.data); return -1; }

    Matrix K;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) K[i][j] = algorithms[type][i][j];

    omp_set_num_threads(threads);
    double t1 = omp_get_wtime();

    // parallelize rows; each thread writes disjoint rows -> no races
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < src.height; row++){
        for (int x = 0; x < src.width; x++){
            for (int bit = 0; bit < src.bpp; bit++){
                dest.data[Index(x,row,src.width,bit,src.bpp)] =
                    getPixelValue(&src, x, row, bit, K);
            }
        }
    }

    double t2 = omp_get_wtime();
    stbi_write_png("output.png", dest.width, dest.height, dest.bpp, dest.data, dest.bpp*dest.width);
    printf("OMP Threads: %d, Took %.3f seconds\n", threads, (t2 - t1));

    free(dest.data);
    stbi_image_free(src.data);
    return 0;
}
