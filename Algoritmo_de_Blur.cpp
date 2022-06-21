//ALGORITMO DE BLUR IMPLEMENTADO EN C++ USANDO LAS LIBRERIAS OPENCV Y CUDA////////////////////////////////
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include<stdlib.h>
#include<time.h>
using namespace std;
using namespace cv;
const int tam_blur = 10;
//////////////////(funcion para generar la imagen principal)/////////////////////////////////////////////
void generar_imagen(unsigned char* output, int height, int width,string filename){
    Mat output_data(height, width, CV_8UC1, (void*)output);
    imshow(filename, output_data);
    imwrite(filename, output_data);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/*void total_blur(float* in, float* out, int w, int h, int r){
    float iarr = 1.f / (r + r + 1);
    for (int i = 0; i < w; i++){
        int ti = i, li = ti, ri = ti + r * w;
        float fv = in[ti], lv = in[ti + w * (h - 1)], val = (r + 1) * fv;
        for (int j = 0; j < r; j++) val += in[ti + j * w];
        for (int j = 0; j <= r; j++) { val += in[ri] - fv; out[ti] = val * iarr; ri += w; ti += w; }
        for (int j = r + 1; j < h - r; j++) { val += in[ri] - in[li]; out[ti] = val * iarr; li += w; ri += w; ti += w; }
        for (int j = h - r; j < h; j++) { val += lv - in[li]; out[ti] = val * iarr; li += w; ti += w; }
    }
}*/
//////////////////(funcion para generar el kernek en el algoritmo)///////////////////////////////////////
__global__
void nucleo_kernel(unsigned char* image, unsigned char* output,int height, int width) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < width && row < height) {
        int pixel_value = 0,pixels = 0;
        for (int blur_row = -tam_blur; blur_row < tam_blur + 1; ++blur_row) {
            for (int blur_col = -tam_blur; blur_col < tam_blur + 1; ++blur_col) {
                int cur_row = row + blur_row,cur_col = col + blur_col;
                if (cur_row > -1 && cur_row < height && cur_col > -1 && cur_col < width) {
                    pixel_value += image[cur_row * width + cur_col];
                    pixels++;
        }}}
        output[row * width + col] = (unsigned char)(pixel_value / pixels);
    }
}
//////////////////(funcion para generar la imagen del algoritmo blur)////////////////////////////////////
void imagen_blur(unsigned char* Image, unsigned char* output, int height, int width,int channels){
    unsigned char* dev_image;
    unsigned char* dev_output;
    cudaMalloc((void**)&dev_image, height * width * channels);
    cudaMalloc((void**)&dev_output, height * width * channels);
    cudaMemcpy(dev_image, Image, height * width, cudaMemcpyHostToDevice);
    dim3 Grid_image((int)ceil(width / 16.0), (int)ceil(height / 16.0));
    dim3 dimBlock(16, 16);
    nucleo_kernel << <Grid_image, dimBlock >> > (dev_image, dev_output, height, width);
    cudaMemcpy(output, dev_output, height * width * channels,cudaMemcpyDeviceToHost);
    cudaFree(dev_output);
    cudaFree(dev_image);
}
//////////////////(funcion para generar el color a la imagen gris)///////////////////////////////////////
__global__
void conversion_de_color(unsigned char* img,unsigned char* output, int height,int width, int CHANNELS){
    int col = threadIdx.x + blockIdx.x * blockDim.x,row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < width && row < height){
        int grey_offset = row * width + col;
        int rgb_offset = grey_offset * CHANNELS;
        unsigned char r = img[rgb_offset + 0],g = img[rgb_offset + 1],b = img[rgb_offset + 2];
        output[grey_offset] = r * 0.299f + g * 0.587f + b * 0.114f;
    }
}
//////////////////(funcion para generar la imagen gris del blur)////////////////////////////////////////
void imagengris_blur(unsigned char* Image, unsigned char* output, int height, int width,int channels) {
    unsigned char* dev_image;
    unsigned char* dev_output;
    cudaMalloc((void**)&dev_image, height * width * channels);
    cudaMalloc((void**)&dev_output, height * width);
    cudaMemcpy(dev_image, Image, height * width * channels, cudaMemcpyHostToDevice);
    dim3 Grid_image((int)ceil(width / 16.0), (int)ceil(height / 16.0));
    dim3 dimBlock(16, 16);
    conversion_de_color << <Grid_image, dimBlock >> > (dev_image, dev_output,height, width, channels);
    cudaMemcpy(output, dev_output, height * width, cudaMemcpyDeviceToHost);
    cudaFree(dev_output);
    cudaFree(dev_image);
}
//////////////////(menu principal)///////////////////////////////////////////////////////////////////////
int main() {
    Mat image_to_gray = imread("imagen.jpg");
    Mat image_to_blur = imread("imagen.jpg", IMREAD_GRAYSCALE);
    unsigned char* output_grayed =(unsigned char*)malloc(sizeof(unsigned char*) * image_to_gray.rows *
    image_to_gray.cols * image_to_gray.channels());
    imagengris_blur(image_to_gray.data, output_grayed, image_to_gray.rows,image_to_gray.cols, image_to_gray.channels());
    unsigned char* output_blurred =(unsigned char*)malloc(sizeof(unsigned char*) * image_to_blur.rows *
    image_to_blur.cols * image_to_blur.channels());
    imagen_blur(image_to_blur.data, output_blurred, image_to_blur.rows,image_to_blur.cols, image_to_blur.channels());
    generar_imagen(output_grayed, image_to_gray.rows, image_to_gray.cols,"imagen_gris.jpg");
    generar_imagen(output_blurred, image_to_blur.rows, image_to_blur.cols,"imagen_blur.jpg");
    waitKey();
    return 0;
}