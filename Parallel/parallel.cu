#include <stdio.h>
#include <stdlib.h>

typedef short WORD;
typedef int DWORD;
typedef int LONG;

#pragma pack(push, 1)
typedef struct tagBITMAPFILEHEADER
{
    WORD bfType;  //specifies the file type
    DWORD bfSize;  //specifies the size in bytes of the bitmap file
    WORD bfReserved1;  //reserved; must be 0
    WORD bfReserved2;  //reserved; must be 0
    DWORD bOffBits;  //species the offset in bytes from the bitmapfileheader to the bitmap bits
}BITMAPFILEHEADER;
#pragma pack(pop)


#pragma pack(push, 1)
typedef struct tagBITMAPINFOHEADER
{
    DWORD biSize;  //specifies the number of bytes required by the struct
    LONG biWidth;  //specifies width in pixels
    LONG biHeight;  //species height in pixels
    WORD biPlanes; //specifies the number of color planes, must be 1
    WORD biBitCount; //specifies the number of bit per pixel
    DWORD biCompression;//spcifies the type of compression
    DWORD biSizeImage;  //size of image in bytes
    LONG biXPelsPerMeter;  //number of pixels per meter in x axis
    LONG biYPelsPerMeter;  //number of pixels per meter in y axis
    DWORD biClrUsed;  //number of colors used by th ebitmap
    DWORD biClrImportant;  //number of colors that are important
}BITMAPINFOHEADER;
#pragma pack(pop)

__global__ void RB_Swap(unsigned char *imageData, int size)
{
    int imageIdx = threadIdx.x+blockIdx.x*blockDim.x;

    if(imageIdx<size/3)
    {
        unsigned char tempRGB;
        imageIdx = imageIdx*3;
        tempRGB = imageData[imageIdx];
        imageData[imageIdx] = imageData[imageIdx + 2];
        imageData[imageIdx + 2] = tempRGB;
    }
}

unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader, BITMAPFILEHEADER *bitmapFileHeader)
{
    FILE *filePtr; //our file pointer
    unsigned char *bitmapImage;  //store image data

    //open filename in read binary mode
    filePtr = fopen(filename,"rb");
    if (filePtr == NULL)
        return NULL;

    //read the bitmap file header
    fread(bitmapFileHeader, sizeof(BITMAPFILEHEADER),1,filePtr);

    
    //verify that this is a bmp file by check bitmap id
    if (bitmapFileHeader->bfType !=0x4D42)
    {
        fclose(filePtr);
        return NULL;
    }
    
    //read the bitmap info header
    fread(bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr); // small edit. forgot to add the closing bracket at sizeof

    //move file point to the begging of bitmap data
    fseek(filePtr, bitmapFileHeader->bOffBits, SEEK_SET);

    //allocate enough memory for the bitmap image data
    bitmapImage = (unsigned char*)malloc(bitmapInfoHeader->biSizeImage);

    //verify memory allocation
    if (!bitmapImage)
    {
        free(bitmapImage);
        fclose(filePtr);
        return NULL;
    }

    //read in the bitmap image data
    fread(bitmapImage,1,bitmapInfoHeader->biSizeImage,filePtr);

    //make sure bitmap image data was read
    if (bitmapImage == NULL)
    {
        fclose(filePtr);
        return NULL;
    }

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t end;
    cudaEventCreate(&end);
    float swapTime;

    //swap the r and b values to get RGB (bitmap is BGR)    
    unsigned char *d_bitmapImage;  //store image data in device
    
    //Allocate size to array in device memory
    cudaMalloc((void**)&d_bitmapImage, bitmapInfoHeader->biSizeImage);

    //Copy data from host to device
    cudaMemcpy(d_bitmapImage, bitmapImage, bitmapInfoHeader->biSizeImage, cudaMemcpyHostToDevice);

    int B = ceil(bitmapInfoHeader->biSizeImage/1024);
    int T = 1024;

    //Kernel call
    cudaEventRecord(start, 0);
    RB_Swap<<<B, T>>> (d_bitmapImage, bitmapInfoHeader->biSizeImage);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaMemcpy(bitmapImage, d_bitmapImage, bitmapInfoHeader->biSizeImage, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&swapTime, start, end);
    printf("Load Swap Time: %fms\n",swapTime);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    //close file and return bitmap iamge data
    fclose(filePtr);
    return bitmapImage;
}

void ReloadBitmapFile(char *filename, unsigned char *bitmapImage, BITMAPFILEHEADER *bitmapFileHeader, BITMAPINFOHEADER *bitmapInfoHeader)
{
    FILE *filePtr; //our file pointer

    //open filename in write binary mode
    filePtr = fopen(filename,"wb");
    if (filePtr == NULL)
    {
        printf("\nERROR: Cannot open file %s", filename);
        exit(1);
    }
        

    //write the bitmap file header
    fwrite(bitmapFileHeader, sizeof(BITMAPFILEHEADER),1,filePtr);

    //write the bitmap info header
    fwrite(bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr); // small edit. forgot to add the closing bracket at sizeof

    //swap the r and b values to get RGB (bitmap is BGR)

    unsigned char *d_bitmapImage;  //store image data in device
    
    //Allocate size to array in device memory
    cudaMalloc((void**)&d_bitmapImage, bitmapInfoHeader->biSizeImage);

    //Copy data from host to device
    cudaMemcpy(d_bitmapImage, bitmapImage, bitmapInfoHeader->biSizeImage, cudaMemcpyHostToDevice);

    int B = ceil(bitmapInfoHeader->biSizeImage/1024);
    int T = 1024;

    //Kernel call
    
    RB_Swap<<<B, T>>> (d_bitmapImage, bitmapInfoHeader->biSizeImage);

    cudaMemcpy(bitmapImage, d_bitmapImage, bitmapInfoHeader->biSizeImage, cudaMemcpyDeviceToHost);

    //write in the bitmap image data
    fwrite(bitmapImage,bitmapInfoHeader->biSizeImage,1,filePtr);

    //close file
    fclose(filePtr);
}

__global__ void encrypt(unsigned char *bitmapImage, int size, int key)
{
    int threadId = threadIdx.x + blockIdx.x*blockDim.x;
    int half = key/2;
    int index = ((threadId/half)*key) + (threadId%half);
    int swap = index + (key - (2*(index%half)) - 1);

    if((swap)<size)
    {
        unsigned char temp;
        //unsigned mid = bitmapImage[((index/half)*key) + half];
        
            temp = bitmapImage[index];
            bitmapImage[index] = bitmapImage[swap];
            bitmapImage[swap] = temp;
    }
}

__global__ void decrypt(unsigned char *bitmapImage, int size, int key)
{
    int threadId = threadIdx.x + blockIdx.x*blockDim.x;
    int half = key/2;
    int index = ((threadId/half)*key) + (threadId%half);
    int swap = index + (key - (2*(index%half)) - 1);

    if((swap)<size)
    {
        unsigned char temp;
        //unsigned mid = bitmapImage[((index/half)*key) + half];
        
            temp = bitmapImage[index];
            bitmapImage[index] = bitmapImage[swap];
            bitmapImage[swap] = temp;
    }
}

int main()
{
    BITMAPINFOHEADER bitmapInfoHeader;
    BITMAPFILEHEADER bitmapFileHeader;
    unsigned char *bitmapData;
    bitmapData = LoadBitmapFile("mona_lisa.bmp",&bitmapInfoHeader, &bitmapFileHeader);
    printf("%d\n",bitmapInfoHeader.biSizeImage);
    
    /*
    //Print array to file
    FILE *fout = fopen("out.bmp","wb");
    fwrite(bitmapData,bitmapInfoHeader.biSizeImage,1,fout);
    */

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t end;
    cudaEventCreate(&end);
    float encryptionTime, decryptionTime, HostToDevice, DeviceToHost;
    
    //Encryption

    int key = 8000;

    unsigned char *d_bitmapImage;  //store image data in device
    
    //Allocate size to array in device memory
    cudaMalloc((void**)&d_bitmapImage, bitmapInfoHeader.biSizeImage);

    //Copy data from host to device
    cudaEventRecord(start, 0);
    cudaMemcpy(d_bitmapImage, bitmapData, bitmapInfoHeader.biSizeImage, cudaMemcpyHostToDevice);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&HostToDevice, start, end);
    printf("Host to Device Time: %fms\n",HostToDevice);

    int B = ceil(bitmapInfoHeader.biSizeImage/1024);
    int T = 1024;

    //Kernel call
    cudaEventRecord(start, 0);
    encrypt<<<B, T>>> (d_bitmapImage, bitmapInfoHeader.biSizeImage, key);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&encryptionTime, start, end);
    printf("Encryption Time: %fms\n",encryptionTime);

    //Copy data from device to host
    cudaEventRecord(start, 0);
    cudaMemcpy(bitmapData, d_bitmapImage, bitmapInfoHeader.biSizeImage, cudaMemcpyDeviceToHost);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&DeviceToHost, start, end);
    printf("Device to Host Time: %fms\n",DeviceToHost);
    
    ReloadBitmapFile("Encrypted.bmp", bitmapData, &bitmapFileHeader, &bitmapInfoHeader);

    //load encrypted image to array
    bitmapData = LoadBitmapFile("Encrypted.bmp",&bitmapInfoHeader, &bitmapFileHeader);

    //Decryption
    cudaMemcpy(d_bitmapImage, bitmapData, bitmapInfoHeader.biSizeImage, cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    decrypt<<<B, T>>> (d_bitmapImage, bitmapInfoHeader.biSizeImage, key);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&decryptionTime, start, end);
    printf("Decryption Time: %fms\n",decryptionTime);
    cudaMemcpy(bitmapData, d_bitmapImage, bitmapInfoHeader.biSizeImage, cudaMemcpyDeviceToHost);

    //decrypt(bitmapData, bitmapInfoHeader.biSizeImage);

    ReloadBitmapFile("Decrypted.bmp", bitmapData, &bitmapFileHeader, &bitmapInfoHeader);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_bitmapImage);

    return 0;
}