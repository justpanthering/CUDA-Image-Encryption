#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

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

unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader, BITMAPFILEHEADER *bitmapFileHeader)
{
    FILE *filePtr; //our file pointer
    unsigned char *bitmapImage;  //store image data
    int imageIdx=0;  //image index counter
    unsigned char tempRGB;  //our swap variable

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

    cudaEventRecord(start, 0);
    //swap the r and b values to get RGB (bitmap is BGR)
    for (imageIdx = 0; imageIdx < bitmapInfoHeader->biSizeImage;imageIdx+=3)
    {
        tempRGB = bitmapImage[imageIdx];
        bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
        bitmapImage[imageIdx + 2] = tempRGB;
    }
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

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
    int imageIdx=0;  //image index counter
    unsigned char tempRGB;  //our swap variable

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
    for (imageIdx = 0; imageIdx < bitmapInfoHeader->biSizeImage;imageIdx+=3)
    {
        tempRGB = bitmapImage[imageIdx];
        bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
        bitmapImage[imageIdx + 2] = tempRGB;
    }

    //write in the bitmap image data
    fwrite(bitmapImage,bitmapInfoHeader->biSizeImage,1,filePtr);

    //close file
    fclose(filePtr);
}

void encrypt(unsigned char *bitmapImage, int size, int key)
{
    int count;
    unsigned char mid, temp;
    for(int i=0; i<size; i+=key)
    {
        if(i+key>size)
            break;

        //mid = bitmapImage[i+(key/2)];
        for(int j=0; j<key/2; j++)
        {
            temp = bitmapImage[i+j];
            bitmapImage[i+j] = bitmapImage[(((i+j)/key)*key)+key-((i+j)%key)-1];
            bitmapImage[(((i+j)/key)*key)+key-((i+j)%key)-1] = temp;
        }
    }
}

void decrypt(unsigned char *bitmapImage, int size, int key)
{
    int count;
    unsigned char mid, temp;
    for(int i=0; i<size; i+=key)
    {
        if(i+key>size)
            break;

        //mid = bitmapImage[i+(key/2)];
        for(int j=0; j<key/2; j++)
        {
            temp = bitmapImage[i+j];
            bitmapImage[i+j] = bitmapImage[(((i+j)/key)*key)+key-((i+j)%key)-1];
            bitmapImage[(((i+j)/key)*key)+key-((i+j)%key)-1] = temp;
        }
    }
}

int main()
{
    BITMAPINFOHEADER bitmapInfoHeader;
    BITMAPFILEHEADER bitmapFileHeader;
    unsigned char *bitmapData;
    bitmapData = LoadBitmapFile("lena.bmp",&bitmapInfoHeader, &bitmapFileHeader);
    printf("%d\n",bitmapInfoHeader.biSizeImage);

    int key = 8000;
    
    /*
    //Print array to file
    FILE *fout = fopen("out.bmp","wb");
    fwrite(bitmapData,bitmapInfoHeader.biSizeImage,1,fout);
    */

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t end;
    cudaEventCreate(&end);
    float encryptionTime, decryptionTime;
    
    //Encryption
    cudaEventRecord(start, 0);
    encrypt(bitmapData, bitmapInfoHeader.biSizeImage, key);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&encryptionTime, start, end);
    printf("Encryption Time: %fms\n",encryptionTime);
    ReloadBitmapFile("encrypted.bmp", bitmapData, &bitmapFileHeader, &bitmapInfoHeader);

    //load encrypted image to array
    bitmapData = LoadBitmapFile("encrypted.bmp",&bitmapInfoHeader, &bitmapFileHeader);

    //Decryption
    cudaEventRecord(start, 0);
    decrypt(bitmapData, bitmapInfoHeader.biSizeImage, key);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&decryptionTime, start, end);
    printf("Decryption Time: %fms\n",decryptionTime);
    ReloadBitmapFile("Decrypted.bmp", bitmapData, &bitmapFileHeader, &bitmapInfoHeader);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
