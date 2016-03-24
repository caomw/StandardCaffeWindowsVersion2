#pragma once
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

template <class DataType>
DataType BigLittleEndianExchange(DataType data)
{
    DataType d;
    unsigned int byte_count = sizeof(DataType);
    char* data_idx = (char*)(&data);
    char* d_idx = (char*)(&d);
    for (unsigned int i = 0; i < byte_count; i++)
        d_idx[i] = data_idx[byte_count - 1 - i];
    return d;
}

bool LoadMnistLable(const string& file, vector<unsigned char>& labels, bool is_little_endian);

bool LoadMnistImage2Heap(const string& file, vector<unsigned char*>& images, unsigned int& img_width, unsigned int& img_height, bool is_little_endian);

