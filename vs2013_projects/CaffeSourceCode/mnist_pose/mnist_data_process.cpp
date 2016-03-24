#include "mnist_data_process.h"

bool LoadMnistLable(const string& file, vector<unsigned char>& labels, bool is_little_endian)
{
    ifstream fin(file, ios::binary);
    if (!fin)
    {
        cout << "read file failed!" << endl;
        return false;
    }

    int MSB;
    fin.read((char*)(&MSB), sizeof(int));
    if (is_little_endian)
        MSB = BigLittleEndianExchange(MSB);

    int count;
    fin.read((char*)(&count), sizeof(int));
    if (is_little_endian)
        count = BigLittleEndianExchange(count);

    labels.clear();
    labels.resize(count);
    fin.read((char*)(&labels[0]), count);

    return true;
}

bool LoadMnistImage2Heap(const string& file, vector<unsigned char*>& images, unsigned int& img_width, unsigned int& img_height, bool is_little_endian)
{
    ifstream fin(file, ios::binary);
    if (!fin)
    {
        cout << "read file failed!" << endl;
        return false;
    }

    int MSB;
    fin.read((char*)(&MSB), sizeof(int));
    if (is_little_endian)
        MSB = BigLittleEndianExchange(MSB);

    int count;
    fin.read((char*)(&count), sizeof(int));
    if (is_little_endian)
        count = BigLittleEndianExchange(count);

    int width;
    fin.read((char*)(&width), sizeof(int));
    if (is_little_endian)
        width = BigLittleEndianExchange(width);
    img_width = width;

    int height;
    fin.read((char*)(&height), sizeof(int));
    if (is_little_endian)
        height = BigLittleEndianExchange(height);
    img_height = height;

    unsigned int img_size = width * height;

    images.clear();
    images.resize(count);
    for (unsigned int n_img = 0; n_img < images.size(); n_img++)
    {
        images[n_img] = new unsigned char[img_size];
        fin.read((char*)(images[n_img]), img_size);
    }

    return true;
}