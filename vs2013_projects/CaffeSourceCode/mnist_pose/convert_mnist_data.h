#pragma once

#include <string>
using namespace std;


void convert_database(const string& db_backend, const char* db_path, int num_items
    , const unsigned char* const labels, unsigned char** images
    , unsigned int rows, unsigned int cols);