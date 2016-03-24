#include "convert_mnist_data.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>


#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb/lmdb.h>

#include <stdint.h>
#include <sys/stat.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

//uint32_t swap_endian(uint32_t val) {
//    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
//    return (val << 16) | (val >> 16);
//}


void convert_database(const string& db_backend, const char* db_path, int num_items
    , const unsigned char* const labels, unsigned char** images
    , unsigned int rows, unsigned int cols)
{

    // lmdb
    MDB_env *mdb_env = NULL;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn = NULL;
    // leveldb
    leveldb::DB* db = NULL;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;
    leveldb::WriteBatch* batch = NULL;

    // Open db
    if (db_backend == "leveldb") {  // leveldb
        LOG(INFO) << "Opening leveldb " << db_path;
        leveldb::Status status = leveldb::DB::Open(
            options, db_path, &db);
        CHECK(status.ok()) << "Failed to open leveldb " << db_path
            << ". Is it already existing?";
        batch = new leveldb::WriteBatch();
    }
    else if (db_backend == "lmdb") {  // lmdb
        LOG(INFO) << "Opening lmdb " << db_path;
        //CHECK_EQ(mkdir(db_path, 0744), 0)
        //    << "mkdir " << db_path << "failed";
        CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
            << "mdb_env_set_mapsize failed";
        CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
            << "mdb_env_open failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
            << "mdb_open failed. Does the lmdb already exist? ";
    }
    else {
        LOG(FATAL) << "Unknown db backend " << db_backend;
    }

    // Storing to db
    //char label;
    //char* pixels = new char[rows * cols];
    int count = 0;
    string value;

    Datum datum;
    datum.set_channels(1);
    datum.set_height(rows);
    datum.set_width(cols);
    LOG(INFO) << "A total of " << num_items << " items.";
    LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
    for (int item_id = 0; item_id < num_items; ++item_id) {
        //image_file.read(pixels, rows * cols);
        //label_file.read(&label, 1);
        datum.set_data(images[item_id], rows*cols);
        datum.set_label(labels[item_id]);
        string key_str = caffe::format_int(item_id, 8);
        datum.SerializeToString(&value);

        // Put in db
        if (db_backend == "leveldb") {  // leveldb
            batch->Put(key_str, value);
        }
        else if (db_backend == "lmdb") {  // lmdb
            mdb_data.mv_size = value.size();
            mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
            mdb_key.mv_size = key_str.size();
            mdb_key.mv_data = reinterpret_cast<void*>(&key_str[0]);
            CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
                << "mdb_put failed";
        }
        else {
            LOG(FATAL) << "Unknown db backend " << db_backend;
        }

        if (++count % 1000 == 0) {
            // Commit txn
            if (db_backend == "leveldb") {  // leveldb
                db->Write(leveldb::WriteOptions(), batch);
                delete batch;
                batch = new leveldb::WriteBatch();
            }
            else if (db_backend == "lmdb") {  // lmdb
                CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
                    << "mdb_txn_commit failed";
                CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
                    << "mdb_txn_begin failed";
            }
            else {
                LOG(FATAL) << "Unknown db backend " << db_backend;
            }
        }
    }
    // write the last batch
    if (count % 1000 != 0) {
        if (db_backend == "leveldb") {  // leveldb
            db->Write(leveldb::WriteOptions(), batch);
            delete batch;
            delete db;
        }
        else if (db_backend == "lmdb") {  // lmdb
            CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
            mdb_close(mdb_env, mdb_dbi);
            mdb_env_close(mdb_env);
        }
        else {
            LOG(FATAL) << "Unknown db backend " << db_backend;
        }
        LOG(ERROR) << "Processed " << count << " files.";
    }
    //delete[] pixels;

}



