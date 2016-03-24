#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb/lmdb.h>

#include <stdint.h>
#include <sys/stat.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

using namespace caffe;

#include "mnist_data_process.h"
#include "convert_mnist_data.h"


int run_create_mnist_leveldb()
{
    {
        // load test labels
        string test_label_file = "C:\\IMP\\projects\\Caffe\\CaffeSourceCode\\caffe\\data\\mnist\\MNIST\\t10k-labels.idx1-ubyte";
        vector<unsigned char> test_labels;
        if (!LoadMnistLable(test_label_file, test_labels, true))
            return 0;;

        // load test iamges;
        string test_image_file = "C:\\IMP\\projects\\Caffe\\CaffeSourceCode\\caffe\\data\\mnist\\MNIST\\t10k-images.idx3-ubyte";
        vector<unsigned char*> test_images;
        unsigned int test_width, test_height;
        if (!LoadMnistImage2Heap(test_image_file, test_images, test_width, test_height, true))
            return 0;

        string test_db_path = "test_db";
        convert_database("leveldb", test_db_path.c_str(), test_labels.size(), &test_labels[0], &test_images[0], test_height, test_width);

        for (unsigned int i = 0; i < test_images.size(); i++)
            delete test_images[i];
    }

    {
        // load train labels
        string train_label_file = "C:\\IMP\\projects\\Caffe\\CaffeSourceCode\\caffe\\data\\mnist\\MNIST\\train-labels.idx1-ubyte";
        vector<unsigned char> train_labels;
        if (!LoadMnistLable(train_label_file, train_labels, true))
            return 0;;

        // load train iamges;
        string train_image_file = "C:\\IMP\\projects\\Caffe\\CaffeSourceCode\\caffe\\data\\mnist\\MNIST\\train-images.idx3-ubyte";
        vector<unsigned char*> train_images;
        unsigned int train_width, train_height;
        if (!LoadMnistImage2Heap(train_image_file, train_images, train_width, train_height, true))
            return 0;

        string train_db_path = "train_db";
        convert_database("leveldb", train_db_path.c_str(), train_labels.size(), &train_labels[0], &train_images[0], train_height, train_width);

        for (unsigned int i = 0; i < train_images.size(); i++)
            delete train_images[i];
    }

    return 1;
}



string FLAGS_gpu = "";
string FLAGS_solver = "lenet_solver.prototxt";
string FLAGS_model = "";
string FLAGS_snapshot = "";
string FLAGS_weights = "";
int FLAGS_iterations = 50;
string FLAGS_sigint_effect = "stop";
string FLAGS_sighup_effect = "snapshot";

// A simple registry for caffe commands.
typedef int(*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
    if (g_brew_map.count(name)) {
        return g_brew_map[name];
    }
    else {
        LOG(ERROR) << "Available caffe actions:";
        for (BrewMap::iterator it = g_brew_map.begin();
            it != g_brew_map.end(); ++it) {
            LOG(ERROR) << "\t" << it->first;
        }
        LOG(FATAL) << "Unknown action: " << name;
        return NULL;  // not reachable, just to suppress old compiler warnings.
    }
}

// Parse GPU ids or use all available devices
static void get_gpus(vector<int>* gpus) {
    if (FLAGS_gpu == "all") {
        int count = 0;
#ifndef CPU_ONLY
        CUDA_CHECK(cudaGetDeviceCount(&count));
#else
        NO_GPU;
#endif
        for (int i = 0; i < count; ++i) {
            gpus->push_back(i);
        }
    }
    else if (FLAGS_gpu.size()) {
        vector<string> strings;
        boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
        for (int i = 0; i < strings.size(); ++i) {
            gpus->push_back(boost::lexical_cast<int>(strings[i]));
        }
    }
    else {
        CHECK_EQ(gpus->size(), 0);
    }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
static caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
    if (flag_value == "stop") {
        return caffe::SolverAction::STOP;
    }
    if (flag_value == "snapshot") {
        return caffe::SolverAction::SNAPSHOT;
    }
    if (flag_value == "none") {
        return caffe::SolverAction::NONE;
    }
    LOG(FATAL) << "Invalid signal effect \"" << flag_value << "\" was specified";
}



// Load the weights from the specified caffemodel(s) into the train and
// test nets.
static void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(","));
    for (int i = 0; i < model_names.size(); ++i) {
        LOG(INFO) << "Finetuning from " << model_names[i];
        solver->net()->CopyTrainedLayersFrom(model_names[i]);
        for (int j = 0; j < solver->test_nets().size(); ++j) {
            solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}

int my_train()
{
    CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
    CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
        << "Give a snapshot to resume training or weights to finetune "
        "but not both.";

    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

    // If the gpus flag is not provided, allow the mode and device to be set
    // in the solver prototxt.
    if (FLAGS_gpu.size() == 0
        && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
        if (solver_param.has_device_id()) {
            FLAGS_gpu = "" +
                boost::lexical_cast<string>(solver_param.device_id());
        }
        else {  // Set default GPU if unspecified
            FLAGS_gpu = "" + boost::lexical_cast<string>(0);
        }
    }

    vector<int> gpus;
    get_gpus(&gpus);
    if (gpus.size() == 0) {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    }
    else {
        ostringstream s;
        for (int i = 0; i < gpus.size(); ++i) {
            s << (i ? ", " : "") << gpus[i];
        }
        LOG(INFO) << "Using GPUs " << s.str();

        solver_param.set_device_id(gpus[0]);
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_solver_count(gpus.size());
    }

    caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

    caffe::shared_ptr<caffe::Solver<float> >
        solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

    solver->SetActionFunction(signal_handler.GetActionFunction());

    if (FLAGS_snapshot.size()) {
        LOG(INFO) << "Resuming from " << FLAGS_snapshot;
        solver->Restore(FLAGS_snapshot.c_str());
    }
    else if (FLAGS_weights.size()) {
        CopyLayers(solver.get(), FLAGS_weights);
    }

    if (gpus.size() > 1) {
        caffe::P2PSync<float> sync(solver, NULL, solver->param());
        sync.run(gpus);
    }
    else {
        LOG(INFO) << "Starting Optimization";
        solver->Solve();
    }
    LOG(INFO) << "Optimization Done.";
    return 0;
}
RegisterBrewFunction(my_train);

int main(int argc, char** argv)
{
    // Print output to stderr (while still logging).
    FLAGS_alsologtostderr = 1;
    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);

    return GetBrewFunction(caffe::string(argv[1]))();
}
