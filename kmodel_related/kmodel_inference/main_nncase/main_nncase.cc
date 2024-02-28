#include <chrono>
#include <fstream>
#include <iostream>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::detail;

#define USE_CACHE 1

template <class T>
std::vector<T> read_binary_file(const char *file_name)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    std::vector<T> vec(len / sizeof(T), 0);
    ifs.seekg(0, ifs.beg);
    ifs.read(reinterpret_cast<char *>(vec.data()), len);
    ifs.close();
    return vec;
}

void read_binary_file(const char *file_name, char *buffer)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    ifs.read(buffer, len);
    ifs.close();
}

template <typename T>
double dot(const T *v1, const T *v2, size_t size)
{
    double ret = 0.f;
    for (size_t i = 0; i < size; i++)
    {
        ret += v1[i] * v2[i];
    }

    return ret;
}

template <typename T>
double cosine(const T *v1, const T *v2, size_t size)
{
    return dot(v1, v2, size) / ((sqrt(dot(v1, v1, size)) * sqrt(dot(v2, v2, size))));
}

void dump(const std::string &info, volatile float *p, size_t size)
{
    std::cout << info << " dump: p = " << std::hex << (void *)p << std::dec << ", size = " << size << std::endl;
    volatile unsigned int *q = reinterpret_cast<volatile unsigned int *>(p);
    for (size_t i = 0; i < size; i++)
    {
        if ((i != 0) && (i % 4 == 0))
        {
            std::cout << std::endl;
        }

        std::cout << std::hex << q[i] << " ";
    }
    std::cout << std::dec << std::endl;
}

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " build " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <kmodel> <input_0.bin> <input_1.bin> ... <input_N.bin> <output_0.bin> <output_1.bin> ... <output_N.bin>" << std::endl;
        return -1;
    }

    interpreter interp;                             

    // 1. load model
    std::ifstream in_before_load_kmodel("/proc/media-mem");
    std::string line_before_load_kmodel;
    // 逐行读取文件内容，查看MMZ使用情况
    while (std::getline(in_before_load_kmodel, line_before_load_kmodel)) { 
        std::cout << line_before_load_kmodel << std::endl;
    }

    std::ifstream ifs(argv[1], std::ios::binary);
    interp.load_model(ifs).expect("Invalid kmodel");

    std::ifstream in_after_load_kmodel("/proc/media-mem");
    std::string line_after_load_kmodel;
    // 逐行读取文件内容，查看MMZ使用情况
    while (std::getline(in_after_load_kmodel, line_after_load_kmodel)) {  
        std::cout << line_after_load_kmodel << std::endl;  
    }

    // 2. set inputs
    for (size_t i = 2, j = 0; i < 2 + interp.inputs_size(); i++, j++)
    {
        auto desc = interp.input_desc(j);
        auto shape = interp.input_shape(j);
        auto tensor = host_runtime_tensor::create(desc.datatype, shape, hrt::pool_shared).expect("cannot create input tensor");
        auto mapped_buf = std::move(hrt::map(tensor, map_access_::map_write).unwrap());
#if USE_CACHE
        read_binary_file(argv[i], reinterpret_cast<char *>(mapped_buf.buffer().data()));
#else
        auto vec = read_binary_file<unsigned char>(argv[i]);
        memcpy(reinterpret_cast<void *>(mapped_buf.buffer().data()), reinterpret_cast<void *>(vec.data()), vec.size());
        // dump("app dump input vector", (volatile float *)vec.data(), 32);
#endif
        auto ret = mapped_buf.unmap();
        ret = hrt::sync(tensor, sync_op_t::sync_write_back, true);
        if (!ret.is_ok())
        {
            std::cerr << "hrt::sync failed" << std::endl;
            std::abort();
        }

        // dump("app dump input block", (volatile float *)block.virtual_address, 32);
        interp.input_tensor(j, tensor).expect("cannot set input tensor");
    }

    // 3. set outputs
    for (size_t i = 0; i < interp.outputs_size(); i++)
    {
        auto desc = interp.output_desc(i);
        auto shape = interp.output_shape(i);
        auto tensor = host_runtime_tensor::create(desc.datatype, shape, hrt::pool_shared).expect("cannot create output tensor");
        interp.output_tensor(i, tensor).expect("cannot set output tensor");
    }

    // 4. run
    auto start = std::chrono::steady_clock::now();
    interp.run().expect("error occurred in running model");
    auto stop = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(stop - start).count();
    std::cout << "interp run: " << duration << " ms, fps = " << 1000 / duration << std::endl;

    // 5. get outputs
    for (int i = 2 + interp.inputs_size(), j = 0; i < argc; i++, j++)
    {
        auto out = interp.output_tensor(j).expect("cannot get output tensor");
        auto mapped_buf = std::move(hrt::map(out, map_access_::map_read).unwrap());
        auto expected = read_binary_file<unsigned char>(argv[i]);

        // 6. compare
        int ret = memcmp((void *)mapped_buf.buffer().data(), (void *)expected.data(), expected.size());
        if (!ret)
        {
            std::cout << "compare output " << j << " Pass!" << std::endl;
        }
        else
        {
            auto cos = cosine((const float *)mapped_buf.buffer().data(), (const float *)expected.data(), expected.size()/sizeof(float));
            std::cerr << "compare output " << j << " Fail: cosine similarity = " << cos << std::endl;
        }
    }

    return 0;
}