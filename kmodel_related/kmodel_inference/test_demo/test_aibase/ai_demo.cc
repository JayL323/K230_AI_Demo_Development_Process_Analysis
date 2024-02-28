/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "ai_demo.h"
#include "k230_math.h"

// for image
AIDemo::AIDemo(const char *kmodel_file, const int debug_mode) : AIBase(kmodel_file,"AIDemo", debug_mode)
{
    model_name_ = "AIDemo";

    for (int i = 0;i<input_shapes_.size();++i)
    {    
        runtime_tensor tensor = get_input_tensor(i);
        input_tensors_.push_back(tensor);            //input_tensors_[i]和get_input_tensor(i)指向同一块内存
    }
}

void AIDemo::read_binary_file(const char *file_name, char *buffer)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    ifs.read(buffer, len);
    ifs.close();
}

void AIDemo::pre_process(char *argv[])
{
    // need to implement oneself
    for (int i = 0 ;i<input_shapes_.size(); ++i)
    {
        auto in_buf = input_tensors_[i].impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
        read_binary_file(argv[i+2], reinterpret_cast<char *>(in_buf.data()));
        hrt::sync(input_tensors_[i], sync_op_t::sync_write_back, true).expect("sync write_back failed");
    }
    
}

void AIDemo::inference()
{
    this->run();
    this->get_output();
}

void AIDemo::post_process(char *argv[])
{
    // need to implement oneself
    int start_out_index = 2 + input_shapes_.size();    //2 for elf and kmodel
    for (int i = 0; i<p_outputs_.size(); ++i)
    {
        auto expected = Utils::read_binary_file<unsigned char>(argv[start_out_index + i]);
        int ret = memcmp((void *)p_outputs_[i], (void *)expected.data(), expected.size());
        if (!ret)
        {
            std::cout << "compare output " << i << " Pass!" << std::endl;
        }
        else
        {
            auto cos = cosine((const float *)p_outputs_[i], (const float *)expected.data(), expected.size()/sizeof(float));
            std::cerr << "compare output " << i << " Fail: cosine similarity = " << cos << std::endl;
        }
    }
    
}



AIDemo::~AIDemo()
{
}
