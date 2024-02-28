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
#ifndef _AI_DEMO_H
#define _AI_DEMO_H

#include <iostream>
#include <vector>

#include "ai_base.h"

using std::vector;

#define LOC_SIZE 4
#define CONF_SIZE 2
#define LAND_SIZE 10
#define PI 3.1415926

/**
 * @brief AIDemo
 * 主要封装了对于kmodel的前处理、推理、后处理过程
 */
class AIDemo : public AIBase
{
public:
    /**
     * @brief AIDemo构造函数，加载kmodel,并初始化kmodel输入、输出
     * @param kmodel_file kmodel文件路径
     * @param debug_mode  0（不调试）、 1（只显示时间）、2（显示所有打印信息）
     * @return None
     */
    AIDemo(const char *kmodel_file, const int debug_mode = 1);

    /**
     * @brief AIDemo析构函数
     * @return None
     */
    ~AIDemo();

    /**
     * @brief 视频流预处理
     * @param argv 命令行参数
     * @return None
     */
    void pre_process(char *argv[]);

    /**
     * @brief kmodel推理
     * @return None
     */
    void inference();

    /**
     * @brief kmodel推理结果后处理
     * @param argv 命令行参数
     * @return None
     */
    void post_process(char *argv[]);

private:
    /**
     * @brief 读取二进制文件，并将文件内容拷贝到buffer
     * @param file_name 文件名称
     * @param buffer 数据保存地址
     * @return None
     */
    void read_binary_file(const char *file_name, char *buffer);

    /**
     * @brief 向量乘
     * @param v1 向量1
     * @param v2 向量1
     * @param size 向量长度
     * @return None
     */
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

    /**
     * @brief 相似度计算
     * @param v1 数据1
     * @param v2 数据2
     * @param v2 数据大小
     * @return None
     */
    template <typename T>
    double cosine(const T *v1, const T *v2, size_t size)
    {
        return dot(v1, v2, size) / ((sqrt(dot(v1, v1, size)) * sqrt(dot(v2, v2, size))));
    }

    vector<runtime_tensor> input_tensors_;             // kmodel输入tensor，执行kmodel输入
};

#endif