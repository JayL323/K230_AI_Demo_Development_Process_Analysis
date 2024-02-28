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
#ifndef _FACE_DETECTION_H
#define _FACE_DETECTION_H

#include <iostream>
#include <vector>
#include <array>

#include "utils.h"
#include "ai_base.h"

using std::vector;
using std::array;

#define LOC_SIZE 4
#define CONF_SIZE 2
#define LAND_SIZE 10

/**
 * @brief 用于NMS排序的roi对象
 */
typedef struct NMSRoiObj
{
	int ori_roi_index;				 // 从kmodel到得到roi原始类别索引，用于过滤locs、landms；nms时定位anchor
	int before_sort_conf_index;           // roi对象所在过滤后的conf列表索引，用于记录sort之后conf，在未sort时conf位置，用于nms时定位locs、landms
	float confidence;    // roi对象的置信度
} NMSRoiObj;

/**
 * @brief 人脸五官点
 */
typedef struct SparseLandmarks
{
    float points[10]; // 人脸五官点,依次是图片的左眼（x,y）、右眼（x,y）,鼻子（x,y）,左嘴角（x,y）,右嘴角
} SparseLandmarks;

/**
 * @brief 预测人脸roi信息
 */
typedef struct FaceDetectionInfo
{
    Bbox bbox;                  // 人脸检测框
    SparseLandmarks sparse_kps; // 人脸五官关键点
    float score;                // 人脸检测框置信度
} FaceDetectionInfo;



/**
 * @brief 人脸检测
 * 主要封装了对于每一帧图片，从预处理、运行到后处理给出结果的过程
 */
class FaceDetection : public AIBase
{
public:
    /**
     * @brief FaceDetection构造函数，加载kmodel,并初始化kmodel输入、输出和人脸检测阈值
     * @param kmodel_file kmodel文件路径
     * @param obj_thresh 人脸检测阈值，用于过滤roi
     * @param nms_thresh 人脸检测nms阈值
     * @param debug_mode  0（不调试）、 1（只显示时间）、2（显示所有打印信息）
     * @return None
     */
    FaceDetection(const char *kmodel_file, float obj_thresh,float nms_thresh, const int debug_mode = 1);

    /**
     * @brief FaceDetection构造函数，加载kmodel,并初始化kmodel输入、输出和人脸检测阈值
     * @param kmodel_file kmodel文件路径
     * @param obj_thresh  人脸检测阈值，用于过滤roi
     * @param nms_thresh 人脸检测nms阈值
     * @param isp_shape   isp输入大小（chw）
     * @param vaddr       isp对应虚拟地址
     * @param paddr       isp对应物理地址
     * @param debug_mode  0（不调试）、 1（只显示时间）、2（显示所有打印信息）
     * @return None
     */
    FaceDetection(const char *kmodel_file, float obj_thresh,float nms_thresh, FrameCHWSize isp_shape, uintptr_t vaddr, uintptr_t paddr, const int debug_mode);

    /**
     * @brief FaceDetection析构函数
     * @return None
     */
    ~FaceDetection();

    /**
     * @brief 图片预处理，（ai2d for image）
     * @param ori_img 原始图片
     * @return None
     */
    void pre_process(cv::Mat ori_img);

    /**
     * @brief 视频流预处理（ai2d for video）
     * @return None
     */
    void pre_process();

    /**
     * @brief kmodel推理
     * @return None
     */
    void inference();

    /**
     * @brief kmodel推理结果后处理
     * @param frame_size 原始图像/帧宽高，用于将结果放到原始图像大小
     * @param results 后处理之后的基于原始图像的{检测框、五官点和得分}集合
     * @return None
     */
    void post_process(FrameSize frame_size, vector<FaceDetectionInfo> &results);

     /**
     * @brief 将检测结果画到原图
     * @param src_img     原图
     * @param results     人脸检测结果
     * @param pic_mode    ture(原图片)，false(osd)
     * @return None
     */
    void draw_result(cv::Mat& src_img,vector<FaceDetectionInfo>& results, bool pic_mode = true);

private:   
    /********************根据检测阈值kmodel数据结果***********************/
    /**
     * @brief 根据检测阈值过滤roi置信度
     * @param confs    kmodel输入结果，confs
     * @return None
     */
    void filter_confs(float *confs);

    /**
     * @brief 根据检测阈值过滤roi loc
     * @param locs     kmodel输入结果，locs
     * @return None
     */
    void filter_locs(float *locs);

    /**
     * @brief 根据检测阈值过滤roi landms
     * @param landms  kmodel输入结果，landms
     * @return None
     */
    void filter_landms(float *landms);

    /********************根据anchor解码***********************/
    /**
     * @brief 根据anchor解码人脸检测框
     * @param obj_index   需要获取的roi索引
     * @return None
     */
    Bbox decode_box(int obj_index);

    /**
     * @brief 根据anchor解码人脸关键点
     * @param obj_index   需要获取的roi索引
     * @return None
     */
    SparseLandmarks decode_landmark(int obj_index);

    /********************iou计算***********************/
    /**
     * @brief 获取2个检测框重叠区域的左上角或右下角
     * @param x1   第1个检测框的中心点x或y坐标
     * @param w1   第1个检测框的宽（w）或高（h）
     * @param x2   第2个检测框的中心点x或y坐标
     * @param w2   第2个检测框的宽（w）或高（h）
     * @return 2个检测框重叠区域的宽或高
     */
    float overlap(float x1, float w1, float x2, float w2);

    /**
     * @brief 获取2个检测框重叠区域的面积
     * @param a   第1个检测框
     * @param b   第2个检测框
     * @return 2个检测框重叠区域的面积
     */
    float box_intersection(Bbox a, Bbox b);

    /**
     * @brief 获取2个检测框联合区域的面积
     * @param a   第1个检测框
     * @param b   第2个检测框
     * @return 2个检测框重叠联合区域的面积
     */
    float box_union(Bbox a, Bbox b);

    /**
     * @brief 获取2个检测框的iou
     * @param a   第1个检测框
     * @param b   第2个检测框
     * @return 2个检测框的iou
     */
    float box_iou(Bbox a, Bbox b);

    /********************nms***********************/
    /**
     * @brief nms
     * @param results     后处理之后的基于原始图像比例(0~1)的{检测框、五官点和得分}集合
     * @return None
     */
    void nms(vector<FaceDetectionInfo> &results);

    /**
     * @brief 将人脸检测结果变换到原图
     * @param frame_size  原始图像/帧宽高，用于将结果放到原始图像大小
     * @param results     后处理之后的基于原始图像的{检测框、五官点和得分}集合
     * @return None
     */
    void transform_result_to_src_size(FrameSize &frame_size, vector<FaceDetectionInfo> &results);

private:
    std::unique_ptr<ai2d_builder> ai2d_builder_; // ai2d构建器
    runtime_tensor ai2d_in_tensor_;              // ai2d输入tensor
    runtime_tensor ai2d_out_tensor_;             // ai2d输出tensor
    uintptr_t vaddr_;                            // isp的虚拟地址
    FrameCHWSize isp_shape_;                     // isp对应的地址大小

    float obj_thresh_; // 人脸检测阈值
    float nms_thresh_; // nms阈值
    int objs_num_;     // roi个数

	vector<NMSRoiObj> confs_;                    // 根据obj_thresh_过滤后的得分列表
	vector<array<float, LOC_SIZE>> boxes_;       // 根据obj_thresh_过滤后roi检测框列表
	vector<array<float, LAND_SIZE>> landmarks_;  // 根据obj_thresh_过滤后roi对应五官点列表
};

#endif