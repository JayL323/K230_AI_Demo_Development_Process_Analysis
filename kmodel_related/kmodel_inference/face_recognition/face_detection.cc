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
#include <algorithm>
#include "face_detection.h"
#include "k230_math.h"

extern float kAnchors320[4200][4];
extern float kAnchors640[16800][4];
static float (*g_anchors)[4];

cv::Scalar color_list_for_det[] = {
    cv::Scalar(0, 0, 255),
    cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 255),
    cv::Scalar(0, 255, 0),
    cv::Scalar(255, 0, 0)};

cv::Scalar color_list_for_osd_det[] = {
    cv::Scalar(255, 0, 0, 255),
    cv::Scalar(255, 0, 255, 255),
    cv::Scalar(255, 255, 0, 255),
    cv::Scalar(255, 0, 255, 0),
    cv::Scalar(255, 255, 0, 0)};

bool nms_comparator(NMSRoiObj &a, NMSRoiObj &b)
{
    return a.confidence > b.confidence;
}

// for image
FaceDetection::FaceDetection(const char *kmodel_file, float obj_thresh, float nms_thresh, const int debug_mode) : obj_thresh_(obj_thresh), AIBase(kmodel_file, "FaceDetection", debug_mode)
{
    model_name_ = "FaceDetection";
    nms_thresh_ = nms_thresh;

    int net_len = input_shapes_[0][2]; // input_shapes_[0][2]==input_shapes_[0][3]
    g_anchors = (net_len == 320 ? kAnchors320 : kAnchors640);
    objs_num_ = output_shapes_[0][1];

    ai2d_out_tensor_ = get_input_tensor(0);
}

// for video
FaceDetection::FaceDetection(const char *kmodel_file, float obj_thresh, float nms_thresh, FrameCHWSize isp_shape, uintptr_t vaddr, uintptr_t paddr, const int debug_mode) : obj_thresh_(obj_thresh), AIBase(kmodel_file, "FaceDetection", debug_mode)
{
    model_name_ = "FaceDetection";
    nms_thresh_ = nms_thresh;
    
    int net_len = input_shapes_[0][2]; 
    g_anchors = (net_len == 320 ? kAnchors320 : kAnchors640);
    objs_num_ = output_shapes_[0][1];
    vaddr_ = vaddr;

    // ai2d_in_tensor to isp
    isp_shape_ = isp_shape;
    dims_t in_shape{1, isp_shape.channel, isp_shape.height, isp_shape.width};
    int isp_size = isp_shape.channel * isp_shape.height * isp_shape.width;
    ai2d_in_tensor_ = hrt::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("create ai2d input tensor failed");
    ai2d_out_tensor_ = get_input_tensor(0);

    // fixed padding resize param
    Utils::padding_resize_one_side(isp_shape, {input_shapes_[0][3], input_shapes_[0][2]}, ai2d_builder_, ai2d_in_tensor_, ai2d_out_tensor_, cv::Scalar(104, 117, 123));
}

// ai2d for image
void FaceDetection::pre_process(cv::Mat ori_img)
{
    ScopedTiming st(model_name_ + " pre_process image", debug_mode_);
    std::vector<uint8_t> chw_vec;
    Utils::bgr2rgb_and_hwc2chw(ori_img, chw_vec);
    Utils::padding_resize_one_side({ori_img.channels(), ori_img.rows, ori_img.cols}, chw_vec, {input_shapes_[0][3], input_shapes_[0][2]}, ai2d_out_tensor_, cv::Scalar(104, 117, 123));
	if (debug_mode_ > 1)
	{
		auto vaddr_out_buf = ai2d_out_tensor_.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_read).unwrap().buffer();
		unsigned char *output = reinterpret_cast<unsigned char *>(vaddr_out_buf.data());
		Utils::dump_color_image("FaceDetection_input_padding.png",{input_shapes_[0][3],input_shapes_[0][2]},output);
	}
}

// ai2d for video
void FaceDetection::pre_process()
{
    ScopedTiming st(model_name_ + " pre_process video", debug_mode_);
    size_t isp_size = isp_shape_.channel * isp_shape_.height * isp_shape_.width;
    auto buf = ai2d_in_tensor_.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
    memcpy(reinterpret_cast<char *>(buf.data()), (void *)vaddr_, isp_size);
    hrt::sync(ai2d_in_tensor_, sync_op_t::sync_write_back, true).expect("sync write_back failed");
    ai2d_builder_->invoke(ai2d_in_tensor_, ai2d_out_tensor_).expect("error occurred in ai2d running");

	if (debug_mode_ > 1)
	{
		auto vaddr_out_buf = ai2d_out_tensor_.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_read).unwrap().buffer();
		unsigned char *output = reinterpret_cast<unsigned char *>(vaddr_out_buf.data());
		Utils::dump_color_image("FaceDetection_input_padding.png",{input_shapes_[0][3],input_shapes_[0][2]},output);
	}
}

void FaceDetection::inference()
{
    this->run();
    this->get_output();
}

void FaceDetection::post_process(FrameSize frame_size, vector<FaceDetectionInfo> &results)
{
	ScopedTiming st(model_name_ + " post_process", debug_mode_);
	filter_confs(p_outputs_[1]);
	filter_locs(p_outputs_[0]);
	filter_landms(p_outputs_[2]);
	
	std::sort(confs_.begin(), confs_.end(), nms_comparator);
	nms(results);
	transform_result_to_src_size(frame_size, results);
}

void FaceDetection::draw_result(cv::Mat& src_img,vector<FaceDetectionInfo>& results, bool pic_mode)
{   
    int src_w = src_img.cols;
    int src_h = src_img.rows;
    int max_src_size = std::max(src_w,src_h);
    for (int i = 0; i < results.size(); ++i)
    {
        auto& l = results[i].sparse_kps;
        for (uint32_t ll = 0; ll < 5; ll++)
        {
            if(pic_mode)
            {
                int32_t x0 = l.points[2 * ll + 0];
                int32_t y0 = l.points[2 * ll + 1];
                cv::circle(src_img, cv::Point(x0, y0), 2, color_list_for_det[ll], 4);  
            }
            else
            {
                int32_t x0 = l.points[2 * ll]/isp_shape_.width*src_w;
                int32_t y0 = l.points[2 * ll+1]/isp_shape_.height*src_h;
                cv::circle(src_img, cv::Point(x0, y0), 4, color_list_for_osd_det[ll], 8); 
            }
        }

        auto& b = results[i].bbox;
        char text[10];
        sprintf(text, "%.2f", results[i].score);
        if(pic_mode)
        {
            cv::rectangle(src_img, cv::Rect(b.x, b.y , b.w, b.h), cv::Scalar(255, 255, 255), 2, 2, 0);
            cv::putText(src_img, text , {b.x,b.y}, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, 8, 0);
		}
        else
        {
            int x = b.x / isp_shape_.width * src_w;
            int y = b.y / isp_shape_.height * src_h;
            int w = b.w / isp_shape_.width * src_w;
            int h = b.h / isp_shape_.height * src_h;
            cv::rectangle(src_img, cv::Rect(x, y , w, h), cv::Scalar(255,255, 255, 255), 6, 2, 0);
			cv::putText(src_img, text , {x,y}, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,0, 255, 255), 1, 8, 0);
        }        
    }
}

/********************根据检测阈值kmodel数据结果***********************/
void FaceDetection::filter_confs(float *conf)
{
	NMSRoiObj inter_obj;
	confs_.clear();
	for (uint32_t roi_index = 0; roi_index < objs_num_; roi_index++)
	{
		float score = conf[roi_index * CONF_SIZE + 1];
		if (score > obj_thresh_)
		{
			inter_obj.ori_roi_index = roi_index;
			inter_obj.before_sort_conf_index = confs_.size();
			inter_obj.confidence = score;
			confs_.push_back(inter_obj);
		}
	}
}

void FaceDetection::filter_locs(float *loc)
{
	boxes_.clear();
	boxes_.resize(confs_.size());
	int roi_index = 0;
	for (uint32_t conf_index = 0; conf_index < boxes_.size(); conf_index++)
	{
		roi_index = confs_[conf_index].ori_roi_index;
		int start = roi_index * LOC_SIZE;
		for (int i = 0; i < LOC_SIZE; ++i)
		{
			boxes_[conf_index][i] = loc[start + i];
		}
	}
}

void FaceDetection::filter_landms(float *landms)
{
	landmarks_.clear();
	landmarks_.resize(confs_.size());
	int roi_index = 0;
	for (uint32_t conf_index = 0; conf_index < boxes_.size(); conf_index++)
	{
		roi_index = confs_[conf_index].ori_roi_index;
		int start = roi_index * LAND_SIZE;
		for (int i = 0; i < LAND_SIZE; ++i)
		{
			landmarks_[conf_index][i] = landms[start + i];
		}
	}
}

/********************根据anchor解码检测框、五官点***********************/
Bbox FaceDetection::decode_box(int obj_index)
{
	float cx, cy, w, h;

	int box_index = confs_[obj_index].before_sort_conf_index;
	int anchor_index = confs_[obj_index].ori_roi_index;

	cx = boxes_[box_index][0];
	cy = boxes_[box_index][1];
	w = boxes_[box_index][2];
	h = boxes_[box_index][3];
	cx = g_anchors[anchor_index][0] + cx * 0.1 * g_anchors[anchor_index][2];
	cy = g_anchors[anchor_index][1] + cy * 0.1 * g_anchors[anchor_index][3];
	w = g_anchors[anchor_index][2] * std::exp(w * 0.2);
	h = g_anchors[anchor_index][3] * std::exp(h * 0.2);
	Bbox box;
	box.x = cx - w / 2;
	box.y = cy - h / 2;
	box.w = w;
	box.h = h;
	return box;
}

SparseLandmarks FaceDetection::decode_landmark(int obj_index)
{
	SparseLandmarks landmark;
	int landm_index = confs_[obj_index].before_sort_conf_index;
	int anchor_index = confs_[obj_index].ori_roi_index;
	for (uint32_t ll = 0; ll < 5; ll++)
	{
		landmark.points[2 * ll + 0] = g_anchors[anchor_index][0] + landmarks_[landm_index][2 * ll + 0] * 0.1 * g_anchors[anchor_index][2];
		landmark.points[2 * ll + 1] = g_anchors[anchor_index][1] + landmarks_[landm_index][2 * ll + 1] * 0.1 * g_anchors[anchor_index][3];
	}
	return landmark;
}

/********************iou计算***********************/
float FaceDetection::overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float FaceDetection::box_intersection(Bbox a, Bbox b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);

	if (w < 0 || h < 0)
		return 0;
	return w * h;
}

float FaceDetection::box_union(Bbox a, Bbox b)
{
	float i = box_intersection(a, b);
	float u = a.w * a.h + b.w * b.h - i;

	return u;
}

float FaceDetection::box_iou(Bbox a, Bbox b)
{
	return box_intersection(a, b) / box_union(a, b);
}

/********************nms***********************/
void FaceDetection::nms(vector<FaceDetectionInfo> &results)
{
	// nms
	for (int conf_index = 0; conf_index < confs_.size(); ++conf_index)
	{
		if (confs_[conf_index].confidence < 0)
			continue;

		FaceDetectionInfo obj;
		obj.bbox = decode_box(conf_index);
		obj.sparse_kps = decode_landmark(conf_index);
		obj.score = confs_[conf_index].confidence;
		results.push_back(obj);

		for (int j = conf_index + 1; j < confs_.size(); ++j)
		{
			if (confs_[j].confidence < 0)
				continue;
			Bbox b = decode_box(j);
			if (box_iou(obj.bbox, b) >= nms_thresh_) // iou大于nms阈值的，之后循环将会忽略
				confs_[j].confidence = -1;
		}
	}
}

/********************将人脸检测结果变换到原图***********************/
void FaceDetection::transform_result_to_src_size(FrameSize &frame_size, vector<FaceDetectionInfo> &results)
{
	// transform result to dispaly size
	int max_src_size = std::max(frame_size.width, frame_size.height);
	for (int i = 0; i < results.size(); ++i)
	{
		auto &l = results[i].sparse_kps;
		for (uint32_t ll = 0; ll < 5; ll++)
		{
			l.points[2 * ll + 0] = l.points[2 * ll + 0] * max_src_size;
			l.points[2 * ll + 1] = l.points[2 * ll + 1] * max_src_size;
		}

		auto &b = results[i].bbox;
		float x0 = b.x * max_src_size;
		float x1 = (b.x + b.w) * max_src_size;
		float y0 = b.y * max_src_size;
		float y1 = (b.y + b.h) * max_src_size;
		x0 = std::max(float(0), std::min(x0, float(frame_size.width)));
		x1 = std::max(float(0), std::min(x1, float(frame_size.width)));
		y0 = std::max(float(0), std::min(y0, float(frame_size.height)));
		y1 = std::max(float(0), std::min(y1, float(frame_size.height)));
		b.x = x0;
		b.y = y0;
		b.w = x1 - x0;
		b.h = y1 - y0;
	}
}

FaceDetection::~FaceDetection()
{
}
