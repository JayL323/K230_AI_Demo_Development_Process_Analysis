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
#include <iostream>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <thread>
#include <chrono>
#include "utils.h"
#include "vi_vo.h"
#include "face_detection.h"
#include "face_recognition.h"

using std::cerr;
using std::cout;
using std::endl;

std::atomic<bool> isp_stop(false);
std::atomic<bool> reg_stop(false);
int flags;

//设置终端属性
void set_terminal_mode(bool buffered) {
    struct termios t;
    tcgetattr(STDIN_FILENO, &t);

    if (buffered) {
        t.c_lflag |= ICANON | ECHO;     // 启用缓冲和回显,输入时按回车结束
    } else {
        t.c_lflag &= ~(ICANON | ECHO);  // 禁用缓冲和回显,输入时读取一个字符解决
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &t);
}

void set_read_block_mode(bool blocked)
{
    if (flags != -1)
    {
        if (blocked)
        {   
            flags &= ~O_NONBLOCK;
            fcntl(STDIN_FILENO, F_SETFL, flags);   //设置为阻塞模式，输入时，需等待输入完成再进行其它操作
        }
        else
        {
            
            flags |= O_NONBLOCK;
            fcntl(STDIN_FILENO, F_SETFL, flags);   //设置为非阻塞模式，输入时，读一次数据，没有数据则继续其它操作
        }
    }
}

void print_usage(const char *name)
{
    cout << "Usage: " << name << "<kmodel_det> <det_thres> <nms_thres> <kmodel_recg> <max_register_face> <recg_thres> <input_mode> <debug_mode> <db_dir>" << endl
         << "Options:" << endl
         << "  kmodel_det               人脸检测kmodel路径\n"
         << "  det_thres                人脸检测阈值\n"
         << "  nms_thres                人脸检测nms阈值\n"
         << "  kmodel_recg              人脸识别kmodel路径\n"
         << "  max_register_face        人脸识别数据库最大容量\n"
         << "  recg_thres               人脸识别阈值\n"
         << "  input_mode               本地图片(图片路径)/ 摄像头(None) \n"
         << "  debug_mode               是否需要调试，0、1、2分别表示不调试、耗时统计调试、预处理调试\n"
         << "  db_dir                   数据库目录\n"
         << "\n"
         << endl;
}

void video_proc(char *argv[])
{
    vivcap_start();
    // 设置osd参数
    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;       //osd
    memset(&vf_info, 0, sizeof(vf_info));
    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory,get isp memory
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    //only for face reg
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (flags == -1)
    {
        cerr << "Failed to get flags for stdin." << endl;
    }
    set_terminal_mode(false);
    set_read_block_mode(false);

    FaceDetection face_det(argv[1], atof(argv[2]),atof(argv[3]), {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[8]));
    
    int max_register_face = atoi(argv[5]);
    float recg_thres = atof(argv[6]);
    FaceRecognition face_recg(argv[4],atoi(argv[5]),recg_thres, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[8]));
    face_recg.database_init(argv[9]);

    vector<FaceDetectionInfo> det_results;
    while (!isp_stop)
    {       
        ScopedTiming st("total time", 1);
        {
            ScopedTiming st("read capture", atoi(argv[8]));
            // 从vivcap中读取一帧图像到dump_info
            memset(&dump_info, 0, sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }

        {
            ScopedTiming st("isp copy", atoi(argv[8]));
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
            kd_mpi_sys_munmap(vbvaddr, size);
        }

        det_results.clear();

        face_det.pre_process();
        face_det.inference();
        face_det.post_process({SENSOR_WIDTH, SENSOR_HEIGHT}, det_results);

        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        char ch;
        if (read(STDIN_FILENO, &ch, 1) > 0) 
        {
            if (ch == 'i')      //for i key
            {
                float max_area_face = 0;
                int max_id_face = -1;
                for (int i = 0; i < det_results.size(); ++i)
                {
                    float area_i = det_results[i].bbox.w * det_results[i].bbox.h;
                    if (area_i > max_area_face)
                    {
                        max_area_face = area_i;
                        max_id_face = i;
                    }
                }
           
                //***for face recg***
                face_recg.pre_process(det_results[max_id_face].sparse_kps.points);
                face_recg.inference();

                FaceRecognitionInfo recg_result;
                face_recg.database_search(recg_result); 
                face_recg.draw_result(osd_frame,det_results[max_id_face].bbox,recg_result,false);

                string ret_name = "unknown";
                if(recg_result.score>recg_thres)
                    ret_name = recg_result.name;
                
                set_terminal_mode(true);
                set_read_block_mode(true);
                if(ret_name == "unknown" && face_recg.valid_register_face_ < max_register_face)
                {    
                    face_recg.database_insert(argv[9]);
                }
                else
                {
                    cerr<<"registration failed"<<endl;
                    if(ret_name != "unknown")
                    {
                        cerr<<"face registered"<<endl;
                    }
                    else if(face_recg.valid_register_face_ > max_register_face)
                    {
                        cerr<<"face database full"<<endl;
                    }
                    
                }
                std::this_thread::sleep_for(std::chrono::seconds(3));
                set_read_block_mode(false);
                set_terminal_mode(false);
            }
            else if (ch == 'r')        //for r key
            {
                face_recg.database_reset(argv[9]);
                std::this_thread::sleep_for(std::chrono::seconds(3));
            }
            else if(ch == 27)         //for ESC key
            {
                reg_stop = true;
            }
        }
        else
        {
            for (int i = 0; i < det_results.size(); ++i)
            {
                //***for face recg***
                face_recg.pre_process(det_results[i].sparse_kps.points);
                face_recg.inference();

                FaceRecognitionInfo recg_result;
                face_recg.database_search(recg_result); 
                face_recg.draw_result(osd_frame,det_results[i].bbox,recg_result,false);
            }
        }
        

        {
            ScopedTiming st("osd copy", atoi(argv[8]));
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            // 显示通道插入帧
            kd_mpi_vo_chn_insert_frame(osd_id + 3, &vf_info); // K_VO_OSD0
            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
            }
        }
    }

    vo_osd_release_block();
    vivcap_stop();

    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }
}

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Press 'i' to register." << std::endl;
    std::cout << "Press 'r' to reset." << std::endl;
    std::cout << "Press 'ESC' to exit." << std::endl;
    if (argc != 10)
    {
        print_usage(argv[0]);
        return -1;
    }

    if (strcmp(argv[7], "None") == 0)
    {
        std::thread thread_isp(video_proc, argv);
        while(!reg_stop)
        {
            usleep(10000);
        }
        
        isp_stop = true;
        thread_isp.join();
        set_read_block_mode(true);
        set_terminal_mode(true);
    }
    else
    {   
        // If necessary, it can be realized according to face_verification
    }
    return 0;
}