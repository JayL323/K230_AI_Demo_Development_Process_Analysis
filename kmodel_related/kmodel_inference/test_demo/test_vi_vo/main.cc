#include <iostream>
#include "utils.h"
#include "vi_vo.h"

using std::cout;
using std::endl;

int main()
{
    //sensor有2路输出，
    //第0路用于显示，输出大小设置1080p,图像格式为PIXEL_FORMAT_YVU_PLANAR_420,直接绑定到vo
    //第1路用于AI计算，输出大小720p,图像格式为PIXEL_FORMAT_BGR_888_PLANAR（实际为rgb,chw,uint8）
    vivcap_start();    

    // osd大小为1080p,图像格式为PIXEL_FORMAT_ARGB_8888
    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;
    memset(&vf_info, 0, sizeof(vf_info));
    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory for sensor，将sensor对应AI的一路数据拷贝到vaddr，以备AI使用
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    for (int i = 0; i < 600; ++i)
    {
        {
            // read capture：从vicap通道1中dump一帧图像到dump_info
            memset(&dump_info, 0, sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }

        {
            // sensor copy：将dump_info从物理地址映射到虚拟地址，并把虚拟地址数据拷贝到固定地址，以备后续使用
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);
            kd_mpi_sys_munmap(vbvaddr, size);
            if (i == 5)
                Utils::dump_color_image("ori_5.png", {SENSOR_WIDTH, SENSOR_HEIGHT}, reinterpret_cast<unsigned char *>(vaddr));
        }

        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0)); // argb
        {
            // draw result：将框或文字画到osd对应大小cv::Mat
            if (i % 10 == 0)
                cv::rectangle(osd_frame, cv::Rect(10, 10, 100, 100), cv::Scalar(255, 255, 0, 0), 2, 2, 0);
            else
                cv::putText(osd_frame, "hello world!", {50, 50}, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0, 255), 1, 8, 0);
        }

        {
           // insert osd：将上述cv::Mat（帧）插入到vo指定通道
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            kd_mpi_vo_chn_insert_frame(osd_id + 3, &vf_info);  
        }

        {
            // release frame：释放从sensor读取的帧
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

    return 0;
}