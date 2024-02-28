#include <iostream>
#include <nncase/runtime/interpreter.h>
#include "utils.h"
#include "vi_vo.h"


using std::cout;
using std::endl;
using namespace nncase::runtime;

int main()
{
    vivcap_start();

    // 设置osd参数
    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;
    memset(&vf_info, 0, sizeof(vf_info));
    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory for sensor
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    /**********************padding resize预处理demo*************************/
    //功能：将从sensor原图（1280,720），chw, rgb888-> （640,640），chw, rgb888
    // create ai2d_in_tensor
    dims_t ai2d_in_shape{1, SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH};
    int ai2d_in_size = ai2d_in_shape[0] * ai2d_in_shape[1] * ai2d_in_shape[2] * ai2d_in_shape[3];
    runtime_tensor ai2d_in_tensor = hrt::create(typecode_t::dt_uint8, ai2d_in_shape, hrt::pool_shared).expect("create ai2d input tensor failed");
    
    // create ai2d_out_tensor
    dims_t ai2d_out_shape{1, 3, 640, 640};
    int ai2d_out_size = ai2d_out_shape[0] * ai2d_out_shape[1] * ai2d_out_shape[2] * ai2d_out_shape[3];
    runtime_tensor ai2d_out_tensor = hrt::create(typecode_t::dt_uint8, ai2d_out_shape, hrt::pool_shared).expect("create ai2d input tensor failed");

    // create ai2d_builder，执行预处理操作
    std::unique_ptr<ai2d_builder> ai2d_builder;       //参数不变时，ai2d_buidler可以反复使用，无需重新创建
    
    // set ai2d_builder：input_size, output_size，padding
    Utils::padding_resize_one_side({SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, {ai2d_out_shape[3], ai2d_out_shape[2]}, ai2d_builder, ai2d_in_tensor, ai2d_out_tensor, cv::Scalar(104, 117, 123));
    for (int i = 0; i < 100; ++i)
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
            if (i == 10)
                Utils::dump_color_image("ori_10.png", {SENSOR_WIDTH, SENSOR_HEIGHT}, reinterpret_cast<unsigned char *>(vaddr));
        }

        {
            // set ai2d input：将sensor原图拷贝到ai2d输入
            auto buf = ai2d_in_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
            memcpy(reinterpret_cast<char *>(buf.data()), (void *)vaddr, ai2d_in_size);
            hrt::sync(ai2d_in_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
            ai2d_builder->invoke(ai2d_in_tensor,ai2d_out_tensor).expect("error occurred in ai2d running");

            if (i == 10)
            {
                auto vaddr_out_buf = ai2d_out_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_read).unwrap().buffer();
                unsigned char *output = reinterpret_cast<unsigned char *>(vaddr_out_buf.data());
                Utils::dump_color_image("ori_padding_resize_10.png",{ai2d_out_shape[3],ai2d_out_shape[2]},output);
            }   
           
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


    /**********************crop resize预处理demo*************************/
    //功能：从sensor原图（1280,720）,chw, rgb888，随机crop一部分区域进行resize
    //（w,h）,chw, rgb888 -> (320,320), chw, rgb888
    // create crop_reisze_in_tensor
    dims_t crop_reisze_in_shape{1, SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH};
    int crop_reisze_in_size = crop_reisze_in_shape[0] * crop_reisze_in_shape[1] * crop_reisze_in_shape[2] * crop_reisze_in_shape[3];
    runtime_tensor crop_reisze_in_tensor = hrt::create(typecode_t::dt_uint8, crop_reisze_in_shape, hrt::pool_shared).expect("create ai2d input tensor failed");
    
    // create crop_reisze_out_tensor
    dims_t crop_reisze_out_shape{1, 3, 640, 640};
    int crop_reisze_out_size = crop_reisze_out_shape[0] * crop_reisze_out_shape[1] * crop_reisze_out_shape[2] * crop_reisze_out_shape[3];
    runtime_tensor crop_reisze_out_tensor = hrt::create(typecode_t::dt_uint8, crop_reisze_out_shape, hrt::pool_shared).expect("create ai2d input tensor failed");

    for (int i = 0; i < 100; ++i)
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
            if (i == 20)
                Utils::dump_color_image("ori_20.png", {SENSOR_WIDTH, SENSOR_HEIGHT}, reinterpret_cast<unsigned char *>(vaddr));
        }

        {
            // set ai2d input：将sensor原图拷贝到ai2d输入
            auto buf = crop_reisze_in_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
            memcpy(reinterpret_cast<char *>(buf.data()), (void *)vaddr, crop_reisze_in_size);
            hrt::sync(crop_reisze_in_tensor, sync_op_t::sync_write_back, true).expect("sync write_back failed");
            Bbox crop_info;
            crop_info.x = 0;
            crop_info.y = 0;
            crop_info.w = i * 5 + 150;
            crop_info.h = i * 5 + 150;
            Utils::crop_resize(crop_info, ai2d_builder, crop_reisze_in_tensor, crop_reisze_out_tensor);

            if (i == 20)
            {
                auto vaddr_out_buf = crop_reisze_out_tensor.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_read).unwrap().buffer();
                unsigned char *output = reinterpret_cast<unsigned char *>(vaddr_out_buf.data());
                Utils::dump_color_image("ori_crop_resize_20.png",{crop_reisze_out_shape[3],crop_reisze_out_shape[2]},output);
            }   
           
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