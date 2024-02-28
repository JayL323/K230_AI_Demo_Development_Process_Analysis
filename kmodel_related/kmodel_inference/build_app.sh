#!/bin/bash
set -x

# set cross build toolchain
export PATH=$PATH:/opt/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin/

clear
rm -rf out
mkdir out
pushd out
cmake -DCMAKE_BUILD_TYPE=Release                 \
      -DCMAKE_INSTALL_PREFIX=`pwd`               \
      -DCMAKE_TOOLCHAIN_FILE=cmake/Riscv64.cmake \
      ..

make -j && make install
popd

k230_bin=`pwd`/k230_bin
rm -rf ${k230_bin}
mkdir -p ${k230_bin}
k230_kmodel=`pwd`/../kmodel_export/k230_kmodel
k230_utils=`pwd`/../kmodel_export/k230_utils

if [ "$1" == "debug" ]; then
    echo "Debug mode"
    mkdir -p ${k230_bin}/debug

    cp -a ${k230_utils}/* ${k230_bin}/debug
    cp -a ${k230_kmodel}/*.kmodel ${k230_bin}/debug
    cp -a shell/face_detect_main_nncase_with_aibase.sh ${k230_bin}/debug
    cp -a shell/face_detect_main_nncase.sh ${k230_bin}/debug
    cp -a shell/face_recognize_main_nncase.sh ${k230_bin}/debug

    if [ -f out/bin/main_nncase.elf ]; then
      cp out/bin/main_nncase.elf ${k230_bin}/debug
    fi

    if [ -f out/bin/test_scoped_timing.elf ]; then
      cp out/bin/test_scoped_timing.elf ${k230_bin}/debug
    fi

    if [ -f out/bin/test_vi_vo.elf ]; then
      cp out/bin/test_vi_vo.elf ${k230_bin}/debug
    fi

    if [ -f out/bin/test_utils.elf ]; then
      cp out/bin/test_utils.elf ${k230_bin}/debug
    fi

    if [ -f out/bin/test_aibase.elf ]; then
      cp out/bin/test_aibase.elf ${k230_bin}/debug
    fi
else
    echo "Release mode"
fi

if [ -f out/bin/face_detection.elf ]; then
      mkdir -p ${k230_bin}/face_detect
      cp -a shell/face_detect_image.sh ${k230_bin}/face_detect
      cp -a shell/face_detect_isp.sh ${k230_bin}/face_detect
      cp -a ${k230_utils}/face_detect.jpg ${k230_bin}/face_detect
      cp -a ${k230_kmodel}/face_detect_640.kmodel ${k230_bin}/face_detect
      cp out/bin/face_detection.elf ${k230_bin}/face_detect
fi

if [ -f out/bin/face_recognition.elf ]; then
      mkdir -p ${k230_bin}/face_recognize
      mkdir -p ${k230_bin}/face_recognize/db
      cp -a shell/face_recognize_isp.sh ${k230_bin}/face_recognize
      cp -a ${k230_kmodel}/face_detect_640.kmodel ${k230_bin}/face_recognize
      cp -a ${k230_kmodel}/face_recognize.kmodel ${k230_bin}/face_recognize
      cp out/bin/face_recognition.elf ${k230_bin}/face_recognize
fi

rm -rf out
