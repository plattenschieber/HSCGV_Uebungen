########################################################################
#  CUDA
# see: http://forums.nvidia.com/index.php?showtopic=29539
########################################################################
win32 {
    INCLUDEPATH += $(CUDA_INC_DIR)
    QMAKE_LIBDIR += $(CUDA_LIB_DIR)
    LIBS += -lcudart

    cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
}


unix:!macx:{
    CUDA_DIR = /usr/local/cuda-5.5
    QMAKE_LIBDIR += $$CUDA_DIR/lib64
}
macx:{
    CUDA_DIR = /Developer/NVIDIA/CUDA-7.0
    QMAKE_LIBDIR += $$CUDA_DIR/lib
}
unix {
# auto-detect CUDA path
    INCLUDEPATH += $$CUDA_DIR/include
    LIBS += -lcudart
    NVCCFLAGS="-use_fast_math --ptxas-options=-v"
    NVCCFLAGS+="-Xptxas -fastimul -arch=sm_20" # 24 bit integer multiplication should be sufficient
    debug:NVCCFLAGS+="-g"
    release:NVCCFLAGS+="-O3"
    macx:NVCCFLAGS+="-m64"
    FLAGS=$$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') $$join(DEFINES,'" -D"','-D"','"')
    cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc -c $$NVCCFLAGS -Xcompiler $$FLAGS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

    cuda.depend_command = $$CUDA_DIR/bin/nvcc -M $$NVCCFLAGS -Xcompiler $$FLAGS ${QMAKE_FILE_NAME} | sed "'s,^.*: ,,'" | tr -d '\\\\\\n'
}
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_COMPILERS += cuda
########################################################################
