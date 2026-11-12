# CMakeList.txt: проект CMake для NeRF++ 
cmake_minimum_required(VERSION 4.4 FATAL_ERROR)
project(NeRF++)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NERFPP_CUDA_ARCHITECTURES "75 86" CACHE STRING "CUDA architectures used by NeRF++")

set(TORCH_CUDA_ARCH_LIST "7.5 8.6")

# CUDA 13.x CCCL requires the standard-conforming MSVC preprocessor
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    )
endif()


find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
#if (USE_COLMAP)
	#set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
	find_package(gflags REQUIRED)
	set(OPENSSL_ROOT_DIR "C:/vcpkg/installed/x64-windows")
	find_package(OpenSSL REQUIRED)
	#add_definitions(-USE_COLMAP)
	find_package(CURL REQUIRED)      # Resolve CURL::libcurl_shared
	find_package(Qt5 COMPONENTS Core OpenGL REQUIRED) 		#for Colmap
	find_package(colmap REQUIRED)
	add_compile_definitions(GLOG_USE_GLOG_EXPORT)
#endif()

include("CMakeLists.Files.txt")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

source_group("Headers" FILES ${HEADERS})
set(SOURCES ${SOURCES} ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${NERFPP_CUDA_ARCHITECTURES}")

if (MSVC) 
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /openmp /MP")
endif (MSVC)