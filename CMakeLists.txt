# CMakeList.txt: проект CMake для NeRF++ 
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(NeRF++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
find_package(OpenCV 4.6 REQUIRED)

include("CMakeLists.Files.txt")

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

source_group("Headers" FILES ${HEADERS})
set(SOURCES ${SOURCES} ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${LIBS})

if (MSVC) 
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET NeRF++
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:NeRF++>)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /openmp")
endif (MSVC)