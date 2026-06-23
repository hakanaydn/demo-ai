cmake_minimum_required(VERSION 3.14)
project(add_model CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
  COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TF_LIB_DIR
)

execute_process(
  COMMAND python3 -c "
import tensorflow as tf
flags = tf.sysconfig.get_compile_flags()
for f in flags:
    print(f)
"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TF_COMPILE_FLAGS
)

string(REPLACE "\n" ";" TF_COMPILE_FLAGS_LIST ${TF_COMPILE_FLAGS})

add_executable(add_app main.cpp)
target_compile_options(add_app PRIVATE ${TF_COMPILE_FLAGS_LIST})
target_include_directories(add_app PRIVATE ${TF_LIB_DIR}/include)
target_link_libraries(add_app PRIVATE
  ${TF_LIB_DIR}/libtensorflow_cc.so.2
  ${TF_LIB_DIR}/libtensorflow_framework.so.2
  dl
)
