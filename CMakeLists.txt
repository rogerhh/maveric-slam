cmake_minimum_required(VERSION 3.10)

# Project Name
project(maveric-slam C)

# Include directories
include_directories(include 
                    include/data/tracking
                    include/data/LCD
                    include/data/quantized
                    include/svd)

# Create the executable
add_executable(tracking_main
               src/tracking_main.c
               src/pnp_solver.c
               src/top_N.c)

add_executable(lcd_main
               src/lcd_main.c
               src/top_N.c)

add_executable(run_nms
               src/run_nms.c
               src/top_N.c)

add_executable(run_relin
               src/run_relin.c)

target_link_libraries(tracking_main m)
target_link_libraries(lcd_main m)
target_link_libraries(run_nms m)
target_link_libraries(run_relin m)

