cmake_minimum_required(VERSION 3.10)

# Project Name
project(maveric-slam C)

# Include directories
include_directories(include include/data/tracking)

# Source files
set(SOURCES
    src/tracking_main.c
)

# Create the executable
add_executable(tracking_main ${SOURCES})

target_link_libraries(tracking_main m)

