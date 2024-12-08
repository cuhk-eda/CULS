cmake_minimum_required(VERSION 3.8.0)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

function(addprefix var prefix)
    foreach( s ${ARGN} )
        list(APPEND tmp "${prefix}${s}")
    endforeach()
    set(${var} ${tmp} PARENT_SCOPE)
endfunction()

# filter out flags that are not appropriate for the compiler being used
function(target_compile_options_filtered target visibility)
    foreach( flag ${ARGN} )
        if( flag MATCHES "^-D.*" )
            target_compile_options( ${target} ${visibility} ${flag} )
        else()
            check_c_compiler_flag( ${flag} C_COMPILER_SUPPORTS__${flag} )
            if( C_COMPILER_SUPPORTS__${flag} )
                target_compile_options( ${target} ${visibility} 
                    $<$<COMPILE_LANGUAGE:C>:${flag}> )
            endif()

            check_cxx_compiler_flag( ${flag} CXX_COMPILER_SUPPORTS__${flag} )
            if( CXX_COMPILER_SUPPORTS__${flag} )
                target_compile_options( ${target} ${visibility} 
                    $<$<COMPILE_LANGUAGE:CXX>:${flag}>  )
            endif()
        endif()
    endforeach()
endfunction()

# for extracting sources and options from make output
function(extract_var SEPARATOR DEST_VARIABLE MAKE_OUTPUT)
    string(REGEX MATCH "${SEPARATOR} .* ${SEPARATOR}" TMP "${MAKE_OUTPUT}")
    string(REGEX REPLACE "${SEPARATOR} (.*) ${SEPARATOR}" "\\1" TMP "${TMP}")

    separate_arguments(TMP)

    set(${DEST_VARIABLE} ${TMP} PARENT_SCOPE)
endfunction()

function(abc_properties target visibility)
    target_include_directories(${target} ${visibility} ${ABC_ROOT_DIR}/src )
    target_compile_options_filtered(${target} ${visibility} 
        ${ABC_CFLAGS} ${ABC_CXXFLAGS} -Wno-unused-but-set-variable )
    target_link_libraries(${target} ${visibility} ${ABC_LIBS})
endfunction()



set(GPULS_CUDA_FLAGS -arch=compute_60 -code=compute_60,sm_60,sm_86)
set(GPULS_CUDA_FLAGS ${GPULS_CUDA_FLAGS} -std=c++17 -O3 --extended-lambda)
# set(GPULS_CUDA_FLAGS ${GPULS_CUDA_FLAGS} -rdc=true)
set(GPULS_CUDA_DEBUG_FLAGS )

set(CMAKE_CXX_STANDARD 17)

file(GLOB GPULS_CPU_SOURCES "src/*.cpp")
file(GLOB GPULS_GPU_SOURCES 
    "src/*.cu"
    "src/aig/*.cu"
    "src/test/*.cu"
    "src/mem/*.cu"
    "src/misc/*.cu"
    "src/algorithms/*.cu"
    "src/algorithms/map/*.cu"
    "src/algorithms/rec_opt/*.cu")

# Define debug flags
set(C_DEBUG_FLAGS "")
set(CXX_DEBUG_FLAGS "")

if(DEFINED PATCH_ABC)
    # as a patch of ABC
    message("Building in ABC patch mode ...")
    project(abcg C CXX CUDA)

    unset(PATCH_ABC)

    # handle readline library
    find_path(READLINE_ROOT_DIR
        NAMES include/readline/readline.h
    )
    find_path(READLINE_INCLUDE_DIR
        NAMES readline/readline.h
        HINTS ${READLINE_ROOT_DIR}/include
    )
    find_path(READLINE_LIBRARY_DIR
        NAMES libreadline.so
        HINTS ${READLINE_ROOT_DIR}/lib
        PATHS /usr/lib64 /usr/lib
    )

    if(READLINE_INCLUDE_DIR AND READLINE_LIBRARY_DIR)
        set(READLINE_FOUND TRUE)
        string(APPEND READLINE_LIBRARY_DIR " -lreadline -lncurses")
    else()
        set(READLINE_FOUND FALSE)
    endif()

    if(READLINE_FOUND)
        message("readline found!")
        addprefix(ABC_READLINE_INCLUDES_FLAGS "-I" ${READLINE_INCLUDE_DIR})
        string(REPLACE ";" " " ABC_READLINE_INCLUDES_FLAGS 
            "${ABC_READLINE_INCLUDES_FLAGS}")
        list(APPEND ABC_READLINE_FLAGS 
            "ABC_READLINE_INCLUDES=${ABC_READLINE_INCLUDES_FLAGS}")

        addprefix(ABC_READLINE_LIBRARIES_FLAGS "-L" ${READLINE_LIBRARY_DIR})
        string(REPLACE ";" " " ABC_READLINE_LIBRARIES_FLAGS 
            "${ABC_READLINE_LIBRARIES_FLAGS}")
        list(APPEND ABC_READLINE_FLAGS 
            "ABC_READLINE_LIBRARIES=${ABC_READLINE_LIBRARIES_FLAGS}")
    elseif(ABC_USE_NO_READLINE)
        message("Manually disable readline library!")
        list(APPEND ABC_READLINE_FLAGS "ABC_USE_NO_READLINE=1")
    else()
        message("readline not found!")
        list(APPEND ABC_READLINE_FLAGS "ABC_USE_NO_READLINE=1")
    endif()


    if(ABC_USE_NAMESPACE)
        set(ABC_USE_NAMESPACE_FLAGS "ABC_USE_NAMESPACE=${ABC_USE_NAMESPACE}")
    endif()

    # check whether ABC is properly cloned
    set(ABC_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/abc")
    if((NOT EXISTS "${ABC_ROOT_DIR}/Makefile") OR 
       (NOT EXISTS "${ABC_ROOT_DIR}/src/demo.c"))
        message(FATAL_ERROR "The ABC submodule might not be properly cloned.")
    endif()

    # run make to extract compiler options, linker options and list of source files
    execute_process(
        COMMAND
        make
            ${ABC_READLINE_FLAGS}
            ${ABC_USE_NAMESPACE_FLAGS}
            ARCHFLAGS_EXE=${CMAKE_CURRENT_BINARY_DIR}/abc_arch_flags_program.exe
            ABC_MAKE_NO_DEPS=1
            CC=${CMAKE_C_COMPILER}
            CXX=${CMAKE_CXX_COMPILER}
            LD=${CMAKE_CXX_COMPILER}
            cmake_info
        WORKING_DIRECTORY ${ABC_ROOT_DIR}
        OUTPUT_VARIABLE MAKE_OUTPUT
    )

    extract_var(SEPARATOR_SRC ABC_SRC ${MAKE_OUTPUT})
    extract_var(SEPARATOR_LIBS ABC_LIBS ${MAKE_OUTPUT})
    extract_var(SEPARATOR_CFLAGS ABC_CFLAGS ${MAKE_OUTPUT})
    extract_var(SEPARATOR_CXXFLAGS ABC_CXXFLAGS ${MAKE_OUTPUT})

    addprefix(ABC_SRC "abc/" ${ABC_SRC})


    if(ABC_USE_NAMESPACE)
        set_source_files_properties(${ABC_SRC} PROPERTIES LANGUAGE CXX)
    endif()

    # compile libabc
    set(ABC_MAIN_SRC abc/src/base/main/main.c)
    list(REMOVE_ITEM ABC_SRC ${ABC_MAIN_SRC})

    add_library(libabc EXCLUDE_FROM_ALL ${ABC_SRC})
    abc_properties(libabc PUBLIC)
    set_property(TARGET libabc PROPERTY OUTPUT_NAME abc)

    # compile the main binary executable
    file(GLOB GPULS_PATCH_SOURCES "src/abc_patch/*.cpp")
    list(FILTER GPULS_CPU_SOURCES EXCLUDE REGEX ".*/src/main\.cpp")
    add_executable(abcg ${GPULS_CPU_SOURCES} ${GPULS_GPU_SOURCES} 
                        ${GPULS_PATCH_SOURCES} ${ABC_MAIN_SRC})
    target_link_libraries(abcg PRIVATE libabc)

    # gpuls headers
    target_include_directories(abcg PRIVATE "src")
    target_include_directories(abcg PRIVATE "src/hash_table")
    target_include_directories(abcg PRIVATE "include") # header only libraries
    
    # abc headers and libs
    abc_properties(abcg PRIVATE)

    # compile options
    target_compile_options(abcg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GPULS_CUDA_FLAGS}>)
    add_definitions(-DPATCH_ABC)

    # Add debug flags based on build type
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(abcg PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_DEBUG_FLAGS}>)
        target_compile_options(abcg PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_DEBUG_FLAGS}>)
    endif()
else()
    # standalone executable
    message("Building in standalone mode ...")
    project(gpuls C CXX CUDA)

    add_executable(gpuls ${GPULS_CPU_SOURCES} ${GPULS_GPU_SOURCES})

    target_include_directories(gpuls PRIVATE "src")
    target_include_directories(gpuls PRIVATE "src/hash_table")
    target_include_directories(gpuls PRIVATE "include") # header only libraries

    target_compile_options(gpuls PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GPULS_CUDA_FLAGS}>)

    # Add debug flags based on build type
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(gpuls PRIVATE $<$<COMPILE_LANGUAGE:C>:${C_DEBUG_FLAGS}>)
        target_compile_options(gpuls PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_DEBUG_FLAGS}>)
    endif()
endif()
