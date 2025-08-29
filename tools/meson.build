custom_include_dir = join_paths(get_option('prefix'), 'include')
link_shared = get_option('link_shared')

cmake = import('cmake')

backward_opts = cmake.subproject_options()
backward_opts.add_cmake_defines(
    {
        'BACKWARD_SHARED': link_shared,
        'BUILD_SHARED_LIBS': link_shared,
        'CMAKE_CXX_FLAGS': '-I' + custom_include_dir,
        'CMAKE_INSTALL_LIBDIR': get_option('libdir'),
        'CMAKE_INSTALL_PREFIX': get_option('prefix'),
        'CMAKE_POLICY_VERSION_MINIMUM': '3.5',
    },
)
backward_proj = cmake.subproject('backward-cpp', options: backward_opts)
backward_dep = backward_proj.dependency('backward')

cpp_logger_opts = cmake.subproject_options()
cpp_logger_opts.add_cmake_defines(
    {
        'BUILD_SHARED_LIBS': link_shared,
        'CMAKE_CXX_FLAGS': '-I' + custom_include_dir,
        'CMAKE_INSTALL_LIBDIR': get_option('libdir'),
        'CMAKE_INSTALL_PREFIX': get_option('prefix'),
        'CMAKE_POLICY_VERSION_MINIMUM': '3.5',
    },
)
cpp_logger_proj = cmake.subproject('cpp-logger', options: cpp_logger_opts)
cpp_logger_dep = cpp_logger_proj.dependency('cpp_logger')

recorder_opts = cmake.subproject_options()
recorder_opts.add_cmake_defines(
    {
        'BUILD_SHARED_LIBS': link_shared,
        'CMAKE_CXX_FLAGS': '-I' + custom_include_dir,
        'CMAKE_INSTALL_LIBDIR': get_option('libdir'),
        'CMAKE_INSTALL_PREFIX': get_option('prefix'),
        'CMAKE_POLICY_VERSION_MINIMUM': '3.5',
        'RECORDER_INSTALL_LIB_DIR': get_option('libdir'),
    },
)
recorder_proj = cmake.subproject('recorder', options: recorder_opts)
reader_dep = recorder_proj.dependency('reader')

# 3rd party dependencies
arrow_dep = dependency('arrow', required: true)
libiberty_dep = dependency('libiberty', required: false)
libstdcxxfs_dep = cpp.find_library('stdc++fs', required: true)
mpi_dep = dependency('mpi', language: 'cpp', required: true)
ncurses_dep = dependency('ncurses', required: true)
nlohmann_json_dep = dependency('nlohmann_json', required: true)
parquet_dep = dependency('parquet', required: true)
readline_dep = dependency('readline', required: true)
zlib_dep = dependency('zlib', required: false)

recorder2parquet_exe = executable(
    meson.project_name() + '-recorder2parquet',
    'recorder2parquet.cpp',
    include_directories: include_directories(custom_include_dir),
    install: true,
    dependencies: [
        arrow_dep,
        backward_dep,
        cpp_logger_dep,
        libiberty_dep,
        libstdcxxfs_dep,
        mpi_dep,
        ncurses_dep,
        nlohmann_json_dep,
        parquet_dep,
        reader_dep,
        readline_dep,
        zlib_dep,
    ],
)

if get_option('enable_tests')
    test_dir = 'tests'
    raw_data_dir = 'tests/data'
    raw_data_name = 'recorder-posix-raw'
    extracted_dir = raw_data_dir + '/extracted'

    prepare_test_data = custom_target(
        'prepare_test_data',
        output: '0_0.parquet',
        command: [
            'sh',
            meson.project_source_root() / test_dir / 'prepare_test_data.sh',  
            meson.project_source_root() / raw_data_dir,
            raw_data_name,
            meson.project_source_root() / extracted_dir,
        ],
        build_by_default: false,
    )

    test(
        'test_recorder2parquet',
        recorder2parquet_exe,
        args: [meson.project_source_root() / extracted_dir / raw_data_name],
        depends: prepare_test_data,
        timeout: 0,
    )

    add_test_setup('mpi', exe_wrapper: ['mpirun', '--allow-run-as-root', '--oversubscribe', '-np', '2'])
endif
