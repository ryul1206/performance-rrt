from setuptools import setup, Extension
import pybind11


# C++ 확장 모듈
ext_modules = [
    # RRT* C++ 바인딩 모듈
    Extension(
        "core.cpp_impl.rrt_star_cpp_fn_bind",
        sources=["core/cpp_impl/rrt_star_cpp_fn_bind.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    ),
    Extension(
        "core.cpp_impl.rrt_star_cpp_full_bind",
        sources=["core/cpp_impl/rrt_star_cpp_full_bind.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    ),
    Extension(
        "core.cpp_impl.rrt_star_simd_full_bind",
        sources=["core/cpp_impl/rrt_star_simd_full_bind.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="performance-rrt",
    version="0.1.0",
    description="Various RRT* implementations for educational purposes",
    ext_modules=ext_modules,
    packages=["core", "core.cpp_impl", "core.python_impl"],  # Python 패키지들
)
