from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="student_gemm",
    version="0.1.0",
    description="Custom CPU GEMM operator integrated into PyTorch",
    ext_modules=[
        CppExtension(
            name="src",
            sources=["src/gemm.cpp"],
            extra_compile_args=[
                "-O3",
                "-march=native",
                "-fopenmp",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    packages=["src"],
    package_dir={"src": "src"},
    python_requires=">=3.12",
)
