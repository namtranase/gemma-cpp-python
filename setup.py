from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import pybind11


class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Customize the build process here if needed
        build_ext.build_extensions(self)

# Include the pybind11 include directory
pybind11_include_dir = pybind11.get_include()

ext_modules = [
    Extension(
        'gemma_cpp_extension',  # Name of the module to import in Python
        ['src/binding.cc', 'vendor/gemma.cpp/gemma.cc', 'vendor/gemma.cpp/run.cc'],  # Source files
        include_dirs=[pybind11_include_dir, './src/include', './vendor/gemma.cpp/compression', './vendor/gemma.cpp/util'],  # Include directories
        language='c++',
        extra_compile_args=['-std=c++11'],  # Replace with your required C++ version
    ),
]

setup(
    name='gemma-cpp-python',
    version='0.1.0',
    author='Nam Tran',
    author_email='trannam.ase@gmail.com',
    description='A Python wrapper for the GEMMA C++ library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CustomBuildExt,
    },
    # Add all other necessary package metadata
    install_requires=[
        'pybind11>=2.6',
    ],
)