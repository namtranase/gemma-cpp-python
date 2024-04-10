import os
import platform
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        os.makedirs(self.build_temp, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        # Allow gemma.cpp to be built on Windows with ClangCL
        # Refer to https://github.com/google/gemma.cpp/pull/6
        if platform.system() == "Windows":
            cmake_args += ["-T", "ClangCL"]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            build_args += ["--", "/m:12"]
        else:
            build_args += ["--", "-j12"]

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", ext.name] + build_args,
            cwd=self.build_temp,
        )


setup(
    ext_modules=[CMakeExtension("_pygemma")],
    cmdclass=dict(build_ext=CMakeBuild),
)
