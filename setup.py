#! /usr/bin/env python
#
# Copyright (C) 2007-2009 Cournapeau David <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
# License: 3-clause BSD

import sys
import os
import platform
import shutil
from distutils.command.clean import clean as Clean
from pkg_resources import parse_version
import traceback

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

# 这是一个 hack 位  bit (!) hackish: 我们正在设置一个全局变量
# 以便 sklearn __init__ 可以检测到它是否被安装例程加载
# 以避免尝试加载尚未生成的组件:
# 使用的 numpy 健壮扩展
# scikit-learn以递归方式生成子包中的已编译扩展
# 基于 Python 导入机制
builtins.__SKLEARN_SETUP__ = True

DISTNAME = 'scikit-learn'
DESCRIPTION = '一组用于机器学习和数据挖掘的python模块'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Andreas Mueller'
MAINTAINER_EMAIL = 'amueller@ais.uni-bonn.de'
URL = 'http://scikit-learn.org'
DOWNLOAD_URL = 'https://pypi.org/project/scikit-learn/#files'
LICENSE = 'new BSD'

# 我们实际上可以导入一个
# 不需要编译代码的限制版sklearn
import sklearn

VERSION = sklearn.__version__

if platform.python_implementation() == 'PyPy':
    SCIPY_MIN_VERSION = '1.1.0'
    NUMPY_MIN_VERSION = '1.14.0'
else:
    SCIPY_MIN_VERSION = '0.13.3'
    NUMPY_MIN_VERSION = '1.8.2'


# 可选的setuptools功能
# 如果我们需要setuptools功能，我们需要尽早导入setuptools
# 因为它修补了'setup'功能
# 对于某些命令，请使用setuptools
SETUPTOOLS_COMMANDS = set([
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed',
])
if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools

    extra_setuptools_args = dict(
        zip_safe=False,  # 包可以不需要.egg文件运行
        include_package_data=True,
        extras_require={
            'alldeps': (
                'numpy >= {0}'.format(NUMPY_MIN_VERSION),
                'scipy >= {0}'.format(SCIPY_MIN_VERSION),
            ),
        },
    )
else:
    extra_setuptools_args = dict()


# 自定义clean命令删除构建工件

class CleanCommand(Clean):
    description = "从源码树中删除构建工件"

    def run(self):
        Clean.run(self)
        # 如果我们不在sdist包中，请删除c文件
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('sklearn'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand}

# 可选的wheelhouse-uploader功能
# 为了自动发布scikit-learn的二进制包
# 我们需要一个工具来下载由travis和appveyor worker生成的包
# （版本号与当前版本匹配）
# 并在发布时将它们全部上传到PyPI。
# 工件存储库的URL在setup.cfg文件中配置。

WHEELHOUSE_UPLOADER_COMMANDS = set(['fetch_artifacts', 'upload_all'])
if WHEELHOUSE_UPLOADER_COMMANDS.intersection(sys.argv):
    import wheelhouse_uploader.cmd

    cmdclass.update(vars(wheelhouse_uploader.cmd))


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # 避免无用的消息：
    # “忽略尝试设置'名称'（来自...” 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('sklearn')

    return config


def get_numpy_status():
    """
    返回一个字典，其中包含一个布尔值
    指定NumPy是否是最新的，以及版本字符串。
    （如果未安装，则为空字符串）
    """
    numpy_status = {}
    try:
        import numpy
        numpy_version = numpy.__version__
        numpy_status['up_to_date'] = parse_version(
            numpy_version) >= parse_version(NUMPY_MIN_VERSION)
        numpy_status['version'] = numpy_version
    except ImportError:
        traceback.print_exc()
        numpy_status['up_to_date'] = False
        numpy_status['version'] = ""
    return numpy_status


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 2',
                                 'Programming Language :: Python :: 2.7',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.4',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 'Programming Language :: Python :: 3.7',
                                 ('Programming Language :: Python :: '
                                  'Implementation :: CPython'),
                                 ('Programming Language :: Python :: '
                                  'Implementation :: PyPy')
                                 ],
                    cmdclass=cmdclass,
                    install_requires=[
                        'numpy>={0}'.format(NUMPY_MIN_VERSION),
                        'scipy>={0}'.format(SCIPY_MIN_VERSION)
                    ],
                    **extra_setuptools_args)

    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg_info',
                                                    '--version',
                                                    'clean'))):
        # 对于这些操作，不需要NumPy
        #
        # 他们需要在没有Numpy的情况下成功
        # 例如当系统中还没有Numpy时
        # 使用pip来安装 Scikit-learn。
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION
    else:
        numpy_status = get_numpy_status()
        numpy_req_str = "scikit-learn requires NumPy >= {0}.\n".format(
            NUMPY_MIN_VERSION)

        instructions = ("Installation instructions are available on the "
                        "scikit-learn website: "
                        "http://scikit-learn.org/stable/install.html\n")

        if numpy_status['up_to_date'] is False:
            if numpy_status['version']:
                raise ImportError("Your installation of Numerical Python "
                                  "(NumPy) {0} is out-of-date.\n{1}{2}"
                                  .format(numpy_status['version'],
                                          numpy_req_str, instructions))
            else:
                raise ImportError("Numerical Python (NumPy) is not "
                                  "installed.\n{0}{1}"
                                  .format(numpy_req_str, instructions))

        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
