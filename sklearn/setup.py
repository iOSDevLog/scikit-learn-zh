import os
from os.path import join
import warnings

from sklearn._build_utils import maybe_cythonize_extensions


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, BlasNotFoundError
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('sklearn', parent_package, top_path)

    # 子模块与构建实用程序
    config.add_subpackage('__check_build')
    config.add_subpackage('_build_utils')

    # 没有自己的setup.py的子模块
    # 我们必须手动添加子子模块 & 测试
    config.add_subpackage('compose')
    config.add_subpackage('compose/tests')
    config.add_subpackage('covariance')
    config.add_subpackage('covariance/tests')
    config.add_subpackage('cross_decomposition')
    config.add_subpackage('cross_decomposition/tests')
    config.add_subpackage('feature_selection')
    config.add_subpackage('feature_selection/tests')
    config.add_subpackage('gaussian_process')
    config.add_subpackage('gaussian_process/tests')
    config.add_subpackage('mixture')
    config.add_subpackage('mixture/tests')
    config.add_subpackage('model_selection')
    config.add_subpackage('model_selection/tests')
    config.add_subpackage('neural_network')
    config.add_subpackage('neural_network/tests')
    config.add_subpackage('preprocessing')
    config.add_subpackage('preprocessing/tests')
    config.add_subpackage('semi_supervised')
    config.add_subpackage('semi_supervised/tests')

    # 有自己setup.py子模块
    # 暂时省略“linear_model”和“utils”; 在下面的cblas之后添加它们
    config.add_subpackage('cluster')
    config.add_subpackage('datasets')
    config.add_subpackage('decomposition')
    config.add_subpackage('ensemble')
    config.add_subpackage('externals')
    config.add_subpackage('feature_extraction')
    config.add_subpackage('manifold')
    config.add_subpackage('metrics')
    config.add_subpackage('neighbors')
    config.add_subpackage('tree')
    config.add_subpackage('svm')

    # 添加用于保序回归(isotonic regression)的Cython扩展模块
    config.add_extension('_isotonic',
                         sources=['_isotonic.pyx'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         )

    # 一些libs需要cblas，fortran编译的BLAS(fortran-compiled BLAS)是不够的
    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or (
            ('NO_ATLAS_INFO', 1) in blas_info.get('define_macros', [])):
        config.add_library('cblas',
                           sources=[join('src', 'cblas', '*.c')])
        warnings.warn(BlasNotFoundError.__doc__)

    # 以下软件包依赖于 cblas
    # 因此必须在上述版本之后进行构建
    config.add_subpackage('linear_model')
    config.add_subpackage('utils')

    # 添加测试目录
    config.add_subpackage('tests')

    maybe_cythonize_extensions(top_path, config)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
