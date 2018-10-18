"""
Python的机器学习模块
==================================

sklearn 是一个 python 模块
它将经典机器学习算法集成在科学 Python 包 (numpy、scipy、matplotlib)
的紧密结合的世界中。

它旨在为学习问题提供简单有效的解决方案，
这些问题可供所有人使用，并可在各种情况下重复使用：
机器学习作为科学和工程的多功能工具。

有关完整文档，请访问 http://scikit-learn.org。
"""
import sys
import re
import warnings
import logging

from ._config import get_config, set_config, config_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# 确保始终打印此包中的DeprecationWarning
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}\.'.format(re.escape(__name__)))

# PEP0440 兼容格式化版本，请参阅：
# https://www.python.org/dev/peps/pep-0440/
#
# 通用发布标记：
#   X.Y
#   X.Y.Z   # 用于修正bug
#
# 允许的预发布标记：
#   X.YaN   # Alpha 版本
#   X.YbN   # Beta 版本
#   X.YrcN  # 候选版本
#   X.Y     # 最终版本
#
# 开发（Dev）分支标志是: 'X.Y.dev' 或 'X.Y.devN' 其中N为整数。
# 'X.Y.dev0' 是 'X.Y.dev' 的规范版本
#
__version__ = '0.20.0'


try:
    # 此变量在 __builtins__ 中由生成过程注入
    # 当未生成二进制文件时,
    # 它用于启用导入子包 sklearn
    __SKLEARN_SETUP__
except NameError:
    __SKLEARN_SETUP__ = False

if __SKLEARN_SETUP__:
    sys.stderr.write('Partial import of sklearn during the build process.\n')
    # 我们不会在构建过程中导入 scikit-learn 的其余部分
    # 因为它可能尚未编译
else:
    from . import __check_build
    from .base import clone
    from .utils._show_versions import show_versions

    __check_build  # 避免报未使用的变量错误

    __all__ = ['calibration', 'cluster', 'covariance', 'cross_decomposition',
               'datasets', 'decomposition', 'dummy', 'ensemble', 'exceptions',
               'externals', 'feature_extraction', 'feature_selection',
               'gaussian_process', 'isotonic', 'kernel_approximation',
               'kernel_ridge', 'linear_model', 'manifold', 'metrics',
               'mixture', 'model_selection', 'multiclass', 'multioutput',
               'naive_bayes', 'neighbors', 'neural_network', 'pipeline',
               'preprocessing', 'random_projection', 'semi_supervised',
               'svm', 'tree', 'discriminant_analysis', 'impute', 'compose',
               # 非-modules:
               'clone', 'get_config', 'set_config', 'config_context',
               'show_versions']


def setup_module(module):
    """用于测试的夹具以确保全局可控制的RNGs种子"""
    import os
    import numpy as np
    import random

    # 它可能是在环境中提供的
    _random_seed = os.environ.get('SKLEARN_SEED', None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * (2 ** 31 - 1)
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
