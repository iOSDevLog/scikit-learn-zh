"""全局状态配置和功能管理
"""
import os
from contextlib import contextmanager as contextmanager

_global_config = {
    'assume_finite': bool(os.environ.get('SKLEARN_ASSUME_FINITE', False)),
    'working_memory': int(os.environ.get('SKLEARN_WORKING_MEMORY', 1024))
}


def get_config():
    """检索配置的当前值 by :func:`set_config`

    Returns
    -------
    config : dict
        键是可以传递给  :func:`set_config` 的参数名
    """
    return _global_config.copy()


def set_config(assume_finite=None, working_memory=None):
    """设置全局 scikit-learn 配置

    参数
    ----------
    assume_finite : bool, 可选
        如果为True，将跳过有限性验证，
        节省时间，但导致潜在的崩溃。
        如果为False，将执行有限性验证，避免错误。
        全局默认值： False。

    working_memory : int, 可选
        如果设置
        scikit-learn将尝试将临时数组的大小限制为此数量的MiB（并行化每个作业）
        通常可以节省可以在块中执行的昂贵操作的计算时间和内存。
        全局默认值：1024。
    """
    if assume_finite is not None:
        _global_config['assume_finite'] = assume_finite
    if working_memory is not None:
        _global_config['working_memory'] = working_memory


@contextmanager
def config_context(**new_config):
    """用于全局scikit-learn配置的上下文管理器

    参数
    ----------
    assume_finite : bool, 可选
        如果为True，
        将跳过有限性验证，从而节省时间，但会导致潜在的崩溃。
        如果为False，将执行有限性验证，避免错误。
        全局默认值： False。

    working_memory : int, 可选
        如果设置
        scikit-learn将尝试将临时数组的大小限制为此数量的MiB（并行化每个作业）
        通常可以节省可以在块中执行的昂贵操作的计算时间和内存。
        全局默认值：1024。

    注释
    -----
    退出上下文管理器时
    所有设置（不仅是当前修改的设置）都将返回到先前的值。
    这不是线程安全的。

    Examples
    --------
    >>> import sklearn
    >>> from sklearn.utils.validation import assert_all_finite
    >>> with sklearn.config_context(assume_finite=True):
    ...     assert_all_finite([float('nan')])
    >>> with sklearn.config_context(assume_finite=True):
    ...     with sklearn.config_context(assume_finite=False):
    ...         assert_all_finite([float('nan')])
    ... # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Input contains NaN, ...
    """
    old_config = get_config().copy()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
