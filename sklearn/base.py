"""所有估计器(estimators)的基类"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause

import copy
import warnings
from collections import defaultdict

import numpy as np
from scipy import sparse
from .externals import six
from .utils.fixes import signature
from . import __version__


##############################################################################
def _first_and_last_element(arr):
    """返回numpy数组或稀疏矩阵(Sparse matrices)的第一个和最后一个元素。"""
    if isinstance(arr, np.ndarray) or hasattr(arr, 'data'):
        # 有.data属性的numpy的数组或稀疏矩阵
        data = arr.data if sparse.issparse(arr) else arr
        return data.flat[0], data.flat[-1]
    else:
        # 没有.data属性的稀疏矩阵 只有在dok_matrix写时
        # 在这种情况下，索引很快
        return arr[0, 0], arr[-1, -1]


def clone(estimator, safe=True):
    """构造一个具有相同参数的新估计器。

    clone在estimators中执行模型的深拷贝,
    而不实际复制附加数据。
    它生成一个新的estimators, 其参数与任何数据都不匹配。

    参数
    ----------
    estimator : 估算器对象，或对象的列表，元组，集合
        要clone的estimators或estimators组

    safe : boolean, optional
        如果 safe 是 false,
        不是estimators的对象clone将回退到深拷贝。

    """
    estimator_type = type(estimator)
    # XXX: 没有处理词典
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params'):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a scikit-learn estimator "
                            "as it does not implement a 'get_params' methods."
                            % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in six.iteritems(new_object_params):
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # 快速完整性检查克隆的参数
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'either does not set or modifies parameter %s' %
                               (estimator, name))
    return new_object


###############################################################################
def _pprint(params, offset=0, printer=repr):
    """漂亮打印字典'params'

    参数
    ----------
    params : dict
        需要漂亮打印的字典

    offset : int
        要在每行开头添加的字符偏移量。

    printer : callable
        通常，将条目转换为字符串的函数
        内置的str或repr

    """
    # 做多行合理的repr：
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            # 使用 str 表示浮点数字
            # 通过这种方式
            # 我们可以在体系结构和版本之间实现一致的表示。
            this_repr = '%s=%s' % (k, str(v))
        else:
            # 使用其余的repr
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # 去除结尾空格以避免doctests中的噩梦
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


###############################################################################
class BaseEstimator(object):
    """scikit-learn中所有estimators的基类

    Notes
    -----
    所有estimators都应指定
    所有可在类级别`__init__`设置为显式关键字参数
    (no ``*args`` or ``**kwargs``)
    """

    @classmethod
    def _get_param_names(cls):
        """获取estimator的参数名称"""
        # 获取构造函数或之前的原始构造
        # 弃用包装如有
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # 没有内省的显式构造函数
            return []

        # 反思构造函数参数
        # 以查找要表示的模型参数
        init_signature = signature(init)
        # 考虑不包括'self'的构造函数参数
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # 提取和排序参数名称，不包括'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """获取此estimator的参数。

        Parameters
        ----------
        deep : boolean, optional
            如果为 True
            将返回此estimators的参数并包含估计值的子对象。

        Returns
        -------
        params : 将字符串映射到任意字符串
            映射到其值的参数名称。
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """设置此estimator的参数。

        该方法适用于简单estimators和嵌套对象
        (如管道)。
        后者具有窗体 ``<component>__<parameter>`` 的参数,
        以便可以更新嵌套对象的每个组件。

        Returns
        -------
        self
        """
        if not params:
            # 简单优化以获得速度（检查速度慢）
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # 按前缀分组
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

    def __getstate__(self):
        try:
            state = super(BaseEstimator, self).__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        if type(self).__module__.startswith('sklearn.'):
            return dict(state.items(), _sklearn_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        if type(self).__module__.startswith('sklearn.'):
            pickle_version = state.pop("_sklearn_version", "pre-0.18")
            if pickle_version != __version__:
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, __version__),
                    UserWarning)
        try:
            super(BaseEstimator, self).__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)


###############################################################################
class ClassifierMixin(object):
    """用于scikit-learn中所有分类器的Mixin类。"""
    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        """返回给定测试数据和标签的平均精度。

        在多标签分类中,
        这是一个苛刻指标的子集精度,
        因为每个样本都需要正确预测每个标签集。

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            测试样品。

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            X的真实标签。

        sample_weight : array-like, shape = [n_samples], optional
            样品权重。

        Returns
        -------
        score : float
            self.predict（X）和 y 的平均精度。

        """
        from .metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


###############################################################################
class RegressorMixin(object):
    """用于scikit-learn中所有回归估计器的Mixin类。"""
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """返回预测的确定系数 R^2.

        系数 R^2 定义为 (1-u/v),
        其中 u 是 ((y_true - y_pred) ** 2).sum() 剩余的平方和
        v 是 ((y_true - y_true.mean()) ** 2).sum() 总平方和.
        最好的评分是 1.0, 它可以是负数
         (因为模型可以是任意恶化)。
         始终预测 y 的预期值 (无视输入要素) 的常量模型
        将获得 R^2 评分0.0。

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            测试样品。 这可能是预计的内核矩阵,
            shape = (n_samples, n_samples_fitted],
            其中 n_samples_fitted
            是用于估计的拟合中使用的样本数。

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            X的真值。

        sample_weight : array-like, shape = [n_samples], optional
            样品权重。

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """

        from .metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')


###############################################################################
class ClusterMixin(object):
    """用于scikit-learn中所有聚类估计器的Mixin类。"""
    _estimator_type = "clusterer"

    def fit_predict(self, X, y=None):
        """在X上执行聚类并返回聚类标签。

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            输入数据。

        y : Ignored
            未使用，按惯例提供API一致性。

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            聚类标签
        """
        # 非优化的默认实现;
        # 当给定的聚类算法可能有更好的方法时重写
        self.fit(X)
        return self.labels_


class BiclusterMixin(object):
    """scikit-learn中用于混合类所有双向估计器"""

    @property
    def biclusters_(self):
        """将行和列索引放在一起的便捷方式。

        返回``rows_``和``columns_``成员。
        """
        return self.rows_, self.columns_

    def get_indices(self, i):
        """第i个双向聚类的行和列索引。

        仅当存在``rows_``和``columns_``属性时才有效。

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        row_ind : np.array, dtype=np.intp
            数据集中属于双向聚类的行的索引。
        col_ind : np.array, dtype=np.intp
            数据集中属于双向聚类的列的索引。

        """
        rows = self.rows_[i]
        columns = self.columns_[i]
        return np.nonzero(rows)[0], np.nonzero(columns)[0]

    def get_shape(self, i):
        """聚类 i 的形状。

        Parameters
        ----------
        i : int
            聚类的索引。

        Returns
        -------
        shape : (int, int)
            双向聚类中的行数和列数（分别）。
        """
        indices = self.get_indices(i)
        return tuple(len(i) for i in indices)

    def get_submatrix(self, i, data):
        """返回对应于双向聚类`i`的子矩阵。

        Parameters
        ----------
        i : int
            聚类的索引。
        data : array
            数据。

        Returns
        -------
        submatrix : array
            子矩阵对应于双向聚类i。

        Notes
        -----
        适用于稀疏矩阵。
        只有``rows_``和 ``columns_``属性存在。
        """
        from .utils.validation import check_array
        data = check_array(data, accept_sparse='csr')
        row_ind, col_ind = self.get_indices(i)
        return data[row_ind[:, np.newaxis], col_ind]


###############################################################################
class TransformerMixin(object):
    """scikit-learn中所有转换器的Mixin类。"""

    def fit_transform(self, X, y=None, **fit_params):
        """拟合数据，然后转换它。

        使用可选参数fit_params转换转换X和y
        并返回X的转换版本。

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            训练集。

        y : numpy array of shape [n_samples]
            目标值。

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            转换后的numpy数组

        """
        # 非优化的默认实现
        # 当给定的聚类算法可能有更好的方法时重写
        if y is None:
            # 元数1的拟合方法（无监督转换）
            return self.fit(X, **fit_params).transform(X)
        else:
            # 元数2的拟合方法（监督变换）
            return self.fit(X, y, **fit_params).transform(X)


class DensityMixin(object):
    """用于scikit-learn中所有密度估计器的Mixin类。"""
    _estimator_type = "DensityEstimator"

    def score(self, X, y=None):
        """返回数据X上模型的分数

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        score : float
        """
        pass


class OutlierMixin(object):
    """用于scikit-learn中所有异常值检测估计器的Mixin类。"""
    _estimator_type = "outlier_detector"

    def fit_predict(self, X, y=None):
        """在X上执行异常值检测。

        对于异常值，返回-1，对于正常值，返回1。

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            输入数据。

        y : Ignored
            未使用，按惯例提供API一致性。

        Returns
        -------
        y : ndarray, shape (n_samples,)
           正常值返回1，异常值返回-1。
        """
        # override for transductive outlier detectors like LocalOulierFactor
        return self.fit(X).predict(X)


###############################################################################
class MetaEstimatorMixin(object):
    """用于scikit-learn中所有元估计的Mixin类。"""
    # 这只是暂时的标签


###############################################################################

def is_classifier(estimator):
    """如果给定的估计器（可能）是分类器，则返回True。

    Parameters
    ----------
    estimator : object
        要测试的Estimator对象。

    Returns
    -------
    out : bool
        如果estimator是分类器，则为True，否则为False。
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"


def is_regressor(estimator):
    """如果给定的估计器（可能）是回归器，则返回True。

    Parameters
    ----------
    estimator : object
        要测试的Estimator对象。

    Returns
    -------
    out : bool
        如果estimator是回归器，则为True，否则为False。
    """
    return getattr(estimator, "_estimator_type", None) == "regressor"


def is_outlier_detector(estimator):
    """如果给定的estimator（可能）是异常值检测器，则返回True。

    Parameters
    ----------
    estimator : object
        要测试的Estimator对象。

    Returns
    -------
    out : bool
        如果estimator是异常值检测器，则为True，否则为False。
    """
    return getattr(estimator, "_estimator_type", None) == "outlier_detector"
