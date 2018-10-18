# scikit-learn
---

scikit-learn是一个用于机器学习的Python模块，建立在SciPy之上，并根据3-Clause BSD许可证进行分发。

该项目由David Cournapeau于2007年开始，作为Google Summer of Code项目，从那时起，许多志愿者都做出了贡献。有关贡献者的完整列表，请参阅`AUTHORS.rst <AUTHORS.rst>`文件。

它目前由一个志愿者团队维护。

网站： http://scikit-learn.org


## 安装

### 依赖

scikit-learn 需要:

- Python (>= 2.7 or >= 3.4)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)

**Scikit-learn 0.20是支持Python2.7的最后一个版本。**
Scikit-learn 0.21及更高版本将需要Python 3.5或更高版本。

要运行示例，需要Matplotlib> = 1.4。一些例子需要scikit-image> = 0.11.3，一些例子要求pandas> = 0.17.1。

scikit-learn还使用CBLAS，它是Basic Linear Algebra Subprograms库的C接口。scikit-learn带有参考实现，但系统CBLAS将由构建系统检测并在存在时使用。CBLAS存在于许多实现中; 有关已知问题，请参阅线`性代数库 Linear algebra libraries
<http://scikit-learn.org/stable/modules/computational_performance.html#linear-algebra-libraries>`。

### 用户安装

如果您已经安装了numpy和scipy，那么安装scikit-learn的最简单方法就是使用 `pip` ::

    pip install -U scikit-learn

或者 `conda`::

    conda install scikit-learn

该文档包含更详细的`安装说明`<http://scikit-learn.org/stable/install.html>。


## 更新日志

有关 scikit-learn的显着更改历史记录，请参阅`更改日志 `<http://scikit-learn.org/dev/whats_new.html>。

## 开发

我们欢迎所有经验水平的新贡献者。scikit-learn社区目标应该是有益的，热情的和有效的。`开发指南 Development Guide` <http://scikit-learn.org/stable/developers/index.html> 提供了有关贡献代码，文档，测试等的详细信息。我们在本自述文件中包含了一些基本信息。

### 重要链接

- 官方源代码： https://github.com/scikit-learn/scikit-learn
- 下载版本： https://pypi.python.org/pypi/scikit-learn
- 问题跟踪: https://github.com/scikit-learn/scikit-learn/issues

### 源代码

您可以使用以下命令检查最新的源：

    git clone https://github.com/scikit-learn/scikit-learn.git

### 设置开发环境

有关如何设置环境以便为scikit-learn做出贡献的快速教程：https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md

## 测试


安装后，您可以从源目录外部启动测试套件（您需要安装 `pytest`> = 3.3.0）：

    pytest sklearn

有关 详细信息，请参阅网页：http://scikit-learn.org/dev/developers/advanced_installation.html#testing

    通过设置`SKLEARN_SEED`环境变量，可以在测试期间控制随机数生成。

### 提交拉取请求

在打开Pull Request之前，请查看完整的Contributing页面，确保您的代码符合我们的指南http://scikit-learn.org/stable/developers/index.html


## 项目历史

该项目由David Cournapeau于2007年开始，作为Google Summer of Code项目，从那时起，许多志愿者都做出了贡献。有关贡献者的完整列表，请参阅`AUTHORS.rst <AUTHORS.rst>`文件。

该项目目前由一个志愿者团队维护。

**注意**: `scikit-learn` 以前称为 `scikits.learn`。


## 帮助和支持

### 文档

- HTML 文档（稳定版）： http://scikit-learn.org
- HTML 文档（开发版）： http://scikit-learn.org/dev/
- 常见问题：http://scikit-learn.org/stable/faq.html

### 通讯

- 邮件列表: https://mail.python.org/mailman/listinfo/scikit-learn
- IRC 频道: `#scikit-learn` at `webchat.freenode.net`
- Stack Overflow: http://stackoverflow.com/questions/tagged/scikit-learn
- 网站: http://scikit-learn.org

## 引用

如果您在科学出版物中使用scikit-learn，我们将非常感谢引用： http://scikit-learn.org/stable/about.html#citing-scikit-learn
