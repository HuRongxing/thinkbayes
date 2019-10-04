#!/usr/bin/env python
# coding: utf-8


import bisect
import copy
import logging
import math
import numpy
import random

#from matplotlib.figure import Figure
#from matplotlib.axes import Axes

import scipy.stats
from scipy.special import erf, erfinv, gammaln
import numpy as np



class _DictWrapper(object):
    """包含字典的对象。"""

    def __init__(self, values=None, name=''):
        """
        values: 假设的值（字典对象各键的名称)。
        """
        self.name = name
        self.d = {}

        # 分布是否为对数形式。
        self.log = False

        if values is None:
            return
        
        # 根据不同的values，选择合适的初始化函数。
        init_methods = [
            self.InitPmf,
            self.InitMapping,
            self.InitSequence,
            self.InitFailure,
            ]
        for method in init_methods:
            try:
                method(values)
                break
            except AttributeError:
                continue
        
        # 对字典值进行标准化（归一化）处理。
        if len(self) > 0:
            self.Normalize()

    def InitSequence(self, values):
        """对列表值按相同权重进行初始化。
        values: 列表。
        """
        for value in values:
            self.Set(value, 1) # 自定义的set方法。

    def InitMapping(self, values):
        """根据字典进行初始化。
        values: 值和频率（概率）组成的字典。
        """
        ####################################################
        # 好像也可以self.d = values，但这样，如果values不是字典，
        # 在init_method中采用试错法调用初始化函数时，就不会报错。
        ####################################################
        for value, prob in values.items():
            self.Set(value, prob)

    def InitPmf(self, values):
        """根据Pmf对象进行初始化。
        values: Pmf对象。
        """
        for value, prob in values.Items():
            self.Set(value, prob)

    def InitFailure(self, values):
        """触发错误。"""
        raise ValueError('字典没有进行初始化。')

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def Copy(self, name=None):
        """返回自身的一个拷贝。

        只是对self.d时行浅层拷贝。如果需要一个深层拷贝，
        可对整个对象使用copy.deepcopy。
        
        Args:
            name: 新实例name属性的值。
        """
        new = copy.copy(self)
        new.d = copy.copy(self.d)  # 对self.d进行复制
        new.name = name if name is not None else self.name
        return new   # 返回一个新的对象实例

    def Scale(self, factor):
        """对键值的标度进行调整。注意是factor乘以键，而不是频率或概率。
        factor: 权重。
        Returns: 新对象。
        """
        new = self.Copy()
        new.d.clear()

        for val, prob in self.Items():
            new.Set(val * factor, prob)
        return new

    def Log(self, m=None):
        """Log transforms the probabilities.
        
        Removes values with probability 0.

        Normalizes so that the largest logprob is 0.
        """
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            if p:
                self.Set(x, math.log(p / m))
            else:
                self.Remove(x)

    def Exp(self, m=None):
        """Exponentiates the probabilities.

        m: how much to shift the ps before exponentiating

        If m is None, normalizes so that the largest prob is 1.
        """
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            self.Set(x, math.exp(p - m))

    def GetDict(self):
        """Gets the dictionary."""
        return self.d

    def SetDict(self, d):
        """Sets the dictionary."""
        self.d = d

    def Values(self):
        """Gets an unsorted sequence of values.

        Note: one source of confusion is that the keys of this
        dictionary are the values of the Hist/Pmf, and the
        values of the dictionary are frequencies/probabilities.
        """
        return self.d.keys()

    def Items(self):
        """Gets an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def Render(self):
        """Generates a sequence of points suitable for plotting.

        Returns:
            tuple of (sorted value sequence, freq/prob sequence)
        """
        return zip(*sorted(self.Items()))

    def Print(self):
        """Prints the values and freqs/probs in ascending order."""
        for val, prob in sorted(self.d.items()):
            print(val, prob)
            
    def __hash__(self):
        return hash(tuple([(val,prob) for val, prob in sorted(self.d.items())]))
  
    def Set(self, x, y=0):
        """设置x的频率（或概率）值为y。
        参数:
            x: 字典键名
            y: 频率（或概率）值，默认为0。
        """
        self.d[x] = y

    def Incr(self, x, term=1):
        """Increments the freq/prob associated with the value x.

        Args:
            x: number value
            term: how much to increment by
        """
        self.d[x] = self.d.get(x, 0) + term

    def Mult(self, x, factor):
        """Scales the freq/prob associated with the value x.

        Args:
            x: number value
            factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor

    def Remove(self, x):
        """Removes a value.

        Throws an exception if the value is not there.

        Args:
            x: value to remove
        """
        del self.d[x]

    def Total(self):
        """Returns the total of the frequencies/probabilities in the map."""
        total = sum(self.d.values())
        return total

    def MaxLike(self):
        """Returns the largest frequency/probability in the map."""
        return max(self.d.values())



class Pmf(_DictWrapper):
    """概率质量函数。
    
    概率值是任何可哈希类型，概率为浮点数。
    Pmfs不一定要标准化
    """

    def Prob(self, x, default=0):
        """根据x的取值，返回该点上的概率.

        Args:
            x: 概率点
            default: 如果概率点不存在，其取值。

        Returns:
            概率值，浮点数。
        """
        return self.d.get(x, default)

    def Probs(self, xs):
        """根据列表xs中的自变量，返回对应的概率值，结果为列表。"""
        return [self.Prob(x) for x in xs]

    def MakeCdf(self, name=None):
        """生成累积分布."""
        return MakeCdfFromPmf(self, name=name)

    def ProbGreater(self, x):
        """定义域大于x的所有点的概率.

        x: number

        returns: float probability
        """
        t = [prob for (val, prob) in self.d.items() if val > x]
        return sum(t)

    def ProbLess(self, x):
        """Probability that a sample from this Pmf is less than x.

        x: number

        returns: float probability
        """
        t = [prob for (val, prob) in self.d.items() if val < x]
        return sum(t)

    def __lt__(self, obj):
        """Less than.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return PmfProbLess(self, obj)
        else:
            return self.ProbLess(obj)

    def __gt__(self, obj):
        """Greater than.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return PmfProbGreater(self, obj)
        else:
            return self.ProbGreater(obj)

    def __ge__(self, obj):
        """Greater than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self < obj)

    def __le__(self, obj):
        """Less than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self > obj)

    def __eq__(self, obj):
        """Equal to.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return PmfProbEqual(self, obj)
        else:
            return self.Prob(obj)

    def __ne__(self, obj):
        """Not equal to.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self == obj)

    def Normalize(self, fraction=1.0):
        """Normalizes this PMF so the sum of all probs is fraction.

        Args:
            fraction: what the total should be after normalization

        Returns: the total probability before normalizing
        """
        if self.log:
            raise ValueError("Pmf is under a log transform")

        total = self.Total()
        if total == 0.0:
            raise ValueError('total probability is zero.')
            logging.warning('Normalize: total probability is zero.')
            return total

        factor = float(fraction) / total
        for x in self.d:
            self.d[x] *= factor

        return total

    def Random(self):
        """Chooses a random element from this PMF.

        Returns:
            float value from the Pmf
        """
        if len(self.d) == 0:
            raise ValueError('Pmf contains no values.')

        target = random.random()
        total = 0.0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        assert False

    def Mean(self):
        """Computes the mean of a PMF.

        Returns:
            float mean
        """
        mu = 0.0
        for x, p in self.d.items():
            mu += p * x
        return mu

    def Var(self, mu=None):
        """Computes the variance of a PMF.

        Args:
            mu: the point around which the variance is computed;
                if omitted, computes the mean

        Returns:
            float variance
        """
        if mu is None:
            mu = self.Mean()

        var = 0.0
        for x, p in self.d.iteritems():
            var += p * (x - mu) ** 2
        return var

    def MaximumLikelihood(self):
        """Returns the value with the highest probability.

        Returns: float probability
        """
        prob, val = max((prob, val) for val, prob in self.Items())
        return val

    def CredibleInterval(self, percentage=90):
        """Computes the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        cdf = self.MakeCdf()
        return cdf.CredibleInterval(percentage)

    def __add__(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        try:
            return self.AddPmf(other)
        except AttributeError:
            return self.AddConstant(other)

    def AddPmf(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 + v2, p1 * p2)
        return pmf

    def AddConstant(self, other):
        """Computes the Pmf of the sum a constant and  values from self.

        other: a number

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            pmf.Set(v1 + other, p1)
        return pmf

    def __sub__(self, other):
        """Computes the Pmf of the diff of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 - v2, p1 * p2)
        return pmf

    def Max(self, k):
        """Computes the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.MakeCdf()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf
    
    def __hash__(self):
        return hash(tuple([(val,prob) for val, prob in sorted(self.d.items())]))


# ## 类Suite$\to$Pmf

# In[4]:


class Suite(Pmf):
    """一系列假设及其概率类。"""

    def Update(self, data):
        """根据数据更新假设。
        data: 新出现的证据。
        returns: 标准化常数。
        """
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            # 似然函数的定义方法需要在子类中定义。
            self.Mult(hypo, like)
            # Mult方法在_DictWrapper中定义，用先验概率乘以似然概率
        return self.Normalize()

    def LogUpdate(self, data):
        """Updates a suite of hypotheses based on new data.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        Note: unlike Update, LogUpdate does not normalize.

        Args:
            data: any representation of the data
        """
        for hypo in self.Values():
            like = self.LogLikelihood(data, hypo)
            self.Incr(hypo, like)

    def UpdateSet(self, dataset):
        """Updates each hypothesis based on the dataset.

        This is more efficient than calling Update repeatedly because
        it waits until the end to Normalize.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: the normalizing constant
        """
        for data in dataset:
            for hypo in self.Values():
                like = self.Likelihood(data, hypo)
                self.Mult(hypo, like)
        return self.Normalize()

    def LogUpdateSet(self, dataset):
        """Updates each hypothesis based on the dataset.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: None
        """
        for data in dataset:
            self.LogUpdate(data)

    def Likelihood(self, data, hypo):
        """根据假设计算似然值。
        hypo: 假设
        data: 证据
        """
        raise UnimplementedMethodException()

    def LogLikelihood(self, data, hypo):
        """Computes the log likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()

    def Print(self):
        """Prints the hypotheses and their probabilities."""
        for hypo, prob in sorted(self.Items()):
            print(hypo, prob)

    def MakeOdds(self):
        """Transforms from probabilities to odds.

        Values with prob=0 are removed.
        """
        for hypo, prob in self.Items():
            if prob:
                self.Set(hypo, Odds(prob))
            else:
                self.Remove(hypo)

    def MakeProbs(self):
        """Transforms from odds to probabilities."""
        for hypo, odds in self.Items():
            self.Set(hypo, Probability(odds))

            
            
            
def MakeCdfFromHist(hist, name=''):
    """Makes a CDF from a Hist object.

    Args:
       hist: Pmf.Hist object
       name: string name for the data.

    Returns:
        Cdf object
    """
    return MakeCdfFromItems(hist.Items(), name)

# ## 类：Cdf

# In[5]:


class Cdf(object):
    """累积分布函数.

    Attributes:
        xs: 定义域
        ps: 概率值。
        name: 命名标签。
    """

    def __init__(self, xs=None, ps=None, name=''):
        self.xs = [] if xs is None else xs
        self.ps = [] if ps is None else ps
        self.name = name

    def Copy(self, name=None):
        """生成Cdf实例的拷贝。

        Args:
            name: 新的Cdf实例的名称。
        """
        if name is None:
            name = self.name
        return Cdf(list(self.xs), list(self.ps), name)

    def MakePmf(self, name=None):
        """从Ddf生成Pmf."""
        return MakePmfFromCdf(self, name=name)

    def Values(self):
        """返回排序后的定义域。
        """
        return self.xs

    def Items(self):
        """返回排序后的(value,probility)键—值对列表。

        Note: 在python3中，返回的是迭代对象。
        """
        return zip(self.xs, self.ps)

    def Append(self, x, p):
        """在CDF尾部添加(x, p)键值对。

        Note: 不会生成一个新的累积分布，仅用于对原累积分布的修补。
              使用该函数时，要注意生成的累积分布是否合理。
        """
        self.xs.append(x)
        self.ps.append(p)

    def Shift(self, term):
        """等距移动值域。

        term: 移动距离。
        """
        new = self.Copy()
        new.xs = [x + term for x in self.xs]
        return new

    def Scale(self, factor):
        """对值域乘以一个固定因子，放大或缩小值域。

        factor: 因子值。
        """
        new = self.Copy()
        new.xs = [x * factor for x in self.xs]
        return new

    def Prob(self, x):
        """返回点x的累积分布值。

        Args:
            x: number

        Returns:
            float probability
        """
        if x < self.xs[0]: return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p

    def Value(self, p):
        """根据累积分布的概率值，返回对应的定义域中的点。
        Args:
            p: number in the range [0, 1]

        Returns:
            number value
        """
        if p < 0 or p > 1:
            raise ValueError('Probability p must be in range [0, 1]')

        if p == 0: return self.xs[0]
        if p == 1: return self.xs[-1]
        index = bisect.bisect(self.ps, p)
        if p == self.ps[index - 1]:
            return self.xs[index - 1]
        else:
            return self.xs[index]

    def Percentile(self, p):
        """百分位数对应的值。

        Args:
            p: number in the range [0, 100]

        Returns:
            number value
        """
        return self.Value(p / 100.0)

    def Random(self):
        """随机返回一个分布的值"""
        return self.Value(random.random())

    def Sample(self, n):
        """随机进行抽样。
        
        Args:
            n: int length of the sample
        """
        return [self.Random() for i in range(n)]

    def Mean(self):
        """从累积分布生成概率函数的数学期望。

        Returns:
            float mean
        """
        old_p = 0
        total = 0.0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def CredibleInterval(self, percentage=90):
        """计算两侧置信区间。

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        prob = (1 - percentage / 100.0) / 2
        interval = self.Value(prob), self.Value(1 - prob)
        return interval

    def _Round(self, multiplier=1000.0):
        """
        An entry is added to the cdf only if the percentile differs
        from the previous value in a significant digit, where the number
        of significant digits is determined by multiplier.  The
        default is 1000, which keeps log10(1000) = 3 significant digits.
        """
        # TODO(write this method)
        raise UnimplementedMethodException()

    def Render(self):
        """Generates a sequence of points suitable for plotting.

        An empirical CDF is a step function; linear interpolation
        can be misleading.

        Returns:
            tuple of (xs, ps)
        """
        xs = [self.xs[0]]
        ps = [0.0]
        for i, p in enumerate(self.ps):
            xs.append(self.xs[i])
            ps.append(p)

            try:
                xs.append(self.xs[i + 1])
                ps.append(p)
            except IndexError:
                pass
        return xs, ps

    def Max(self, k):
        """Computes the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.Copy()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf


# ## 函数MakeCdfFromPmf：生成累积分布。

# In[6]:


def MakeCdfFromPmf(pmf, name=None):
    """从Pmf对象生成累积分布。

    Args:
       pmf: Pmf对象
       name: 数据的名称。

    Returns:
        Cdf对象
    """
    if name == None:
        name = pmf.name
    return MakeCdfFromItems(pmf.Items(), name)


# ## 函数MakeCdfFromItems：

# In[7]:


def MakeCdfFromItems(items, name=''):
    """从无序元组对(value, frequency)生成累积分布。

    Args:
        items: 从无序元组对(value, frequency)
        name: 累积分布CDF的name属性值。

    Returns:
        cdf: 元组对(value, fraction)组成的列表。
    """
    runsum = 0
    xs = []
    cs = []

    for value, count in sorted(items):
        runsum += count
        xs.append(value)
        cs.append(runsum)
    
    # 标准化（归一化）处理。
    total = float(runsum)
    ps = [c / total for c in cs]

    cdf = Cdf(xs, ps, name)
    return cdf


# ## 函数MakePmfFromCdf：从累积分布生成概率质量函数。

# In[8]:


def MakePmfFromCdf(cdf, name=None):
    """Makes a normalized Pmf from a Cdf object.

    Args:
        cdf: Cdf object
        name: string name for the new Pmf

    Returns:
        Pmf object
    """
    if name is None:
        name = cdf.name

    pmf = Pmf(name=name)

    prev = 0.0
    for val, prob in cdf.Items():
        pmf.Incr(val, prob - prev)
        prev = prob

    return pmf


# ## 类Joint$\to$Pmf
def MakeMixture(metapmf, name='mix'):
    """Make a mixture distribution.

    Args:
      metapmf: Pmf that maps from Pmfs to probs.
      name: string name for the new Pmf.

    Returns: Pmf object.
    """
    mix = Pmf(name=name)
    print('******')
    for pmf, p1 in metapmf.Items():
        if np.isnan(p1): p1=0
        for x, p2 in pmf.Items():
            if np.isnan(p2): p2=0
            mix.Incr(x, p1 * p2)            
    return mix

def MakePmfFromItems(t, name=''):
    """Makes a PMF from a sequence of value-probability pairs

    Args:
        t: sequence of value-probability pairs
        name: string name for this PMF

    Returns:
        Pmf object
    """
    pmf = Pmf(dict(t), name)
    pmf.Normalize()
    return pmf

class Joint(Pmf):
    """联合分布.

    值是序列(通常为元组)
    """

    def Marginal(self, i, name=''):
        """生成指定变量的边缘分布

        i: 变量的序号
        Returns: Pmf
        """
        pmf = Pmf(name=name)
        for vs, prob in self.Items():
            pmf.Incr(vs[i], prob)
        return pmf

    def Conditional(self, i, j, val, name=''):
        """获取指定变量的条件分布

        Distribution of vs[i], conditioned on vs[j] = val.

        i: index of the variable we want
        j: which variable is conditioned on
        val: the value the jth variable has to have

        Returns: Pmf
        """
        pmf = Pmf(name=name)
        for vs, prob in self.Items():
            if vs[j] != val: continue
            pmf.Incr(vs[i], prob)

        pmf.Normalize()
        return pmf

    def MaxLikeInterval(self, percentage=90):
        """Returns the maximum-likelihood credible interval.

        If percentage=90, computes a 90% CI containing the values
        with the highest likelihoods.

        percentage: float between 0 and 100

        Returns: list of values from the suite
        """
        interval = []
        total = 0

        t = [(prob, val) for val, prob in self.Items()]
        t.sort(reverse=True)

        for prob, val in t:
            interval.append(val)
            total += prob
            if total >= percentage / 100.0:
                break

        return interval


def MakeCdfFromList(seq, name=''):
    """Creates a CDF from an unsorted sequence.

    Args:
        seq: unsorted sequence of sortable values
        name: string name for the cdf

    Returns:
       Cdf object
    """
    hist = MakeHistFromList(seq)
    return MakeCdfFromHist(hist, name)

def MakeHistFromList(t, name=''):
    """Makes a histogram from an unsorted sequence of values.

    Args:
        t: sequence of numbers
        name: string name for this histogram

    Returns:
        Hist object
    """
    hist = Hist(name=name)
    [hist.Incr(x) for x in t]
    return hist


class Hist(_DictWrapper):
    """Represents a histogram, which is a map from values to frequencies.

    Values can be any hashable type; frequencies are integer counters.
    """

    def Freq(self, x):
        """Gets the frequency associated with the value x.

        Args:
            x: number value

        Returns:
            int frequency
        """
        return self.d.get(x, 0)

    def Freqs(self, xs):
        """Gets frequencies for a sequence of values."""
        return [self.Freq(x) for x in xs]

    def IsSubset(self, other):
        """Checks whether the values in this histogram are a subset of
        the values in the given histogram."""
        for val, freq in self.Items():
            if freq > other.Freq(val):
                return False
        return True

    def Subtract(self, other):
        """Subtracts the values in the given histogram from this histogram."""
        for val, freq in other.Items():
            self.Incr(val, -freq)



class Beta(object):
    """Represents a Beta distribution.

    See http://en.wikipedia.org/wiki/Beta_distribution
    """
    def __init__(self, alpha=1, beta=1, name=''):
        """Initializes a Beta distribution."""
        self.alpha = alpha
        self.beta = beta
        self.name = name

    def Update(self, data):
        """Updates a Beta distribution.

        data: pair of int (heads, tails)
        """
        heads, tails = data
        self.alpha += heads
        self.beta += tails

    def Mean(self):
        """Computes the mean of this distribution."""
        return float(self.alpha) / (self.alpha + self.beta)

    def Random(self):
        """Generates a random variate from this distribution."""
        return random.betavariate(self.alpha, self.beta)

    def Sample(self, n):
        """Generates a random sample from this distribution.

        n: int sample size
        """
        size = n,
        return numpy.random.beta(self.alpha, self.beta, size)

    def EvalPdf(self, x):
        """Evaluates the PDF at x."""
        return x ** (self.alpha - 1) * (1 - x) ** (self.beta - 1)

    def MakePmf(self, steps=101, name=''):
        """Returns a Pmf of this distribution.

        Note: Normally, we just evaluate the PDF at a sequence
        of points and treat the probability density as a probability
        mass.

        But if alpha or beta is less than one, we have to be
        more careful because the PDF goes to infinity at x=0
        and x=1.  In that case we evaluate the CDF and compute
        differences.
        """
        if self.alpha < 1 or self.beta < 1:
            cdf = self.MakeCdf()
            pmf = cdf.MakePmf()
            return pmf

        xs = [i / (steps - 1.0) for i in xrange(steps)]
        probs = [self.EvalPdf(x) for x in xs]
        pmf = MakePmfFromDict(dict(zip(xs, probs)), name)
        return pmf

    def MakeCdf(self, steps=101):
        """Returns the CDF of this distribution."""
        xs = [i / (steps - 1.0) for i in xrange(steps)]
        ps = [scipy.special.betainc(self.alpha, self.beta, x) for x in xs]
        cdf = Cdf(xs, ps)
        return cdf



# ## 函数MakeLocationPmf，位置、夹角与位置

# In[10]:


def MakeLocationPmf(alpha, beta, locations):
    """Computes the Pmf of the locations, given alpha and beta. 

    Given that the shooter is at coordinates (alpha, beta),
    the probability of hitting any spot is inversely proportionate
    to the strafe speed.

    alpha: x position
    beta: y position
    locations: x locations where the pmf is evaluated

    Returns: Pmf object
    """
    pmf = Pmf()
    for x in locations:
        prob = 1.0 / StrafingSpeed(alpha, beta, x)
        pmf.Set(x, prob)
    pmf.Normalize()
    return pmf


# ## 彩弹对象Paintball$\to\begin{cases} 
# Suite \\
# Joint
# \end{cases}$

# In[11]:


class Paintball(Suite, Joint):
    """对手站位的假设."""

    def __init__(self, alphas, betas, locations):
        """Makes a joint suite of parameters alpha and beta.

        Enumerates all pairs of alpha and beta.
        Stores locations for use in Likelihood.

        alphas: possible values for alpha
        betas: possible values for beta
        locations: possible locations along the wall
        """
        self.locations = locations
        pairs = [(alpha, beta) 
                 for alpha in alphas 
                 for beta in betas]
        Suite.__init__(self, pairs)

    def Likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: pair of alpha, beta
        data: location of a hit

        Returns: float likelihood
        """
        alpha, beta = hypo
        x = data
        pmf = MakeLocationPmf(alpha, beta, self.locations)
        like = pmf.Prob(x)
        return like


# ## 函数

# In[12]:


def StrafingSpeed(alpha, beta, x):
    """Computes strafing speed, given location of shooter and impact.

    alpha: x location of shooter
    beta: y location of shooter
    x: location of impact

    Returns: derivative of x with respect to theta
    """
    theta = math.atan2(x - alpha, beta)
    speed = beta / math.cos(theta)**2
    return speed


# ## 函数pmf_plot

# In[13]:


import matplotlib.pyplot as plt
def pmf_plot(ax,pmf):
    x,y = zip(*sorted([x for x in pmf.Items()]))
    ax.plot(x,y,label=pmf.name)
    return ax
    
