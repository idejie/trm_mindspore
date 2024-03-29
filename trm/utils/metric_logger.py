# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

from mindspore import Tensor

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        #self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        #self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = Tensor(list(self.deque))
        return d.asnumpy()

    @property
    def last(self):
        d = Tensor(list(self.deque)[-1])
        return d.asnumpy()

    @property
    def avg(self):
        d = Tensor(list(self.deque))
        return d.mean().asnumpy()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                v = v.asnumpy().item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.2f} ".format(name, meter.avg))  # ({:.2f}) meter.median,
        return self.delimiter.join(loss_str)
