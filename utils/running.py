import numpy as np

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/


class NormaliseObservation(object):
    def __init__(self, shape):
        self._counter = 0
        self._mean = np.zeros(shape)
        self._standdev = np.zeros(shape)

    def update_running_statistics(self, sample_value):
        sample_value = np.asarray(sample_value)
        assert sample_value.shape == self._mean.shape
        self._counter += 1
        if self._counter == 1:
            self._mean[...] = sample_value
        else:
            old_mean = self._mean.copy()
            self._mean[...] = old_mean + (sample_value - old_mean) / self._counter
            self._standdev[...] = self._standdev + (sample_value - old_mean) * (sample_value - self._mean)
    @property
    def n(self):
        return self._counter
    @property
    def mean(self):
        return self._mean
    @property
    def var(self):
        return self._standdev / (self._counter - 1) if self._counter > 1 else np.square(self._mean)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._mean.shape


class ZFilter:
    """
    y = (sample_value-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, run_mean=True, run_std=True, clip=10.0):
        self.run_mean = run_mean
        self.run_std = run_std
        self.clip = clip
        self.running_observation = NormaliseObservation(shape)
        self.freeze_running_statistics = False

    def __call__(self, sample_value, update=True):
        if update and not self.freeze_running_statistics:
            self.running_observation.update_running_statistics(sample_value)
        if self.run_mean:
            sample_value = sample_value - self.running_observation.mean
        if self.run_std:
            sample_value = sample_value / (self.running_observation.std + 1e-8)
        if self.clip:
            sample_value = np.clip(sample_value, -self.clip, self.clip)
        return sample_value

