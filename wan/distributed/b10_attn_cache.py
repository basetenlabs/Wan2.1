import torch
import torch.fft

from ..b10_config import enable_b10_attn_cache


@torch.no_grad()
def fft(tensor):
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    b, c, h, w = tensor.size()
    radius = min(h, w) // 5

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    center_x, center_y = w // 2, h // 2
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask

    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft


class B10UncondCache:

    def __init__(self):
        self.current_step_id = None
        self.max_step_id = None

        self.step_id_threshold = 10
        self.interval = 4
        self.low_freq_weight = 1.1
        self.high_freq_weight = 1.1
        self.low_freq_threshold = 45
        self.high_freq_threshold = 35
        self.delta_high_freq = None
        self.delta_low_freq = None
        self.force_update = False

    def _if_use_cache(self, step_id):
        assert step_id is not None, "step_id is None"
        return (enable_b10_attn_cache() and step_id < self.max_step_id
                and step_id > self.step_id_threshold
                and step_id % self.interval != 0)

    def if_use_cache_this_step(self):
        return self._if_use_cache(self.current_step_id) and not self.force_update

    def if_use_cache_next_step(self):
        return self._if_use_cache(self.current_step_id + 1)

    def set_cached_output(self, cond, uncond):
        low_freq_cond, high_freq_cond = fft(cond.float())
        low_freq_uncond, high_freq_uncond = fft(uncond.float())

        self.delta_high_freq = high_freq_uncond - high_freq_cond
        self.delta_low_freq = low_freq_uncond - low_freq_cond

    def get_cached_output(self, cond):
        t, c, h, w = cond.shape
        low_freq_cond, high_freq_cond = fft(cond.float())
        if self.current_step_id < self.low_freq_threshold:
            self.delta_low_freq = self.delta_low_freq * self.low_freq_weight
        elif self.current_step_id > self.high_freq_threshold:
            self.delta_high_freq = self.delta_high_freq * self.high_freq_weight

        new_high_freq_uncond = self.delta_high_freq + high_freq_cond
        new_low_freq_uncond = self.delta_low_freq + low_freq_cond

        combine_uncond = new_low_freq_uncond + new_high_freq_uncond
        combined_fft = torch.fft.ifftshift(combine_uncond)
        recovered_uncond = torch.fft.ifft2(combined_fft).real
        return recovered_uncond.to(cond.dtype).view(*cond.shape)


class B10CondCache:

    def __init__(self):
        self.current_step_id = 0
        self.max_step_id = None

        self.current_layer_id = None
        self.current_ts = None
        self.interval = 2
        # int -> tuple((ts, tensor), (ts, tensor))
        self.layer_id2attn_output = {}

        self.step_id_threshold = 10
        self.alpha = 0.5
        self.force_update = False

    def _if_use_cache(self, step_id):
        assert step_id is not None, "step_id is None"
        return (enable_b10_attn_cache() and step_id < self.max_step_id
                and step_id > self.step_id_threshold
                and step_id % self.interval != 0)

    def if_use_cache_this_step(self):
        return self._if_use_cache(self.current_step_id) and not self.force_update

    def if_use_cache_next_step(self):
        return self._if_use_cache(self.current_step_id + 1)

    def set_cached_output(self, attn_output):
        if self.current_layer_id not in self.layer_id2attn_output:
            self.layer_id2attn_output[self.current_layer_id] = (
                None, (self.current_ts, attn_output))
        else:
            _, prev_ts_attn_output = self.layer_id2attn_output[
                self.current_layer_id]
            self.layer_id2attn_output[self.current_layer_id] = (
                prev_ts_attn_output, (self.current_ts, attn_output))

    def get_cached_output(self):
        (ts1, attn_output1), (ts2,
                              attn_output2) = self.layer_id2attn_output[
                                  self.current_layer_id]
        return attn_output2 + (attn_output2 - attn_output1) * (
            self.current_ts - ts2) / (ts2 - ts1) * self.alpha


B10UNCONDCACHE = B10UncondCache()
B10CONDNCACHE = B10CondCache()
