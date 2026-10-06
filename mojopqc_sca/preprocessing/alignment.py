from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def align_xcorr(traces: np.ndarray, reference: np.ndarray, max_shift: int = 500, device: str | None = None) -> np.ndarray:
    traces_np = np.asarray(traces, dtype=np.float32)
    reference_np = np.asarray(reference, dtype=np.float32)
    if traces_np.ndim != 2 or reference_np.ndim != 1 or traces_np.shape[1] != reference_np.size:
        raise ValueError("traces must be 2-D and reference must match its sample length")

    B, L = traces_np.shape
    limit = min(max_shift, L - 1)
    if limit <= 0:
        return traces_np.copy()

    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    t = torch.from_numpy(traces_np).to(dev)
    ref = torch.from_numpy(reference_np).to(dev)

    ref_c = ref - ref.mean()
    t_c = t - t.mean(dim=1, keepdim=True)

    win_len = min(L - 2 * limit, 10000)
    if win_len >= 50 and L >= 2 * limit + win_len:
        kernel = ref_c[:win_len].view(1, 1, -1)
        signal = t_c[:, : win_len + 2 * limit].unsqueeze(1)
        corr = F.conv1d(signal, kernel)
        shifts = torch.argmax(corr, dim=-1).squeeze(1) - limit
    else:
        n_fft = 2 * L - 1
        t_f = torch.fft.rfft(t_c, n=n_fft)
        ref_f = torch.fft.rfft(ref_c.flip(0), n=n_fft)
        corr = torch.fft.irfft(t_f * ref_f, n=n_fft)
        center = L - 1
        window = corr[:, center - limit : center + limit + 1]
        shifts = torch.argmax(window, dim=1) - limit

    out = torch.empty_like(t)
    for i in range(B):
        s = int(shifts[i].item())
        if s != 0:
            out[i] = torch.roll(t[i], -s)
        else:
            out[i] = t[i]

    return out.cpu().numpy()

