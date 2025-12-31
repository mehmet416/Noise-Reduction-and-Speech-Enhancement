import numpy as np

# --------------------------------------------------
# Frame a signal (causal-friendly)
# --------------------------------------------------
def frame_signal(x, frame_len, hop_len):
    num_frames = 1 + (len(x) - frame_len) // hop_len
    frames = np.zeros((num_frames, frame_len))

    for i in range(num_frames):
        start = i * hop_len
        frames[i] = x[start:start + frame_len]

    return frames

# --------------------------------------------------
# Overlap-add reconstruction
# --------------------------------------------------
def overlap_add(frames, hop_len):
    frame_len = frames.shape[1]
    out_len = frame_len + hop_len * (frames.shape[0] - 1)

    x = np.zeros(out_len)
    for i, frame in enumerate(frames):
        start = i * hop_len
        x[start:start + frame_len] += frame

    return x
