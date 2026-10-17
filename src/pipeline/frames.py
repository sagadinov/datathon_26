"""Frame extraction: ffmpeg -> sampled frames (2 fps, long side 640) with timestamps.

Frames are cached on disk under cache/<sha1-of-file>/ so repeated runs are free.
"""
