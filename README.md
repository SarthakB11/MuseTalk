# MuseTalk Super Resolution Video Processing

## Overview
This script applies super-resolution to videos using a pre-trained model from TensorFlow Hub. It enhances the quality of the input video by upscaling each frame using a super-resolution model and outputs the processed video.

## Features
- Supports super-resolution using the ESRGAN model.
- Processes input videos frame by frame.
- Maintains the original audio from the input video.
- Outputs a high-quality video with enhanced resolution.

## How to Use
Run the following command to apply super-resolution on the video:

```bash
python super_resolution.py --superres GFPGAN -iv input_video.mp4 -ia input.mp3 -o output.mp4
```

### Arguments:
- `--superres`: The super-resolution model to use (e.g., `GFPGAN`).
- `input_video.mp4`: Path to the input video.
- `-ia input.mp3`: Path to the input audio (will be kept intact in the output).
- `-o output.mp4`: Path for the output video with enhanced resolution.

## Contribution to MuseTalk
This script is my contribution to the MuseTalk repository, where I implemented a video processing pipeline that enhances video resolution using advanced super-resolution techniques. The pipeline processes each video frame, applies the super-resolution model, and reassembles the frames into a high-resolution video while preserving the original audio.

---
## Please refer to original project for all the details and installation.

