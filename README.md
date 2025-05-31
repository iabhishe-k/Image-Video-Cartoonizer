
# Cartoonizer

A simple Python program to convert images and videos into cartoon-style visuals using OpenCV and KMeans clustering.

---

## Features

- Cartoonize an input image by reducing colors and emphasizing edges.
- Cartoonize video frames in real-time.
- Adjustable number of color clusters for quantization.
- View original and cartoonized images side-by-side.

---

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- scikit-learn

You can install the required packages via pip:

```bash
pip install opencv-python numpy scikit-learn
```

---

## Usage

Run the script with either an image file or a video file input.

### Cartoonize an Image

```bash
python cartoonizer.py --image path/to/image.jpg
```

This will display a window showing the original and cartoonized image side-by-side.

### Cartoonize a Video

```bash
python cartoonizer.py --video path/to/video.mp4
```

This will open a window showing the cartoonized video frames in real time.

- Press `d` to exit the video window.

---

## Parameters

- `--image`: Path to the input image file.
- `--video`: Path to the input video file.

You can specify either one or both inputs (image and video).

---

## How it works

1. **Read and resize image/video frame.**  
2. **Edge detection:** Using adaptive thresholding on a blurred grayscale version.  
3. **Color quantization:** Reducing colors with KMeans clustering.  
4. **Combine edges and quantized colors** to create a cartoon effect.

---

## Example

```bash
python cartoonizer.py --image 'image_path'
python cartoonizer.py --video 'video_path'
```

---

## Notes

- For videos, the cartoonized output plays in a window; press `d` to quit.
- The default number of color clusters for images is 9, for video is 7. You can modify this in the code if needed.