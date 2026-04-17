import io
import numpy as np
import string
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


class RandomText:
    def __init__(self, p=0.5):
        self.ascii_chars = np.array(
            list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        )
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        font = ImageFont.load_default()
        x = np.random.randint(0, img.width / 2)
        y = np.random.randint(0, img.height)
        strlen = np.random.randint(5, 15)
        text = "".join(np.random.choice(self.ascii_chars, strlen))
        ImageDraw.Draw(img).text(
            (x, y), text, fill=np.random.randint(0, 256), font=font
        )
        return img


class RandomRect:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        width, height = img.size
        x = np.random.randint(0, width / 2)
        y = np.random.randint(0, height / 2)
        rect_width = np.random.randint(width / 3, width / 2)
        rect_height = np.random.randint(height / 3, height / 2)
        ImageDraw.Draw(img).rectangle(
            [(x, y), (x + rect_width, y + rect_height)],
            fill=None,
            width=np.random.randint(1, 3),
            outline=np.random.randint(0, 256),
        )
        return img


class RandomErase:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        width, height = img.size
        r = np.random.randint(0.05 * width, 0.2 * width)
        x = np.random.randint(r, width - r)
        y = np.random.randint(r, height - r)
        ImageDraw.Draw(img).ellipse(
            [(x - r, y - r), (x + r, y + r)], outline=None, fill=(0,)
        )
        return img


class JPEGCompression:
    """Simulate JPEG re-save artifacts by encoding to a JPEG buffer at a random
    quality level and decoding back. This is one of the most common degradation
    patterns in plagiarized scientific images that are re-exported or
    screenshot-captured."""

    def __init__(self, quality_range=(30, 95), p=0.4):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        quality = int(np.random.randint(self.quality_range[0], self.quality_range[1] + 1))
        buffer = io.BytesIO()
        # JPEG requires RGB or L mode
        save_mode = img.mode if img.mode in ("RGB", "L") else "RGB"
        img.convert(save_mode).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        compressed.load()  # force full decode before buffer is GC'd
        # Return in same mode as input
        if compressed.mode != img.mode:
            compressed = compressed.convert(img.mode)
        return compressed


class GaussianNoise:
    """Add pixel-level Gaussian noise to simulate sensor noise, scanner
    artifacts, and general signal degradation."""

    def __init__(self, std_range=(5, 25), p=0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        arr = np.array(img, dtype=np.float32)
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode=img.mode)


class ColorShift:
    """Randomly adjust brightness and contrast to simulate color-space
    manipulations beyond simple brightness jitter. Covers scenarios where
    images are re-processed with different exposure or display settings."""

    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), p=0.3):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        brightness_factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast_factor = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        img = ImageEnhance.Brightness(img).enhance(float(brightness_factor))
        img = ImageEnhance.Contrast(img).enhance(float(contrast_factor))
        return img
