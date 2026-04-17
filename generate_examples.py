import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

INPUT_DIR = Path('inputs')

def generate_examples_for_image(base_path: Path):
    print(f"\nProcessing: {base_path.name}")
    base_img = cv2.imread(str(base_path))
    if base_img is None:
        print(f"  Error: Could not read {base_path.name}")
        return

    stem = base_path.stem
    h, w = base_img.shape[:2]

    # --- Manipulation 1: Affine Splicing (Rotate, Scale, Flip) ---
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.2)
    rotated = cv2.warpAffine(base_img, M, (w, h))
    flipped = cv2.flip(rotated, 1)
    
    out_ex1 = INPUT_DIR / f"{stem}_ex1_affine.png"
    cv2.imwrite(str(out_ex1), flipped)
    print(f"  -> Generated {out_ex1.name}")

    # --- Manipulation 2: Signal Degradation (Noise + JPEG) ---
    img_pil = Image.open(base_path).convert('RGB')
    
    arr = np.array(img_pil, dtype=np.float32)
    noise = np.random.normal(0, 20, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img_noisy = Image.fromarray(noisy)
    
    out_ex2 = INPUT_DIR / f"{stem}_ex2_degraded.jpg"
    img_noisy.save(str(out_ex2), 'JPEG', quality=25)
    print(f"  -> Generated {out_ex2.name}")

    # --- Manipulation 3: Intra-Image CMFD (Copy-Move) ---
    cmfd_img = base_img.copy()
    patch_size = min(w, h) // 4
    src_y, src_x = int(h*0.1), int(w*0.1)
    dst_y, dst_x = int(h*0.6), int(w*0.6)
    
    patch = cmfd_img[src_y:src_y+patch_size, src_x:src_x+patch_size].copy()
    cmfd_img[dst_y:dst_y+patch_size, dst_x:dst_x+patch_size] = patch
    
    out_ex3 = INPUT_DIR / f"{stem}_ex3_copymove.png"
    cv2.imwrite(str(out_ex3), cmfd_img)
    print(f"  -> Generated {out_ex3.name}")

    # --- Manipulation 4: Extreme Exposure Shift ---
    # Simulating saving the image with different exposure/brightness settings
    img_pil4 = Image.open(base_path).convert('RGB')
    enhancer = ImageEnhance.Contrast(img_pil4)
    img_pil4 = enhancer.enhance(1.8) # High contrast
    bright_enhancer = ImageEnhance.Brightness(img_pil4)
    img_pil4 = bright_enhancer.enhance(0.5) # Darkened
    
    out_ex4 = INPUT_DIR / f"{stem}_ex4_exposure.png"
    img_pil4.save(str(out_ex4))
    print(f"  -> Generated {out_ex4.name}")

    # --- Manipulation 5: Malicious Blackout Masking ---
    # Simulates covering up a spot or cell intentionally
    blackout_img = base_img.copy()
    mask_y, mask_x = int(h*0.4), int(w*0.4)
    mask_size = int(min(w,h) * 0.15)
    
    cv2.rectangle(blackout_img, (mask_x, mask_y), (mask_x+mask_size, mask_y+mask_size), (0,0,0), -1)
    
    out_ex5 = INPUT_DIR / f"{stem}_ex5_blackout.png"
    cv2.imwrite(str(out_ex5), blackout_img)
    print(f"  -> Generated {out_ex5.name}")


def main():
    print("Scanning inputs/ directory for base files...")
    all_bases = list(INPUT_DIR.glob('base*.png')) + list(INPUT_DIR.glob('base*.jpg'))
    base_files = [f for f in all_bases if "_ex" not in f.name]
    
    if not base_files:
        print("No base files starting with 'base_' found in inputs/")
        return

    for base_file in base_files:
        generate_examples_for_image(base_file)

    print(f"\nGeneration complete. Processed {len(base_files)} images, 5 manipulations each.")

if __name__ == "__main__":
    main()
