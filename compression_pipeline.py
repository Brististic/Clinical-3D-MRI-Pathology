import numpy as np
import cv2
import heapq
import nibabel as nib
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. Lossless Huffman Coding for Binary Masks ---
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text_or_bytes):
    frequency = Counter(text_or_bytes)
    heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        node = heapq.heappop(heap)
        root = HuffmanNode(None, node.freq)
        root.left = node
        return root

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

def build_codes_helper(root, current_code, codes):
    if root is None:
        return
    if root.char is not None:
        codes[root.char] = current_code
        return
    build_codes_helper(root.left, current_code + "0", codes)
    build_codes_helper(root.right, current_code + "1", codes)

def huffman_compress_mask(binary_mask):
    """Compresses a flattened 2D/3D binary mask using Huffman coding."""
    flat_data = binary_mask.flatten().tobytes()
    original_size_bytes = len(flat_data)
    
    root = build_huffman_tree(flat_data)
    codes = {}
    build_codes_helper(root, "", codes)
    
    encoded_bitstring = "".join(codes[byte] for byte in flat_data)
    compressed_size_bytes = (len(encoded_bitstring) + 7) // 8
    compression_ratio = original_size_bytes / compressed_size_bytes
    
    return encoded_bitstring, original_size_bytes, compressed_size_bytes, compression_ratio

# --- 2. DCT-Based JPEG-Style Image Compression ---
def block_dct_compress(img, quality_factor=50):
    """
    Applies standard 8x8 block Discrete Cosine Transform (DCT),
    quantization based on JPEG luminance tables, and IDCT reconstruction.
    """
    # Standard JPEG Luminance Quantization Table
    q_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    # Scale quantization matrix according to Quality Factor (1-100)
    if quality_factor < 50:
        scale = 5000 / quality_factor
    else:
        scale = 200 - 2 * quality_factor
    scale = scale / 100.0
    q_table_scaled = np.clip(np.round(q_table * scale), 1, 255)

    h, w = img.shape
    # Pad to nearest multiple of 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge').astype(np.float32) - 128.0

    reconstructed_img = np.zeros_like(padded_img)

    for i in range(0, padded_img.shape[0], 8):
        for j in range(0, padded_img.shape[1], 8):
            block = padded_img[i:i+8, j:j+8]
            # Forward 2D-DCT
            dct_block = cv2.dct(block)
            # Quantization (Lossy Step)
            quantized = np.round(dct_block / q_table_scaled)
            # De-quantization
            dequantized = quantized * q_table_scaled
            # Inverse 2D-DCT
            idct_block = cv2.idct(dequantized)
            reconstructed_img[i:i+8, j:j+8] = idct_block

    # Unpad and shift back
    reconstructed_img = np.clip(reconstructed_img[:h, :w] + 128.0, 0, 255).astype(np.uint8)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((img.astype(np.float32) - reconstructed_img.astype(np.float32)) ** 2)
    psnr = 10 * np.log10((255.0 ** 2) / mse) if mse != 0 else float('inf')
    
    return reconstructed_img, psnr

if __name__ == "__main__":
    patient_id = "BraTS2021_00621"
    base_path = f"data/{patient_id}/{patient_id}"
    slice_idx = 80

    flair_vol = nib.load(f"{base_path}_flair.nii.gz").get_fdata(dtype=np.float32)
    seg_vol = (nib.load(f"{base_path}_seg.nii.gz").get_fdata(dtype=np.float32) > 0).astype(np.uint8)

    slice_flair = flair_vol[:, :, slice_idx]
    norm_flair = ((slice_flair - np.min(slice_flair)) / (np.max(slice_flair) - np.min(slice_flair)) * 255).astype(np.uint8)
    slice_seg = seg_vol[:, :, slice_idx]

    # 1. Test Huffman Compression on Binary Segmentation Mask
    bitstring, orig_bytes, comp_bytes, ratio = huffman_compress_mask(slice_seg)
    print("=" * 50)
    print("1. Lossless Huffman Mask Compression:")
    print(f"Original Uncompressed Mask Size : {orig_bytes} bytes")
    print(f"Compressed Mask Payload Size   : {comp_bytes} bytes")
    print(f"Compression Ratio (CR)         : {ratio:.2f}:1 ({(1 - comp_bytes/orig_bytes)*100:.1f}% space saved)")

    # 2. Test DCT JPEG Compression on FLAIR Slice
    reconstructed, psnr_val = block_dct_compress(norm_flair, quality_factor=40)
    print("\n2. DCT-Based JPEG Reconstruction:")
    print(f"Reconstruction PSNR            : {psnr_val:.2f} dB")
    print("=" * 50)

    # 3. Visual Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(norm_flair, cmap="gray")
    axes[0].set_title("Original Input Slice")
    
    axes[1].imshow(reconstructed, cmap="gray")
    axes[1].set_title(f"DCT Reconstructed (PSNR: {psnr_val:.2f} dB)")
    
    # Absolute difference map showing quantization error
    diff_map = np.abs(norm_flair.astype(np.int16) - reconstructed.astype(np.int16)).astype(np.uint8)
    axes[2].imshow(diff_map * 5, cmap="hot") # Multiplied by 5 to highlight residual error
    axes[2].set_title("Residual Error Map (Amplified x5)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()