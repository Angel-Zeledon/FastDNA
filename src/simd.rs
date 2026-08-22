// src/simd.rs

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

/// Encodes a 32-byte ASCII DNA chunk into 32 2-bit values simultaneously.
pub fn encode_32_bytes_simd(chunk: &[u8; 32], output: &mut [u8; 32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                encode_32_bytes_avx2(chunk, output);
                return;
            }
        }
    }

    encode_32_bytes_scalar(chunk, output);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn encode_32_bytes_avx2(chunk: &[u8; 32], output: &mut [u8; 32]) {
    let lut = _mm256_setr_epi8(
        -1, 0, -1, 1, 3, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 0, -1, 1, 3, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1,
    );

    let raw_bytes = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
    let low_nibble_mask = _mm256_set1_epi8(0x0F);
    let indices = _mm256_and_si256(raw_bytes, low_nibble_mask);
    let result = _mm256_shuffle_epi8(lut, indices);

    _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
}

#[inline(always)]
fn encode_32_bytes_scalar(chunk: &[u8; 32], output: &mut [u8; 32]) {
    for i in 0..32 {
        output[i] = match chunk[i] {
            b'A' | b'a' => 0b00,
            b'C' | b'c' => 0b01,
            b'G' | b'g' => 0b10,
            b'T' | b't' | b'U' | b'u' => 0b11,
            _ => 255,
        };
    }
}