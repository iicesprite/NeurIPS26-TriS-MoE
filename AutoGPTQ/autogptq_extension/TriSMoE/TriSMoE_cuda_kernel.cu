#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef USE_ROCM
#include <mma.h>
#endif
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700) || defined(USE_ROCM)
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int *address_as_ui = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        hsum += val;
        old = reinterpret_cast<size_t>(address) & 2
                 ? (old & 0xffff) | (hsum << 16)
                 : (old & 0xffff0000) | hsum;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}
__device__ __forceinline__ void atomicAdd(__half* address, c10::Half val) {
    unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(reinterpret_cast<char*>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        __half_raw hsum;
        hsum.x = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        half tmpres = __hadd(hsum, val);
        hsum = __half_raw(tmpres);
        old = reinterpret_cast<size_t>(address) & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}
#endif

namespace {

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Half, #x " must be float16")
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be int32")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be int64")
#define CHECK_BOOL(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Bool, #x " must be bool")

constexpr int PACKED4_TILE_COLS = 64;
constexpr int PACKED4_BLOCK_ROWS = 64;
constexpr int PACKED4_FAST_BLOCKWIDTH = 64;
constexpr int PACKED4_FAST_BLOCKHEIGHT = 8;
constexpr int STRUCTURED_TILE_K = 64;
constexpr int STRUCTURED_BLOCK_THREADS = 128;
constexpr int STRUCTURED_BLOCK_OUT = 64;
constexpr int BLOCK_GR_TILE_OUT = 64;
constexpr int BLOCK_GR_PACKED_ROWS = STRUCTURED_TILE_K / 32;
constexpr int BLOCK_GR_WORDS_PER_BLOCK = BLOCK_GR_PACKED_ROWS * BLOCK_GR_TILE_OUT;
constexpr int BLOCK_GR_META_K_MASK = 0xF;
constexpr int BLOCK_GR_META_RAW_FLAG = 1 << 4;
constexpr int BLOCK_GR_META_ONE_COUNT_SHIFT = 8;
constexpr int COMPACT3_SINGLE_BLOCK_GR_K_WORKERS = 2;
constexpr int STRUCTURED_TC_M_TILE = 16;
constexpr int STRUCTURED_TC_N_TILE = 64;
constexpr int STRUCTURED_TC_K_TILE = 64;
constexpr int STRUCTURED_TC_K_STEP = 16;
constexpr int STRUCTURED_TC_WARPS = 4;

#ifndef TRIS_GROUPED_M_TILE
#define TRIS_GROUPED_M_TILE 4
#endif
constexpr int STRUCTURED_GROUPED_M_TILE = TRIS_GROUPED_M_TILE;
constexpr int PACKED4_DEQ_LUT_SIZE = 256;
constexpr int NS_FUSED_CHUNK = 64;
constexpr int GROUPED_NS_FUSED_BLOCK_THREADS = 256;

__host__ __device__ __forceinline__ int ceil_div_int(int num, int den) {
  return (num + den - 1) / den;
}

constexpr int PREFILL_AUTO_SPLIT_K_MIN_K_TILES_PER_SLICE = 4;
constexpr int BLOCK_GR_DECODE_AUTO_SPLIT_K_TARGET_MULTIPLIER = 32;

int resolve_auto_split_k_slices(int num_k_tiles, int base_tiles, bool decode_phase, int sm_count) {
  const int target_multiplier = decode_phase ? 8 : 2;
  const int target_tiles = sm_count * target_multiplier;
  if (base_tiles >= target_tiles) {
    return 1;
  }

  if (!decode_phase && base_tiles >= sm_count) {
    return 1;
  }

  const int needed = ceil_div_int(target_tiles, base_tiles);
  int max_split_k = num_k_tiles;
  if (!decode_phase) {
    max_split_k = num_k_tiles / PREFILL_AUTO_SPLIT_K_MIN_K_TILES_PER_SLICE;
    if (max_split_k <= 1) {
      return 1;
    }
  }

  const int split_k = needed < max_split_k ? needed : max_split_k;
  return split_k > 1 ? split_k : 1;
}

int resolve_grouped_split_k_slices(int rank, int total_problem_tiles, int out_tiles, bool decode_phase) {
  const int num_k_tiles = ceil_div_int(rank, STRUCTURED_TILE_K);
  if (num_k_tiles <= 1 || total_problem_tiles <= 0 || out_tiles <= 0) {
    return 1;
  }

  const char* phase_env = std::getenv(decode_phase ? "TRIS_GROUPED_DECODE_SPLIT_K" : "TRIS_GROUPED_PREFILL_SPLIT_K");
  if (phase_env != nullptr && phase_env[0] != 0) {
    const int requested = std::atoi(phase_env);
    if (requested > 1) {
      return requested < num_k_tiles ? requested : num_k_tiles;
    }
    return 1;
  }

  const char* env = std::getenv("TRIS_GROUPED_SPLIT_K");
  if (env != nullptr && env[0] != 0) {
    const int requested = std::atoi(env);
    if (requested > 1) {
      return requested < num_k_tiles ? requested : num_k_tiles;
    }
    return 1;
  }

  const auto* props = at::cuda::getCurrentDeviceProperties();
  if (props == nullptr || props->multiProcessorCount <= 0) {
    return 1;
  }

  const int base_tiles = total_problem_tiles * out_tiles;
  return resolve_auto_split_k_slices(num_k_tiles, base_tiles, decode_phase, props->multiProcessorCount);
}

int resolve_grouped_block_gr_split_k_slices(int rank, int total_problem_tiles, int out_tiles, bool decode_phase) {
  const int num_k_tiles = ceil_div_int(rank, STRUCTURED_TILE_K);
  if (num_k_tiles <= 1 || total_problem_tiles <= 0 || out_tiles <= 0) {
    return 1;
  }

  const char* block_gr_phase_env = std::getenv(decode_phase ? "TRIS_GROUPED_BLOCK_GR_DECODE_SPLIT_K" : "TRIS_GROUPED_BLOCK_GR_PREFILL_SPLIT_K");
  if (block_gr_phase_env != nullptr && block_gr_phase_env[0] != 0) {
    const int requested = std::atoi(block_gr_phase_env);
    if (requested > 1) {
      return requested < num_k_tiles ? requested : num_k_tiles;
    }
    return 1;
  }

  const char* block_gr_env = std::getenv("TRIS_GROUPED_BLOCK_GR_SPLIT_K");
  if (block_gr_env != nullptr && block_gr_env[0] != 0) {
    const int requested = std::atoi(block_gr_env);
    if (requested > 1) {
      return requested < num_k_tiles ? requested : num_k_tiles;
    }
    return 1;
  }

  const char* phase_env = std::getenv(decode_phase ? "TRIS_GROUPED_DECODE_SPLIT_K" : "TRIS_GROUPED_PREFILL_SPLIT_K");
  if (phase_env != nullptr && phase_env[0] != 0) {
    const int requested = std::atoi(phase_env);
    if (requested > 1) {
      return requested < num_k_tiles ? requested : num_k_tiles;
    }
    return 1;
  }

  const char* env = std::getenv("TRIS_GROUPED_SPLIT_K");
  if (env != nullptr && env[0] != 0) {
    const int requested = std::atoi(env);
    if (requested > 1) {
      return requested < num_k_tiles ? requested : num_k_tiles;
    }
    return 1;
  }

  const auto* props = at::cuda::getCurrentDeviceProperties();
  if (props == nullptr || props->multiProcessorCount <= 0) {
    return 1;
  }

  const int base_tiles = total_problem_tiles * out_tiles;
  if (!decode_phase) {
    return resolve_auto_split_k_slices(num_k_tiles, base_tiles, false, props->multiProcessorCount);
  }

  const int target_tiles = props->multiProcessorCount * BLOCK_GR_DECODE_AUTO_SPLIT_K_TARGET_MULTIPLIER;
  if (base_tiles >= target_tiles) {
    return 1;
  }

  const int needed = ceil_div_int(target_tiles, base_tiles);
  int split_k = 1;
  while (split_k < needed && split_k < num_k_tiles) {
    split_k <<= 1;
  }
  if (split_k > num_k_tiles) {
    split_k = num_k_tiles;
  }
  return split_k > 1 ? split_k : 1;
}

bool compact3_prefill_tensor_core_disabled() {
  const char* env = std::getenv("TRIS_DISABLE_COMPACT3_PREFILL_TC");
  if (env == nullptr || env[0] == 0) {
    return false;
  }
  return std::atoi(env) != 0;
}

bool can_use_compact3_prefill_tensor_core(bool requested, int batch, int split_k_slices) {
#if defined(USE_ROCM)
  (void)requested;
  (void)batch;
  (void)split_k_slices;
  return false;
#else
  if (!requested || batch < STRUCTURED_TC_M_TILE || split_k_slices != 1 || compact3_prefill_tensor_core_disabled()) {
    return false;
  }
  const auto* props = at::cuda::getCurrentDeviceProperties();
  return props != nullptr && props->major >= 7;
#endif
}

int resolve_grouped_ns_split_k_slices(int cols, int batch, int row_tiles, bool decode_phase) {
  const int num_k_tiles = ceil_div_int(cols, PACKED4_TILE_COLS);
  if (num_k_tiles <= 1 || batch <= 0 || row_tiles <= 0) {
    return 1;
  }

  const char* phase_env = std::getenv(decode_phase ? "TRIS_GROUPED_NS_DECODE_SPLIT_K" : "TRIS_GROUPED_NS_PREFILL_SPLIT_K");
  if (phase_env != nullptr && phase_env[0] != 0) {
    const int requested = std::atoi(phase_env);
    if (requested > 1) {
      return requested < num_k_tiles ? requested : num_k_tiles;
    }
    return 1;
  }

  const char* env = std::getenv("TRIS_GROUPED_NS_SPLIT_K");
  if (env != nullptr && env[0] != 0) {
    const int requested = std::atoi(env);
    if (requested > 1) {
      return requested < num_k_tiles ? requested : num_k_tiles;
    }
    return 1;
  }

  const auto* props = at::cuda::getCurrentDeviceProperties();
  if (props == nullptr || props->multiProcessorCount <= 0) {
    return 1;
  }

  const int base_tiles = batch * row_tiles;
  return resolve_auto_split_k_slices(num_k_tiles, base_tiles, decode_phase, props->multiProcessorCount);
}


int resolve_packed4_split_k_slices(int cols, int batch, int row_tiles, bool already_k_parallel) {
  if (already_k_parallel) {
    return 1;
  }

  const int num_k_tiles = ceil_div_int(cols, PACKED4_TILE_COLS);
  if (num_k_tiles <= 1 || batch <= 0 || row_tiles <= 0) {
    return 1;
  }

  const char* env = std::getenv("TRIS_PACKED4_SPLIT_K");
  if (env != nullptr && env[0] != '\0') {
    const int requested = std::atoi(env);
    if (requested > 1) {
      return requested < num_k_tiles ? requested : num_k_tiles;
    }
    return 1;
  }

  const auto* props = at::cuda::getCurrentDeviceProperties();
  if (props == nullptr || props->multiProcessorCount <= 0) {
    return 1;
  }

  const int base_tiles = batch * row_tiles;
  return resolve_auto_split_k_slices(num_k_tiles, base_tiles, false, props->multiProcessorCount);
}

__device__ __forceinline__ unsigned low_bit_mask(int valid_bits);

__device__ __forceinline__ half unpack_half_from_u64(unsigned long long packed, unsigned shift) {
  return __ushort_as_half(static_cast<unsigned short>((packed >> shift) & 0xFFFFull));
}

__device__ __forceinline__ float reduce_half2(half2 v) {
  return __half2float(__low2half(v)) + __half2float(__high2half(v));
}

__device__ __forceinline__ void init_packed4_deq_lut(half2* lut) {
  const int linear_tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int linear_stride = blockDim.x * blockDim.y;
  for (int val = linear_tid; val < PACKED4_DEQ_LUT_SIZE; val += linear_stride) {
    lut[val] = __halves2half2(
        __int2half_rn(val & 0x0F),
        __int2half_rn((val >> 4) & 0x0F));
  }
}

__device__ __forceinline__ unsigned read_block_gr_bit(const int* __restrict__ words, int bit_idx) {
  return (static_cast<unsigned>(words[bit_idx >> 5]) >> (bit_idx & 31)) & 1U;
}

__device__ __forceinline__ unsigned read_block_gr_bits_upto_32(
    const int* __restrict__ words,
    int total_bits,
    int bit_pos,
    int count) {
  if (count <= 0 || bit_pos >= total_bits) {
    return 0U;
  }

  const int total_words = (total_bits + 31) >> 5;
  const int word_idx = bit_pos >> 5;
  const int bit_idx = bit_pos & 31;
  unsigned bits = static_cast<unsigned>(words[word_idx]) >> bit_idx;
  const int bits_in_word = 32 - bit_idx;
  if (bits_in_word < count && word_idx + 1 < total_words) {
    bits |= static_cast<unsigned>(words[word_idx + 1]) << bits_in_word;
  }
  if (count < 32) {
    bits &= (1U << count) - 1U;
  }
  return bits;
}

__device__ __forceinline__ int decode_block_gr_zero_run_k3(
    const int* __restrict__ words,
    int total_bits,
    int* bit_pos) {
  if (*bit_pos >= total_bits) {
    return 0;
  }

  const int total_words = (total_bits + 31) >> 5;
  int quotient = 0;
  int word_idx = *bit_pos >> 5;
  int bit_idx = *bit_pos & 31;
  unsigned word = static_cast<unsigned>(words[word_idx]);
  if (bit_idx != 0) {
    word >>= bit_idx;
  }

  while (true) {
    if (word != 0U) {
      const int first_one = __ffs(static_cast<int>(word)) - 1;
      quotient += first_one;
      *bit_pos += first_one + 1;
      word >>= (first_one + 1);
      const int bits_left_in_word = 32 - bit_idx - (first_one + 1);
      const int remainder_bits = total_bits - *bit_pos;
      const int bits_to_take = remainder_bits < 3 ? remainder_bits : 3;
      unsigned remainder = 0U;
      if (bits_to_take > 0) {
        if (bits_left_in_word >= bits_to_take) {
          remainder = word & ((1U << bits_to_take) - 1U);
        } else {
          remainder = read_block_gr_bits_upto_32(words, total_bits, *bit_pos, bits_to_take) & 0x7U;
        }
      }
      *bit_pos += bits_to_take;
      return (quotient << 3) | static_cast<int>(remainder);
    }

    const int remaining_bits = total_bits - *bit_pos;
    const int available = remaining_bits < (32 - bit_idx) ? remaining_bits : (32 - bit_idx);
    quotient += available;
    *bit_pos += available;
    if (*bit_pos >= total_bits) {
      return quotient << 3;
    }

    ++word_idx;
    bit_idx = 0;
    if (word_idx >= total_words) {
      return quotient << 3;
    }
    word = static_cast<unsigned>(words[word_idx]);
  }
}

__device__ __forceinline__ int decode_block_gr_zero_run(
    const int* __restrict__ words,
    int total_bits,
    int* bit_pos,
    int k) {
  int quotient = 0;
  while (*bit_pos < total_bits) {
    const unsigned bit = read_block_gr_bit(words, *bit_pos);
    ++(*bit_pos);
    if (bit != 0U) {
      break;
    }
    ++quotient;
  }

  int remainder = 0;
  for (int shift = 0; shift < k && *bit_pos < total_bits; ++shift) {
    remainder |= static_cast<int>(read_block_gr_bit(words, *bit_pos)) << shift;
    ++(*bit_pos);
  }
  return (quotient << k) | remainder;
}

__device__ __forceinline__ void decode_block_gr_block(
    const int* __restrict__ words,
    const int* __restrict__ offsets,
    const int* __restrict__ meta,
    int block_idx,
    unsigned (*decoded_words)[BLOCK_GR_TILE_OUT]) {
  unsigned* decoded_words_flat = &decoded_words[0][0];
  for (int word_idx = 0; word_idx < BLOCK_GR_WORDS_PER_BLOCK; ++word_idx) {
    decoded_words_flat[word_idx] = 0U;
  }

  const int meta_word = meta[block_idx];
  const int word_begin = offsets[block_idx];
  const int word_end = offsets[block_idx + 1];
  const int word_count = word_end - word_begin;
  if (word_count <= 0) {
    return;
  }

  const int* payload = words + word_begin;
  if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
    for (int word_idx = 0; word_idx < BLOCK_GR_WORDS_PER_BLOCK && word_idx < word_count; ++word_idx) {
      decoded_words_flat[word_idx] = static_cast<unsigned>(payload[word_idx]);
    }
    return;
  }

  const int k = meta_word & BLOCK_GR_META_K_MASK;
  const int one_count = meta_word >> BLOCK_GR_META_ONE_COUNT_SHIFT;
  const int total_bits = word_count * 32;
  constexpr int decoded_total_bits = BLOCK_GR_WORDS_PER_BLOCK * 32;
  int bit_pos = 0;
  int flat_pos = 0;
  if (k == 3) {
    for (int one_idx = 0; one_idx < one_count; ++one_idx) {
      flat_pos += decode_block_gr_zero_run_k3(payload, total_bits, &bit_pos);
      if (flat_pos >= decoded_total_bits) {
        break;
      }
      decoded_words_flat[flat_pos >> 5] |= 1U << (flat_pos & 31);
      ++flat_pos;
    }
    return;
  }

  for (int one_idx = 0; one_idx < one_count; ++one_idx) {
    flat_pos += decode_block_gr_zero_run(payload, total_bits, &bit_pos, k);
    if (flat_pos >= decoded_total_bits) {
      break;
    }
    decoded_words_flat[flat_pos >> 5] |= 1U << (flat_pos & 31);
    ++flat_pos;
  }
}

__device__ __forceinline__ void decode_block_gr_tile(
    const int* __restrict__ words,
    const int* __restrict__ offsets,
    const int* __restrict__ meta,
    int packed_rows,
    int outfeatures,
    int tile_k,
    int out_block,
    unsigned (*decoded_words)[BLOCK_GR_TILE_OUT]) {
  const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
  const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  const int k_block = tile_k / STRUCTURED_TILE_K;
  if (words == nullptr || offsets == nullptr || meta == nullptr || k_block < 0 || k_block >= num_k_blocks || out_block < 0 || out_block >= num_out_blocks) {
    for (int local_packed_row = 0; local_packed_row < BLOCK_GR_PACKED_ROWS; ++local_packed_row) {
      for (int local_out = 0; local_out < BLOCK_GR_TILE_OUT; ++local_out) {
        decoded_words[local_packed_row][local_out] = 0U;
      }
    }
    return;
  }
  decode_block_gr_block(words, offsets, meta, k_block * num_out_blocks + out_block, decoded_words);
}

__device__ __forceinline__ void load_predecoded_block_gr_tile(
    const int* __restrict__ decoded_q_gmap_words,
    int decoded_gmap_block_count,
    int flat_block_idx,
    bool expert_valid,
    unsigned (*decoded_words)[BLOCK_GR_TILE_OUT]) {
  for (int idx = threadIdx.x; idx < BLOCK_GR_WORDS_PER_BLOCK; idx += blockDim.x) {
    unsigned value = 0U;
    if (expert_valid && decoded_q_gmap_words != nullptr && flat_block_idx >= 0 && flat_block_idx < decoded_gmap_block_count) {
      value = static_cast<unsigned>(decoded_q_gmap_words[flat_block_idx * BLOCK_GR_WORDS_PER_BLOCK + idx]);
    }
    decoded_words[idx / BLOCK_GR_TILE_OUT][idx % BLOCK_GR_TILE_OUT] = value;
  }
}

__device__ __forceinline__ float dot8_packed4_shared(
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    const int* __restrict__ g_idx,
    const half* __restrict__ x_tile,
    const half2* __restrict__ qbyte_lut,
    int rows,
    int qzeros_cols,
    int row,
    int global_col,
    int tile_valid_cols) {
  const unsigned qword = static_cast<unsigned>(qweight[(global_col >> 3) * rows + row]);
  float acc = 0.0f;

#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const int lane0 = pair * 2;
    if (lane0 >= tile_valid_cols) {
      break;
    }

    const int c0 = global_col + lane0;
    const int c1 = c0 + 1;
    const bool has_c1 = (lane0 + 1) < tile_valid_cols;

    const half x0 = x_tile[lane0];
    const half x1 = has_c1 ? x_tile[lane0 + 1] : __float2half(0.0f);

    const int g0 = g_idx[c0];
    const int g1 = has_c1 ? g_idx[c1] : g0;
    const int zshift = (row & 7) * 4;
    const unsigned zword0 = static_cast<unsigned>(qzeros[g0 * qzeros_cols + (row >> 3)]);
    const unsigned zword1 = static_cast<unsigned>(qzeros[g1 * qzeros_cols + (row >> 3)]);
    const half scale0 = scales[g0 * rows + row];
    const half scale1 = has_c1 ? scales[g1 * rows + row] : __float2half(0.0f);
    const half zero0 = __int2half_rn(((zword0 >> zshift) & 0xF) + 1);
    const half zero1 = has_c1 ? __int2half_rn(((zword1 >> zshift) & 0xF) + 1) : __float2half(0.0f);

    const unsigned qbyte = (qword >> (pair * 8)) & 0xFF;
    const half2 decoded = __hmul2(
        __hsub2(qbyte_lut[qbyte], __halves2half2(zero0, zero1)),
        __halves2half2(scale0, scale1));
    const half2 xpair = __halves2half2(x0, x1);
    acc += reduce_half2(__hmul2(decoded, xpair));
  }

  return acc;
}

__device__ __forceinline__ float load_packed4_weight(
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    const int* __restrict__ g_idx,
    int rows,
    int qzeros_cols,
    int row,
    int col) {
  const int group = g_idx[col];
  if (group < 0) {
    return 0.0f;
  }

  const unsigned qword = static_cast<unsigned>(qweight[(col >> 3) * rows + row]);
  const unsigned qvalue = (qword >> ((col & 7) * 4)) & 0x0F;
  const unsigned zword = static_cast<unsigned>(qzeros[group * qzeros_cols + (row >> 3)]);
  const unsigned zvalue = (zword >> ((row & 7) * 4)) & 0x0F;
  const float zero = static_cast<float>(zvalue + 1);
  const float scale = __half2float(scales[group * rows + row]);
  return (static_cast<float>(qvalue) - zero) * scale;
}

__device__ __forceinline__ unsigned low_bit_mask(int valid_bits) {
  if (valid_bits >= 32) {
    return 0xFFFFFFFFu;
  }
  return (1u << valid_bits) - 1u;
}

__device__ __forceinline__ bool get_sparse_qweight2_bit(
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    unsigned c_map_word,
    int packed_row,
    int bit_idx,
    int outfeatures,
    int out_idx) {
  if (((c_map_word >> bit_idx) & 1U) == 0U) {
    return false;
  }
  const unsigned prefix_mask = bit_idx == 0 ? 0U : low_bit_mask(bit_idx);
  const int sparse_bit_idx = qweight2_sparse_prefix[packed_row] + __popc(c_map_word & prefix_mask);
  const int sparse_packed_row = sparse_bit_idx >> 5;
  const int sparse_bit = sparse_bit_idx & 31;
  const unsigned qword = static_cast<unsigned>(qweight2_sparse[sparse_packed_row * outfeatures + out_idx]);
  return ((qword >> sparse_bit) & 1U) != 0U;
}


__device__ __forceinline__ bool get_sparse_inactive_bit(
    const int* __restrict__ packed_sparse,
    const int* __restrict__ sparse_prefix,
    unsigned c_map_word,
    int packed_row,
    int bit_idx,
    int outfeatures,
    int out_idx) {
  if (((c_map_word >> bit_idx) & 1U) != 0U) {
    return false;
  }
  const unsigned prefix_mask = bit_idx == 0 ? 0U : low_bit_mask(bit_idx);
  const int inactive_before = bit_idx - __popc(c_map_word & prefix_mask);
  const int sparse_bit_idx = sparse_prefix[packed_row] + inactive_before;
  const int sparse_packed_row = sparse_bit_idx >> 5;
  const int sparse_bit = sparse_bit_idx & 31;
  const unsigned qword = static_cast<unsigned>(packed_sparse[sparse_packed_row * outfeatures + out_idx]);
  return ((qword >> sparse_bit) & 1U) != 0U;
}

__device__ __forceinline__ unsigned reconstruct_sparse_qweight2_dense_word(
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    unsigned c_map_word,
    int packed_row,
    int outfeatures,
    int out_idx) {
  unsigned dense_word = 0U;
  if (c_map_word == 0U) {
    return dense_word;
  }

  unsigned pending = c_map_word;
  int sparse_bit_idx = qweight2_sparse_prefix[packed_row];
  while (pending != 0U) {
    const int bit_idx = __ffs(pending) - 1;
    const int sparse_packed_row = sparse_bit_idx >> 5;
    const int sparse_bit = sparse_bit_idx & 31;
    const unsigned qword = static_cast<unsigned>(qweight2_sparse[sparse_packed_row * outfeatures + out_idx]);
    dense_word |= (((qword >> sparse_bit) & 1U) << bit_idx);
    pending &= pending - 1;
    ++sparse_bit_idx;
  }
  return dense_word;
}

__device__ __forceinline__ half load_compact3_weight_from_packs_sparse(
    unsigned qweight1_word,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_sparse,
    const int* __restrict__ q_gmap_sparse_prefix,
    unsigned c_map_word,
    int packed_row,
    int outfeatures,
    int out_idx,
    int bit_idx,
    unsigned long long col_pack3,
    unsigned long long row_pack3) {
  const unsigned salient = (c_map_word >> bit_idx) & 1U;
  const unsigned sign1 = (qweight1_word >> bit_idx) & 1U;

  if (salient != 0U) {
    half out = __hmul(unpack_half_from_u64(col_pack3, 0), unpack_half_from_u64(row_pack3, 0));
    if (sign1 == 0U) {
      out = __hneg(out);
    }
    half term2 = __hmul(unpack_half_from_u64(col_pack3, 16), unpack_half_from_u64(row_pack3, 16));
    if (!get_sparse_qweight2_bit(qweight2_sparse, qweight2_sparse_prefix, c_map_word, packed_row, bit_idx, outfeatures, out_idx)) {
      term2 = __hneg(term2);
    }
    return __hadd(out, term2);
  }

  const bool large = get_sparse_inactive_bit(
      q_gmap_sparse,
      q_gmap_sparse_prefix,
      c_map_word,
      packed_row,
      bit_idx,
      outfeatures,
      out_idx);
  const unsigned shift = large ? 32U : 48U;
  half out = __hmul(unpack_half_from_u64(col_pack3, shift), unpack_half_from_u64(row_pack3, shift));
  if (sign1 == 0U) {
    out = __hneg(out);
  }
  return out;
}

__device__ __forceinline__ half load_compact3_weight_from_packs_gmap_word(
    unsigned qweight1_word,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    unsigned gmap_word,
    unsigned c_map_word,
    int packed_row,
    int outfeatures,
    int out_idx,
    int bit_idx,
    unsigned long long col_pack3,
    unsigned long long row_pack3) {
  const unsigned salient = (c_map_word >> bit_idx) & 1U;
  const unsigned sign1 = (qweight1_word >> bit_idx) & 1U;

  if (salient != 0U) {
    half out = __hmul(unpack_half_from_u64(col_pack3, 0), unpack_half_from_u64(row_pack3, 0));
    if (sign1 == 0U) {
      out = __hneg(out);
    }
    half term2 = __hmul(unpack_half_from_u64(col_pack3, 16), unpack_half_from_u64(row_pack3, 16));
    if (!get_sparse_qweight2_bit(qweight2_sparse, qweight2_sparse_prefix, c_map_word, packed_row, bit_idx, outfeatures, out_idx)) {
      term2 = __hneg(term2);
    }
    return __hadd(out, term2);
  }

  const bool large = ((gmap_word >> bit_idx) & 1U) != 0U;
  const unsigned shift = large ? 32U : 48U;
  half out = __hmul(unpack_half_from_u64(col_pack3, shift), unpack_half_from_u64(row_pack3, shift));
  if (sign1 == 0U) {
    out = __hneg(out);
  }
  return out;
}

__device__ __forceinline__ half load_compact3_weight_from_dense_packs_gmap_word(
    unsigned qweight1_word,
    unsigned qweight2_word,
    unsigned gmap_word,
    unsigned c_map_word,
    int bit_idx,
    unsigned long long col_pack3,
    unsigned long long row_pack3) {
  const unsigned salient = (c_map_word >> bit_idx) & 1U;
  const unsigned sign1 = (qweight1_word >> bit_idx) & 1U;

  if (salient != 0U) {
    half out = __hmul(unpack_half_from_u64(col_pack3, 0), unpack_half_from_u64(row_pack3, 0));
    if (sign1 == 0U) {
      out = __hneg(out);
    }
    half term2 = __hmul(unpack_half_from_u64(col_pack3, 16), unpack_half_from_u64(row_pack3, 16));
    if (((qweight2_word >> bit_idx) & 1U) == 0U) {
      term2 = __hneg(term2);
    }
    return __hadd(out, term2);
  }

  const unsigned shift = ((gmap_word >> bit_idx) & 1U) != 0U ? 32U : 48U;
  half out = __hmul(unpack_half_from_u64(col_pack3, shift), unpack_half_from_u64(row_pack3, shift));
  if (sign1 == 0U) {
    out = __hneg(out);
  }
  return out;
}

__device__ __forceinline__ half2 load_compact3_weight_pair_tiled_uniform_groups(
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_sparse,
    const int* __restrict__ q_gmap_sparse_prefix,
    int out_idx,
    int k0,
    int local_k0,
    bool has_k1,
    int outfeatures,
    unsigned c_map_word,
    const unsigned long long* __restrict__ tile_col_pack3,
    unsigned long long row_pack3_0,
    unsigned long long row_pack3_1) {
  const int packed_row = k0 >> 5;
  const int bit0 = k0 & 31;
  const unsigned qweight1_word = static_cast<unsigned>(qweight1[packed_row * outfeatures + out_idx]);

  const half w0 = load_compact3_weight_from_packs_sparse(
      qweight1_word,
      qweight2_sparse,
      qweight2_sparse_prefix,
      q_gmap_sparse,
      q_gmap_sparse_prefix,
      c_map_word,
      packed_row,
      outfeatures,
      out_idx,
      bit0,
      tile_col_pack3[local_k0],
      row_pack3_0);

  if (!has_k1) {
    return __halves2half2(w0, __float2half(0.0f));
  }

  const int local_k1 = local_k0 + 1;
  const half w1 = load_compact3_weight_from_packs_sparse(
      qweight1_word,
      qweight2_sparse,
      qweight2_sparse_prefix,
      q_gmap_sparse,
      q_gmap_sparse_prefix,
      c_map_word,
      packed_row,
      outfeatures,
      out_idx,
      bit0 + 1,
      tile_col_pack3[local_k1],
      row_pack3_1);
  return __halves2half2(w0, w1);
}

// A K64 tile can contain multiple quantization groups. Reuse the tile's
// cached row pack for the common uniform-group case, and resolve every other
// K position independently (g_idx need not be monotonic).
template <bool UniformK64Groups>
__device__ __forceinline__ unsigned long long select_compact3_row_pack(
    const int64_t* __restrict__ row_packs,
    const int* __restrict__ g_idx,
    int global_k,
    int out_idx,
    int num_groups,
    int tile_group,
    unsigned long long tile_row_pack) {
  if constexpr (UniformK64Groups) {
    return tile_row_pack;
  } else {
    const int group = g_idx[global_k];
    if (group == tile_group) {
      return tile_row_pack;
    }
    return (group >= 0 && group < num_groups)
        ? static_cast<unsigned long long>(row_packs[out_idx * num_groups + group])
        : 0ULL;
  }
}

template <bool UniformK64Groups>
__device__ __forceinline__ half2 load_compact3_weight_pair_tiled_uniform_group_gmap_word(
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    unsigned gmap_word,
    int out_idx,
    int k0,
    int local_k0,
    bool has_k1,
    int outfeatures,
    unsigned c_map_word,
    const unsigned long long* __restrict__ tile_col_pack3,
    unsigned long long row_pack3,
    const int64_t* __restrict__ row_packs,
    const int* __restrict__ g_idx,
    int num_groups,
    int tile_group) {
  const int packed_row = k0 >> 5;
  const int bit0 = k0 & 31;
  const unsigned qweight1_word = static_cast<unsigned>(qweight1[packed_row * outfeatures + out_idx]);

  const unsigned long long row_pack0 = select_compact3_row_pack<UniformK64Groups>(
      row_packs, g_idx, k0, out_idx, num_groups, tile_group, row_pack3);
  const half w0 = load_compact3_weight_from_packs_gmap_word(
      qweight1_word,
      qweight2_sparse,
      qweight2_sparse_prefix,
      gmap_word,
      c_map_word,
      packed_row,
      outfeatures,
      out_idx,
      bit0,
      tile_col_pack3[local_k0],
      row_pack0);

  if (!has_k1) {
    return __halves2half2(w0, __float2half(0.0f));
  }

  const int local_k1 = local_k0 + 1;
  const unsigned long long row_pack1 = select_compact3_row_pack<UniformK64Groups>(
      row_packs, g_idx, k0 + 1, out_idx, num_groups, tile_group, row_pack3);
  const half w1 = load_compact3_weight_from_packs_gmap_word(
      qweight1_word,
      qweight2_sparse,
      qweight2_sparse_prefix,
      gmap_word,
      c_map_word,
      packed_row,
      outfeatures,
      out_idx,
      bit0 + 1,
      tile_col_pack3[local_k1],
      row_pack1);
  return __halves2half2(w0, w1);
}

__global__ void Packed4LinearKernelTiledLop3(
    const half* __restrict__ input,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    const int* __restrict__ g_idx,
    half* __restrict__ output,
    int batch,
    int rows,
    int cols,
    int qzeros_cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  const bool row_valid = batch_idx < batch && row < rows;

  __shared__ half x_tile[PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  float acc = 0.0f;
  const half* input_row = input + batch_idx * cols;

  for (int tile_col = 0; tile_col < cols; tile_col += PACKED4_TILE_COLS) {
    const int tile_cols = min(PACKED4_TILE_COLS, cols - tile_col);
    for (int idx = threadIdx.x; idx < tile_cols; idx += blockDim.x) {
      x_tile[idx] = input_row[tile_col + idx];
    }
    __syncthreads();

    if (row_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
        acc += dot8_packed4_shared(
            qweight,
            qzeros,
            scales,
            g_idx,
            x_tile + local_col,
            qbyte_lut,
            rows,
            qzeros_cols,
            row,
            tile_col + local_col,
            valid);
      }
    }
    __syncthreads();
  }

  if (row_valid) {
    output[batch_idx * rows + row] = __float2half(acc);
  }
}

__global__ void Packed4LinearKernelTiledLop3SplitK(
    const half* __restrict__ input,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    const int* __restrict__ g_idx,
    float* __restrict__ output,
    int batch,
    int rows,
    int cols,
    int qzeros_cols,
    int split_k_slices) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool row_valid = batch_idx < batch && row < rows;

  const int num_k_tiles = ceil_div_int(cols, PACKED4_TILE_COLS);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  __shared__ half x_tile[PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  float acc = 0.0f;
  const half* input_row = input + batch_idx * cols;

  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_col = tile_iter * PACKED4_TILE_COLS;
    const int tile_cols = min(PACKED4_TILE_COLS, cols - tile_col);
    for (int idx = threadIdx.x; idx < tile_cols; idx += blockDim.x) {
      x_tile[idx] = input_row[tile_col + idx];
    }
    __syncthreads();

    if (row_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
        acc += dot8_packed4_shared(
            qweight,
            qzeros,
            scales,
            g_idx,
            x_tile + local_col,
            qbyte_lut,
            rows,
            qzeros_cols,
            row,
            tile_col + local_col,
            valid);
      }
    }
    __syncthreads();
  }

  if (row_valid) {
    atomicAdd(output + batch_idx * rows + row, acc);
  }
}

__global__ void Packed4LinearKernelFasterUniformGroups(
    const half2* __restrict__ input,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    float* __restrict__ output,
    int batch,
    int input_half2_stride,
    int packed_cols,
    int rows,
    int qzeros_cols,
    int groupsize,
    int num_groups) {
  const int blockwidth2 = PACKED4_FAST_BLOCKWIDTH / 2;
  const int batch_idx = blockIdx.z;
  const int packed_col_base = PACKED4_FAST_BLOCKHEIGHT * blockIdx.x;
  const int row = PACKED4_FAST_BLOCKWIDTH * blockIdx.y + threadIdx.x;
  const bool row_valid = batch_idx < batch && row < rows;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2) {
    const int input_idx = blockIdx.x * blockwidth2 + threadIdx.x;
    blockvec[threadIdx.x] = input_idx < input_half2_stride
        ? input[batch_idx * input_half2_stride + input_idx]
        : __float2half2_rn(0.0f);
  }

  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE][8];
  const int val = threadIdx.x / 8;
  const int off = threadIdx.x % 8;
  for (int lut_val = val; lut_val < PACKED4_DEQ_LUT_SIZE; lut_val += PACKED4_FAST_BLOCKWIDTH / 8) {
    qbyte_lut[lut_val][off] = __halves2half2(
        __int2half_rn(lut_val & 0x0F),
        __int2half_rn((lut_val >> 4) & 0x0F));
  }
  __syncthreads();

  float acc = 0.0f;
  if (row_valid) {
    const int z_w = row >> 3;
    const int z_mod = (row & 7) * 4;
    int q_index = rows * packed_col_base + row;

    for (int k = 0; k < blockwidth2; k += 4) {
      const int col0 = packed_col_base * 8 + k * 2;
      if (col0 >= packed_cols * 8) {
        break;
      }
      int group = col0 / groupsize;
      if (group >= num_groups) {
        group = num_groups - 1;
      }
      const half scale_h = scales[group * rows + row];
      const float scale_f = __half2float(scale_h);
      const half2 scale = __float2half2_rn(scale_f);
      const unsigned zero_word = static_cast<unsigned>(qzeros[group * qzeros_cols + z_w]);
      const half2 zero = __float2half2_rn(-(scale_f * static_cast<float>(((zero_word >> z_mod) & 0xF) + 1)));

      half2 res2 = __float2half2_rn(0.0f);
      const unsigned tmp = static_cast<unsigned>(qweight[q_index]);
      res2 = __hfma2(__hfma2(qbyte_lut[(tmp >> 0) & 0xFF][off], scale, zero), blockvec[k + 0], res2);
      res2 = __hfma2(__hfma2(qbyte_lut[(tmp >> 8) & 0xFF][off], scale, zero), blockvec[k + 1], res2);
      res2 = __hfma2(__hfma2(qbyte_lut[(tmp >> 16) & 0xFF][off], scale, zero), blockvec[k + 2], res2);
      res2 = __hfma2(__hfma2(qbyte_lut[(tmp >> 24) & 0xFF][off], scale, zero), blockvec[k + 3], res2);
      acc += __low2float(res2) + __high2float(res2);
      q_index += rows;
    }
  }

  if (row_valid) {
    atomicAdd(output + batch_idx * rows + row, acc);
  }
}

__global__ void Packed4LinearCastFloatToHalf(
    const float* __restrict__ input,
    half* __restrict__ output,
    int numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    output[idx] = __float2half_rn(input[idx]);
  }
}


__device__ __forceinline__ int find_grouped_problem_segment(const int* __restrict__ problem_offsets, int num_segments, int problem_tile_idx);

__device__ __forceinline__ float dot8_packed4_uniform_shared_stride(
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    const half* __restrict__ x_tile,
    const half2* __restrict__ qbyte_lut,
    int row_stride,
    int qzeros_stride,
    int row,
    int global_col,
    int tile_valid_cols,
    int group_size,
    int num_groups) {
  if (num_groups <= 0) {
    return 0.0f;
  }

  const unsigned qword = static_cast<unsigned>(qweight[(global_col >> 3) * row_stride + row]);
  const int row_word = row >> 3;
  const int zshift = (row & 7) * 4;
  half2 res2 = __float2half2_rn(0.0f);

#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const int lane0 = pair * 2;
    if (lane0 >= tile_valid_cols) {
      break;
    }

    const int c0 = global_col + lane0;
    const int c1 = c0 + 1;
    const bool has_c1 = (lane0 + 1) < tile_valid_cols;

    int g0 = group_size > 0 ? (c0 / group_size) : 0;
    int g1 = has_c1 ? (group_size > 0 ? (c1 / group_size) : 0) : g0;
    if (g0 >= num_groups) {
      g0 = num_groups - 1;
    }
    if (g1 >= num_groups) {
      g1 = num_groups - 1;
    }

    const unsigned zword0 = static_cast<unsigned>(qzeros[g0 * qzeros_stride + row_word]);
    const unsigned zword1 = static_cast<unsigned>(qzeros[g1 * qzeros_stride + row_word]);
    const half scale0_h = scales[g0 * row_stride + row];
    const half scale1_h = scales[g1 * row_stride + row];
    const float scale0_f = __half2float(scale0_h);
    const float scale1_f = __half2float(scale1_h);
    const half2 scale01 = __halves2half2(scale0_h, scale1_h);
    const half zero0 = __float2half_rn(-(scale0_f * static_cast<float>(((zword0 >> zshift) & 0xF) + 1)));
    const half zero1 = __float2half_rn(-(scale1_f * static_cast<float>(((zword1 >> zshift) & 0xF) + 1)));
    const unsigned qbyte = (qword >> (pair * 8)) & 0xFF;
    const half2 decoded = __hfma2(qbyte_lut[qbyte], scale01, __halves2half2(zero0, zero1));
    const half x0 = x_tile[lane0];
    const half x1 = has_c1 ? x_tile[lane0 + 1] : __float2half(0.0f);
    res2 = __hfma2(decoded, __halves2half2(x0, x1), res2);
  }

  return __low2float(res2) + __high2float(res2);
}

__device__ __forceinline__ float load_packed4_weight_uniform(
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    int row_stride,
    int qzeros_stride,
    int row,
    int col,
    int group_size,
    int num_groups) {
  if (num_groups <= 0) {
    return 0.0f;
  }

  int group = group_size > 0 ? (col / group_size) : 0;
  if (group >= num_groups) {
    group = num_groups - 1;
  }

  const unsigned qword = static_cast<unsigned>(qweight[(col >> 3) * row_stride + row]);
  const unsigned qvalue = (qword >> ((col & 7) * 4)) & 0x0F;
  const unsigned zword = static_cast<unsigned>(qzeros[group * qzeros_stride + (row >> 3)]);
  const unsigned zvalue = (zword >> ((row & 7) * 4)) & 0x0F;
  const float zero = static_cast<float>(zvalue + 1);
  const float scale = __half2float(scales[group * row_stride + row]);
  return (static_cast<float>(qvalue) - zero) * scale;
}

__global__ void GroupedPacked4LinearOffsetSegmentC2Kernel(
    const half* __restrict__ input,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ ns_rank_per_expert,
    const int* __restrict__ rank_offsets,
    const int* __restrict__ rank_word_offsets,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int num_experts,
    int num_segments,
    int infeatures,
    int max_rank,
    int total_rank,
    int total_rank_words,
    int group_count,
    int group_size) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool row_in_bounds = row < max_rank;

  __shared__ half x_tile[STRUCTURED_GROUPED_M_TILE][PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ int expert_idx_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    row_start_shared = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start_shared);
    expert_idx_shared = expert_ids[segment_idx];
  }
  __syncthreads();

  const int expert_idx = expert_idx_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_idx >= 0 && expert_idx < num_experts;
  const int rank = expert_valid ? ns_rank_per_expert[expert_idx] : 0;
  const bool row_valid = expert_valid && row_in_bounds && row < rank;

  const int rank_offset = expert_valid ? rank_offsets[expert_idx] : 0;
  const int rank_word_offset = expert_valid ? rank_word_offsets[expert_idx] : 0;
  const int* expert_qweight = expert_valid ? (qweight + rank_offset) : nullptr;
  const int* expert_qzeros = expert_valid ? (qzeros + rank_word_offset) : nullptr;
  const half* expert_scales = expert_valid ? (scales + rank_offset) : nullptr;

  float acc[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int tile_col = 0; tile_col < infeatures; tile_col += PACKED4_TILE_COLS) {
    const int tile_cols = min(PACKED4_TILE_COLS, infeatures - tile_col);
    const int load_items = row_count * tile_cols;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int local_row = idx / tile_cols;
      const int local_col = idx - local_row * tile_cols;
      x_tile[local_row][local_col] = input[(row_start + local_row) * infeatures + tile_col + local_col];
    }
    __syncthreads();

    if (row_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
#pragma unroll
        for (int local_row = 0; local_row < STRUCTURED_GROUPED_M_TILE; ++local_row) {
          if (local_row >= row_count) {
            break;
          }
          acc[local_row] += dot8_packed4_uniform_shared_stride(
              expert_qweight,
              expert_qzeros,
              expert_scales,
              x_tile[local_row] + local_col,
              qbyte_lut,
              total_rank,
              total_rank_words,
              row,
              tile_col + local_col,
              valid,
              group_size,
              group_count);
        }
      }
    }
    __syncthreads();
  }

  if (row_valid) {
#pragma unroll
    for (int local_row = 0; local_row < STRUCTURED_GROUPED_M_TILE; ++local_row) {
      if (local_row >= row_count) {
        break;
      }
      output[(row_start + local_row) * max_rank + row] = __float2half_rn(acc[local_row]);
    }
  }
}

__global__ void GroupedPacked4LinearOffsetDecodeC2Kernel(
    const half* __restrict__ input,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ ns_rank_per_expert,
    const int* __restrict__ rank_offsets,
    const int* __restrict__ rank_word_offsets,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int num_experts,
    int num_segments,
    int infeatures,
    int max_rank,
    int total_rank,
    int total_rank_words,
    int group_count,
    int group_size) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool row_in_bounds = row < max_rank;

  __shared__ half x_tile[PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ int expert_idx_shared;
  __shared__ int row_idx_shared;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_idx_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_idx = expert_idx_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_idx >= 0 && expert_idx < num_experts;
  const int rank = expert_valid ? ns_rank_per_expert[expert_idx] : 0;
  const bool row_valid = expert_valid && row_in_bounds && row < rank;

  const int rank_offset = expert_valid ? rank_offsets[expert_idx] : 0;
  const int rank_word_offset = expert_valid ? rank_word_offsets[expert_idx] : 0;
  const int* expert_qweight = expert_valid ? (qweight + rank_offset) : nullptr;
  const int* expert_qzeros = expert_valid ? (qzeros + rank_word_offset) : nullptr;
  const half* expert_scales = expert_valid ? (scales + rank_offset) : nullptr;

  float acc = 0.0f;
  for (int tile_col = 0; tile_col < infeatures; tile_col += PACKED4_TILE_COLS) {
    const int tile_cols = min(PACKED4_TILE_COLS, infeatures - tile_col);
    for (int idx = threadIdx.x; idx < tile_cols; idx += blockDim.x) {
      x_tile[idx] = input[row_idx * infeatures + tile_col + idx];
    }
    __syncthreads();

    if (row_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
        acc += dot8_packed4_uniform_shared_stride(
            expert_qweight,
            expert_qzeros,
            expert_scales,
            x_tile + local_col,
            qbyte_lut,
            total_rank,
            total_rank_words,
            row,
            tile_col + local_col,
            valid,
            group_size,
            group_count);
      }
    }
    __syncthreads();
  }

  if (row_valid) {
    output[row_idx * max_rank + row] = __float2half_rn(acc);
  }
}

__global__ void GroupedPacked4LinearOffsetSegmentC1Kernel(
    const half* __restrict__ input,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ ns_rank_per_expert,
    const int* __restrict__ rank_word_offsets,
    const int* __restrict__ group_offsets,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int num_experts,
    int num_segments,
    int rows,
    int max_rank,
    int qzeros_cols,
    int group_size) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_valid = out_idx < rows;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ int expert_idx_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    row_start_shared = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start_shared);
    expert_idx_shared = expert_ids[segment_idx];
  }
  __syncthreads();

  const int expert_idx = expert_idx_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_idx >= 0 && expert_idx < num_experts;
  const int rank = expert_valid ? ns_rank_per_expert[expert_idx] : 0;
  if (!expert_valid || rank <= 0) {
    return;
  }

  const int expert_group_size = rank < group_size ? rank : group_size;
  const int expert_group_count = ceil_div_int(rank, expert_group_size > 0 ? expert_group_size : 1);
  const int rank_word_offset = rank_word_offsets[expert_idx];
  const int group_offset = group_offsets[expert_idx];
  const int* expert_qweight = qweight + static_cast<long long>(rank_word_offset) * rows;
  const int* expert_qzeros = qzeros + static_cast<long long>(group_offset) * qzeros_cols;
  const half* expert_scales = scales + static_cast<long long>(group_offset) * rows;

  float acc[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int tile_col = 0; tile_col < rank; tile_col += PACKED4_TILE_COLS) {
    const int tile_cols = min(PACKED4_TILE_COLS, rank - tile_col);
    const int load_items = row_count * tile_cols;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int local_row = idx / tile_cols;
      const int local_col = idx - local_row * tile_cols;
      hidden_tile[local_row][local_col] = input[(row_start + local_row) * max_rank + tile_col + local_col];
    }
    __syncthreads();

    if (out_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
#pragma unroll
        for (int local_row = 0; local_row < STRUCTURED_GROUPED_M_TILE; ++local_row) {
          if (local_row >= row_count) {
            break;
          }
          acc[local_row] += dot8_packed4_uniform_shared_stride(
              expert_qweight,
              expert_qzeros,
              expert_scales,
              hidden_tile[local_row] + local_col,
              qbyte_lut,
              rows,
              qzeros_cols,
              out_idx,
              tile_col + local_col,
              valid,
              expert_group_size,
              expert_group_count);
        }
      }
    }
    __syncthreads();
  }

  if (out_valid) {
#pragma unroll
    for (int local_row = 0; local_row < STRUCTURED_GROUPED_M_TILE; ++local_row) {
      if (local_row >= row_count) {
        break;
      }
      output[(row_start + local_row) * rows + out_idx] = __float2half_rn(acc[local_row]);
    }
  }
}

__global__ void GroupedPacked4LinearOffsetDecodeC1Kernel(
    const half* __restrict__ input,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ ns_rank_per_expert,
    const int* __restrict__ rank_word_offsets,
    const int* __restrict__ group_offsets,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int num_experts,
    int num_segments,
    int rows,
    int max_rank,
    int qzeros_cols,
    int group_size) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_valid = out_idx < rows;

  __shared__ half hidden_tile[PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ int expert_idx_shared;
  __shared__ int row_idx_shared;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_idx_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_idx = expert_idx_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_idx >= 0 && expert_idx < num_experts;
  const int rank = expert_valid ? ns_rank_per_expert[expert_idx] : 0;
  if (!expert_valid || rank <= 0) {
    return;
  }

  const int expert_group_size = rank < group_size ? rank : group_size;
  const int expert_group_count = ceil_div_int(rank, expert_group_size > 0 ? expert_group_size : 1);
  const int rank_word_offset = rank_word_offsets[expert_idx];
  const int group_offset = group_offsets[expert_idx];
  const int* expert_qweight = qweight + static_cast<long long>(rank_word_offset) * rows;
  const int* expert_qzeros = qzeros + static_cast<long long>(group_offset) * qzeros_cols;
  const half* expert_scales = scales + static_cast<long long>(group_offset) * rows;

  float acc = 0.0f;
  for (int tile_col = 0; tile_col < rank; tile_col += PACKED4_TILE_COLS) {
    const int tile_cols = min(PACKED4_TILE_COLS, rank - tile_col);
    for (int idx = threadIdx.x; idx < tile_cols; idx += blockDim.x) {
      hidden_tile[idx] = input[row_idx * max_rank + tile_col + idx];
    }
    __syncthreads();

    if (out_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
        acc += dot8_packed4_uniform_shared_stride(
            expert_qweight,
            expert_qzeros,
            expert_scales,
            hidden_tile + local_col,
            qbyte_lut,
            rows,
            qzeros_cols,
            out_idx,
            tile_col + local_col,
            valid,
            expert_group_size,
            expert_group_count);
      }
    }
    __syncthreads();
  }

  if (out_valid) {
    output[row_idx * rows + out_idx] = __float2half_rn(acc);
  }
}


__global__ void GroupedPacked4LinearOffsetSegmentC2SplitKKernel(
    const half* __restrict__ input,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ ns_rank_per_expert,
    const int* __restrict__ rank_offsets,
    const int* __restrict__ rank_word_offsets,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    float* __restrict__ output_accum,
    int num_experts,
    int num_segments,
    int infeatures,
    int max_rank,
    int total_rank,
    int total_rank_words,
    int group_count,
    int group_size,
    int split_k_slices) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool row_in_bounds = row < max_rank;

  __shared__ half x_tile[STRUCTURED_GROUPED_M_TILE][PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ int expert_idx_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    row_start_shared = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start_shared);
    expert_idx_shared = expert_ids[segment_idx];
  }
  __syncthreads();

  const int expert_idx = expert_idx_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_idx >= 0 && expert_idx < num_experts;
  const int rank = expert_valid ? ns_rank_per_expert[expert_idx] : 0;
  const bool row_valid = expert_valid && row_in_bounds && row < rank;

  const int rank_offset = expert_valid ? rank_offsets[expert_idx] : 0;
  const int rank_word_offset = expert_valid ? rank_word_offsets[expert_idx] : 0;
  const int* expert_qweight = expert_valid ? (qweight + rank_offset) : nullptr;
  const int* expert_qzeros = expert_valid ? (qzeros + rank_word_offset) : nullptr;
  const half* expert_scales = expert_valid ? (scales + rank_offset) : nullptr;

  const int num_k_tiles = ceil_div_int(infeatures, PACKED4_TILE_COLS);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  float acc[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_col = tile_iter * PACKED4_TILE_COLS;
    const int tile_cols = min(PACKED4_TILE_COLS, infeatures - tile_col);
    const int load_items = row_count * tile_cols;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int local_row = idx / tile_cols;
      const int local_col = idx - local_row * tile_cols;
      x_tile[local_row][local_col] = input[(row_start + local_row) * infeatures + tile_col + local_col];
    }
    __syncthreads();

    if (row_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
#pragma unroll
        for (int local_row = 0; local_row < STRUCTURED_GROUPED_M_TILE; ++local_row) {
          if (local_row >= row_count) {
            break;
          }
          acc[local_row] += dot8_packed4_uniform_shared_stride(
              expert_qweight,
              expert_qzeros,
              expert_scales,
              x_tile[local_row] + local_col,
              qbyte_lut,
              total_rank,
              total_rank_words,
              row,
              tile_col + local_col,
              valid,
              group_size,
              group_count);
        }
      }
    }
    __syncthreads();
  }

  if (row_valid) {
#pragma unroll
    for (int local_row = 0; local_row < STRUCTURED_GROUPED_M_TILE; ++local_row) {
      if (local_row >= row_count) {
        break;
      }
      atomicAdd(output_accum + (row_start + local_row) * max_rank + row, acc[local_row]);
    }
  }
}

__global__ void GroupedPacked4LinearOffsetDecodeC2SplitKKernel(
    const half* __restrict__ input,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ ns_rank_per_expert,
    const int* __restrict__ rank_offsets,
    const int* __restrict__ rank_word_offsets,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    float* __restrict__ output_accum,
    int num_experts,
    int num_segments,
    int infeatures,
    int max_rank,
    int total_rank,
    int total_rank_words,
    int group_count,
    int group_size,
    int split_k_slices) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool row_in_bounds = row < max_rank;

  __shared__ half x_tile[PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ int expert_idx_shared;
  __shared__ int row_idx_shared;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_idx_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_idx = expert_idx_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_idx >= 0 && expert_idx < num_experts;
  const int rank = expert_valid ? ns_rank_per_expert[expert_idx] : 0;
  const bool row_valid = expert_valid && row_in_bounds && row < rank;

  const int rank_offset = expert_valid ? rank_offsets[expert_idx] : 0;
  const int rank_word_offset = expert_valid ? rank_word_offsets[expert_idx] : 0;
  const int* expert_qweight = expert_valid ? (qweight + rank_offset) : nullptr;
  const int* expert_qzeros = expert_valid ? (qzeros + rank_word_offset) : nullptr;
  const half* expert_scales = expert_valid ? (scales + rank_offset) : nullptr;

  const int num_k_tiles = ceil_div_int(infeatures, PACKED4_TILE_COLS);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  float acc = 0.0f;
  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_col = tile_iter * PACKED4_TILE_COLS;
    const int tile_cols = min(PACKED4_TILE_COLS, infeatures - tile_col);
    for (int idx = threadIdx.x; idx < tile_cols; idx += blockDim.x) {
      x_tile[idx] = input[row_idx * infeatures + tile_col + idx];
    }
    __syncthreads();

    if (row_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
        acc += dot8_packed4_uniform_shared_stride(
            expert_qweight,
            expert_qzeros,
            expert_scales,
            x_tile + local_col,
            qbyte_lut,
            total_rank,
            total_rank_words,
            row,
            tile_col + local_col,
            valid,
            group_size,
            group_count);
      }
    }
    __syncthreads();
  }

  if (row_valid) {
    atomicAdd(output_accum + row_idx * max_rank + row, acc);
  }
}

__global__ void GroupedPacked4LinearOffsetSegmentC1SplitKKernel(
    const half* __restrict__ input,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ ns_rank_per_expert,
    const int* __restrict__ rank_word_offsets,
    const int* __restrict__ group_offsets,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    float* __restrict__ output_accum,
    int num_experts,
    int num_segments,
    int rows,
    int max_rank,
    int qzeros_cols,
    int group_size,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool out_valid = out_idx < rows;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ int expert_idx_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    row_start_shared = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start_shared);
    expert_idx_shared = expert_ids[segment_idx];
  }
  __syncthreads();

  const int expert_idx = expert_idx_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_idx >= 0 && expert_idx < num_experts;
  const int rank = expert_valid ? ns_rank_per_expert[expert_idx] : 0;
  if (!expert_valid || rank <= 0) {
    return;
  }

  const int expert_group_size = rank < group_size ? rank : group_size;
  const int expert_group_count = ceil_div_int(rank, expert_group_size > 0 ? expert_group_size : 1);
  const int rank_word_offset = rank_word_offsets[expert_idx];
  const int group_offset = group_offsets[expert_idx];
  const int* expert_qweight = qweight + static_cast<long long>(rank_word_offset) * rows;
  const int* expert_qzeros = qzeros + static_cast<long long>(group_offset) * qzeros_cols;
  const half* expert_scales = scales + static_cast<long long>(group_offset) * rows;

  const int num_k_tiles = ceil_div_int(rank, PACKED4_TILE_COLS);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  float acc[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_col = tile_iter * PACKED4_TILE_COLS;
    const int tile_cols = min(PACKED4_TILE_COLS, rank - tile_col);
    const int load_items = row_count * tile_cols;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int local_row = idx / tile_cols;
      const int local_col = idx - local_row * tile_cols;
      hidden_tile[local_row][local_col] = input[(row_start + local_row) * max_rank + tile_col + local_col];
    }
    __syncthreads();

    if (out_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
#pragma unroll
        for (int local_row = 0; local_row < STRUCTURED_GROUPED_M_TILE; ++local_row) {
          if (local_row >= row_count) {
            break;
          }
          acc[local_row] += dot8_packed4_uniform_shared_stride(
              expert_qweight,
              expert_qzeros,
              expert_scales,
              hidden_tile[local_row] + local_col,
              qbyte_lut,
              rows,
              qzeros_cols,
              out_idx,
              tile_col + local_col,
              valid,
              expert_group_size,
              expert_group_count);
        }
      }
    }
    __syncthreads();
  }

  if (out_valid) {
#pragma unroll
    for (int local_row = 0; local_row < STRUCTURED_GROUPED_M_TILE; ++local_row) {
      if (local_row >= row_count) {
        break;
      }
      atomicAdd(output_accum + (row_start + local_row) * rows + out_idx, acc[local_row]);
    }
  }
}

__global__ void GroupedPacked4LinearOffsetDecodeC1SplitKKernel(
    const half* __restrict__ input,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ ns_rank_per_expert,
    const int* __restrict__ rank_word_offsets,
    const int* __restrict__ group_offsets,
    const int* __restrict__ qweight,
    const int* __restrict__ qzeros,
    const half* __restrict__ scales,
    float* __restrict__ output_accum,
    int num_experts,
    int num_segments,
    int rows,
    int max_rank,
    int qzeros_cols,
    int group_size,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool out_valid = out_idx < rows;

  __shared__ half hidden_tile[PACKED4_TILE_COLS];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ int expert_idx_shared;
  __shared__ int row_idx_shared;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_idx_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_idx = expert_idx_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_idx >= 0 && expert_idx < num_experts;
  const int rank = expert_valid ? ns_rank_per_expert[expert_idx] : 0;
  if (!expert_valid || rank <= 0) {
    return;
  }

  const int expert_group_size = rank < group_size ? rank : group_size;
  const int expert_group_count = ceil_div_int(rank, expert_group_size > 0 ? expert_group_size : 1);
  const int rank_word_offset = rank_word_offsets[expert_idx];
  const int group_offset = group_offsets[expert_idx];
  const int* expert_qweight = qweight + static_cast<long long>(rank_word_offset) * rows;
  const int* expert_qzeros = qzeros + static_cast<long long>(group_offset) * qzeros_cols;
  const half* expert_scales = scales + static_cast<long long>(group_offset) * rows;

  const int num_k_tiles = ceil_div_int(rank, PACKED4_TILE_COLS);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  float acc = 0.0f;
  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_col = tile_iter * PACKED4_TILE_COLS;
    const int tile_cols = min(PACKED4_TILE_COLS, rank - tile_col);
    for (int idx = threadIdx.x; idx < tile_cols; idx += blockDim.x) {
      hidden_tile[idx] = input[row_idx * max_rank + tile_col + idx];
    }
    __syncthreads();

    if (out_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
        acc += dot8_packed4_uniform_shared_stride(
            expert_qweight,
            expert_qzeros,
            expert_scales,
            hidden_tile + local_col,
            qbyte_lut,
            rows,
            qzeros_cols,
            out_idx,
            tile_col + local_col,
            valid,
            expert_group_size,
            expert_group_count);
      }
    }
    __syncthreads();
  }

  if (out_valid) {
    atomicAdd(output_accum + row_idx * rows + out_idx, acc);
  }
}

__global__ void GroupedPacked4LinearOffsetDecodeFusedKernel(
    const half* __restrict__ input,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ ns_rank_per_expert,
    const int* __restrict__ c2_rank_offsets,
    const int* __restrict__ c2_rank_word_offsets,
    const int* __restrict__ c1_rank_word_offsets,
    const int* __restrict__ c1_group_offsets,
    const int* __restrict__ ns_c2_qweight,
    const int* __restrict__ ns_c2_qzeros,
    const half* __restrict__ ns_c2_scales,
    const int* __restrict__ ns_c1_qweight,
    const int* __restrict__ ns_c1_qzeros,
    const half* __restrict__ ns_c1_scales,
    float* __restrict__ output_accum,
    int num_experts,
    int num_segments,
    int infeatures,
    int outfeatures,
    int total_rank,
    int total_rank_words,
    int group_count,
    int ns_c1_qzeros_cols,
    int group_size) {
  const int problem_tile_idx = static_cast<int>(blockIdx.x);
  const int rank_chunk_idx = static_cast<int>(blockIdx.y);

  __shared__ half x_tile[PACKED4_TILE_COLS];
  __shared__ float hidden_chunk[NS_FUSED_CHUNK];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ int expert_idx_shared;
  __shared__ int row_idx_shared;
  __shared__ int rank_shared;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_idx_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
    rank_shared = (expert_idx_shared >= 0 && expert_idx_shared < num_experts) ? ns_rank_per_expert[expert_idx_shared] : 0;
  }
  __syncthreads();

  const int expert_idx = expert_idx_shared;
  const int row_idx = row_idx_shared;
  const int rank = rank_shared;
  if (expert_idx < 0 || expert_idx >= num_experts || rank <= 0) {
    return;
  }

  const int rank_base = rank_chunk_idx * NS_FUSED_CHUNK;
  if (rank_base >= rank) {
    return;
  }
  const int rank_chunk_cols = min(NS_FUSED_CHUNK, rank - rank_base);
  const int c2_rank_offset = c2_rank_offsets[expert_idx];
  const int c2_rank_word_offset = c2_rank_word_offsets[expert_idx];
  const int c1_rank_word_offset = c1_rank_word_offsets[expert_idx];
  const int c1_group_offset = c1_group_offsets[expert_idx];
  const int expert_group_size = rank < group_size ? rank : group_size;
  const int expert_group_count = ceil_div_int(rank, expert_group_size > 0 ? expert_group_size : 1);

  const int* expert_c2_qweight = ns_c2_qweight + c2_rank_offset;
  const int* expert_c2_qzeros = ns_c2_qzeros + c2_rank_word_offset;
  const half* expert_c2_scales = ns_c2_scales + c2_rank_offset;
  const int* expert_c1_qweight = ns_c1_qweight + static_cast<long long>(c1_rank_word_offset) * outfeatures;
  const int* expert_c1_qzeros = ns_c1_qzeros + static_cast<long long>(c1_group_offset) * ns_c1_qzeros_cols;
  const half* expert_c1_scales = ns_c1_scales + static_cast<long long>(c1_group_offset) * outfeatures;

  if (threadIdx.x < NS_FUSED_CHUNK) {
    hidden_chunk[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  const bool hidden_valid = static_cast<int>(threadIdx.x) < rank_chunk_cols;
  const int rank_row = rank_base + static_cast<int>(threadIdx.x);
  float hidden_acc = 0.0f;
  for (int tile_col = 0; tile_col < infeatures; tile_col += PACKED4_TILE_COLS) {
    const int tile_cols = min(PACKED4_TILE_COLS, infeatures - tile_col);
    for (int idx = threadIdx.x; idx < tile_cols; idx += blockDim.x) {
      x_tile[idx] = input[row_idx * infeatures + tile_col + idx];
    }
    __syncthreads();
    if (hidden_valid) {
      for (int local_col = 0; local_col < tile_cols; local_col += 8) {
        const int valid = min(8, tile_cols - local_col);
        hidden_acc += dot8_packed4_uniform_shared_stride(
            expert_c2_qweight,
            expert_c2_qzeros,
            expert_c2_scales,
            x_tile + local_col,
            qbyte_lut,
            total_rank,
            total_rank_words,
            rank_row,
            tile_col + local_col,
            valid,
            group_size,
            group_count);
      }
    }
    __syncthreads();
  }
  if (hidden_valid) {
    hidden_chunk[threadIdx.x] = hidden_acc;
  }
  __syncthreads();

  for (int out_base = 0; out_base < outfeatures; out_base += blockDim.x) {
    const int out_idx = out_base + static_cast<int>(threadIdx.x);
    if (out_idx < outfeatures) {
      float acc = 0.0f;
#pragma unroll 8
      for (int local_rank = 0; local_rank < NS_FUSED_CHUNK; ++local_rank) {
        if (local_rank >= rank_chunk_cols) {
          break;
        }
        const float c1 = load_packed4_weight_uniform(
            expert_c1_qweight,
            expert_c1_qzeros,
            expert_c1_scales,
            outfeatures,
            ns_c1_qzeros_cols,
            out_idx,
            rank_base + local_rank,
            expert_group_size,
            expert_group_count);
        acc += hidden_chunk[local_rank] * c1;
      }
      atomicAdd(output_accum + row_idx * outfeatures + out_idx, acc);
    }
  }
}

__global__ void Compact3LinearKernelPacked(
    const half* __restrict__ input,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_sparse,
    const int* __restrict__ q_gmap_sparse_prefix,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    const half* __restrict__ input_scale,
    const half* __restrict__ output_scale,
    const half* __restrict__ bias,
    const int* __restrict__ ns_c1_qweight,
    const int* __restrict__ ns_c1_qzeros,
    const half* __restrict__ ns_c1_scales,
    const int* __restrict__ ns_c1_g_idx,
    const int* __restrict__ ns_c2_qweight,
    const int* __restrict__ ns_c2_qzeros,
    const half* __restrict__ ns_c2_scales,
    const int* __restrict__ ns_c2_g_idx,
    half* __restrict__ output,
    int batch,
    int rank,
    int outfeatures,
    int num_groups,
    int ns_rank,
    int ns_c1_qzeros_cols,
    int ns_c2_qzeros_cols,
    bool has_input_scale,
    bool has_output_scale,
    bool has_bias,
    bool has_ns_matrix) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  const bool out_valid = out_idx < outfeatures && batch_idx < batch;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ int tile_group_values[STRUCTURED_TILE_K];
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  float acc = 0.0f;
  const half* input_row = input + batch_idx * rank;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      const half input_value = input_row[global_k];
      hidden_tile[idx] = has_input_scale ? __hmul(input_value, input_scale[global_k]) : input_value;
      tile_col_pack3[idx] = static_cast<unsigned long long>(col_pack3[global_k]);
      tile_group_values[idx] = g_idx[global_k];
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    __syncthreads();

    if (out_valid) {
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const int group0 = tile_group_values[local_k];
        const int group1 = has_k1 ? tile_group_values[local_k + 1] : group0;
        const unsigned long long row_pack3_value0 =
            (group0 >= 0 && group0 < num_groups)
                ? static_cast<unsigned long long>(row_pack3[out_idx * num_groups + group0])
                : 0ULL;
        const unsigned long long row_pack3_value1 =
            (has_k1 && group1 >= 0 && group1 < num_groups)
                ? static_cast<unsigned long long>(row_pack3[out_idx * num_groups + group1])
                : 0ULL;
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_groups(
            qweight1,
            qweight2_sparse,
            qweight2_sparse_prefix,
            q_gmap_sparse,
            q_gmap_sparse_prefix,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value0,
            row_pack3_value1);
        acc += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  // Every lane must participate in the shared C2 input loads and barriers,
  // including lanes outside a partial output tile. Only C1 reads need N bounds.
  if (has_ns_matrix) {
    for (int ns_k = 0; ns_k < ns_rank; ns_k += 2) {
      const int ns_k1 = ns_k + 1;
      float hidden0 = 0.0f;
      float hidden1 = 0.0f;
      for (int tile_col = 0; tile_col < rank; tile_col += PACKED4_TILE_COLS) {
        const int tile_cols = min(PACKED4_TILE_COLS, rank - tile_col);
        for (int idx = threadIdx.x; idx < tile_cols; idx += blockDim.x) {
          hidden_tile[idx] = input_row[tile_col + idx];
        }
        __syncthreads();
        for (int local_col = 0; local_col < tile_cols; local_col += 8) {
          const int valid = min(8, tile_cols - local_col);
          hidden0 += dot8_packed4_shared(
              ns_c2_qweight,
              ns_c2_qzeros,
              ns_c2_scales,
              ns_c2_g_idx,
              hidden_tile + local_col,
              qbyte_lut,
              ns_rank,
              ns_c2_qzeros_cols,
              ns_k,
              tile_col + local_col,
              valid);
          if (ns_k1 < ns_rank) {
            hidden1 += dot8_packed4_shared(
                ns_c2_qweight,
                ns_c2_qzeros,
                ns_c2_scales,
                ns_c2_g_idx,
                hidden_tile + local_col,
                qbyte_lut,
                ns_rank,
                ns_c2_qzeros_cols,
                ns_k1,
                tile_col + local_col,
                valid);
          }
        }
        __syncthreads();
      }
      if (out_valid) {
        const float c10 = load_packed4_weight(
            ns_c1_qweight,
            ns_c1_qzeros,
            ns_c1_scales,
            ns_c1_g_idx,
            outfeatures,
            ns_c1_qzeros_cols,
            out_idx,
            ns_k);
        const float c11 = (ns_k1 < ns_rank)
            ? load_packed4_weight(
                  ns_c1_qweight,
                  ns_c1_qzeros,
                  ns_c1_scales,
                  ns_c1_g_idx,
                  outfeatures,
                  ns_c1_qzeros_cols,
                  out_idx,
                  ns_k1)
            : 0.0f;
        acc += hidden0 * c10 + hidden1 * c11;
      }
    }
  }

  if (out_valid) {
    float out_value = has_output_scale ? (acc * __half2float(output_scale[out_idx])) : acc;
    if (has_bias) {
      out_value += __half2float(bias[out_idx]);
    }
    output[batch_idx * outfeatures + out_idx] = __float2half(out_value);
  }
}

template <bool UniformK64Groups>
__global__ void Compact3LinearKernelPackedBlockGR(
    const half* __restrict__ input,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    const half* __restrict__ input_scale,
    const half* __restrict__ output_scale,
    const half* __restrict__ bias,
    const int* __restrict__ ns_c1_qweight,
    const int* __restrict__ ns_c1_qzeros,
    const half* __restrict__ ns_c1_scales,
    const int* __restrict__ ns_c1_g_idx,
    const int* __restrict__ ns_c2_qweight,
    const int* __restrict__ ns_c2_qzeros,
    const half* __restrict__ ns_c2_scales,
    const int* __restrict__ ns_c2_g_idx,
    half* __restrict__ output,
    int batch,
    int rank,
    int outfeatures,
    int num_groups,
    int ns_rank,
    int ns_c1_qzeros_cols,
    int ns_c2_qzeros_cols,
    bool has_input_scale,
    bool has_output_scale,
    bool has_bias,
    bool has_ns_matrix) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  const bool out_valid = out_idx < outfeatures && batch_idx < batch;
  const int packed_rows = (rank + 31) / 32;
  const int out_block = static_cast<int>(blockIdx.x);
  const int out_block_offset = threadIdx.x;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ half2 qbyte_lut[PACKED4_DEQ_LUT_SIZE];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_q_gmap_raw;
  __shared__ int tile_q_gmap_word_begin;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  init_packed4_deq_lut(qbyte_lut);
  __syncthreads();

  float acc = 0.0f;
  const half* input_row = input + batch_idx * rank;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      const half input_value = input_row[global_k];
      hidden_tile[idx] = has_input_scale ? __hmul(input_value, input_scale[global_k]) : input_value;
      tile_col_pack3[idx] = static_cast<unsigned long long>(col_pack3[global_k]);
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      tile_q_gmap_raw = 0;
      tile_q_gmap_word_begin = 0;
      const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
      const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
      const int k_block = tile_k / STRUCTURED_TILE_K;
      if (k_block >= 0 && k_block < num_k_blocks && out_block >= 0 && out_block < num_out_blocks) {
        const int block_idx = k_block * num_out_blocks + out_block;
        const int meta_word = q_gmap_gr_meta[block_idx];
        if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
          tile_q_gmap_raw = 1;
          tile_q_gmap_word_begin = q_gmap_gr_offsets[block_idx];
        } else {
          decode_block_gr_tile(
              q_gmap_gr_words,
              q_gmap_gr_offsets,
              q_gmap_gr_meta,
              packed_rows,
              outfeatures,
              tile_k,
              out_block,
              tile_q_gmap_words);
        }
      } else {
        decode_block_gr_tile(
            q_gmap_gr_words,
            q_gmap_gr_offsets,
            q_gmap_gr_meta,
            packed_rows,
            outfeatures,
            tile_k,
            out_block,
            tile_q_gmap_words);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = tile_k_size > 0 ? g_idx[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    if (out_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_raw != 0
            ? static_cast<unsigned>(q_gmap_gr_words[tile_q_gmap_word_begin + (local_k >> 5) * BLOCK_GR_TILE_OUT + out_block_offset])
            : tile_q_gmap_words[local_k >> 5][out_block_offset];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1,
            qweight2_sparse,
            qweight2_sparse_prefix,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3,
            g_idx,
            num_groups,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  // Every lane must participate in the shared C2 input loads and barriers,
  // including lanes outside a partial output tile. Only C1 reads need N bounds.
  if (has_ns_matrix) {
    for (int ns_k = 0; ns_k < ns_rank; ns_k += 2) {
      const int ns_k1 = ns_k + 1;
      float hidden0 = 0.0f;
      float hidden1 = 0.0f;
      for (int tile_col = 0; tile_col < rank; tile_col += PACKED4_TILE_COLS) {
        const int tile_cols = min(PACKED4_TILE_COLS, rank - tile_col);
        for (int idx = threadIdx.x; idx < tile_cols; idx += blockDim.x) {
          hidden_tile[idx] = input_row[tile_col + idx];
        }
        __syncthreads();
        for (int local_col = 0; local_col < tile_cols; local_col += 8) {
          const int valid = min(8, tile_cols - local_col);
          hidden0 += dot8_packed4_shared(
              ns_c2_qweight,
              ns_c2_qzeros,
              ns_c2_scales,
              ns_c2_g_idx,
              hidden_tile + local_col,
              qbyte_lut,
              ns_rank,
              ns_c2_qzeros_cols,
              ns_k,
              tile_col + local_col,
              valid);
          if (ns_k1 < ns_rank) {
            hidden1 += dot8_packed4_shared(
                ns_c2_qweight,
                ns_c2_qzeros,
                ns_c2_scales,
                ns_c2_g_idx,
                hidden_tile + local_col,
                qbyte_lut,
                ns_rank,
                ns_c2_qzeros_cols,
                ns_k1,
                tile_col + local_col,
                valid);
          }
        }
        __syncthreads();
      }
      if (out_valid) {
        const float c10 = load_packed4_weight(
            ns_c1_qweight,
            ns_c1_qzeros,
            ns_c1_scales,
            ns_c1_g_idx,
            outfeatures,
            ns_c1_qzeros_cols,
            out_idx,
            ns_k);
        const float c11 = (ns_k1 < ns_rank)
            ? load_packed4_weight(
                  ns_c1_qweight,
                  ns_c1_qzeros,
                  ns_c1_scales,
                  ns_c1_g_idx,
                  outfeatures,
                  ns_c1_qzeros_cols,
                  out_idx,
                  ns_k1)
            : 0.0f;
        acc += hidden0 * c10 + hidden1 * c11;
      }
    }
  }

  if (out_valid) {
    float out_value = has_output_scale ? (acc * __half2float(output_scale[out_idx])) : acc;
    if (has_bias) {
      out_value += __half2float(bias[out_idx]);
    }
    output[batch_idx * outfeatures + out_idx] = __float2half(out_value);
  }
}

#ifndef USE_ROCM
template <int N_TILE, int WARPS, bool UniformK64Groups>
__global__ void Compact3LinearPrefillTensorCoreBlockGRKernel(
    const half* __restrict__ input,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    half* __restrict__ output,
    int batch,
    int rank,
    int outfeatures,
    int num_groups) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const int linear_tid = threadIdx.x;
  const int warp_id = linear_tid >> 5;
  const int row_start = static_cast<int>(blockIdx.y) * STRUCTURED_TC_M_TILE;
  const int out_col_start = static_cast<int>(blockIdx.x) * N_TILE;
  const int out_block = out_col_start / BLOCK_GR_TILE_OUT;
  const int out_block_col_start = out_col_start - out_block * BLOCK_GR_TILE_OUT;
  const int packed_rows = (rank + 31) / 32;

  __shared__ half hidden_tile[STRUCTURED_TC_M_TILE][STRUCTURED_TC_K_TILE];
  __shared__ half weight_stage[N_TILE][STRUCTURED_TC_K_STEP];
  __shared__ unsigned tile_c_map_words[STRUCTURED_TC_K_TILE / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TC_K_TILE];
  __shared__ unsigned long long row_pack3_tile[N_TILE];
  __shared__ unsigned qweight1_word_tile[STRUCTURED_TC_K_TILE / 32][N_TILE];
  __shared__ unsigned qweight2_word_tile[STRUCTURED_TC_K_TILE / 32][N_TILE];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ float output_tile[WARPS][STRUCTURED_TC_M_TILE][STRUCTURED_TC_K_STEP];
  __shared__ int tile_q_gmap_raw;
  __shared__ int tile_q_gmap_word_begin;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;

  nvcuda::wmma::fragment<
      nvcuda::wmma::accumulator,
      STRUCTURED_TC_M_TILE,
      STRUCTURED_TC_K_STEP,
      STRUCTURED_TC_K_STEP,
      float> acc_frag;
  if (warp_id < WARPS) {
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
  }

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TC_K_TILE) {
    const int tile_k_size = min(STRUCTURED_TC_K_TILE, rank - tile_k);
    const int packed_row_base = tile_k >> 5;
    const int tile_word_count = ceil_div_int(tile_k_size, 32);

    const int hidden_items = STRUCTURED_TC_M_TILE * STRUCTURED_TC_K_TILE;
    for (int idx = linear_tid; idx < hidden_items; idx += blockDim.x) {
      const int row = idx / STRUCTURED_TC_K_TILE;
      const int local_k = idx - row * STRUCTURED_TC_K_TILE;
      half value = __float2half(0.0f);
      if (row_start + row < batch && local_k < tile_k_size) {
        value = input[(row_start + row) * rank + tile_k + local_k];
      }
      hidden_tile[row][local_k] = value;
    }

    for (int idx = linear_tid; idx < STRUCTURED_TC_K_TILE; idx += blockDim.x) {
      tile_col_pack3[idx] = idx < tile_k_size
          ? static_cast<unsigned long long>(col_pack3[tile_k + idx])
          : 0ULL;
    }

    if (linear_tid < STRUCTURED_TC_K_TILE / 32) {
      const int word_idx = linear_tid;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (word_base < tile_k_size) {
        word = static_cast<unsigned>(c_map[packed_row_base + word_idx])
            & low_bit_mask(min(32, tile_k_size - word_base));
      }
      tile_c_map_words[word_idx] = word;
    }

    if (linear_tid == 0) {
      const int group = tile_k_size > 0 ? g_idx[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;

      tile_q_gmap_raw = 0;
      tile_q_gmap_word_begin = 0;
      const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
      const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
      const int k_block = tile_k / STRUCTURED_TC_K_TILE;
      if (k_block >= 0 && k_block < num_k_blocks && out_block >= 0 && out_block < num_out_blocks) {
        const int block_idx = k_block * num_out_blocks + out_block;
        const int meta_word = q_gmap_gr_meta[block_idx];
        if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
          tile_q_gmap_raw = 1;
          tile_q_gmap_word_begin = q_gmap_gr_offsets[block_idx];
        } else {
          decode_block_gr_tile(
              q_gmap_gr_words,
              q_gmap_gr_offsets,
              q_gmap_gr_meta,
              packed_rows,
              outfeatures,
              tile_k,
              out_block,
              tile_q_gmap_words);
        }
      } else {
        decode_block_gr_tile(
            q_gmap_gr_words,
            q_gmap_gr_offsets,
            q_gmap_gr_meta,
            packed_rows,
            outfeatures,
            tile_k,
            out_block,
            tile_q_gmap_words);
      }
    }
    __syncthreads();

    if (tile_q_gmap_raw != 0) {
      for (int idx = linear_tid; idx < BLOCK_GR_WORDS_PER_BLOCK; idx += blockDim.x) {
        tile_q_gmap_words[idx / BLOCK_GR_TILE_OUT][idx % BLOCK_GR_TILE_OUT] =
            static_cast<unsigned>(q_gmap_gr_words[tile_q_gmap_word_begin + idx]);
      }
    }

    for (int local_out = linear_tid; local_out < N_TILE; local_out += blockDim.x) {
      const int out_idx = out_col_start + local_out;
      unsigned long long row_value = 0ULL;
      if (tile_group_valid != 0 && out_idx < outfeatures) {
        row_value = static_cast<unsigned long long>(row_pack3[out_idx * num_groups + tile_group_value]);
      }
      row_pack3_tile[local_out] = row_value;
    }
    __syncthreads();

    const int qword_items = (STRUCTURED_TC_K_TILE / 32) * N_TILE;
    for (int idx = linear_tid; idx < qword_items; idx += blockDim.x) {
      const int local_packed_row = idx / N_TILE;
      const int local_out = idx - local_packed_row * N_TILE;
      const int out_idx = out_col_start + local_out;
      unsigned qweight1_word = 0U;
      unsigned qweight2_word = 0U;
      if (local_packed_row < tile_word_count && out_idx < outfeatures) {
        const int packed_row = packed_row_base + local_packed_row;
        qweight1_word = static_cast<unsigned>(qweight1[packed_row * outfeatures + out_idx]);
        qweight2_word = reconstruct_sparse_qweight2_dense_word(
            qweight2_sparse,
            qweight2_sparse_prefix,
            tile_c_map_words[local_packed_row],
            packed_row,
            outfeatures,
            out_idx);
      }
      qweight1_word_tile[local_packed_row][local_out] = qweight1_word;
      qweight2_word_tile[local_packed_row][local_out] = qweight2_word;
    }
    __syncthreads();

    const int num_k_steps = ceil_div_int(tile_k_size, STRUCTURED_TC_K_STEP);
    const int stage_items = N_TILE * STRUCTURED_TC_K_STEP;
    for (int step_idx = 0; step_idx < num_k_steps; ++step_idx) {
      for (int idx = linear_tid; idx < stage_items; idx += blockDim.x) {
        const int local_out = idx / STRUCTURED_TC_K_STEP;
        const int stage_k = idx - local_out * STRUCTURED_TC_K_STEP;
        const int local_k = step_idx * STRUCTURED_TC_K_STEP + stage_k;
        half w = __float2half(0.0f);
        if (local_k < tile_k_size && out_col_start + local_out < outfeatures) {
          const int local_packed_row = local_k >> 5;
          w = load_compact3_weight_from_dense_packs_gmap_word(
              qweight1_word_tile[local_packed_row][local_out],
              qweight2_word_tile[local_packed_row][local_out],
              tile_q_gmap_words[local_packed_row][out_block_col_start + local_out],
              tile_c_map_words[local_packed_row],
              local_k & 31,
              tile_col_pack3[local_k],
              select_compact3_row_pack<UniformK64Groups>(
                  row_pack3, g_idx, tile_k + local_k, out_col_start + local_out,
                  num_groups, tile_group_value, row_pack3_tile[local_out]));
        }
        weight_stage[local_out][stage_k] = w;
      }
      __syncthreads();

      if (warp_id < WARPS) {
        nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_a,
            STRUCTURED_TC_M_TILE,
            STRUCTURED_TC_K_STEP,
            STRUCTURED_TC_K_STEP,
            half,
            nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_b,
            STRUCTURED_TC_M_TILE,
            STRUCTURED_TC_K_STEP,
            STRUCTURED_TC_K_STEP,
            half,
            nvcuda::wmma::col_major> b_frag;
        nvcuda::wmma::load_matrix_sync(
            a_frag,
            &hidden_tile[0][step_idx * STRUCTURED_TC_K_STEP],
            STRUCTURED_TC_K_TILE);
        nvcuda::wmma::load_matrix_sync(
            b_frag,
            &weight_stage[warp_id * STRUCTURED_TC_K_STEP][0],
            STRUCTURED_TC_K_STEP);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      __syncthreads();
    }
  }

  if (warp_id < WARPS) {
    nvcuda::wmma::store_matrix_sync(
        &output_tile[warp_id][0][0],
        acc_frag,
        STRUCTURED_TC_K_STEP,
        nvcuda::wmma::mem_row_major);
  }
  __syncthreads();

  const int output_items = STRUCTURED_TC_M_TILE * N_TILE;
  for (int idx = linear_tid; idx < output_items; idx += blockDim.x) {
    const int row = idx / N_TILE;
    const int local_out = idx - row * N_TILE;
    const int out_idx = out_col_start + local_out;
    if (row_start + row < batch && out_idx < outfeatures) {
      output[(row_start + row) * outfeatures + out_idx] = __float2half(
          output_tile[local_out / STRUCTURED_TC_K_STEP][row][local_out % STRUCTURED_TC_K_STEP]);
    }
  }
#else
  (void)input;
  (void)qweight1;
  (void)qweight2_sparse;
  (void)qweight2_sparse_prefix;
  (void)q_gmap_gr_words;
  (void)q_gmap_gr_offsets;
  (void)q_gmap_gr_meta;
  (void)c_map;
  (void)g_idx;
  (void)col_pack3;
  (void)row_pack3;
  (void)output;
  (void)batch;
  (void)rank;
  (void)outfeatures;
  (void)num_groups;
#endif
}
#endif

template <bool UniformK64Groups>
__global__ void Compact3LinearKernelPackedFastNoScaleNoBiasBlockGR(
    const half* __restrict__ input,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    half* __restrict__ output,
    int batch,
    int rank,
    int outfeatures,
    int num_groups) {
  const int out_lane = threadIdx.x;
  const int worker_lane = threadIdx.y;
  const int linear_tid = worker_lane * blockDim.x + out_lane;
  const int threads_per_block = blockDim.x * blockDim.y;
  const int out_idx = blockIdx.x * blockDim.x + out_lane;
  const int out_block = static_cast<int>(blockIdx.x);
  const int out_block_offset = out_lane;
  const int batch_idx = static_cast<int>(blockIdx.y);
  const bool out_valid = out_idx < outfeatures && batch_idx < batch;
  const int packed_rows = (rank + 31) / 32;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_q_gmap_raw;
  __shared__ int tile_q_gmap_word_begin;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ float partial[COMPACT3_SINGLE_BLOCK_GR_K_WORKERS][BLOCK_GR_TILE_OUT];

  float acc = 0.0f;
  const half* input_row = input + batch_idx * rank;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = linear_tid; idx < tile_k_size; idx += threads_per_block) {
      hidden_tile[idx] = input_row[tile_k + idx];
    }
    for (int idx = linear_tid; idx < tile_k_size; idx += threads_per_block) {
      tile_col_pack3[idx] = static_cast<unsigned long long>(col_pack3[tile_k + idx]);
    }
    if (linear_tid < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = linear_tid;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (linear_tid == 0) {
      tile_q_gmap_raw = 0;
      tile_q_gmap_word_begin = 0;
      const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
      const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
      const int k_block = tile_k / STRUCTURED_TILE_K;
      if (k_block >= 0 && k_block < num_k_blocks && out_block >= 0 && out_block < num_out_blocks) {
        const int block_idx = k_block * num_out_blocks + out_block;
        const int meta_word = q_gmap_gr_meta[block_idx];
        if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
          tile_q_gmap_raw = 1;
          tile_q_gmap_word_begin = q_gmap_gr_offsets[block_idx];
        } else {
          decode_block_gr_tile(
              q_gmap_gr_words,
              q_gmap_gr_offsets,
              q_gmap_gr_meta,
              packed_rows,
              outfeatures,
              tile_k,
              out_block,
              tile_q_gmap_words);
        }
      } else {
        decode_block_gr_tile(
            q_gmap_gr_words,
            q_gmap_gr_offsets,
            q_gmap_gr_meta,
            packed_rows,
            outfeatures,
            tile_k,
            out_block,
            tile_q_gmap_words);
      }
    }
    __syncthreads();

    if (linear_tid == 0) {
      const int group = tile_k_size > 0 ? g_idx[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    float local_partial = 0.0f;
    if (out_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3[row_offset])
          : 0ULL;
      for (int local_k = 2 * worker_lane; local_k < tile_k_size; local_k += 2 * COMPACT3_SINGLE_BLOCK_GR_K_WORKERS) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_raw != 0
            ? static_cast<unsigned>(q_gmap_gr_words[tile_q_gmap_word_begin + (local_k >> 5) * BLOCK_GR_TILE_OUT + out_block_offset])
            : tile_q_gmap_words[local_k >> 5][out_block_offset];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1,
            qweight2_sparse,
            qweight2_sparse_prefix,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3,
            g_idx,
            num_groups,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        local_partial += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    partial[worker_lane][out_lane] = local_partial;
    __syncthreads();

    if (worker_lane == 0 && out_valid) {
      float tile_sum = 0.0f;
#pragma unroll
      for (int worker = 0; worker < COMPACT3_SINGLE_BLOCK_GR_K_WORKERS; ++worker) {
        tile_sum += partial[worker][out_lane];
      }
      acc += tile_sum;
    }
    __syncthreads();
  }

  if (worker_lane == 0 && out_valid) {
    output[batch_idx * outfeatures + out_idx] = __float2half(acc);
  }
}

template <bool UniformK64Groups>
__global__ void Compact3LinearKernelPackedFastNoScaleNoBiasSplitKBlockGR(
    const half* __restrict__ input,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    float* __restrict__ output,
    int batch,
    int rank,
    int outfeatures,
    int num_groups,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool out_valid = out_idx < outfeatures && batch_idx < batch;
  const int packed_rows = (rank + 31) / 32;
  const int out_block = static_cast<int>(blockIdx.x);
  const int out_block_offset = threadIdx.x;

  const int num_k_tiles = ceil_div_int(rank, STRUCTURED_TILE_K);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_q_gmap_raw;
  __shared__ int tile_q_gmap_word_begin;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;

  float acc = 0.0f;
  const half* input_row = input + batch_idx * rank;

  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = input_row[tile_k + idx];
      tile_col_pack3[idx] = static_cast<unsigned long long>(col_pack3[tile_k + idx]);
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      tile_q_gmap_raw = 0;
      tile_q_gmap_word_begin = 0;
      const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
      const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
      const int k_block = tile_k / STRUCTURED_TILE_K;
      if (k_block >= 0 && k_block < num_k_blocks && out_block >= 0 && out_block < num_out_blocks) {
        const int block_idx = k_block * num_out_blocks + out_block;
        const int meta_word = q_gmap_gr_meta[block_idx];
        if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
          tile_q_gmap_raw = 1;
          tile_q_gmap_word_begin = q_gmap_gr_offsets[block_idx];
        } else {
          decode_block_gr_tile(
              q_gmap_gr_words,
              q_gmap_gr_offsets,
              q_gmap_gr_meta,
              packed_rows,
              outfeatures,
              tile_k,
              out_block,
              tile_q_gmap_words);
        }
      } else {
        decode_block_gr_tile(
            q_gmap_gr_words,
            q_gmap_gr_offsets,
            q_gmap_gr_meta,
            packed_rows,
            outfeatures,
            tile_k,
            out_block,
            tile_q_gmap_words);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = tile_k_size > 0 ? g_idx[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    if (out_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_raw != 0
            ? static_cast<unsigned>(q_gmap_gr_words[tile_q_gmap_word_begin + (local_k >> 5) * BLOCK_GR_TILE_OUT + out_block_offset])
            : tile_q_gmap_words[local_k >> 5][out_block_offset];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1,
            qweight2_sparse,
            qweight2_sparse_prefix,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3,
            g_idx,
            num_groups,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  if (out_valid) {
    atomicAdd(output + batch_idx * outfeatures + out_idx, acc);
  }
}

template <bool UniformK64Groups>
__global__ void Compact3QKVLinearKernelPackedFastNoScaleBiasBlockGR(
    const half* __restrict__ input,
    const int* __restrict__ q_qweight1,
    const int* __restrict__ q_qweight2_sparse,
    const int* __restrict__ q_qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    const int* __restrict__ q_c_map,
    const int* __restrict__ q_g_idx,
    const int64_t* __restrict__ q_col_pack3,
    const int64_t* __restrict__ q_row_pack3,
    const half* __restrict__ q_bias,
    const int* __restrict__ k_qweight1,
    const int* __restrict__ k_qweight2_sparse,
    const int* __restrict__ k_qweight2_sparse_prefix,
    const int* __restrict__ k_gmap_gr_words,
    const int* __restrict__ k_gmap_gr_offsets,
    const int* __restrict__ k_gmap_gr_meta,
    const int* __restrict__ k_c_map,
    const int* __restrict__ k_g_idx,
    const int64_t* __restrict__ k_col_pack3,
    const int64_t* __restrict__ k_row_pack3,
    const half* __restrict__ k_bias,
    const int* __restrict__ v_qweight1,
    const int* __restrict__ v_qweight2_sparse,
    const int* __restrict__ v_qweight2_sparse_prefix,
    const int* __restrict__ v_gmap_gr_words,
    const int* __restrict__ v_gmap_gr_offsets,
    const int* __restrict__ v_gmap_gr_meta,
    const int* __restrict__ v_c_map,
    const int* __restrict__ v_g_idx,
    const int64_t* __restrict__ v_col_pack3,
    const int64_t* __restrict__ v_row_pack3,
    const half* __restrict__ v_bias,
    half* __restrict__ output,
    int batch,
    int rank,
    int outfeatures,
    int num_groups) {
  const int out_lane = threadIdx.x;
  const int worker_lane = threadIdx.y;
  const int linear_tid = worker_lane * blockDim.x + out_lane;
  const int threads_per_block = blockDim.x * blockDim.y;
  const int out_idx = blockIdx.x * blockDim.x + out_lane;
  const int batch_idx = blockIdx.y;
  const int proj_idx = blockIdx.z;
  const bool out_valid = out_idx < outfeatures && batch_idx < batch && proj_idx < 3;
  const int packed_rows = (rank + 31) / 32;
  const int out_block = static_cast<int>(blockIdx.x);
  const int out_block_offset = out_lane;

  const int* qweight1 = proj_idx == 0 ? q_qweight1 : (proj_idx == 1 ? k_qweight1 : v_qweight1);
  const int* qweight2_sparse = proj_idx == 0 ? q_qweight2_sparse : (proj_idx == 1 ? k_qweight2_sparse : v_qweight2_sparse);
  const int* qweight2_sparse_prefix = proj_idx == 0 ? q_qweight2_sparse_prefix : (proj_idx == 1 ? k_qweight2_sparse_prefix : v_qweight2_sparse_prefix);
  const int* gmap_gr_words = proj_idx == 0 ? q_gmap_gr_words : (proj_idx == 1 ? k_gmap_gr_words : v_gmap_gr_words);
  const int* gmap_gr_offsets = proj_idx == 0 ? q_gmap_gr_offsets : (proj_idx == 1 ? k_gmap_gr_offsets : v_gmap_gr_offsets);
  const int* gmap_gr_meta = proj_idx == 0 ? q_gmap_gr_meta : (proj_idx == 1 ? k_gmap_gr_meta : v_gmap_gr_meta);
  const int* c_map = proj_idx == 0 ? q_c_map : (proj_idx == 1 ? k_c_map : v_c_map);
  const int* g_idx = proj_idx == 0 ? q_g_idx : (proj_idx == 1 ? k_g_idx : v_g_idx);
  const int64_t* col_pack3 = proj_idx == 0 ? q_col_pack3 : (proj_idx == 1 ? k_col_pack3 : v_col_pack3);
  const int64_t* row_pack3 = proj_idx == 0 ? q_row_pack3 : (proj_idx == 1 ? k_row_pack3 : v_row_pack3);
  const half* bias = proj_idx == 0 ? q_bias : (proj_idx == 1 ? k_bias : v_bias);

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_q_gmap_raw;
  __shared__ int tile_q_gmap_word_begin;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ float partial[COMPACT3_SINGLE_BLOCK_GR_K_WORKERS][BLOCK_GR_TILE_OUT];

  float acc = 0.0f;
  const half* input_row = input + batch_idx * rank;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = linear_tid; idx < tile_k_size; idx += threads_per_block) {
      hidden_tile[idx] = input_row[tile_k + idx];
      tile_col_pack3[idx] = static_cast<unsigned long long>(col_pack3[tile_k + idx]);
    }
    if (linear_tid < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = linear_tid;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (linear_tid == 0) {
      tile_q_gmap_raw = 0;
      tile_q_gmap_word_begin = 0;
      const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
      const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
      const int k_block = tile_k / STRUCTURED_TILE_K;
      if (k_block >= 0 && k_block < num_k_blocks && out_block >= 0 && out_block < num_out_blocks) {
        const int block_idx = k_block * num_out_blocks + out_block;
        const int meta_word = gmap_gr_meta[block_idx];
        if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
          tile_q_gmap_raw = 1;
          tile_q_gmap_word_begin = gmap_gr_offsets[block_idx];
        } else {
          decode_block_gr_tile(
              gmap_gr_words,
              gmap_gr_offsets,
              gmap_gr_meta,
              packed_rows,
              outfeatures,
              tile_k,
              out_block,
              tile_q_gmap_words);
        }
      } else {
        decode_block_gr_tile(
            gmap_gr_words,
            gmap_gr_offsets,
            gmap_gr_meta,
            packed_rows,
            outfeatures,
            tile_k,
            out_block,
            tile_q_gmap_words);
      }
    }
    __syncthreads();

    if (linear_tid == 0) {
      const int group = tile_k_size > 0 ? g_idx[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    float local_partial = 0.0f;
    if (out_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3[row_offset])
          : 0ULL;
      for (int local_k = 2 * worker_lane; local_k < tile_k_size; local_k += 2 * COMPACT3_SINGLE_BLOCK_GR_K_WORKERS) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_raw != 0
            ? static_cast<unsigned>(gmap_gr_words[tile_q_gmap_word_begin + (local_k >> 5) * BLOCK_GR_TILE_OUT + out_block_offset])
            : tile_q_gmap_words[local_k >> 5][out_block_offset];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1,
            qweight2_sparse,
            qweight2_sparse_prefix,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3,
            g_idx,
            num_groups,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        local_partial += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    partial[worker_lane][out_lane] = local_partial;
    __syncthreads();

    if (worker_lane == 0 && out_valid) {
      float tile_sum = 0.0f;
#pragma unroll
      for (int worker = 0; worker < COMPACT3_SINGLE_BLOCK_GR_K_WORKERS; ++worker) {
        tile_sum += partial[worker][out_lane];
      }
      acc += tile_sum;
    }
    __syncthreads();
  }

  if (worker_lane == 0 && out_valid) {
    output[(proj_idx * batch + batch_idx) * outfeatures + out_idx] = __hadd(__float2half(acc), bias[out_idx]);
  }
}

template <bool UniformK64Groups>
__global__ void Compact3QKVLinearKernelPackedFastNoScaleBiasSplitKBlockGR(
    const half* __restrict__ input,
    const int* __restrict__ q_qweight1,
    const int* __restrict__ q_qweight2_sparse,
    const int* __restrict__ q_qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    const int* __restrict__ q_c_map,
    const int* __restrict__ q_g_idx,
    const int64_t* __restrict__ q_col_pack3,
    const int64_t* __restrict__ q_row_pack3,
    const int* __restrict__ k_qweight1,
    const int* __restrict__ k_qweight2_sparse,
    const int* __restrict__ k_qweight2_sparse_prefix,
    const int* __restrict__ k_gmap_gr_words,
    const int* __restrict__ k_gmap_gr_offsets,
    const int* __restrict__ k_gmap_gr_meta,
    const int* __restrict__ k_c_map,
    const int* __restrict__ k_g_idx,
    const int64_t* __restrict__ k_col_pack3,
    const int64_t* __restrict__ k_row_pack3,
    const int* __restrict__ v_qweight1,
    const int* __restrict__ v_qweight2_sparse,
    const int* __restrict__ v_qweight2_sparse_prefix,
    const int* __restrict__ v_gmap_gr_words,
    const int* __restrict__ v_gmap_gr_offsets,
    const int* __restrict__ v_gmap_gr_meta,
    const int* __restrict__ v_c_map,
    const int* __restrict__ v_g_idx,
    const int64_t* __restrict__ v_col_pack3,
    const int64_t* __restrict__ v_row_pack3,
    float* __restrict__ output,
    int batch,
    int rank,
    int outfeatures,
    int num_groups,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  const int z_idx = static_cast<int>(blockIdx.z);
  const int proj_idx = z_idx / split_k_slices;
  const int split_k_idx = z_idx - proj_idx * split_k_slices;
  const bool out_valid = out_idx < outfeatures && batch_idx < batch && proj_idx < 3;
  const int packed_rows = (rank + 31) / 32;
  const int out_block = static_cast<int>(blockIdx.x);
  const int out_block_offset = threadIdx.x;

  const int* qweight1 = proj_idx == 0 ? q_qweight1 : (proj_idx == 1 ? k_qweight1 : v_qweight1);
  const int* qweight2_sparse = proj_idx == 0 ? q_qweight2_sparse : (proj_idx == 1 ? k_qweight2_sparse : v_qweight2_sparse);
  const int* qweight2_sparse_prefix = proj_idx == 0 ? q_qweight2_sparse_prefix : (proj_idx == 1 ? k_qweight2_sparse_prefix : v_qweight2_sparse_prefix);
  const int* gmap_gr_words = proj_idx == 0 ? q_gmap_gr_words : (proj_idx == 1 ? k_gmap_gr_words : v_gmap_gr_words);
  const int* gmap_gr_offsets = proj_idx == 0 ? q_gmap_gr_offsets : (proj_idx == 1 ? k_gmap_gr_offsets : v_gmap_gr_offsets);
  const int* gmap_gr_meta = proj_idx == 0 ? q_gmap_gr_meta : (proj_idx == 1 ? k_gmap_gr_meta : v_gmap_gr_meta);
  const int* c_map = proj_idx == 0 ? q_c_map : (proj_idx == 1 ? k_c_map : v_c_map);
  const int* g_idx = proj_idx == 0 ? q_g_idx : (proj_idx == 1 ? k_g_idx : v_g_idx);
  const int64_t* col_pack3 = proj_idx == 0 ? q_col_pack3 : (proj_idx == 1 ? k_col_pack3 : v_col_pack3);
  const int64_t* row_pack3 = proj_idx == 0 ? q_row_pack3 : (proj_idx == 1 ? k_row_pack3 : v_row_pack3);

  const int num_k_tiles = ceil_div_int(rank, STRUCTURED_TILE_K);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_q_gmap_raw;
  __shared__ int tile_q_gmap_word_begin;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;

  float acc = 0.0f;
  const half* input_row = input + batch_idx * rank;

  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = input_row[tile_k + idx];
      tile_col_pack3[idx] = static_cast<unsigned long long>(col_pack3[tile_k + idx]);
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      tile_q_gmap_raw = 0;
      tile_q_gmap_word_begin = 0;
      const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
      const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
      const int k_block = tile_k / STRUCTURED_TILE_K;
      if (k_block >= 0 && k_block < num_k_blocks && out_block >= 0 && out_block < num_out_blocks) {
        const int block_idx = k_block * num_out_blocks + out_block;
        const int meta_word = gmap_gr_meta[block_idx];
        if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
          tile_q_gmap_raw = 1;
          tile_q_gmap_word_begin = gmap_gr_offsets[block_idx];
        } else {
          decode_block_gr_tile(
              gmap_gr_words,
              gmap_gr_offsets,
              gmap_gr_meta,
              packed_rows,
              outfeatures,
              tile_k,
              out_block,
              tile_q_gmap_words);
        }
      } else {
        decode_block_gr_tile(
            gmap_gr_words,
            gmap_gr_offsets,
            gmap_gr_meta,
            packed_rows,
            outfeatures,
            tile_k,
            out_block,
            tile_q_gmap_words);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = tile_k_size > 0 ? g_idx[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    if (out_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_raw != 0
            ? static_cast<unsigned>(gmap_gr_words[tile_q_gmap_word_begin + (local_k >> 5) * BLOCK_GR_TILE_OUT + out_block_offset])
            : tile_q_gmap_words[local_k >> 5][out_block_offset];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1,
            qweight2_sparse,
            qweight2_sparse_prefix,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3,
            g_idx,
            num_groups,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  if (out_valid) {
    atomicAdd(output + (proj_idx * batch + batch_idx) * outfeatures + out_idx, acc);
  }
}

template <bool UniformK64Groups>
__global__ void Compact3DualLinearKernelPackedFastNoScaleNoBiasSplitKBlockGR(
    const half* __restrict__ input,
    const int* __restrict__ a_qweight1,
    const int* __restrict__ a_qweight2_sparse,
    const int* __restrict__ a_qweight2_sparse_prefix,
    const int* __restrict__ a_gmap_gr_words,
    const int* __restrict__ a_gmap_gr_offsets,
    const int* __restrict__ a_gmap_gr_meta,
    const int* __restrict__ a_c_map,
    const int* __restrict__ a_g_idx,
    const int64_t* __restrict__ a_col_pack3,
    const int64_t* __restrict__ a_row_pack3,
    const int* __restrict__ b_qweight1,
    const int* __restrict__ b_qweight2_sparse,
    const int* __restrict__ b_qweight2_sparse_prefix,
    const int* __restrict__ b_gmap_gr_words,
    const int* __restrict__ b_gmap_gr_offsets,
    const int* __restrict__ b_gmap_gr_meta,
    const int* __restrict__ b_c_map,
    const int* __restrict__ b_g_idx,
    const int64_t* __restrict__ b_col_pack3,
    const int64_t* __restrict__ b_row_pack3,
    float* __restrict__ output,
    int batch,
    int rank,
    int outfeatures,
    int num_groups,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  const int z_idx = static_cast<int>(blockIdx.z);
  const int proj_idx = z_idx / split_k_slices;
  const int split_k_idx = z_idx - proj_idx * split_k_slices;
  const bool out_valid = out_idx < outfeatures && batch_idx < batch && proj_idx < 2;
  const int packed_rows = (rank + 31) / 32;
  const int out_block = static_cast<int>(blockIdx.x);
  const int out_block_offset = threadIdx.x;

  const int* qweight1 = proj_idx == 0 ? a_qweight1 : b_qweight1;
  const int* qweight2_sparse = proj_idx == 0 ? a_qweight2_sparse : b_qweight2_sparse;
  const int* qweight2_sparse_prefix = proj_idx == 0 ? a_qweight2_sparse_prefix : b_qweight2_sparse_prefix;
  const int* gmap_gr_words = proj_idx == 0 ? a_gmap_gr_words : b_gmap_gr_words;
  const int* gmap_gr_offsets = proj_idx == 0 ? a_gmap_gr_offsets : b_gmap_gr_offsets;
  const int* gmap_gr_meta = proj_idx == 0 ? a_gmap_gr_meta : b_gmap_gr_meta;
  const int* c_map = proj_idx == 0 ? a_c_map : b_c_map;
  const int* g_idx = proj_idx == 0 ? a_g_idx : b_g_idx;
  const int64_t* col_pack3 = proj_idx == 0 ? a_col_pack3 : b_col_pack3;
  const int64_t* row_pack3 = proj_idx == 0 ? a_row_pack3 : b_row_pack3;

  const int num_k_tiles = ceil_div_int(rank, STRUCTURED_TILE_K);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_q_gmap_raw;
  __shared__ int tile_q_gmap_word_begin;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;

  float acc = 0.0f;
  const half* input_row = input + batch_idx * rank;

  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = input_row[tile_k + idx];
      tile_col_pack3[idx] = static_cast<unsigned long long>(col_pack3[tile_k + idx]);
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      tile_q_gmap_raw = 0;
      tile_q_gmap_word_begin = 0;
      const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
      const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
      const int k_block = tile_k / STRUCTURED_TILE_K;
      if (k_block >= 0 && k_block < num_k_blocks && out_block >= 0 && out_block < num_out_blocks) {
        const int block_idx = k_block * num_out_blocks + out_block;
        const int meta_word = gmap_gr_meta[block_idx];
        if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
          tile_q_gmap_raw = 1;
          tile_q_gmap_word_begin = gmap_gr_offsets[block_idx];
        } else {
          decode_block_gr_tile(
              gmap_gr_words,
              gmap_gr_offsets,
              gmap_gr_meta,
              packed_rows,
              outfeatures,
              tile_k,
              out_block,
              tile_q_gmap_words);
        }
      } else {
        decode_block_gr_tile(
            gmap_gr_words,
            gmap_gr_offsets,
            gmap_gr_meta,
            packed_rows,
            outfeatures,
            tile_k,
            out_block,
            tile_q_gmap_words);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = tile_k_size > 0 ? g_idx[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    if (out_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_raw != 0
            ? static_cast<unsigned>(gmap_gr_words[tile_q_gmap_word_begin + (local_k >> 5) * BLOCK_GR_TILE_OUT + out_block_offset])
            : tile_q_gmap_words[local_k >> 5][out_block_offset];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1,
            qweight2_sparse,
            qweight2_sparse_prefix,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3,
            g_idx,
            num_groups,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  if (out_valid) {
    output[((proj_idx * split_k_slices + split_k_idx) * batch + batch_idx) * outfeatures + out_idx] = acc;
  }
}

__global__ void Compact3DualSplitKFinalizeHalfKernel(
    const float* __restrict__ input,
    half* __restrict__ output,
    int numel_per_projection,
    int split_k_slices) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = numel_per_projection * 2;
  if (idx >= total) {
    return;
  }
  const int proj_idx = idx / numel_per_projection;
  const int projection_offset = proj_idx * split_k_slices * numel_per_projection;
  const int local_idx = idx - proj_idx * numel_per_projection;
  float acc = 0.0f;
  for (int split_k_idx = 0; split_k_idx < split_k_slices; ++split_k_idx) {
    acc += input[projection_offset + split_k_idx * numel_per_projection + local_idx];
  }
  output[idx] = __float2half(acc);
}

__global__ void Compact3QKVFinalizeFloatKernel(
    const float* __restrict__ input,
    const half* __restrict__ q_bias,
    const half* __restrict__ k_bias,
    const half* __restrict__ v_bias,
    half* __restrict__ output,
    int numel_per_projection,
    int outfeatures) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = numel_per_projection * 3;
  if (idx >= total) {
    return;
  }
  const int proj_idx = idx / numel_per_projection;
  const int out_idx = idx % outfeatures;
  const half* bias = proj_idx == 0 ? q_bias : (proj_idx == 1 ? k_bias : v_bias);
  output[idx] = __float2half(input[idx] + __half2float(bias[out_idx]));
}

__global__ void Compact3LinearFinalizeFloatKernel(
    const float* __restrict__ input,
    const half* __restrict__ output_scale,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int numel,
    int outfeatures,
    bool has_output_scale,
    bool has_bias) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  const int out_idx = idx % outfeatures;
  float out_value = input[idx];
  if (has_output_scale) {
    out_value *= __half2float(output_scale[out_idx]);
  }
  if (has_bias) {
    out_value += __half2float(bias[out_idx]);
  }
  output[idx] = __float2half(out_value);
}

__global__ void AddBiasHalfKernel(
    half* __restrict__ output,
    const half* __restrict__ bias,
    int numel,
    int outfeatures) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  const int out_idx = idx % outfeatures;
  output[idx] = __hadd(output[idx], bias[out_idx]);
}

__global__ void Compact3TriSMoEPrecomputedKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_sparse,
    const int* __restrict__ q_gmap_sparse_prefix,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    const half* __restrict__ output_scale,
    const half* __restrict__ bias,
    const half* __restrict__ ns_hidden,
    const int* __restrict__ ns_c1_qweight,
    const int* __restrict__ ns_c1_qzeros,
    const half* __restrict__ ns_c1_scales,
    const int* __restrict__ ns_c1_g_idx,
    half* __restrict__ output,
    int batch,
    int rank,
    int outfeatures,
    int num_groups,
    int ns_rank,
    int ns_c1_qzeros_cols,
    bool has_bias) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  const bool out_valid = out_idx < outfeatures && batch_idx < batch;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ int tile_group_values[STRUCTURED_TILE_K];
  __shared__ float ns_hidden_chunk[NS_FUSED_CHUNK];

  float acc = 0.0f;
  const half* hidden_row = hidden + batch_idx * rank;
  const half* ns_hidden_row = ns_rank > 0 ? (ns_hidden + batch_idx * ns_rank) : nullptr;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      hidden_tile[idx] = hidden_row[global_k];
      tile_col_pack3[idx] = static_cast<unsigned long long>(col_pack3[global_k]);
      tile_group_values[idx] = g_idx[global_k];
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    __syncthreads();

    if (out_valid) {
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const int group0 = tile_group_values[local_k];
        const int group1 = has_k1 ? tile_group_values[local_k + 1] : group0;
        const unsigned long long row_pack3_value0 =
            (group0 >= 0 && group0 < num_groups)
                ? static_cast<unsigned long long>(row_pack3[out_idx * num_groups + group0])
                : 0ULL;
        const unsigned long long row_pack3_value1 =
            (has_k1 && group1 >= 0 && group1 < num_groups)
                ? static_cast<unsigned long long>(row_pack3[out_idx * num_groups + group1])
                : 0ULL;
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_groups(
            qweight1,
            qweight2_sparse,
            qweight2_sparse_prefix,
            q_gmap_sparse,
            q_gmap_sparse_prefix,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value0,
            row_pack3_value1);
        acc += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  for (int ns_base = 0; ns_base < ns_rank; ns_base += NS_FUSED_CHUNK) {
    const int chunk = min(NS_FUSED_CHUNK, ns_rank - ns_base);
    if (threadIdx.x < chunk) {
      ns_hidden_chunk[threadIdx.x] = __half2float(ns_hidden_row[ns_base + threadIdx.x]);
    }
    __syncthreads();
    if (out_valid) {
      for (int idx = 0; idx < chunk; ++idx) {
        const float c1 = load_packed4_weight(
            ns_c1_qweight,
            ns_c1_qzeros,
            ns_c1_scales,
            ns_c1_g_idx,
            outfeatures,
            ns_c1_qzeros_cols,
            out_idx,
            ns_base + idx);
        acc += ns_hidden_chunk[idx] * c1;
      }
    }
    __syncthreads();
  }

  if (out_valid) {
    float out_value = acc * __half2float(output_scale[out_idx]);
    if (has_bias) {
      out_value += __half2float(bias[out_idx]);
    }
    output[batch_idx * outfeatures + out_idx] = __float2half(out_value);
  }
}

__device__ __forceinline__ int find_grouped_problem_segment(const int* __restrict__ problem_offsets, int num_segments, int problem_tile_idx) {
  int lo = 0;
  int hi = num_segments - 1;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (problem_offsets[mid + 1] <= problem_tile_idx) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ __forceinline__ int find_grouped_row_segment(const int* __restrict__ segment_offsets, int num_segments, int row_idx) {
  int lo = 0;
  int hi = num_segments - 1;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (segment_offsets[mid + 1] <= row_idx) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__host__ __device__ __forceinline__ bool is_grouped_decode_problem(int batch, int num_segments, int64_t total_problem_tiles) {
  return batch > 0 && num_segments == batch && total_problem_tiles == static_cast<int64_t>(batch);
}



__global__ void Compact3TriSMoEPrecomputedGroupedSegmentKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_sparse,
    const int* __restrict__ q_gmap_sparse_prefix,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    const half* __restrict__ output_scale,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int qweight2_sparse_rows,
    int q_gmap_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups,
    bool has_bias) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_valid = out_idx < outfeatures;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ int tile_group_values[STRUCTURED_TILE_K];
  __shared__ int expert_slot_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    const int row_start = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    expert_slot_shared = expert_ids[segment_idx];
    row_start_shared = row_start;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start);
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int packed_rows = (rank + 31) / 32;
  const int matrix_stride = packed_rows * outfeatures;
  const int matrix_stride_qweight2_sparse = qweight2_sparse_rows * outfeatures;
  const int matrix_stride_q_gmap_sparse = q_gmap_sparse_rows * outfeatures;
  const int col_pack_stride = rank;
  const int row_pack_stride = outfeatures * num_groups;
  const int out_stride = outfeatures;

  const int* qweight1_e = expert_valid ? (qweight1 + expert_slot * matrix_stride) : nullptr;
  const int* qweight2_sparse_e = expert_valid ? (qweight2_sparse + expert_slot * matrix_stride_qweight2_sparse) : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? (qweight2_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* q_gmap_sparse_e = expert_valid ? (q_gmap_sparse + expert_slot * matrix_stride_q_gmap_sparse) : nullptr;
  const int* q_gmap_sparse_prefix_e = expert_valid ? (q_gmap_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* c_map_e = expert_valid ? (c_map + expert_slot * packed_rows) : nullptr;
  const int* g_idx_e = expert_valid ? (g_idx + expert_slot * rank) : nullptr;
  const int64_t* col_pack3_e = expert_valid ? (col_pack3 + expert_slot * col_pack_stride) : nullptr;
  const int64_t* row_pack3_e = expert_valid ? (row_pack3 + expert_slot * row_pack_stride) : nullptr;
  const half* output_scale_e = expert_valid ? (output_scale + expert_slot * out_stride) : nullptr;
  const half* bias_e = (expert_valid && has_bias) ? (bias + expert_slot * out_stride) : nullptr;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden[(row_start + row) * rank + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_e[global_k]) : 0ULL;
      tile_group_values[idx] = expert_valid ? g_idx_e[global_k] : -1;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    __syncthreads();

    if (out_valid && expert_valid) {
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const int group0 = tile_group_values[local_k];
        const int group1 = has_k1 ? tile_group_values[local_k + 1] : group0;
        const unsigned long long row_pack3_value0 =
            (group0 >= 0 && group0 < num_groups)
                ? static_cast<unsigned long long>(row_pack3_e[out_idx * num_groups + group0])
                : 0ULL;
        const unsigned long long row_pack3_value1 =
            (has_k1 && group1 >= 0 && group1 < num_groups)
                ? static_cast<unsigned long long>(row_pack3_e[out_idx * num_groups + group1])
                : 0ULL;
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_groups(
            qweight1_e,
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            q_gmap_sparse_e,
            q_gmap_sparse_prefix_e,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value0,
            row_pack3_value1);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  if (out_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      float out_value = acc[row];
      if (expert_valid) {
        out_value *= __half2float(output_scale_e[out_idx]);
        if (has_bias) {
          out_value += __half2float(bias_e[out_idx]);
        }
      }
      output[(row_start + row) * outfeatures + out_idx] = __float2half(out_value);
    }
  }
}

__global__ void Compact3TriSMoEPrecomputedGroupedSegmentSplitKKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_sparse,
    const int* __restrict__ q_gmap_sparse_prefix,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    float* __restrict__ output_accum,
    int qweight2_sparse_rows,
    int q_gmap_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool out_valid = out_idx < outfeatures;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ int tile_group_values[STRUCTURED_TILE_K];
  __shared__ int expert_slot_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    const int row_start = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    expert_slot_shared = expert_ids[segment_idx];
    row_start_shared = row_start;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start);
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int packed_rows = (rank + 31) / 32;
  const int matrix_stride = packed_rows * outfeatures;
  const int matrix_stride_qweight2_sparse = qweight2_sparse_rows * outfeatures;
  const int matrix_stride_q_gmap_sparse = q_gmap_sparse_rows * outfeatures;
  const int col_pack_stride = rank;
  const int row_pack_stride = outfeatures * num_groups;

  const int* qweight1_e = expert_valid ? (qweight1 + expert_slot * matrix_stride) : nullptr;
  const int* qweight2_sparse_e = expert_valid ? (qweight2_sparse + expert_slot * matrix_stride_qweight2_sparse) : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? (qweight2_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* q_gmap_sparse_e = expert_valid ? (q_gmap_sparse + expert_slot * matrix_stride_q_gmap_sparse) : nullptr;
  const int* q_gmap_sparse_prefix_e = expert_valid ? (q_gmap_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* c_map_e = expert_valid ? (c_map + expert_slot * packed_rows) : nullptr;
  const int* g_idx_e = expert_valid ? (g_idx + expert_slot * rank) : nullptr;
  const int64_t* col_pack3_e = expert_valid ? (col_pack3 + expert_slot * col_pack_stride) : nullptr;
  const int64_t* row_pack3_e = expert_valid ? (row_pack3 + expert_slot * row_pack_stride) : nullptr;

  const int num_k_tiles = ceil_div_int(rank, STRUCTURED_TILE_K);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden[(row_start + row) * rank + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_e[global_k]) : 0ULL;
      tile_group_values[idx] = expert_valid ? g_idx_e[global_k] : -1;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    __syncthreads();

    if (out_valid && expert_valid) {
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const int group0 = tile_group_values[local_k];
        const int group1 = has_k1 ? tile_group_values[local_k + 1] : group0;
        const unsigned long long row_pack3_value0 =
            (group0 >= 0 && group0 < num_groups)
                ? static_cast<unsigned long long>(row_pack3_e[out_idx * num_groups + group0])
                : 0ULL;
        const unsigned long long row_pack3_value1 =
            (has_k1 && group1 >= 0 && group1 < num_groups)
                ? static_cast<unsigned long long>(row_pack3_e[out_idx * num_groups + group1])
                : 0ULL;
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_groups(
            qweight1_e,
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            q_gmap_sparse_e,
            q_gmap_sparse_prefix_e,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value0,
            row_pack3_value1);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  if (out_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      atomicAdd(output_accum + (row_start + row) * outfeatures + out_idx, acc[row]);
    }
  }
}

__global__ void Compact3TriSMoEPrecomputedGroupedDecodeKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_sparse,
    const int* __restrict__ q_gmap_sparse_prefix,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    const half* __restrict__ output_scale,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int qweight2_sparse_rows,
    int q_gmap_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups,
    bool has_bias) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_valid = out_idx < outfeatures;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ int tile_group_values[STRUCTURED_TILE_K];
  __shared__ int expert_slot_shared;
  __shared__ int row_idx_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_slot_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc = 0.0f;

  const int packed_rows = (rank + 31) / 32;
  const int matrix_stride = packed_rows * outfeatures;
  const int matrix_stride_qweight2_sparse = qweight2_sparse_rows * outfeatures;
  const int matrix_stride_q_gmap_sparse = q_gmap_sparse_rows * outfeatures;
  const int col_pack_stride = rank;
  const int row_pack_stride = outfeatures * num_groups;
  const int out_stride = outfeatures;

  const int* qweight1_e = expert_valid ? (qweight1 + expert_slot * matrix_stride) : nullptr;
  const int* qweight2_sparse_e = expert_valid ? (qweight2_sparse + expert_slot * matrix_stride_qweight2_sparse) : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? (qweight2_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* q_gmap_sparse_e = expert_valid ? (q_gmap_sparse + expert_slot * matrix_stride_q_gmap_sparse) : nullptr;
  const int* q_gmap_sparse_prefix_e = expert_valid ? (q_gmap_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* c_map_e = expert_valid ? (c_map + expert_slot * packed_rows) : nullptr;
  const int* g_idx_e = expert_valid ? (g_idx + expert_slot * rank) : nullptr;
  const int64_t* col_pack3_e = expert_valid ? (col_pack3 + expert_slot * col_pack_stride) : nullptr;
  const int64_t* row_pack3_e = expert_valid ? (row_pack3 + expert_slot * row_pack_stride) : nullptr;
  const half* output_scale_e = expert_valid ? (output_scale + expert_slot * out_stride) : nullptr;
  const half* bias_e = (expert_valid && has_bias) ? (bias + expert_slot * out_stride) : nullptr;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = hidden[row_idx * rank + tile_k + idx];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_e[global_k]) : 0ULL;
      tile_group_values[idx] = expert_valid ? g_idx_e[global_k] : -1;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    __syncthreads();

    if (out_valid && expert_valid) {
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const int group0 = tile_group_values[local_k];
        const int group1 = has_k1 ? tile_group_values[local_k + 1] : group0;
        const unsigned long long row_pack3_value0 =
            (group0 >= 0 && group0 < num_groups)
                ? static_cast<unsigned long long>(row_pack3_e[out_idx * num_groups + group0])
                : 0ULL;
        const unsigned long long row_pack3_value1 =
            (has_k1 && group1 >= 0 && group1 < num_groups)
                ? static_cast<unsigned long long>(row_pack3_e[out_idx * num_groups + group1])
                : 0ULL;
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_groups(
            qweight1_e,
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            q_gmap_sparse_e,
            q_gmap_sparse_prefix_e,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value0,
            row_pack3_value1);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  if (out_valid) {
    float out_value = acc;
    if (expert_valid) {
      out_value *= __half2float(output_scale_e[out_idx]);
      if (has_bias) {
        out_value += __half2float(bias_e[out_idx]);
      }
    }
    output[row_idx * outfeatures + out_idx] = __float2half(out_value);
  }
}

__global__ void Compact3TriSMoEPrecomputedGroupedDecodeSplitKKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_sparse,
    const int* __restrict__ q_gmap_sparse_prefix,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    float* __restrict__ output_accum,
    int qweight2_sparse_rows,
    int q_gmap_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool out_valid = out_idx < outfeatures;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ int tile_group_values[STRUCTURED_TILE_K];
  __shared__ int expert_slot_shared;
  __shared__ int row_idx_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_slot_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc = 0.0f;

  const int packed_rows = (rank + 31) / 32;
  const int matrix_stride = packed_rows * outfeatures;
  const int matrix_stride_qweight2_sparse = qweight2_sparse_rows * outfeatures;
  const int matrix_stride_q_gmap_sparse = q_gmap_sparse_rows * outfeatures;
  const int col_pack_stride = rank;
  const int row_pack_stride = outfeatures * num_groups;

  const int* qweight1_e = expert_valid ? (qweight1 + expert_slot * matrix_stride) : nullptr;
  const int* qweight2_sparse_e = expert_valid ? (qweight2_sparse + expert_slot * matrix_stride_qweight2_sparse) : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? (qweight2_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* q_gmap_sparse_e = expert_valid ? (q_gmap_sparse + expert_slot * matrix_stride_q_gmap_sparse) : nullptr;
  const int* q_gmap_sparse_prefix_e = expert_valid ? (q_gmap_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* c_map_e = expert_valid ? (c_map + expert_slot * packed_rows) : nullptr;
  const int* g_idx_e = expert_valid ? (g_idx + expert_slot * rank) : nullptr;
  const int64_t* col_pack3_e = expert_valid ? (col_pack3 + expert_slot * col_pack_stride) : nullptr;
  const int64_t* row_pack3_e = expert_valid ? (row_pack3 + expert_slot * row_pack_stride) : nullptr;

  const int num_k_tiles = ceil_div_int(rank, STRUCTURED_TILE_K);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = hidden[row_idx * rank + tile_k + idx];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_e[global_k]) : 0ULL;
      tile_group_values[idx] = expert_valid ? g_idx_e[global_k] : -1;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    __syncthreads();

    if (out_valid && expert_valid) {
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const int group0 = tile_group_values[local_k];
        const int group1 = has_k1 ? tile_group_values[local_k + 1] : group0;
        const unsigned long long row_pack3_value0 =
            (group0 >= 0 && group0 < num_groups)
                ? static_cast<unsigned long long>(row_pack3_e[out_idx * num_groups + group0])
                : 0ULL;
        const unsigned long long row_pack3_value1 =
            (has_k1 && group1 >= 0 && group1 < num_groups)
                ? static_cast<unsigned long long>(row_pack3_e[out_idx * num_groups + group1])
                : 0ULL;
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_groups(
            qweight1_e,
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            q_gmap_sparse_e,
            q_gmap_sparse_prefix_e,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value0,
            row_pack3_value1);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  if (out_valid) {
    atomicAdd(output_accum + row_idx * outfeatures + out_idx, acc);
  }
}

template <bool HasOutputScale, bool HasBias, bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedSegmentBlockGRKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    int q_gmap_gr_words_stride,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    const half* __restrict__ output_scale,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int qweight2_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_valid = out_idx < outfeatures;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    const int row_start = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    expert_slot_shared = expert_ids[segment_idx];
    row_start_shared = row_start;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start);
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int packed_rows = (rank + 31) / 32;
  const int out_block = static_cast<int>(blockIdx.x);
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  const int matrix_stride = packed_rows * outfeatures;
  const int matrix_stride_qweight2_sparse = qweight2_sparse_rows * outfeatures;
  const int col_pack_stride = rank;
  const int row_pack_stride = outfeatures * num_groups;
  const int out_stride = outfeatures;

  const int* qweight1_e = expert_valid ? (qweight1 + expert_slot * matrix_stride) : nullptr;
  const int* qweight2_sparse_e = expert_valid ? (qweight2_sparse + expert_slot * matrix_stride_qweight2_sparse) : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? (qweight2_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* q_gmap_gr_words_e = expert_valid ? (q_gmap_gr_words + expert_slot * q_gmap_gr_words_stride) : nullptr;
  const int* q_gmap_gr_offsets_e = expert_valid ? (q_gmap_gr_offsets + expert_slot * (gmap_block_count + 1)) : nullptr;
  const int* q_gmap_gr_meta_e = expert_valid ? (q_gmap_gr_meta + expert_slot * gmap_block_count) : nullptr;
  const int* c_map_e = expert_valid ? (c_map + expert_slot * packed_rows) : nullptr;
  const int* g_idx_e = expert_valid ? (g_idx + expert_slot * rank) : nullptr;
  const int64_t* col_pack3_e = expert_valid ? (col_pack3 + expert_slot * col_pack_stride) : nullptr;
  const int64_t* row_pack3_e = expert_valid ? (row_pack3 + expert_slot * row_pack_stride) : nullptr;
  const half* output_scale_e = expert_valid ? (output_scale + expert_slot * out_stride) : nullptr;
  const half* bias_e = (expert_valid && HasBias) ? (bias + expert_slot * out_stride) : nullptr;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden[(row_start + row) * rank + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_e,
          q_gmap_gr_offsets_e,
          q_gmap_gr_meta_e,
          packed_rows,
          outfeatures,
          tile_k,
          out_block,
          tile_q_gmap_words);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    if (out_valid && expert_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_e,
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_e,
            g_idx_e,
            num_groups,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  if (out_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      float out_value = acc[row];
      if (expert_valid) {
        if constexpr (HasOutputScale) {
          out_value *= __half2float(output_scale_e[out_idx]);
        }
        if constexpr (HasBias) {
          out_value += __half2float(bias_e[out_idx]);
        }
      }
      output[(row_start + row) * outfeatures + out_idx] = __float2half(out_value);
    }
  }
}

__global__ void Compact3TriSMoEGroupedBlockGRPredecodeKernel(
    const int* __restrict__ expert_ids,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    int q_gmap_gr_words_stride,
    int* __restrict__ decoded_q_gmap_words,
    int gmap_block_count) {
  const int gmap_block_idx = static_cast<int>(blockIdx.x);
  const int active_idx = static_cast<int>(blockIdx.y);
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];

  if (threadIdx.x == 0) {
    const int expert_slot = expert_ids[active_idx];
    const bool expert_valid = expert_slot >= 0;
    const int* q_gmap_gr_words_e = expert_valid ? (q_gmap_gr_words + expert_slot * q_gmap_gr_words_stride) : nullptr;
    const int* q_gmap_gr_offsets_e = expert_valid ? (q_gmap_gr_offsets + expert_slot * (gmap_block_count + 1)) : nullptr;
    const int* q_gmap_gr_meta_e = expert_valid ? (q_gmap_gr_meta + expert_slot * gmap_block_count) : nullptr;
    if (expert_valid && gmap_block_idx >= 0 && gmap_block_idx < gmap_block_count) {
      decode_block_gr_block(q_gmap_gr_words_e, q_gmap_gr_offsets_e, q_gmap_gr_meta_e, gmap_block_idx, tile_q_gmap_words);
    } else {
      for (int local_packed_row = 0; local_packed_row < BLOCK_GR_PACKED_ROWS; ++local_packed_row) {
        for (int local_out = 0; local_out < BLOCK_GR_TILE_OUT; ++local_out) {
          tile_q_gmap_words[local_packed_row][local_out] = 0U;
        }
      }
    }
  }
  __syncthreads();

  const int output_base = (active_idx * gmap_block_count + gmap_block_idx) * BLOCK_GR_WORDS_PER_BLOCK;
  if (threadIdx.x < BLOCK_GR_WORDS_PER_BLOCK) {
    decoded_q_gmap_words[output_base + threadIdx.x] =
        static_cast<int>(tile_q_gmap_words[threadIdx.x / BLOCK_GR_TILE_OUT][threadIdx.x % BLOCK_GR_TILE_OUT]);
  }
}

template <bool TailOnly, bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedSegmentNoScaleNoBiasBlockGRPredecodedKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ decoded_q_gmap_words,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    half* __restrict__ output,
    int qweight2_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups,
    int decoded_gmap_block_count) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_block = static_cast<int>(blockIdx.x);
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_valid = out_idx < outfeatures;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int segment_idx_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    const int segment_rows = segment_row_end - segment_row_begin;
    const int full_rows = (segment_rows / STRUCTURED_TC_M_TILE) * STRUCTURED_TC_M_TILE;
    const int row_start = segment_row_begin
        + (TailOnly ? full_rows : 0)
        + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    expert_slot_shared = expert_ids[segment_idx];
    segment_idx_shared = segment_idx;
    row_start_shared = row_start;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start);
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int segment_idx = segment_idx_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int packed_rows = (rank + 31) / 32;
  const int out_tiles = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  const int matrix_stride = packed_rows * outfeatures;
  const int matrix_stride_qweight2_sparse = qweight2_sparse_rows * outfeatures;
  const int col_pack_stride = rank;
  const int row_pack_stride = outfeatures * num_groups;

  const int* qweight1_e = expert_valid ? (qweight1 + expert_slot * matrix_stride) : nullptr;
  const int* qweight2_sparse_e = expert_valid ? (qweight2_sparse + expert_slot * matrix_stride_qweight2_sparse) : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? (qweight2_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* decoded_q_gmap_words_e = expert_valid ? (decoded_q_gmap_words + segment_idx * decoded_gmap_block_count * BLOCK_GR_WORDS_PER_BLOCK) : nullptr;
  const int* c_map_e = expert_valid ? (c_map + expert_slot * packed_rows) : nullptr;
  const int* g_idx_e = expert_valid ? (g_idx + expert_slot * rank) : nullptr;
  const int64_t* col_pack3_e = expert_valid ? (col_pack3 + expert_slot * col_pack_stride) : nullptr;
  const int64_t* row_pack3_e = expert_valid ? (row_pack3 + expert_slot * row_pack_stride) : nullptr;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden[(row_start + row) * rank + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    load_predecoded_block_gr_tile(
        decoded_q_gmap_words_e,
        decoded_gmap_block_count,
        (tile_k / STRUCTURED_TILE_K) * out_tiles + out_block,
        expert_valid,
        tile_q_gmap_words);
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    if (out_valid && expert_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_e,
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_e,
            g_idx_e,
            num_groups,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  if (out_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      output[(row_start + row) * outfeatures + out_idx] = __float2half(acc[row]);
    }
  }
}

#ifndef USE_ROCM
template <bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedM16TensorCoreBlockGRPredecodedKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ full_problem_offsets,
    const int* __restrict__ decoded_q_gmap_words,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    half* __restrict__ output,
    int qweight2_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups,
    int decoded_gmap_block_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const int linear_tid = threadIdx.x;
  const int warp_id = linear_tid >> 5;
  const int out_col_start = static_cast<int>(blockIdx.x) * STRUCTURED_TC_N_TILE;
  const int out_block = static_cast<int>(blockIdx.x);
  const int problem_tile_idx = static_cast<int>(blockIdx.y);

  __shared__ half hidden_tile[STRUCTURED_TC_M_TILE][STRUCTURED_TC_K_TILE];
  __shared__ half weight_stage[STRUCTURED_TC_N_TILE][STRUCTURED_TC_K_STEP];
  __shared__ unsigned tile_c_map_words[STRUCTURED_TC_K_TILE / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TC_K_TILE];
  __shared__ unsigned long long row_pack3_tile[STRUCTURED_TC_N_TILE];
  __shared__ unsigned qweight1_word_tile[STRUCTURED_TC_K_TILE / 32][STRUCTURED_TC_N_TILE];
  __shared__ unsigned qweight2_word_tile[STRUCTURED_TC_K_TILE / 32][STRUCTURED_TC_N_TILE];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ float output_tile[STRUCTURED_TC_WARPS][STRUCTURED_TC_M_TILE][STRUCTURED_TC_K_STEP];
  __shared__ int expert_slot_shared;
  __shared__ int segment_idx_shared;
  __shared__ int row_start_shared;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;

  if (linear_tid == 0) {
    const int segment_idx = find_grouped_problem_segment(full_problem_offsets, num_segments, problem_tile_idx);
    const int local_tile_idx = problem_tile_idx - full_problem_offsets[segment_idx];
    expert_slot_shared = expert_ids[segment_idx];
    segment_idx_shared = segment_idx;
    row_start_shared = segment_offsets[segment_idx] + local_tile_idx * STRUCTURED_TC_M_TILE;
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int segment_idx = segment_idx_shared;
  const int row_start = row_start_shared;
  const bool expert_valid = expert_slot >= 0;
  const int packed_rows = (rank + 31) / 32;
  const int out_tiles = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  const int matrix_stride = packed_rows * outfeatures;
  const int sparse_matrix_stride = qweight2_sparse_rows * outfeatures;
  const int row_pack_stride = outfeatures * num_groups;

  const int* qweight1_e = expert_valid ? qweight1 + expert_slot * matrix_stride : nullptr;
  const int* qweight2_sparse_e = expert_valid ? qweight2_sparse + expert_slot * sparse_matrix_stride : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? qweight2_sparse_prefix + expert_slot * (packed_rows + 1) : nullptr;
  const int* c_map_e = expert_valid ? c_map + expert_slot * packed_rows : nullptr;
  const int* g_idx_e = expert_valid ? g_idx + expert_slot * rank : nullptr;
  const int64_t* col_pack3_e = expert_valid ? col_pack3 + expert_slot * rank : nullptr;
  const int64_t* row_pack3_e = expert_valid ? row_pack3 + expert_slot * row_pack_stride : nullptr;
  const int* decoded_q_gmap_words_e = expert_valid
      ? decoded_q_gmap_words + segment_idx * decoded_gmap_block_count * BLOCK_GR_WORDS_PER_BLOCK
      : nullptr;

  nvcuda::wmma::fragment<
      nvcuda::wmma::accumulator,
      STRUCTURED_TC_M_TILE,
      STRUCTURED_TC_K_STEP,
      STRUCTURED_TC_K_STEP,
      float> acc_frag;
  if (warp_id < STRUCTURED_TC_WARPS) {
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
  }

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TC_K_TILE) {
    const int tile_k_size = min(STRUCTURED_TC_K_TILE, rank - tile_k);
    const int packed_row_base = tile_k >> 5;
    const int tile_word_count = ceil_div_int(tile_k_size, 32);
    const int hidden_items = STRUCTURED_TC_M_TILE * STRUCTURED_TC_K_TILE;
    for (int idx = linear_tid; idx < hidden_items; idx += blockDim.x) {
      const int row = idx / STRUCTURED_TC_K_TILE;
      const int local_k = idx - row * STRUCTURED_TC_K_TILE;
      hidden_tile[row][local_k] = local_k < tile_k_size
          ? hidden[(row_start + row) * rank + tile_k + local_k]
          : __float2half(0.0f);
    }
    for (int idx = linear_tid; idx < STRUCTURED_TC_K_TILE; idx += blockDim.x) {
      tile_col_pack3[idx] = (expert_valid && idx < tile_k_size)
          ? static_cast<unsigned long long>(col_pack3_e[tile_k + idx])
          : 0ULL;
    }
    if (linear_tid < STRUCTURED_TC_K_TILE / 32) {
      const int word_idx = linear_tid;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        word = static_cast<unsigned>(c_map_e[packed_row_base + word_idx])
            & low_bit_mask(min(32, tile_k_size - word_base));
      }
      tile_c_map_words[word_idx] = word;
    }
    load_predecoded_block_gr_tile(
        decoded_q_gmap_words_e,
        decoded_gmap_block_count,
        (tile_k / STRUCTURED_TC_K_TILE) * out_tiles + out_block,
        expert_valid,
        tile_q_gmap_words);
    if (linear_tid == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    for (int local_out = linear_tid; local_out < STRUCTURED_TC_N_TILE; local_out += blockDim.x) {
      const int out_idx = out_col_start + local_out;
      row_pack3_tile[local_out] = (expert_valid && tile_group_valid != 0 && out_idx < outfeatures)
          ? static_cast<unsigned long long>(row_pack3_e[out_idx * num_groups + tile_group_value])
          : 0ULL;
    }
    const int qword_items = (STRUCTURED_TC_K_TILE / 32) * STRUCTURED_TC_N_TILE;
    for (int idx = linear_tid; idx < qword_items; idx += blockDim.x) {
      const int local_packed_row = idx / STRUCTURED_TC_N_TILE;
      const int local_out = idx - local_packed_row * STRUCTURED_TC_N_TILE;
      const int out_idx = out_col_start + local_out;
      unsigned qweight1_word = 0U;
      unsigned qweight2_word = 0U;
      if (expert_valid && local_packed_row < tile_word_count && out_idx < outfeatures) {
        const int packed_row = packed_row_base + local_packed_row;
        qweight1_word = static_cast<unsigned>(qweight1_e[packed_row * outfeatures + out_idx]);
        qweight2_word = reconstruct_sparse_qweight2_dense_word(
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            tile_c_map_words[local_packed_row],
            packed_row,
            outfeatures,
            out_idx);
      }
      qweight1_word_tile[local_packed_row][local_out] = qweight1_word;
      qweight2_word_tile[local_packed_row][local_out] = qweight2_word;
    }
    __syncthreads();

    const int num_k_steps = ceil_div_int(tile_k_size, STRUCTURED_TC_K_STEP);
    const int stage_items = STRUCTURED_TC_N_TILE * STRUCTURED_TC_K_STEP;
    for (int step_idx = 0; step_idx < num_k_steps; ++step_idx) {
      for (int idx = linear_tid; idx < stage_items; idx += blockDim.x) {
        const int local_out = idx / STRUCTURED_TC_K_STEP;
        const int stage_k = idx - local_out * STRUCTURED_TC_K_STEP;
        const int local_k = step_idx * STRUCTURED_TC_K_STEP + stage_k;
        half w = __float2half(0.0f);
        if (expert_valid && local_k < tile_k_size && out_col_start + local_out < outfeatures) {
          const int local_packed_row = local_k >> 5;
          w = load_compact3_weight_from_dense_packs_gmap_word(
              qweight1_word_tile[local_packed_row][local_out],
              qweight2_word_tile[local_packed_row][local_out],
              tile_q_gmap_words[local_packed_row][local_out],
              tile_c_map_words[local_packed_row],
              local_k & 31,
              tile_col_pack3[local_k],
              select_compact3_row_pack<UniformK64Groups>(
                  row_pack3_e, g_idx_e, tile_k + local_k, out_col_start + local_out,
                  num_groups, tile_group_value, row_pack3_tile[local_out]));
        }
        weight_stage[local_out][stage_k] = w;
      }
      __syncthreads();
      if (warp_id < STRUCTURED_TC_WARPS) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
        nvcuda::wmma::load_matrix_sync(a_frag, &hidden_tile[0][step_idx * STRUCTURED_TC_K_STEP], STRUCTURED_TC_K_TILE);
        nvcuda::wmma::load_matrix_sync(b_frag, &weight_stage[warp_id * STRUCTURED_TC_K_STEP][0], STRUCTURED_TC_K_STEP);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      __syncthreads();
    }
  }

  if (warp_id < STRUCTURED_TC_WARPS) {
    nvcuda::wmma::store_matrix_sync(
        &output_tile[warp_id][0][0], acc_frag, STRUCTURED_TC_K_STEP, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  const int output_items = STRUCTURED_TC_M_TILE * STRUCTURED_TC_N_TILE;
  for (int idx = linear_tid; idx < output_items; idx += blockDim.x) {
    const int row = idx / STRUCTURED_TC_N_TILE;
    const int local_out = idx - row * STRUCTURED_TC_N_TILE;
    const int out_idx = out_col_start + local_out;
    if (out_idx < outfeatures) {
      output[(row_start + row) * outfeatures + out_idx] = __float2half(
          output_tile[local_out / STRUCTURED_TC_K_STEP][row][local_out % STRUCTURED_TC_K_STEP]);
    }
  }
#endif
}
#endif

template <bool TailOnly, bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedDualSegmentNoScaleNoBiasBlockGRPredecodedKernel(
    const half* __restrict__ hidden_a,
    const half* __restrict__ hidden_b,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ decoded_q_gmap_words_a,
    const int* __restrict__ qweight1_a,
    const int* __restrict__ qweight2_sparse_a,
    const int* __restrict__ qweight2_sparse_prefix_a,
    const int* __restrict__ c_map_a,
    const int* __restrict__ g_idx_a,
    const int64_t* __restrict__ col_pack3_a,
    const int64_t* __restrict__ row_pack3_a,
    const int* __restrict__ decoded_q_gmap_words_b,
    const int* __restrict__ qweight1_b,
    const int* __restrict__ qweight2_sparse_b,
    const int* __restrict__ qweight2_sparse_prefix_b,
    const int* __restrict__ c_map_b,
    const int* __restrict__ g_idx_b,
    const int64_t* __restrict__ col_pack3_b,
    const int64_t* __restrict__ row_pack3_b,
    half* __restrict__ output_a,
    half* __restrict__ output_b,
    int qweight2_sparse_rows_a,
    int qweight2_sparse_rows_b,
    int num_segments,
    int rank_a,
    int rank_b,
    int outfeatures_a,
    int outfeatures_b,
    int num_groups_a,
    int num_groups_b,
    int decoded_gmap_block_count_a,
    int decoded_gmap_block_count_b) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_block = static_cast<int>(blockIdx.x);
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_a_valid = out_idx < outfeatures_a;
  const bool out_b_valid = out_idx < outfeatures_b;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int segment_idx_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    const int segment_rows = segment_row_end - segment_row_begin;
    const int full_rows = (segment_rows / STRUCTURED_TC_M_TILE) * STRUCTURED_TC_M_TILE;
    const int row_start = segment_row_begin
        + (TailOnly ? full_rows : 0)
        + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    expert_slot_shared = expert_ids[segment_idx];
    segment_idx_shared = segment_idx;
    row_start_shared = row_start;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start);
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int segment_idx = segment_idx_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc_a[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc_b[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int packed_rows_a = (rank_a + 31) / 32;
  const int packed_rows_b = (rank_b + 31) / 32;
  const int out_tiles_a = ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
  const int out_tiles_b = ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
  const int matrix_stride_a = packed_rows_a * outfeatures_a;
  const int matrix_stride_b = packed_rows_b * outfeatures_b;
  const int matrix_stride_qweight2_sparse_a = qweight2_sparse_rows_a * outfeatures_a;
  const int matrix_stride_qweight2_sparse_b = qweight2_sparse_rows_b * outfeatures_b;
  const int col_pack_stride_a = rank_a;
  const int col_pack_stride_b = rank_b;
  const int row_pack_stride_a = outfeatures_a * num_groups_a;
  const int row_pack_stride_b = outfeatures_b * num_groups_b;

  const int* decoded_q_gmap_words_a_e =
      expert_valid ? (decoded_q_gmap_words_a + segment_idx * decoded_gmap_block_count_a * BLOCK_GR_WORDS_PER_BLOCK) : nullptr;
  const int* qweight1_a_e = expert_valid ? (qweight1_a + expert_slot * matrix_stride_a) : nullptr;
  const int* qweight2_sparse_a_e = expert_valid ? (qweight2_sparse_a + expert_slot * matrix_stride_qweight2_sparse_a) : nullptr;
  const int* qweight2_sparse_prefix_a_e = expert_valid ? (qweight2_sparse_prefix_a + expert_slot * (packed_rows_a + 1)) : nullptr;
  const int* c_map_a_e = expert_valid ? (c_map_a + expert_slot * packed_rows_a) : nullptr;
  const int* g_idx_a_e = expert_valid ? (g_idx_a + expert_slot * rank_a) : nullptr;
  const int64_t* col_pack3_a_e = expert_valid ? (col_pack3_a + expert_slot * col_pack_stride_a) : nullptr;
  const int64_t* row_pack3_a_e = expert_valid ? (row_pack3_a + expert_slot * row_pack_stride_a) : nullptr;

  const int* decoded_q_gmap_words_b_e =
      expert_valid ? (decoded_q_gmap_words_b + segment_idx * decoded_gmap_block_count_b * BLOCK_GR_WORDS_PER_BLOCK) : nullptr;
  const int* qweight1_b_e = expert_valid ? (qweight1_b + expert_slot * matrix_stride_b) : nullptr;
  const int* qweight2_sparse_b_e = expert_valid ? (qweight2_sparse_b + expert_slot * matrix_stride_qweight2_sparse_b) : nullptr;
  const int* qweight2_sparse_prefix_b_e = expert_valid ? (qweight2_sparse_prefix_b + expert_slot * (packed_rows_b + 1)) : nullptr;
  const int* c_map_b_e = expert_valid ? (c_map_b + expert_slot * packed_rows_b) : nullptr;
  const int* g_idx_b_e = expert_valid ? (g_idx_b + expert_slot * rank_b) : nullptr;
  const int64_t* col_pack3_b_e = expert_valid ? (col_pack3_b + expert_slot * col_pack_stride_b) : nullptr;
  const int64_t* row_pack3_b_e = expert_valid ? (row_pack3_b + expert_slot * row_pack_stride_b) : nullptr;

  for (int tile_k = 0; tile_k < rank_a; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_a - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden_a[(row_start + row) * rank_a + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_a_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_a_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    load_predecoded_block_gr_tile(
        decoded_q_gmap_words_a_e,
        decoded_gmap_block_count_a,
        (tile_k / STRUCTURED_TILE_K) * out_tiles_a + out_block,
        expert_valid,
        tile_q_gmap_words);
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_a_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_a) ? 1 : 0;
    }
    __syncthreads();

    if (out_a_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_a + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_a_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_a_e,
            qweight2_sparse_a_e,
            qweight2_sparse_prefix_a_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_a,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_a_e,
            g_idx_a_e,
            num_groups_a,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc_a[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  for (int tile_k = 0; tile_k < rank_b; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_b - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden_b[(row_start + row) * rank_b + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_b_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_b_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    load_predecoded_block_gr_tile(
        decoded_q_gmap_words_b_e,
        decoded_gmap_block_count_b,
        (tile_k / STRUCTURED_TILE_K) * out_tiles_b + out_block,
        expert_valid,
        tile_q_gmap_words);
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_b_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_b) ? 1 : 0;
    }
    __syncthreads();

    if (out_b_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_b + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_b_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_b_e,
            qweight2_sparse_b_e,
            qweight2_sparse_prefix_b_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_b,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_b_e,
            g_idx_b_e,
            num_groups_b,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc_b[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  if (out_a_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      output_a[(row_start + row) * outfeatures_a + out_idx] = __float2half(acc_a[row]);
    }
  }

  if (out_b_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      output_b[(row_start + row) * outfeatures_b + out_idx] = __float2half(acc_b[row]);
    }
  }
}

template <bool HasOutputScale, bool HasBias, bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedDecodeBlockGRKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    int q_gmap_gr_words_stride,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    const half* __restrict__ output_scale,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int qweight2_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups) {
  const int out_lane = threadIdx.x;
  const int worker_lane = threadIdx.y;
  const int linear_tid = worker_lane * blockDim.x + out_lane;
  const int threads_per_block = blockDim.x * blockDim.y;
  const int out_idx = blockIdx.x * blockDim.x + out_lane;
  const int out_block = static_cast<int>(blockIdx.x);
  const int out_block_offset = out_lane;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_valid = out_idx < outfeatures;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_q_gmap_raw;
  __shared__ int tile_q_gmap_word_begin;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_idx_shared;
  __shared__ float partial[COMPACT3_SINGLE_BLOCK_GR_K_WORKERS][BLOCK_GR_TILE_OUT];

  if (linear_tid == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_slot_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc = 0.0f;

  const int packed_rows = (rank + 31) / 32;
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  const int matrix_stride = packed_rows * outfeatures;
  const int matrix_stride_qweight2_sparse = qweight2_sparse_rows * outfeatures;
  const int col_pack_stride = rank;
  const int row_pack_stride = outfeatures * num_groups;
  const int out_stride = outfeatures;

  const int* qweight1_e = expert_valid ? (qweight1 + expert_slot * matrix_stride) : nullptr;
  const int* qweight2_sparse_e = expert_valid ? (qweight2_sparse + expert_slot * matrix_stride_qweight2_sparse) : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? (qweight2_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* q_gmap_gr_words_e = expert_valid ? (q_gmap_gr_words + expert_slot * q_gmap_gr_words_stride) : nullptr;
  const int* q_gmap_gr_offsets_e = expert_valid ? (q_gmap_gr_offsets + expert_slot * (gmap_block_count + 1)) : nullptr;
  const int* q_gmap_gr_meta_e = expert_valid ? (q_gmap_gr_meta + expert_slot * gmap_block_count) : nullptr;
  const int* c_map_e = expert_valid ? (c_map + expert_slot * packed_rows) : nullptr;
  const int* g_idx_e = expert_valid ? (g_idx + expert_slot * rank) : nullptr;
  const int64_t* col_pack3_e = expert_valid ? (col_pack3 + expert_slot * col_pack_stride) : nullptr;
  const int64_t* row_pack3_e = expert_valid ? (row_pack3 + expert_slot * row_pack_stride) : nullptr;
  const half* output_scale_e = expert_valid ? (output_scale + expert_slot * out_stride) : nullptr;
  const half* bias_e = (expert_valid && HasBias) ? (bias + expert_slot * out_stride) : nullptr;

  for (int tile_k = 0; tile_k < rank; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = linear_tid; idx < tile_k_size; idx += threads_per_block) {
      hidden_tile[idx] = hidden[row_idx * rank + tile_k + idx];
    }
    for (int idx = linear_tid; idx < tile_k_size; idx += threads_per_block) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_e[global_k]) : 0ULL;
    }
    if (linear_tid < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = linear_tid;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (linear_tid == 0) {
      tile_q_gmap_raw = 0;
      tile_q_gmap_word_begin = 0;
      const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
      const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
      const int k_block = tile_k / STRUCTURED_TILE_K;
      if (expert_valid && k_block >= 0 && k_block < num_k_blocks && out_block >= 0 && out_block < num_out_blocks) {
        const int block_idx = k_block * num_out_blocks + out_block;
        const int meta_word = q_gmap_gr_meta_e[block_idx];
        if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
          tile_q_gmap_raw = 1;
          tile_q_gmap_word_begin = q_gmap_gr_offsets_e[block_idx];
        } else {
          decode_block_gr_tile(
              q_gmap_gr_words_e,
              q_gmap_gr_offsets_e,
              q_gmap_gr_meta_e,
              packed_rows,
              outfeatures,
              tile_k,
              out_block,
              tile_q_gmap_words);
        }
      } else {
        decode_block_gr_tile(
            q_gmap_gr_words_e,
            q_gmap_gr_offsets_e,
            q_gmap_gr_meta_e,
            packed_rows,
            outfeatures,
            tile_k,
            out_block,
            tile_q_gmap_words);
      }
    }
    __syncthreads();

    if (linear_tid == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    float local_partial = 0.0f;
    if (out_valid && expert_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_e[row_offset])
          : 0ULL;
      for (int local_k = 2 * worker_lane; local_k < tile_k_size; local_k += 2 * COMPACT3_SINGLE_BLOCK_GR_K_WORKERS) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_raw != 0
            ? static_cast<unsigned>(q_gmap_gr_words_e[tile_q_gmap_word_begin + (local_k >> 5) * BLOCK_GR_TILE_OUT + out_block_offset])
            : tile_q_gmap_words[local_k >> 5][out_block_offset];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_e,
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_e,
            g_idx_e,
            num_groups,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        local_partial += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    partial[worker_lane][out_lane] = local_partial;
    __syncthreads();

    if (worker_lane == 0 && out_valid) {
      float tile_sum = 0.0f;
#pragma unroll
      for (int worker = 0; worker < COMPACT3_SINGLE_BLOCK_GR_K_WORKERS; ++worker) {
        tile_sum += partial[worker][out_lane];
      }
      acc += tile_sum;
    }
    __syncthreads();
  }

  if (worker_lane == 0 && out_valid) {
    float out_value = acc;
    if (expert_valid) {
      if constexpr (HasOutputScale) {
        out_value *= __half2float(output_scale_e[out_idx]);
      }
      if constexpr (HasBias) {
        out_value += __half2float(bias_e[out_idx]);
      }
    }
    output[row_idx * outfeatures + out_idx] = __float2half(out_value);
  }
}

__global__ void Compact3TriSMoEGroupedSplitKFinalizeHalfKernel(
    const float* __restrict__ input,
    half* __restrict__ output,
    int numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  output[idx] = __float2half(input[idx]);
}

__global__ void Compact3TriSMoEGroupedSplitKEpilogueKernel(
    const float* __restrict__ output_accum,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const half* __restrict__ output_scale,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch,
    int outfeatures,
    int num_segments,
    bool has_bias) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (row_idx >= batch || out_idx >= outfeatures) {
    return;
  }

  const int segment_idx = find_grouped_row_segment(segment_offsets, num_segments, row_idx);
  const int expert_slot = expert_ids[segment_idx];
  const int offset = row_idx * outfeatures + out_idx;

  float out_value = output_accum[offset];
  out_value *= __half2float(output_scale[expert_slot * outfeatures + out_idx]);
  if (has_bias) {
    out_value += __half2float(bias[expert_slot * outfeatures + out_idx]);
  }
  output[offset] = __float2half(out_value);
}

template <bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedSegmentSplitKBlockGRKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    int q_gmap_gr_words_stride,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    float* __restrict__ output_accum,
    int qweight2_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool out_valid = out_idx < outfeatures;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    const int row_start = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    expert_slot_shared = expert_ids[segment_idx];
    row_start_shared = row_start;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start);
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int packed_rows = (rank + 31) / 32;
  const int out_block = static_cast<int>(blockIdx.x);
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  const int matrix_stride = packed_rows * outfeatures;
  const int matrix_stride_qweight2_sparse = qweight2_sparse_rows * outfeatures;
  const int col_pack_stride = rank;
  const int row_pack_stride = outfeatures * num_groups;

  const int* qweight1_e = expert_valid ? (qweight1 + expert_slot * matrix_stride) : nullptr;
  const int* qweight2_sparse_e = expert_valid ? (qweight2_sparse + expert_slot * matrix_stride_qweight2_sparse) : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? (qweight2_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* q_gmap_gr_words_e = expert_valid ? (q_gmap_gr_words + expert_slot * q_gmap_gr_words_stride) : nullptr;
  const int* q_gmap_gr_offsets_e = expert_valid ? (q_gmap_gr_offsets + expert_slot * (gmap_block_count + 1)) : nullptr;
  const int* q_gmap_gr_meta_e = expert_valid ? (q_gmap_gr_meta + expert_slot * gmap_block_count) : nullptr;
  const int* c_map_e = expert_valid ? (c_map + expert_slot * packed_rows) : nullptr;
  const int* g_idx_e = expert_valid ? (g_idx + expert_slot * rank) : nullptr;
  const int64_t* col_pack3_e = expert_valid ? (col_pack3 + expert_slot * col_pack_stride) : nullptr;
  const int64_t* row_pack3_e = expert_valid ? (row_pack3 + expert_slot * row_pack_stride) : nullptr;

  const int num_k_tiles = ceil_div_int(rank, STRUCTURED_TILE_K);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden[(row_start + row) * rank + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_e,
          q_gmap_gr_offsets_e,
          q_gmap_gr_meta_e,
          packed_rows,
          outfeatures,
          tile_k,
          out_block,
          tile_q_gmap_words);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    if (out_valid && expert_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_e,
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_e,
            g_idx_e,
            num_groups,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  if (out_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      atomicAdd(output_accum + (row_start + row) * outfeatures + out_idx, acc[row]);
    }
  }
}

template <bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedDecodeSplitKBlockGRKernel(
    const half* __restrict__ hidden,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1,
    const int* __restrict__ qweight2_sparse,
    const int* __restrict__ qweight2_sparse_prefix,
    const int* __restrict__ q_gmap_gr_words,
    const int* __restrict__ q_gmap_gr_offsets,
    const int* __restrict__ q_gmap_gr_meta,
    int q_gmap_gr_words_stride,
    const int* __restrict__ c_map,
    const int* __restrict__ g_idx,
    const int64_t* __restrict__ col_pack3,
    const int64_t* __restrict__ row_pack3,
    float* __restrict__ output_accum,
    int qweight2_sparse_rows,
    int num_segments,
    int rank,
    int outfeatures,
    int num_groups,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_block = static_cast<int>(blockIdx.x);
  const int out_block_offset = threadIdx.x;
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool out_valid = out_idx < outfeatures;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_q_gmap_raw;
  __shared__ int tile_q_gmap_word_begin;
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_idx_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_slot_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc = 0.0f;

  const int packed_rows = (rank + 31) / 32;
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  const int matrix_stride = packed_rows * outfeatures;
  const int matrix_stride_qweight2_sparse = qweight2_sparse_rows * outfeatures;
  const int col_pack_stride = rank;
  const int row_pack_stride = outfeatures * num_groups;

  const int* qweight1_e = expert_valid ? (qweight1 + expert_slot * matrix_stride) : nullptr;
  const int* qweight2_sparse_e = expert_valid ? (qweight2_sparse + expert_slot * matrix_stride_qweight2_sparse) : nullptr;
  const int* qweight2_sparse_prefix_e = expert_valid ? (qweight2_sparse_prefix + expert_slot * (packed_rows + 1)) : nullptr;
  const int* q_gmap_gr_words_e = expert_valid ? (q_gmap_gr_words + expert_slot * q_gmap_gr_words_stride) : nullptr;
  const int* q_gmap_gr_offsets_e = expert_valid ? (q_gmap_gr_offsets + expert_slot * (gmap_block_count + 1)) : nullptr;
  const int* q_gmap_gr_meta_e = expert_valid ? (q_gmap_gr_meta + expert_slot * gmap_block_count) : nullptr;
  const int* c_map_e = expert_valid ? (c_map + expert_slot * packed_rows) : nullptr;
  const int* g_idx_e = expert_valid ? (g_idx + expert_slot * rank) : nullptr;
  const int64_t* col_pack3_e = expert_valid ? (col_pack3 + expert_slot * col_pack_stride) : nullptr;
  const int64_t* row_pack3_e = expert_valid ? (row_pack3 + expert_slot * row_pack_stride) : nullptr;

  const int num_k_tiles = ceil_div_int(rank, STRUCTURED_TILE_K);
  const int tiles_per_slice = ceil_div_int(num_k_tiles, split_k_slices);
  const int tile_iter_begin = split_k_idx * tiles_per_slice;
  const int tile_iter_end = min(num_k_tiles, tile_iter_begin + tiles_per_slice);

  for (int tile_iter = tile_iter_begin; tile_iter < tile_iter_end; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = hidden[row_idx * rank + tile_k + idx];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      tile_q_gmap_raw = 0;
      tile_q_gmap_word_begin = 0;
      const int num_k_blocks = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS);
      const int num_out_blocks = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
      const int k_block = tile_k / STRUCTURED_TILE_K;
      if (expert_valid && k_block >= 0 && k_block < num_k_blocks && out_block >= 0 && out_block < num_out_blocks) {
        const int block_idx = k_block * num_out_blocks + out_block;
        const int meta_word = q_gmap_gr_meta_e[block_idx];
        if ((meta_word & BLOCK_GR_META_RAW_FLAG) != 0) {
          tile_q_gmap_raw = 1;
          tile_q_gmap_word_begin = q_gmap_gr_offsets_e[block_idx];
        } else {
          decode_block_gr_tile(
              q_gmap_gr_words_e,
              q_gmap_gr_offsets_e,
              q_gmap_gr_meta_e,
              packed_rows,
              outfeatures,
              tile_k,
              out_block,
              tile_q_gmap_words);
        }
      } else {
        decode_block_gr_tile(
            q_gmap_gr_words_e,
            q_gmap_gr_offsets_e,
            q_gmap_gr_meta_e,
            packed_rows,
            outfeatures,
            tile_k,
            out_block,
            tile_q_gmap_words);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups) ? 1 : 0;
    }
    __syncthreads();

    if (out_valid && expert_valid) {
      const int row_offset = out_idx * num_groups + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_raw != 0
            ? static_cast<unsigned>(q_gmap_gr_words_e[tile_q_gmap_word_begin + (local_k >> 5) * BLOCK_GR_TILE_OUT + out_block_offset])
            : tile_q_gmap_words[local_k >> 5][out_block_offset];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_e,
            qweight2_sparse_e,
            qweight2_sparse_prefix_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_e,
            g_idx_e,
            num_groups,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  if (out_valid) {
    atomicAdd(output_accum + row_idx * outfeatures + out_idx, acc);
  }
}

template <bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedDualSegmentBlockGRKernel(
    const half* __restrict__ hidden_a,
    const half* __restrict__ hidden_b,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1_a,
    const int* __restrict__ qweight2_sparse_a,
    const int* __restrict__ qweight2_sparse_prefix_a,
    const int* __restrict__ q_gmap_gr_words_a,
    const int* __restrict__ q_gmap_gr_offsets_a,
    const int* __restrict__ q_gmap_gr_meta_a,
    int q_gmap_gr_words_stride_a,
    const int* __restrict__ c_map_a,
    const int* __restrict__ g_idx_a,
    const int64_t* __restrict__ col_pack3_a,
    const int64_t* __restrict__ row_pack3_a,
    const half* __restrict__ output_scale_a,
    const half* __restrict__ bias_a,
    const int* __restrict__ qweight1_b,
    const int* __restrict__ qweight2_sparse_b,
    const int* __restrict__ qweight2_sparse_prefix_b,
    const int* __restrict__ q_gmap_gr_words_b,
    const int* __restrict__ q_gmap_gr_offsets_b,
    const int* __restrict__ q_gmap_gr_meta_b,
    int q_gmap_gr_words_stride_b,
    const int* __restrict__ c_map_b,
    const int* __restrict__ g_idx_b,
    const int64_t* __restrict__ col_pack3_b,
    const int64_t* __restrict__ row_pack3_b,
    const half* __restrict__ output_scale_b,
    const half* __restrict__ bias_b,
    half* __restrict__ output_a,
    half* __restrict__ output_b,
    int qweight2_sparse_rows_a,
    int qweight2_sparse_rows_b,
    int num_segments,
    int rank_a,
    int rank_b,
    int outfeatures_a,
    int outfeatures_b,
    int num_groups_a,
    int num_groups_b,
    bool has_bias_a,
    bool has_bias_b) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_block = static_cast<int>(blockIdx.x);
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_a_valid = out_idx < outfeatures_a;
  const bool out_b_valid = out_idx < outfeatures_b;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    const int row_start = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    expert_slot_shared = expert_ids[segment_idx];
    row_start_shared = row_start;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start);
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc_a[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc_b[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int packed_rows_a = (rank_a + 31) / 32;
  const int packed_rows_b = (rank_b + 31) / 32;
  const int gmap_block_count_a = ceil_div_int(packed_rows_a, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
  const int gmap_block_count_b = ceil_div_int(packed_rows_b, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
  const int matrix_stride_a = packed_rows_a * outfeatures_a;
  const int matrix_stride_b = packed_rows_b * outfeatures_b;
  const int matrix_stride_qweight2_sparse_a = qweight2_sparse_rows_a * outfeatures_a;
  const int matrix_stride_qweight2_sparse_b = qweight2_sparse_rows_b * outfeatures_b;
  const int col_pack_stride_a = rank_a;
  const int col_pack_stride_b = rank_b;
  const int row_pack_stride_a = outfeatures_a * num_groups_a;
  const int row_pack_stride_b = outfeatures_b * num_groups_b;
  const int out_stride_a = outfeatures_a;
  const int out_stride_b = outfeatures_b;

  const int* qweight1_a_e = expert_valid ? (qweight1_a + expert_slot * matrix_stride_a) : nullptr;
  const int* qweight2_sparse_a_e = expert_valid ? (qweight2_sparse_a + expert_slot * matrix_stride_qweight2_sparse_a) : nullptr;
  const int* qweight2_sparse_prefix_a_e = expert_valid ? (qweight2_sparse_prefix_a + expert_slot * (packed_rows_a + 1)) : nullptr;
  const int* q_gmap_gr_words_a_e = expert_valid ? (q_gmap_gr_words_a + expert_slot * q_gmap_gr_words_stride_a) : nullptr;
  const int* q_gmap_gr_offsets_a_e = expert_valid ? (q_gmap_gr_offsets_a + expert_slot * (gmap_block_count_a + 1)) : nullptr;
  const int* q_gmap_gr_meta_a_e = expert_valid ? (q_gmap_gr_meta_a + expert_slot * gmap_block_count_a) : nullptr;
  const int* c_map_a_e = expert_valid ? (c_map_a + expert_slot * packed_rows_a) : nullptr;
  const int* g_idx_a_e = expert_valid ? (g_idx_a + expert_slot * rank_a) : nullptr;
  const int64_t* col_pack3_a_e = expert_valid ? (col_pack3_a + expert_slot * col_pack_stride_a) : nullptr;
  const int64_t* row_pack3_a_e = expert_valid ? (row_pack3_a + expert_slot * row_pack_stride_a) : nullptr;
  const half* output_scale_a_e = expert_valid ? (output_scale_a + expert_slot * out_stride_a) : nullptr;
  const half* bias_a_e = (expert_valid && has_bias_a) ? (bias_a + expert_slot * out_stride_a) : nullptr;

  const int* qweight1_b_e = expert_valid ? (qweight1_b + expert_slot * matrix_stride_b) : nullptr;
  const int* qweight2_sparse_b_e = expert_valid ? (qweight2_sparse_b + expert_slot * matrix_stride_qweight2_sparse_b) : nullptr;
  const int* qweight2_sparse_prefix_b_e = expert_valid ? (qweight2_sparse_prefix_b + expert_slot * (packed_rows_b + 1)) : nullptr;
  const int* q_gmap_gr_words_b_e = expert_valid ? (q_gmap_gr_words_b + expert_slot * q_gmap_gr_words_stride_b) : nullptr;
  const int* q_gmap_gr_offsets_b_e = expert_valid ? (q_gmap_gr_offsets_b + expert_slot * (gmap_block_count_b + 1)) : nullptr;
  const int* q_gmap_gr_meta_b_e = expert_valid ? (q_gmap_gr_meta_b + expert_slot * gmap_block_count_b) : nullptr;
  const int* c_map_b_e = expert_valid ? (c_map_b + expert_slot * packed_rows_b) : nullptr;
  const int* g_idx_b_e = expert_valid ? (g_idx_b + expert_slot * rank_b) : nullptr;
  const int64_t* col_pack3_b_e = expert_valid ? (col_pack3_b + expert_slot * col_pack_stride_b) : nullptr;
  const int64_t* row_pack3_b_e = expert_valid ? (row_pack3_b + expert_slot * row_pack_stride_b) : nullptr;
  const half* output_scale_b_e = expert_valid ? (output_scale_b + expert_slot * out_stride_b) : nullptr;
  const half* bias_b_e = (expert_valid && has_bias_b) ? (bias_b + expert_slot * out_stride_b) : nullptr;

  for (int tile_k = 0; tile_k < rank_a; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_a - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden_a[(row_start + row) * rank_a + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_a_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_a_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_a_e,
          q_gmap_gr_offsets_a_e,
          q_gmap_gr_meta_a_e,
          packed_rows_a,
          outfeatures_a,
          tile_k,
          out_block,
          tile_q_gmap_words);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_a_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_a) ? 1 : 0;
    }
    __syncthreads();

    if (out_a_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_a + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_a_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_a_e,
            qweight2_sparse_a_e,
            qweight2_sparse_prefix_a_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_a,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_a_e,
            g_idx_a_e,
            num_groups_a,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc_a[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  __syncthreads();
  for (int tile_k = 0; tile_k < rank_b; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_b - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden_b[(row_start + row) * rank_b + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_b_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_b_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_b_e,
          q_gmap_gr_offsets_b_e,
          q_gmap_gr_meta_b_e,
          packed_rows_b,
          outfeatures_b,
          tile_k,
          out_block,
          tile_q_gmap_words);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_b_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_b) ? 1 : 0;
    }
    __syncthreads();

    if (out_b_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_b + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_b_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_b_e,
            qweight2_sparse_b_e,
            qweight2_sparse_prefix_b_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_b,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_b_e,
            g_idx_b_e,
            num_groups_b,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc_b[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  if (out_a_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      float out_value = acc_a[row];
      if (expert_valid) {
        out_value *= __half2float(output_scale_a_e[out_idx]);
        if (has_bias_a) {
          out_value += __half2float(bias_a_e[out_idx]);
        }
      }
      output_a[(row_start + row) * outfeatures_a + out_idx] = __float2half(out_value);
    }
  }

  if (out_b_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      float out_value = acc_b[row];
      if (expert_valid) {
        out_value *= __half2float(output_scale_b_e[out_idx]);
        if (has_bias_b) {
          out_value += __half2float(bias_b_e[out_idx]);
        }
      }
      output_b[(row_start + row) * outfeatures_b + out_idx] = __float2half(out_value);
    }
  }
}

template <bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedDualDecodeBlockGRKernel(
    const half* __restrict__ hidden_a,
    const half* __restrict__ hidden_b,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1_a,
    const int* __restrict__ qweight2_sparse_a,
    const int* __restrict__ qweight2_sparse_prefix_a,
    const int* __restrict__ q_gmap_gr_words_a,
    const int* __restrict__ q_gmap_gr_offsets_a,
    const int* __restrict__ q_gmap_gr_meta_a,
    int q_gmap_gr_words_stride_a,
    const int* __restrict__ c_map_a,
    const int* __restrict__ g_idx_a,
    const int64_t* __restrict__ col_pack3_a,
    const int64_t* __restrict__ row_pack3_a,
    const half* __restrict__ output_scale_a,
    const half* __restrict__ bias_a,
    const int* __restrict__ qweight1_b,
    const int* __restrict__ qweight2_sparse_b,
    const int* __restrict__ qweight2_sparse_prefix_b,
    const int* __restrict__ q_gmap_gr_words_b,
    const int* __restrict__ q_gmap_gr_offsets_b,
    const int* __restrict__ q_gmap_gr_meta_b,
    int q_gmap_gr_words_stride_b,
    const int* __restrict__ c_map_b,
    const int* __restrict__ g_idx_b,
    const int64_t* __restrict__ col_pack3_b,
    const int64_t* __restrict__ row_pack3_b,
    const half* __restrict__ output_scale_b,
    const half* __restrict__ bias_b,
    half* __restrict__ output_a,
    half* __restrict__ output_b,
    int qweight2_sparse_rows_a,
    int qweight2_sparse_rows_b,
    int num_segments,
    int rank_a,
    int rank_b,
    int outfeatures_a,
    int outfeatures_b,
    int num_groups_a,
    int num_groups_b,
    bool has_bias_a,
    bool has_bias_b) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_block = static_cast<int>(blockIdx.x);
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_a_valid = out_idx < outfeatures_a;
  const bool out_b_valid = out_idx < outfeatures_b;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_idx_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_slot_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc_a = 0.0f;
  float acc_b = 0.0f;

  const int packed_rows_a = (rank_a + 31) / 32;
  const int packed_rows_b = (rank_b + 31) / 32;
  const int gmap_block_count_a = ceil_div_int(packed_rows_a, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
  const int gmap_block_count_b = ceil_div_int(packed_rows_b, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
  const int matrix_stride_a = packed_rows_a * outfeatures_a;
  const int matrix_stride_b = packed_rows_b * outfeatures_b;
  const int matrix_stride_qweight2_sparse_a = qweight2_sparse_rows_a * outfeatures_a;
  const int matrix_stride_qweight2_sparse_b = qweight2_sparse_rows_b * outfeatures_b;
  const int col_pack_stride_a = rank_a;
  const int col_pack_stride_b = rank_b;
  const int row_pack_stride_a = outfeatures_a * num_groups_a;
  const int row_pack_stride_b = outfeatures_b * num_groups_b;
  const int out_stride_a = outfeatures_a;
  const int out_stride_b = outfeatures_b;

  const int* qweight1_a_e = expert_valid ? (qweight1_a + expert_slot * matrix_stride_a) : nullptr;
  const int* qweight2_sparse_a_e = expert_valid ? (qweight2_sparse_a + expert_slot * matrix_stride_qweight2_sparse_a) : nullptr;
  const int* qweight2_sparse_prefix_a_e = expert_valid ? (qweight2_sparse_prefix_a + expert_slot * (packed_rows_a + 1)) : nullptr;
  const int* q_gmap_gr_words_a_e = expert_valid ? (q_gmap_gr_words_a + expert_slot * q_gmap_gr_words_stride_a) : nullptr;
  const int* q_gmap_gr_offsets_a_e = expert_valid ? (q_gmap_gr_offsets_a + expert_slot * (gmap_block_count_a + 1)) : nullptr;
  const int* q_gmap_gr_meta_a_e = expert_valid ? (q_gmap_gr_meta_a + expert_slot * gmap_block_count_a) : nullptr;
  const int* c_map_a_e = expert_valid ? (c_map_a + expert_slot * packed_rows_a) : nullptr;
  const int* g_idx_a_e = expert_valid ? (g_idx_a + expert_slot * rank_a) : nullptr;
  const int64_t* col_pack3_a_e = expert_valid ? (col_pack3_a + expert_slot * col_pack_stride_a) : nullptr;
  const int64_t* row_pack3_a_e = expert_valid ? (row_pack3_a + expert_slot * row_pack_stride_a) : nullptr;
  const half* output_scale_a_e = expert_valid ? (output_scale_a + expert_slot * out_stride_a) : nullptr;
  const half* bias_a_e = (expert_valid && has_bias_a) ? (bias_a + expert_slot * out_stride_a) : nullptr;

  const int* qweight1_b_e = expert_valid ? (qweight1_b + expert_slot * matrix_stride_b) : nullptr;
  const int* qweight2_sparse_b_e = expert_valid ? (qweight2_sparse_b + expert_slot * matrix_stride_qweight2_sparse_b) : nullptr;
  const int* qweight2_sparse_prefix_b_e = expert_valid ? (qweight2_sparse_prefix_b + expert_slot * (packed_rows_b + 1)) : nullptr;
  const int* q_gmap_gr_words_b_e = expert_valid ? (q_gmap_gr_words_b + expert_slot * q_gmap_gr_words_stride_b) : nullptr;
  const int* q_gmap_gr_offsets_b_e = expert_valid ? (q_gmap_gr_offsets_b + expert_slot * (gmap_block_count_b + 1)) : nullptr;
  const int* q_gmap_gr_meta_b_e = expert_valid ? (q_gmap_gr_meta_b + expert_slot * gmap_block_count_b) : nullptr;
  const int* c_map_b_e = expert_valid ? (c_map_b + expert_slot * packed_rows_b) : nullptr;
  const int* g_idx_b_e = expert_valid ? (g_idx_b + expert_slot * rank_b) : nullptr;
  const int64_t* col_pack3_b_e = expert_valid ? (col_pack3_b + expert_slot * col_pack_stride_b) : nullptr;
  const int64_t* row_pack3_b_e = expert_valid ? (row_pack3_b + expert_slot * row_pack_stride_b) : nullptr;
  const half* output_scale_b_e = expert_valid ? (output_scale_b + expert_slot * out_stride_b) : nullptr;
  const half* bias_b_e = (expert_valid && has_bias_b) ? (bias_b + expert_slot * out_stride_b) : nullptr;

  for (int tile_k = 0; tile_k < rank_a; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_a - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = hidden_a[row_idx * rank_a + tile_k + idx];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_a_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_a_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_a_e,
          q_gmap_gr_offsets_a_e,
          q_gmap_gr_meta_a_e,
          packed_rows_a,
          outfeatures_a,
          tile_k,
          out_block,
          tile_q_gmap_words);
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_a_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_a) ? 1 : 0;
    }
    __syncthreads();

    if (out_a_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_a + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_a_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_a_e,
            qweight2_sparse_a_e,
            qweight2_sparse_prefix_a_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_a,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_a_e,
            g_idx_a_e,
            num_groups_a,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc_a += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  for (int tile_k = 0; tile_k < rank_b; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_b - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = hidden_b[row_idx * rank_b + tile_k + idx];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_b_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_b_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_b_e,
          q_gmap_gr_offsets_b_e,
          q_gmap_gr_meta_b_e,
          packed_rows_b,
          outfeatures_b,
          tile_k,
          out_block,
          tile_q_gmap_words);
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_b_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_b) ? 1 : 0;
    }
    __syncthreads();

    if (out_b_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_b + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_b_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_b_e,
            qweight2_sparse_b_e,
            qweight2_sparse_prefix_b_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_b,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_b_e,
            g_idx_b_e,
            num_groups_b,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc_b += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  if (out_a_valid) {
    float out_value = acc_a;
    if (expert_valid) {
      out_value *= __half2float(output_scale_a_e[out_idx]);
      if (has_bias_a) {
        out_value += __half2float(bias_a_e[out_idx]);
      }
    }
    output_a[row_idx * outfeatures_a + out_idx] = __float2half(out_value);
  }

  if (out_b_valid) {
    float out_value = acc_b;
    if (expert_valid) {
      out_value *= __half2float(output_scale_b_e[out_idx]);
      if (has_bias_b) {
        out_value += __half2float(bias_b_e[out_idx]);
      }
    }
    output_b[row_idx * outfeatures_b + out_idx] = __float2half(out_value);
  }
}

template <bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedDualSegmentNoScaleNoBiasBlockGRKernel(
    const half* __restrict__ hidden_a,
    const half* __restrict__ hidden_b,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1_a,
    const int* __restrict__ qweight2_sparse_a,
    const int* __restrict__ qweight2_sparse_prefix_a,
    const int* __restrict__ q_gmap_gr_words_a,
    const int* __restrict__ q_gmap_gr_offsets_a,
    const int* __restrict__ q_gmap_gr_meta_a,
    int q_gmap_gr_words_stride_a,
    const int* __restrict__ c_map_a,
    const int* __restrict__ g_idx_a,
    const int64_t* __restrict__ col_pack3_a,
    const int64_t* __restrict__ row_pack3_a,
    const int* __restrict__ qweight1_b,
    const int* __restrict__ qweight2_sparse_b,
    const int* __restrict__ qweight2_sparse_prefix_b,
    const int* __restrict__ q_gmap_gr_words_b,
    const int* __restrict__ q_gmap_gr_offsets_b,
    const int* __restrict__ q_gmap_gr_meta_b,
    int q_gmap_gr_words_stride_b,
    const int* __restrict__ c_map_b,
    const int* __restrict__ g_idx_b,
    const int64_t* __restrict__ col_pack3_b,
    const int64_t* __restrict__ row_pack3_b,
    half* __restrict__ output_a,
    half* __restrict__ output_b,
    int qweight2_sparse_rows_a,
    int qweight2_sparse_rows_b,
    int num_segments,
    int rank_a,
    int rank_b,
    int outfeatures_a,
    int outfeatures_b,
    int num_groups_a,
    int num_groups_b) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_block = static_cast<int>(blockIdx.x);
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_a_valid = out_idx < outfeatures_a;
  const bool out_b_valid = out_idx < outfeatures_b;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    const int row_start = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    expert_slot_shared = expert_ids[segment_idx];
    row_start_shared = row_start;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start);
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc_a[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc_b[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int packed_rows_a = (rank_a + 31) / 32;
  const int packed_rows_b = (rank_b + 31) / 32;
  const int gmap_block_count_a = ceil_div_int(packed_rows_a, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
  const int gmap_block_count_b = ceil_div_int(packed_rows_b, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
  const int matrix_stride_a = packed_rows_a * outfeatures_a;
  const int matrix_stride_b = packed_rows_b * outfeatures_b;
  const int matrix_stride_qweight2_sparse_a = qweight2_sparse_rows_a * outfeatures_a;
  const int matrix_stride_qweight2_sparse_b = qweight2_sparse_rows_b * outfeatures_b;
  const int col_pack_stride_a = rank_a;
  const int col_pack_stride_b = rank_b;
  const int row_pack_stride_a = outfeatures_a * num_groups_a;
  const int row_pack_stride_b = outfeatures_b * num_groups_b;

  const int* qweight1_a_e = expert_valid ? (qweight1_a + expert_slot * matrix_stride_a) : nullptr;
  const int* qweight2_sparse_a_e = expert_valid ? (qweight2_sparse_a + expert_slot * matrix_stride_qweight2_sparse_a) : nullptr;
  const int* qweight2_sparse_prefix_a_e = expert_valid ? (qweight2_sparse_prefix_a + expert_slot * (packed_rows_a + 1)) : nullptr;
  const int* q_gmap_gr_words_a_e = expert_valid ? (q_gmap_gr_words_a + expert_slot * q_gmap_gr_words_stride_a) : nullptr;
  const int* q_gmap_gr_offsets_a_e = expert_valid ? (q_gmap_gr_offsets_a + expert_slot * (gmap_block_count_a + 1)) : nullptr;
  const int* q_gmap_gr_meta_a_e = expert_valid ? (q_gmap_gr_meta_a + expert_slot * gmap_block_count_a) : nullptr;
  const int* c_map_a_e = expert_valid ? (c_map_a + expert_slot * packed_rows_a) : nullptr;
  const int* g_idx_a_e = expert_valid ? (g_idx_a + expert_slot * rank_a) : nullptr;
  const int64_t* col_pack3_a_e = expert_valid ? (col_pack3_a + expert_slot * col_pack_stride_a) : nullptr;
  const int64_t* row_pack3_a_e = expert_valid ? (row_pack3_a + expert_slot * row_pack_stride_a) : nullptr;

  const int* qweight1_b_e = expert_valid ? (qweight1_b + expert_slot * matrix_stride_b) : nullptr;
  const int* qweight2_sparse_b_e = expert_valid ? (qweight2_sparse_b + expert_slot * matrix_stride_qweight2_sparse_b) : nullptr;
  const int* qweight2_sparse_prefix_b_e = expert_valid ? (qweight2_sparse_prefix_b + expert_slot * (packed_rows_b + 1)) : nullptr;
  const int* q_gmap_gr_words_b_e = expert_valid ? (q_gmap_gr_words_b + expert_slot * q_gmap_gr_words_stride_b) : nullptr;
  const int* q_gmap_gr_offsets_b_e = expert_valid ? (q_gmap_gr_offsets_b + expert_slot * (gmap_block_count_b + 1)) : nullptr;
  const int* q_gmap_gr_meta_b_e = expert_valid ? (q_gmap_gr_meta_b + expert_slot * gmap_block_count_b) : nullptr;
  const int* c_map_b_e = expert_valid ? (c_map_b + expert_slot * packed_rows_b) : nullptr;
  const int* g_idx_b_e = expert_valid ? (g_idx_b + expert_slot * rank_b) : nullptr;
  const int64_t* col_pack3_b_e = expert_valid ? (col_pack3_b + expert_slot * col_pack_stride_b) : nullptr;
  const int64_t* row_pack3_b_e = expert_valid ? (row_pack3_b + expert_slot * row_pack_stride_b) : nullptr;

  for (int tile_k = 0; tile_k < rank_a; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_a - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden_a[(row_start + row) * rank_a + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_a_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_a_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_a_e,
          q_gmap_gr_offsets_a_e,
          q_gmap_gr_meta_a_e,
          packed_rows_a,
          outfeatures_a,
          tile_k,
          out_block,
          tile_q_gmap_words);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_a_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_a) ? 1 : 0;
    }
    __syncthreads();

    if (out_a_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_a + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_a_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_a_e,
            qweight2_sparse_a_e,
            qweight2_sparse_prefix_a_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_a,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_a_e,
            g_idx_a_e,
            num_groups_a,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc_a[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  __syncthreads();
  for (int tile_k = 0; tile_k < rank_b; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_b - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden_b[(row_start + row) * rank_b + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_b_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_b_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_b_e,
          q_gmap_gr_offsets_b_e,
          q_gmap_gr_meta_b_e,
          packed_rows_b,
          outfeatures_b,
          tile_k,
          out_block,
          tile_q_gmap_words);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_b_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_b) ? 1 : 0;
    }
    __syncthreads();

    if (out_b_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_b + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_b_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_b_e,
            qweight2_sparse_b_e,
            qweight2_sparse_prefix_b_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_b,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_b_e,
            g_idx_b_e,
            num_groups_b,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc_b[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  if (out_a_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      output_a[(row_start + row) * outfeatures_a + out_idx] = __float2half(acc_a[row]);
    }
  }

  if (out_b_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      output_b[(row_start + row) * outfeatures_b + out_idx] = __float2half(acc_b[row]);
    }
  }
}

template <bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedDualDecodeNoScaleNoBiasBlockGRKernel(
    const half* __restrict__ hidden_a,
    const half* __restrict__ hidden_b,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1_a,
    const int* __restrict__ qweight2_sparse_a,
    const int* __restrict__ qweight2_sparse_prefix_a,
    const int* __restrict__ q_gmap_gr_words_a,
    const int* __restrict__ q_gmap_gr_offsets_a,
    const int* __restrict__ q_gmap_gr_meta_a,
    int q_gmap_gr_words_stride_a,
    const int* __restrict__ c_map_a,
    const int* __restrict__ g_idx_a,
    const int64_t* __restrict__ col_pack3_a,
    const int64_t* __restrict__ row_pack3_a,
    const int* __restrict__ qweight1_b,
    const int* __restrict__ qweight2_sparse_b,
    const int* __restrict__ qweight2_sparse_prefix_b,
    const int* __restrict__ q_gmap_gr_words_b,
    const int* __restrict__ q_gmap_gr_offsets_b,
    const int* __restrict__ q_gmap_gr_meta_b,
    int q_gmap_gr_words_stride_b,
    const int* __restrict__ c_map_b,
    const int* __restrict__ g_idx_b,
    const int64_t* __restrict__ col_pack3_b,
    const int64_t* __restrict__ row_pack3_b,
    half* __restrict__ output_a,
    half* __restrict__ output_b,
    int qweight2_sparse_rows_a,
    int qweight2_sparse_rows_b,
    int num_segments,
    int rank_a,
    int rank_b,
    int outfeatures_a,
    int outfeatures_b,
    int num_groups_a,
    int num_groups_b) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_block = static_cast<int>(blockIdx.x);
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const bool out_a_valid = out_idx < outfeatures_a;
  const bool out_b_valid = out_idx < outfeatures_b;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_idx_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_slot_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc_a = 0.0f;
  float acc_b = 0.0f;

  const int packed_rows_a = (rank_a + 31) / 32;
  const int packed_rows_b = (rank_b + 31) / 32;
  const int gmap_block_count_a = ceil_div_int(packed_rows_a, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
  const int gmap_block_count_b = ceil_div_int(packed_rows_b, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
  const int matrix_stride_a = packed_rows_a * outfeatures_a;
  const int matrix_stride_b = packed_rows_b * outfeatures_b;
  const int matrix_stride_qweight2_sparse_a = qweight2_sparse_rows_a * outfeatures_a;
  const int matrix_stride_qweight2_sparse_b = qweight2_sparse_rows_b * outfeatures_b;
  const int col_pack_stride_a = rank_a;
  const int col_pack_stride_b = rank_b;
  const int row_pack_stride_a = outfeatures_a * num_groups_a;
  const int row_pack_stride_b = outfeatures_b * num_groups_b;

  const int* qweight1_a_e = expert_valid ? (qweight1_a + expert_slot * matrix_stride_a) : nullptr;
  const int* qweight2_sparse_a_e = expert_valid ? (qweight2_sparse_a + expert_slot * matrix_stride_qweight2_sparse_a) : nullptr;
  const int* qweight2_sparse_prefix_a_e = expert_valid ? (qweight2_sparse_prefix_a + expert_slot * (packed_rows_a + 1)) : nullptr;
  const int* q_gmap_gr_words_a_e = expert_valid ? (q_gmap_gr_words_a + expert_slot * q_gmap_gr_words_stride_a) : nullptr;
  const int* q_gmap_gr_offsets_a_e = expert_valid ? (q_gmap_gr_offsets_a + expert_slot * (gmap_block_count_a + 1)) : nullptr;
  const int* q_gmap_gr_meta_a_e = expert_valid ? (q_gmap_gr_meta_a + expert_slot * gmap_block_count_a) : nullptr;
  const int* c_map_a_e = expert_valid ? (c_map_a + expert_slot * packed_rows_a) : nullptr;
  const int* g_idx_a_e = expert_valid ? (g_idx_a + expert_slot * rank_a) : nullptr;
  const int64_t* col_pack3_a_e = expert_valid ? (col_pack3_a + expert_slot * col_pack_stride_a) : nullptr;
  const int64_t* row_pack3_a_e = expert_valid ? (row_pack3_a + expert_slot * row_pack_stride_a) : nullptr;

  const int* qweight1_b_e = expert_valid ? (qweight1_b + expert_slot * matrix_stride_b) : nullptr;
  const int* qweight2_sparse_b_e = expert_valid ? (qweight2_sparse_b + expert_slot * matrix_stride_qweight2_sparse_b) : nullptr;
  const int* qweight2_sparse_prefix_b_e = expert_valid ? (qweight2_sparse_prefix_b + expert_slot * (packed_rows_b + 1)) : nullptr;
  const int* q_gmap_gr_words_b_e = expert_valid ? (q_gmap_gr_words_b + expert_slot * q_gmap_gr_words_stride_b) : nullptr;
  const int* q_gmap_gr_offsets_b_e = expert_valid ? (q_gmap_gr_offsets_b + expert_slot * (gmap_block_count_b + 1)) : nullptr;
  const int* q_gmap_gr_meta_b_e = expert_valid ? (q_gmap_gr_meta_b + expert_slot * gmap_block_count_b) : nullptr;
  const int* c_map_b_e = expert_valid ? (c_map_b + expert_slot * packed_rows_b) : nullptr;
  const int* g_idx_b_e = expert_valid ? (g_idx_b + expert_slot * rank_b) : nullptr;
  const int64_t* col_pack3_b_e = expert_valid ? (col_pack3_b + expert_slot * col_pack_stride_b) : nullptr;
  const int64_t* row_pack3_b_e = expert_valid ? (row_pack3_b + expert_slot * row_pack_stride_b) : nullptr;

  for (int tile_k = 0; tile_k < rank_a; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_a - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = hidden_a[row_idx * rank_a + tile_k + idx];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_a_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_a_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_a_e,
          q_gmap_gr_offsets_a_e,
          q_gmap_gr_meta_a_e,
          packed_rows_a,
          outfeatures_a,
          tile_k,
          out_block,
          tile_q_gmap_words);
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_a_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_a) ? 1 : 0;
    }
    __syncthreads();

    if (out_a_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_a + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_a_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_a_e,
            qweight2_sparse_a_e,
            qweight2_sparse_prefix_a_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_a,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_a_e,
            g_idx_a_e,
            num_groups_a,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc_a += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  for (int tile_k = 0; tile_k < rank_b; tile_k += STRUCTURED_TILE_K) {
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_b - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = hidden_b[row_idx * rank_b + tile_k + idx];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_b_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_b_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_b_e,
          q_gmap_gr_offsets_b_e,
          q_gmap_gr_meta_b_e,
          packed_rows_b,
          outfeatures_b,
          tile_k,
          out_block,
          tile_q_gmap_words);
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_b_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_b) ? 1 : 0;
    }
    __syncthreads();

    if (out_b_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_b + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_b_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_b_e,
            qweight2_sparse_b_e,
            qweight2_sparse_prefix_b_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_b,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_b_e,
            g_idx_b_e,
            num_groups_b,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc_b += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  if (out_a_valid) {
    output_a[row_idx * outfeatures_a + out_idx] = __float2half(acc_a);
  }

  if (out_b_valid) {
    output_b[row_idx * outfeatures_b + out_idx] = __float2half(acc_b);
  }
}

template <bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedDualSegmentSplitKBlockGRKernel(
    const half* __restrict__ hidden_a,
    const half* __restrict__ hidden_b,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1_a,
    const int* __restrict__ qweight2_sparse_a,
    const int* __restrict__ qweight2_sparse_prefix_a,
    const int* __restrict__ q_gmap_gr_words_a,
    const int* __restrict__ q_gmap_gr_offsets_a,
    const int* __restrict__ q_gmap_gr_meta_a,
    int q_gmap_gr_words_stride_a,
    const int* __restrict__ c_map_a,
    const int* __restrict__ g_idx_a,
    const int64_t* __restrict__ col_pack3_a,
    const int64_t* __restrict__ row_pack3_a,
    const int* __restrict__ qweight1_b,
    const int* __restrict__ qweight2_sparse_b,
    const int* __restrict__ qweight2_sparse_prefix_b,
    const int* __restrict__ q_gmap_gr_words_b,
    const int* __restrict__ q_gmap_gr_offsets_b,
    const int* __restrict__ q_gmap_gr_meta_b,
    int q_gmap_gr_words_stride_b,
    const int* __restrict__ c_map_b,
    const int* __restrict__ g_idx_b,
    const int64_t* __restrict__ col_pack3_b,
    const int64_t* __restrict__ row_pack3_b,
    float* __restrict__ output_accum_a,
    float* __restrict__ output_accum_b,
    int qweight2_sparse_rows_a,
    int qweight2_sparse_rows_b,
    int num_segments,
    int rank_a,
    int rank_b,
    int outfeatures_a,
    int outfeatures_b,
    int num_groups_a,
    int num_groups_b,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_block = static_cast<int>(blockIdx.x);
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool out_a_valid = out_idx < outfeatures_a;
  const bool out_b_valid = out_idx < outfeatures_b;

  __shared__ half hidden_tile[STRUCTURED_GROUPED_M_TILE][STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_start_shared;
  __shared__ int row_count_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    const int segment_problem_base = problem_offsets[segment_idx];
    const int segment_row_begin = segment_offsets[segment_idx];
    const int segment_row_end = segment_offsets[segment_idx + 1];
    const int local_tile_idx = problem_tile_idx - segment_problem_base;
    const int row_start = segment_row_begin + local_tile_idx * STRUCTURED_GROUPED_M_TILE;
    expert_slot_shared = expert_ids[segment_idx];
    row_start_shared = row_start;
    row_count_shared = min(STRUCTURED_GROUPED_M_TILE, segment_row_end - row_start);
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_start = row_start_shared;
  const int row_count = row_count_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc_a[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc_b[STRUCTURED_GROUPED_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int packed_rows_a = (rank_a + 31) / 32;
  const int packed_rows_b = (rank_b + 31) / 32;
  const int gmap_block_count_a = ceil_div_int(packed_rows_a, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
  const int gmap_block_count_b = ceil_div_int(packed_rows_b, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
  const int matrix_stride_a = packed_rows_a * outfeatures_a;
  const int matrix_stride_b = packed_rows_b * outfeatures_b;
  const int matrix_stride_qweight2_sparse_a = qweight2_sparse_rows_a * outfeatures_a;
  const int matrix_stride_qweight2_sparse_b = qweight2_sparse_rows_b * outfeatures_b;
  const int col_pack_stride_a = rank_a;
  const int col_pack_stride_b = rank_b;
  const int row_pack_stride_a = outfeatures_a * num_groups_a;
  const int row_pack_stride_b = outfeatures_b * num_groups_b;

  const int* qweight1_a_e = expert_valid ? (qweight1_a + expert_slot * matrix_stride_a) : nullptr;
  const int* qweight2_sparse_a_e = expert_valid ? (qweight2_sparse_a + expert_slot * matrix_stride_qweight2_sparse_a) : nullptr;
  const int* qweight2_sparse_prefix_a_e = expert_valid ? (qweight2_sparse_prefix_a + expert_slot * (packed_rows_a + 1)) : nullptr;
  const int* q_gmap_gr_words_a_e = expert_valid ? (q_gmap_gr_words_a + expert_slot * q_gmap_gr_words_stride_a) : nullptr;
  const int* q_gmap_gr_offsets_a_e = expert_valid ? (q_gmap_gr_offsets_a + expert_slot * (gmap_block_count_a + 1)) : nullptr;
  const int* q_gmap_gr_meta_a_e = expert_valid ? (q_gmap_gr_meta_a + expert_slot * gmap_block_count_a) : nullptr;
  const int* c_map_a_e = expert_valid ? (c_map_a + expert_slot * packed_rows_a) : nullptr;
  const int* g_idx_a_e = expert_valid ? (g_idx_a + expert_slot * rank_a) : nullptr;
  const int64_t* col_pack3_a_e = expert_valid ? (col_pack3_a + expert_slot * col_pack_stride_a) : nullptr;
  const int64_t* row_pack3_a_e = expert_valid ? (row_pack3_a + expert_slot * row_pack_stride_a) : nullptr;

  const int* qweight1_b_e = expert_valid ? (qweight1_b + expert_slot * matrix_stride_b) : nullptr;
  const int* qweight2_sparse_b_e = expert_valid ? (qweight2_sparse_b + expert_slot * matrix_stride_qweight2_sparse_b) : nullptr;
  const int* qweight2_sparse_prefix_b_e = expert_valid ? (qweight2_sparse_prefix_b + expert_slot * (packed_rows_b + 1)) : nullptr;
  const int* q_gmap_gr_words_b_e = expert_valid ? (q_gmap_gr_words_b + expert_slot * q_gmap_gr_words_stride_b) : nullptr;
  const int* q_gmap_gr_offsets_b_e = expert_valid ? (q_gmap_gr_offsets_b + expert_slot * (gmap_block_count_b + 1)) : nullptr;
  const int* q_gmap_gr_meta_b_e = expert_valid ? (q_gmap_gr_meta_b + expert_slot * gmap_block_count_b) : nullptr;
  const int* c_map_b_e = expert_valid ? (c_map_b + expert_slot * packed_rows_b) : nullptr;
  const int* g_idx_b_e = expert_valid ? (g_idx_b + expert_slot * rank_b) : nullptr;
  const int64_t* col_pack3_b_e = expert_valid ? (col_pack3_b + expert_slot * col_pack_stride_b) : nullptr;
  const int64_t* row_pack3_b_e = expert_valid ? (row_pack3_b + expert_slot * row_pack_stride_b) : nullptr;

  const int num_k_tiles_a = ceil_div_int(rank_a, STRUCTURED_TILE_K);
  const int tiles_per_slice_a = ceil_div_int(num_k_tiles_a, split_k_slices);
  const int tile_iter_begin_a = split_k_idx * tiles_per_slice_a;
  const int tile_iter_end_a = min(num_k_tiles_a, tile_iter_begin_a + tiles_per_slice_a);

  for (int tile_iter = tile_iter_begin_a; tile_iter < tile_iter_end_a; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_a - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden_a[(row_start + row) * rank_a + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_a_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_a_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_a_e,
          q_gmap_gr_offsets_a_e,
          q_gmap_gr_meta_a_e,
          packed_rows_a,
          outfeatures_a,
          tile_k,
          out_block,
          tile_q_gmap_words);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_a_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_a) ? 1 : 0;
    }
    __syncthreads();

    if (out_a_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_a + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_a_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_a_e,
            qweight2_sparse_a_e,
            qweight2_sparse_prefix_a_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_a,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_a_e,
            g_idx_a_e,
            num_groups_a,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc_a[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  __syncthreads();
  const int num_k_tiles_b = ceil_div_int(rank_b, STRUCTURED_TILE_K);
  const int tiles_per_slice_b = ceil_div_int(num_k_tiles_b, split_k_slices);
  const int tile_iter_begin_b = split_k_idx * tiles_per_slice_b;
  const int tile_iter_end_b = min(num_k_tiles_b, tile_iter_begin_b + tiles_per_slice_b);

  for (int tile_iter = tile_iter_begin_b; tile_iter < tile_iter_end_b; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_b - tile_k);
    const int load_items = row_count * tile_k_size;
    for (int idx = threadIdx.x; idx < load_items; idx += blockDim.x) {
      const int row = idx / tile_k_size;
      const int local_k = idx - row * tile_k_size;
      hidden_tile[row][local_k] = hidden_b[(row_start + row) * rank_b + tile_k + local_k];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_b_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_b_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_b_e,
          q_gmap_gr_offsets_b_e,
          q_gmap_gr_meta_b_e,
          packed_rows_b,
          outfeatures_b,
          tile_k,
          out_block,
          tile_q_gmap_words);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_b_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_b) ? 1 : 0;
    }
    __syncthreads();

    if (out_b_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_b + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_b_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_b_e,
            qweight2_sparse_b_e,
            qweight2_sparse_prefix_b_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_b,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_b_e,
            g_idx_b_e,
            num_groups_b,
            tile_group_value);
#pragma unroll
        for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
          if (row >= row_count) {
            break;
          }
          const half x0 = hidden_tile[row][local_k];
          const half x1 = has_k1 ? hidden_tile[row][local_k + 1] : __float2half(0.0f);
          acc_b[row] += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
        }
      }
    }
    __syncthreads();
  }

  if (out_a_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      atomicAdd(output_accum_a + (row_start + row) * outfeatures_a + out_idx, acc_a[row]);
    }
  }

  if (out_b_valid) {
#pragma unroll
    for (int row = 0; row < STRUCTURED_GROUPED_M_TILE; ++row) {
      if (row >= row_count) {
        break;
      }
      atomicAdd(output_accum_b + (row_start + row) * outfeatures_b + out_idx, acc_b[row]);
    }
  }
}

template <bool UniformK64Groups>
__global__ void Compact3TriSMoEPrecomputedGroupedDualDecodeSplitKBlockGRKernel(
    const half* __restrict__ hidden_a,
    const half* __restrict__ hidden_b,
    const int* __restrict__ expert_ids,
    const int* __restrict__ segment_offsets,
    const int* __restrict__ problem_offsets,
    const int* __restrict__ qweight1_a,
    const int* __restrict__ qweight2_sparse_a,
    const int* __restrict__ qweight2_sparse_prefix_a,
    const int* __restrict__ q_gmap_gr_words_a,
    const int* __restrict__ q_gmap_gr_offsets_a,
    const int* __restrict__ q_gmap_gr_meta_a,
    int q_gmap_gr_words_stride_a,
    const int* __restrict__ c_map_a,
    const int* __restrict__ g_idx_a,
    const int64_t* __restrict__ col_pack3_a,
    const int64_t* __restrict__ row_pack3_a,
    const int* __restrict__ qweight1_b,
    const int* __restrict__ qweight2_sparse_b,
    const int* __restrict__ qweight2_sparse_prefix_b,
    const int* __restrict__ q_gmap_gr_words_b,
    const int* __restrict__ q_gmap_gr_offsets_b,
    const int* __restrict__ q_gmap_gr_meta_b,
    int q_gmap_gr_words_stride_b,
    const int* __restrict__ c_map_b,
    const int* __restrict__ g_idx_b,
    const int64_t* __restrict__ col_pack3_b,
    const int64_t* __restrict__ row_pack3_b,
    float* __restrict__ output_accum_a,
    float* __restrict__ output_accum_b,
    int qweight2_sparse_rows_a,
    int qweight2_sparse_rows_b,
    int num_segments,
    int rank_a,
    int rank_b,
    int outfeatures_a,
    int outfeatures_b,
    int num_groups_a,
    int num_groups_b,
    int split_k_slices) {
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_block = static_cast<int>(blockIdx.x);
  const int problem_tile_idx = static_cast<int>(blockIdx.y);
  const int split_k_idx = static_cast<int>(blockIdx.z);
  const bool out_a_valid = out_idx < outfeatures_a;
  const bool out_b_valid = out_idx < outfeatures_b;

  __shared__ half hidden_tile[STRUCTURED_TILE_K];
  __shared__ unsigned tile_c_map_words[(STRUCTURED_TILE_K + 31) / 32];
  __shared__ unsigned long long tile_col_pack3[STRUCTURED_TILE_K];
  __shared__ unsigned tile_q_gmap_words[BLOCK_GR_PACKED_ROWS][BLOCK_GR_TILE_OUT];
  __shared__ int tile_group_value;
  __shared__ int tile_group_valid;
  __shared__ int expert_slot_shared;
  __shared__ int row_idx_shared;

  if (threadIdx.x == 0) {
    const int segment_idx = find_grouped_problem_segment(problem_offsets, num_segments, problem_tile_idx);
    expert_slot_shared = expert_ids[segment_idx];
    row_idx_shared = segment_offsets[segment_idx];
  }
  __syncthreads();

  const int expert_slot = expert_slot_shared;
  const int row_idx = row_idx_shared;
  const bool expert_valid = expert_slot >= 0;

  float acc_a = 0.0f;
  float acc_b = 0.0f;

  const int packed_rows_a = (rank_a + 31) / 32;
  const int packed_rows_b = (rank_b + 31) / 32;
  const int gmap_block_count_a = ceil_div_int(packed_rows_a, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
  const int gmap_block_count_b = ceil_div_int(packed_rows_b, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
  const int matrix_stride_a = packed_rows_a * outfeatures_a;
  const int matrix_stride_b = packed_rows_b * outfeatures_b;
  const int matrix_stride_qweight2_sparse_a = qweight2_sparse_rows_a * outfeatures_a;
  const int matrix_stride_qweight2_sparse_b = qweight2_sparse_rows_b * outfeatures_b;
  const int col_pack_stride_a = rank_a;
  const int col_pack_stride_b = rank_b;
  const int row_pack_stride_a = outfeatures_a * num_groups_a;
  const int row_pack_stride_b = outfeatures_b * num_groups_b;

  const int* qweight1_a_e = expert_valid ? (qweight1_a + expert_slot * matrix_stride_a) : nullptr;
  const int* qweight2_sparse_a_e = expert_valid ? (qweight2_sparse_a + expert_slot * matrix_stride_qweight2_sparse_a) : nullptr;
  const int* qweight2_sparse_prefix_a_e = expert_valid ? (qweight2_sparse_prefix_a + expert_slot * (packed_rows_a + 1)) : nullptr;
  const int* q_gmap_gr_words_a_e = expert_valid ? (q_gmap_gr_words_a + expert_slot * q_gmap_gr_words_stride_a) : nullptr;
  const int* q_gmap_gr_offsets_a_e = expert_valid ? (q_gmap_gr_offsets_a + expert_slot * (gmap_block_count_a + 1)) : nullptr;
  const int* q_gmap_gr_meta_a_e = expert_valid ? (q_gmap_gr_meta_a + expert_slot * gmap_block_count_a) : nullptr;
  const int* c_map_a_e = expert_valid ? (c_map_a + expert_slot * packed_rows_a) : nullptr;
  const int* g_idx_a_e = expert_valid ? (g_idx_a + expert_slot * rank_a) : nullptr;
  const int64_t* col_pack3_a_e = expert_valid ? (col_pack3_a + expert_slot * col_pack_stride_a) : nullptr;
  const int64_t* row_pack3_a_e = expert_valid ? (row_pack3_a + expert_slot * row_pack_stride_a) : nullptr;

  const int* qweight1_b_e = expert_valid ? (qweight1_b + expert_slot * matrix_stride_b) : nullptr;
  const int* qweight2_sparse_b_e = expert_valid ? (qweight2_sparse_b + expert_slot * matrix_stride_qweight2_sparse_b) : nullptr;
  const int* qweight2_sparse_prefix_b_e = expert_valid ? (qweight2_sparse_prefix_b + expert_slot * (packed_rows_b + 1)) : nullptr;
  const int* q_gmap_gr_words_b_e = expert_valid ? (q_gmap_gr_words_b + expert_slot * q_gmap_gr_words_stride_b) : nullptr;
  const int* q_gmap_gr_offsets_b_e = expert_valid ? (q_gmap_gr_offsets_b + expert_slot * (gmap_block_count_b + 1)) : nullptr;
  const int* q_gmap_gr_meta_b_e = expert_valid ? (q_gmap_gr_meta_b + expert_slot * gmap_block_count_b) : nullptr;
  const int* c_map_b_e = expert_valid ? (c_map_b + expert_slot * packed_rows_b) : nullptr;
  const int* g_idx_b_e = expert_valid ? (g_idx_b + expert_slot * rank_b) : nullptr;
  const int64_t* col_pack3_b_e = expert_valid ? (col_pack3_b + expert_slot * col_pack_stride_b) : nullptr;
  const int64_t* row_pack3_b_e = expert_valid ? (row_pack3_b + expert_slot * row_pack_stride_b) : nullptr;

  const int num_k_tiles_a = ceil_div_int(rank_a, STRUCTURED_TILE_K);
  const int num_k_tiles_b = ceil_div_int(rank_b, STRUCTURED_TILE_K);
  const int tiles_per_slice_a = ceil_div_int(num_k_tiles_a, split_k_slices);
  const int tile_iter_begin_a = split_k_idx * tiles_per_slice_a;
  const int tile_iter_end_a = min(num_k_tiles_a, tile_iter_begin_a + tiles_per_slice_a);
  const int tiles_per_slice_b = ceil_div_int(num_k_tiles_b, split_k_slices);
  const int tile_iter_begin_b = split_k_idx * tiles_per_slice_b;
  const int tile_iter_end_b = min(num_k_tiles_b, tile_iter_begin_b + tiles_per_slice_b);

  for (int tile_iter = tile_iter_begin_a; tile_iter < tile_iter_end_a; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_a - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = hidden_a[row_idx * rank_a + tile_k + idx];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_a_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_a_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_a_e,
          q_gmap_gr_offsets_a_e,
          q_gmap_gr_meta_a_e,
          packed_rows_a,
          outfeatures_a,
          tile_k,
          out_block,
          tile_q_gmap_words);
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_a_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_a) ? 1 : 0;
    }
    __syncthreads();

    if (out_a_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_a + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_a_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_a_e,
            qweight2_sparse_a_e,
            qweight2_sparse_prefix_a_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_a,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_a_e,
            g_idx_a_e,
            num_groups_a,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc_a += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  for (int tile_iter = tile_iter_begin_b; tile_iter < tile_iter_end_b; ++tile_iter) {
    const int tile_k = tile_iter * STRUCTURED_TILE_K;
    const int tile_k_size = min(STRUCTURED_TILE_K, rank_b - tile_k);
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      hidden_tile[idx] = hidden_b[row_idx * rank_b + tile_k + idx];
    }
    for (int idx = threadIdx.x; idx < tile_k_size; idx += blockDim.x) {
      const int global_k = tile_k + idx;
      tile_col_pack3[idx] = expert_valid ? static_cast<unsigned long long>(col_pack3_b_e[global_k]) : 0ULL;
    }
    if (threadIdx.x < (STRUCTURED_TILE_K + 31) / 32) {
      const int word_idx = threadIdx.x;
      const int word_base = word_idx * 32;
      unsigned word = 0U;
      if (expert_valid && word_base < tile_k_size) {
        const int valid_bits = min(32, tile_k_size - word_base);
        word = static_cast<unsigned>(c_map_b_e[(tile_k >> 5) + word_idx]) & low_bit_mask(valid_bits);
      }
      tile_c_map_words[word_idx] = word;
    }
    if (threadIdx.x == 0) {
      decode_block_gr_tile(
          q_gmap_gr_words_b_e,
          q_gmap_gr_offsets_b_e,
          q_gmap_gr_meta_b_e,
          packed_rows_b,
          outfeatures_b,
          tile_k,
          out_block,
          tile_q_gmap_words);
      const int group = (expert_valid && tile_k_size > 0) ? g_idx_b_e[tile_k] : -1;
      tile_group_value = group;
      tile_group_valid = (group >= 0 && group < num_groups_b) ? 1 : 0;
    }
    __syncthreads();

    if (out_b_valid && expert_valid) {
      const int row_offset = out_idx * num_groups_b + tile_group_value;
      const unsigned long long row_pack3_value = tile_group_valid != 0
          ? static_cast<unsigned long long>(row_pack3_b_e[row_offset])
          : 0ULL;
      for (int local_k = 0; local_k < tile_k_size; local_k += 2) {
        const int k0 = tile_k + local_k;
        const bool has_k1 = (local_k + 1) < tile_k_size;
        const unsigned gmap_word = tile_q_gmap_words[local_k >> 5][threadIdx.x];
        const half2 wpair = load_compact3_weight_pair_tiled_uniform_group_gmap_word<UniformK64Groups>(
            qweight1_b_e,
            qweight2_sparse_b_e,
            qweight2_sparse_prefix_b_e,
            gmap_word,
            out_idx,
            k0,
            local_k,
            has_k1,
            outfeatures_b,
            tile_c_map_words[local_k >> 5],
            tile_col_pack3,
            row_pack3_value,
            row_pack3_b_e,
            g_idx_b_e,
            num_groups_b,
            tile_group_value);
        const half x0 = hidden_tile[local_k];
        const half x1 = has_k1 ? hidden_tile[local_k + 1] : __float2half(0.0f);
        acc_b += reduce_half2(__hmul2(__halves2half2(x0, x1), wpair));
      }
    }
    __syncthreads();
  }

  if (out_a_valid) {
    atomicAdd(output_accum_a + row_idx * outfeatures_a + out_idx, acc_a);
  }

  if (out_b_valid) {
    atomicAdd(output_accum_b + row_idx * outfeatures_b + out_idx, acc_b);
  }
}

#ifndef USE_ROCM
#endif

void check_packed4_args(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    torch::Tensor g_idx) {
  CHECK_CUDA(input);
  CHECK_CUDA(qweight);
  CHECK_CUDA(qzeros);
  CHECK_CUDA(scales);
  CHECK_CUDA(g_idx);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(qweight);
  CHECK_CONTIGUOUS(qzeros);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(g_idx);

  CHECK_HALF(input);
  CHECK_HALF(scales);
  CHECK_INT(qweight);
  CHECK_INT(qzeros);
  CHECK_INT(g_idx);

  TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, cols]");
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D");
  TORCH_CHECK(qzeros.dim() == 2, "qzeros must be 2D");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2D");
  TORCH_CHECK(g_idx.dim() == 1, "g_idx must be 1D");

  const int cols = static_cast<int>(input.size(1));
  const int rows = static_cast<int>(qweight.size(1));

  TORCH_CHECK(qweight.size(0) == (cols + 7) / 8, "qweight packed height mismatch");
  TORCH_CHECK(g_idx.size(0) == cols, "g_idx length mismatch");
  TORCH_CHECK(scales.size(1) == rows, "scales row dimension mismatch");
  TORCH_CHECK(qzeros.size(0) == scales.size(0), "qzeros/scales group dimension mismatch");
  TORCH_CHECK(qzeros.size(1) == (rows + 7) / 8, "qzeros packed width mismatch");
}

void check_grouped_ns_compensation_args(
    torch::Tensor input,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor ns_rank_per_expert,
    torch::Tensor ns_c2_rank_offsets,
    torch::Tensor ns_c2_rank_word_offsets,
    torch::Tensor ns_c1_rank_word_offsets,
    torch::Tensor ns_c1_group_offsets,
    torch::Tensor ns_c2_qweight,
    torch::Tensor ns_c2_qzeros,
    torch::Tensor ns_c2_scales,
    torch::Tensor ns_c1_qweight,
    torch::Tensor ns_c1_qzeros,
    torch::Tensor ns_c1_scales,
    int64_t max_ns_rank,
    int64_t group_size) {
  CHECK_CUDA(input);
  CHECK_CUDA(expert_ids);
  CHECK_CUDA(segment_offsets);
  CHECK_CUDA(problem_offsets);
  CHECK_CUDA(ns_rank_per_expert);
  CHECK_CUDA(ns_c2_rank_offsets);
  CHECK_CUDA(ns_c2_rank_word_offsets);
  CHECK_CUDA(ns_c1_rank_word_offsets);
  CHECK_CUDA(ns_c1_group_offsets);
  CHECK_CUDA(ns_c2_qweight);
  CHECK_CUDA(ns_c2_qzeros);
  CHECK_CUDA(ns_c2_scales);
  CHECK_CUDA(ns_c1_qweight);
  CHECK_CUDA(ns_c1_qzeros);
  CHECK_CUDA(ns_c1_scales);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(expert_ids);
  CHECK_CONTIGUOUS(segment_offsets);
  CHECK_CONTIGUOUS(problem_offsets);
  CHECK_CONTIGUOUS(ns_rank_per_expert);
  CHECK_CONTIGUOUS(ns_c2_rank_offsets);
  CHECK_CONTIGUOUS(ns_c2_rank_word_offsets);
  CHECK_CONTIGUOUS(ns_c1_rank_word_offsets);
  CHECK_CONTIGUOUS(ns_c1_group_offsets);
  CHECK_CONTIGUOUS(ns_c2_qweight);
  CHECK_CONTIGUOUS(ns_c2_qzeros);
  CHECK_CONTIGUOUS(ns_c2_scales);
  CHECK_CONTIGUOUS(ns_c1_qweight);
  CHECK_CONTIGUOUS(ns_c1_qzeros);
  CHECK_CONTIGUOUS(ns_c1_scales);

  CHECK_HALF(input);
  CHECK_HALF(ns_c2_scales);
  CHECK_HALF(ns_c1_scales);
  CHECK_INT(expert_ids);
  CHECK_INT(segment_offsets);
  CHECK_INT(problem_offsets);
  CHECK_INT(ns_rank_per_expert);
  CHECK_INT(ns_c2_rank_offsets);
  CHECK_INT(ns_c2_rank_word_offsets);
  CHECK_INT(ns_c1_rank_word_offsets);
  CHECK_INT(ns_c1_group_offsets);
  CHECK_INT(ns_c2_qweight);
  CHECK_INT(ns_c2_qzeros);
  CHECK_INT(ns_c1_qweight);
  CHECK_INT(ns_c1_qzeros);

  TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, infeatures]");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be 1D");
  TORCH_CHECK(segment_offsets.dim() == 1, "segment_offsets must be 1D");
  TORCH_CHECK(problem_offsets.dim() == 1, "problem_offsets must be 1D");
  TORCH_CHECK(ns_rank_per_expert.dim() == 1, "ns_rank_per_expert must be 1D");
  TORCH_CHECK(ns_c2_rank_offsets.dim() == 1 && ns_c2_rank_word_offsets.dim() == 1 && ns_c1_rank_word_offsets.dim() == 1 && ns_c1_group_offsets.dim() == 1,
              "grouped NS offset tensors must be 1D");
  TORCH_CHECK(ns_c2_qweight.dim() == 2 && ns_c2_qzeros.dim() == 2 && ns_c2_scales.dim() == 2, "ns_c2 payload tensors must be 2D");
  TORCH_CHECK(ns_c1_qweight.dim() == 2 && ns_c1_qzeros.dim() == 2 && ns_c1_scales.dim() == 2, "ns_c1 payload tensors must be 2D");
  TORCH_CHECK(max_ns_rank >= 0, "max_ns_rank must be non-negative");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(total_problem_tiles >= 0, "total_problem_tiles must be non-negative");

  const int batch = static_cast<int>(input.size(0));
  const int infeatures = static_cast<int>(input.size(1));
  const int num_experts = static_cast<int>(ns_rank_per_expert.size(0));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int outfeatures = static_cast<int>(ns_c1_qweight.size(1));
  const int in_group_count = (infeatures + static_cast<int>(group_size) - 1) / static_cast<int>(group_size);

  TORCH_CHECK(segment_offsets.size(0) == num_segments + 1, "segment_offsets length mismatch");
  TORCH_CHECK(problem_offsets.size(0) == num_segments + 1, "problem_offsets length mismatch");
  TORCH_CHECK(ns_c2_rank_offsets.size(0) == num_experts && ns_c2_rank_word_offsets.size(0) == num_experts &&
              ns_c1_rank_word_offsets.size(0) == num_experts && ns_c1_group_offsets.size(0) == num_experts,
              "grouped NS offset tensor length mismatch");
  TORCH_CHECK(ns_c2_qweight.size(0) == (infeatures + 7) / 8, "ns_c2_qweight packed height mismatch");
  TORCH_CHECK(ns_c2_qzeros.size(0) == in_group_count, "ns_c2_qzeros group dimension mismatch");
  TORCH_CHECK(ns_c2_scales.size(0) == in_group_count, "ns_c2_scales group dimension mismatch");
  TORCH_CHECK(ns_c2_scales.size(1) == ns_c2_qweight.size(1), "ns_c2_scales rank mismatch");
  TORCH_CHECK(ns_c1_scales.size(1) == outfeatures, "ns_c1_scales outfeature mismatch");
  TORCH_CHECK(ns_c1_qzeros.size(1) == (outfeatures + 7) / 8, "ns_c1_qzeros packed width mismatch");
  TORCH_CHECK(batch == 0 || num_segments > 0, "non-empty input requires non-empty grouped route");
}

void check_grouped_ns_c1_args(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor ns_rank_per_expert,
    torch::Tensor ns_c1_rank_word_offsets,
    torch::Tensor ns_c1_group_offsets,
    torch::Tensor ns_c1_qweight,
    torch::Tensor ns_c1_qzeros,
    torch::Tensor ns_c1_scales,
    int64_t max_ns_rank,
    int64_t group_size) {
  CHECK_CUDA(hidden);
  CHECK_CUDA(expert_ids);
  CHECK_CUDA(segment_offsets);
  CHECK_CUDA(problem_offsets);
  CHECK_CUDA(ns_rank_per_expert);
  CHECK_CUDA(ns_c1_rank_word_offsets);
  CHECK_CUDA(ns_c1_group_offsets);
  CHECK_CUDA(ns_c1_qweight);
  CHECK_CUDA(ns_c1_qzeros);
  CHECK_CUDA(ns_c1_scales);

  CHECK_CONTIGUOUS(hidden);
  CHECK_CONTIGUOUS(expert_ids);
  CHECK_CONTIGUOUS(segment_offsets);
  CHECK_CONTIGUOUS(problem_offsets);
  CHECK_CONTIGUOUS(ns_rank_per_expert);
  CHECK_CONTIGUOUS(ns_c1_rank_word_offsets);
  CHECK_CONTIGUOUS(ns_c1_group_offsets);
  CHECK_CONTIGUOUS(ns_c1_qweight);
  CHECK_CONTIGUOUS(ns_c1_qzeros);
  CHECK_CONTIGUOUS(ns_c1_scales);

  CHECK_HALF(hidden);
  CHECK_HALF(ns_c1_scales);
  CHECK_INT(expert_ids);
  CHECK_INT(segment_offsets);
  CHECK_INT(problem_offsets);
  CHECK_INT(ns_rank_per_expert);
  CHECK_INT(ns_c1_rank_word_offsets);
  CHECK_INT(ns_c1_group_offsets);
  CHECK_INT(ns_c1_qweight);
  CHECK_INT(ns_c1_qzeros);

  TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [batch, max_ns_rank]");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be 1D");
  TORCH_CHECK(segment_offsets.dim() == 1, "segment_offsets must be 1D");
  TORCH_CHECK(problem_offsets.dim() == 1, "problem_offsets must be 1D");
  TORCH_CHECK(ns_rank_per_expert.dim() == 1, "ns_rank_per_expert must be 1D");
  TORCH_CHECK(ns_c1_rank_word_offsets.dim() == 1 && ns_c1_group_offsets.dim() == 1, "grouped NS C1 offset tensors must be 1D");
  TORCH_CHECK(ns_c1_qweight.dim() == 2 && ns_c1_qzeros.dim() == 2 && ns_c1_scales.dim() == 2, "ns_c1 payload tensors must be 2D");
  TORCH_CHECK(max_ns_rank >= 0, "max_ns_rank must be non-negative");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(total_problem_tiles >= 0, "total_problem_tiles must be non-negative");

  const int batch = static_cast<int>(hidden.size(0));
  const int hidden_rank = static_cast<int>(hidden.size(1));
  const int num_experts = static_cast<int>(ns_rank_per_expert.size(0));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int outfeatures = static_cast<int>(ns_c1_qweight.size(1));

  TORCH_CHECK(hidden_rank == static_cast<int>(max_ns_rank), "hidden rank mismatch for grouped NS C1 forward");
  TORCH_CHECK(segment_offsets.size(0) == num_segments + 1, "segment_offsets length mismatch");
  TORCH_CHECK(problem_offsets.size(0) == num_segments + 1, "problem_offsets length mismatch");
  TORCH_CHECK(ns_c1_rank_word_offsets.size(0) == num_experts && ns_c1_group_offsets.size(0) == num_experts,
              "grouped NS C1 offset tensor length mismatch");
  TORCH_CHECK(ns_c1_scales.size(1) == outfeatures, "ns_c1_scales outfeature mismatch");
  TORCH_CHECK(ns_c1_qzeros.size(1) == (outfeatures + 7) / 8, "ns_c1_qzeros packed width mismatch");
  TORCH_CHECK(batch == 0 || num_segments > 0, "non-empty hidden requires non-empty grouped route");
}


void check_half_vector(torch::Tensor value, const char* name, int expected_size) {
  CHECK_CUDA(value);
  CHECK_CONTIGUOUS(value);
  CHECK_HALF(value);
  TORCH_CHECK(value.dim() == 1, name, " must be 1D");
  TORCH_CHECK(value.size(0) == expected_size, name, " size mismatch");
}

struct Packed4FastConfig {
  bool valid;
  int groupsize;
};

Packed4FastConfig resolve_packed4_fast_config(
    torch::Tensor g_idx,
    int cols,
    int num_groups) {
  Packed4FastConfig fallback{false, 0};
  if (cols <= 0 || num_groups <= 0) {
    return fallback;
  }

  static std::mutex cache_mutex;
  static std::unordered_map<uintptr_t, Packed4FastConfig> cache;
  const uintptr_t key = reinterpret_cast<uintptr_t>(g_idx.data_ptr<int>());
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
  }

  auto g_idx_cpu = g_idx.to(torch::kCPU, /*non_blocking=*/false, /*copy=*/true);
  const int* values = g_idx_cpu.data_ptr<int>();
  const int groupsize = (cols + num_groups - 1) / num_groups;
  bool valid = true;
  for (int col = 0; col < cols; ++col) {
    int expected = col / groupsize;
    if (expected >= num_groups) {
      expected = num_groups - 1;
    }
    if (values[col] != expected) {
      valid = false;
      break;
    }
  }

  Packed4FastConfig result{valid, groupsize};
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache[key] = result;
  }
  return result;
}

void check_packed4_weight_only(
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    torch::Tensor g_idx,
    int expected_rows,
    int expected_cols,
    const char* name) {
  CHECK_CUDA(qweight);
  CHECK_CUDA(qzeros);
  CHECK_CUDA(scales);
  CHECK_CUDA(g_idx);
  CHECK_CONTIGUOUS(qweight);
  CHECK_CONTIGUOUS(qzeros);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(g_idx);
  CHECK_INT(qweight);
  CHECK_INT(qzeros);
  CHECK_HALF(scales);
  CHECK_INT(g_idx);
  TORCH_CHECK(qweight.dim() == 2, name, " qweight must be 2D");
  TORCH_CHECK(qzeros.dim() == 2, name, " qzeros must be 2D");
  TORCH_CHECK(scales.dim() == 2, name, " scales must be 2D");
  TORCH_CHECK(g_idx.dim() == 1, name, " g_idx must be 1D");
  TORCH_CHECK(qweight.size(1) == expected_rows, name, " qweight row mismatch");
  TORCH_CHECK(qweight.size(0) == (expected_cols + 7) / 8, name, " qweight packed width mismatch");
  TORCH_CHECK(g_idx.size(0) == expected_cols, name, " g_idx length mismatch");
  TORCH_CHECK(scales.size(1) == expected_rows, name, " scales row mismatch");
  TORCH_CHECK(qzeros.size(0) == scales.size(0), name, " qzeros/scales group mismatch");
  TORCH_CHECK(qzeros.size(1) == (expected_rows + 7) / 8, name, " qzeros packed width mismatch");
}

}  // namespace

torch::Tensor packed4_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    torch::Tensor g_idx) {
  check_packed4_args(input, qweight, qzeros, scales, g_idx);

  const int batch = static_cast<int>(input.size(0));
  const int cols = static_cast<int>(input.size(1));
  const int rows = static_cast<int>(qweight.size(1));
  const int packed_cols = static_cast<int>(qweight.size(0));
  const int qzeros_cols = static_cast<int>(qzeros.size(1));
  const int num_groups = static_cast<int>(qzeros.size(0));

  auto output = torch::zeros({batch, rows}, input.options());
  if (batch == 0 || rows == 0 || cols == 0) {
    return output;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const char* enable_fast_env = std::getenv("TRIS_ENABLE_PACKED4_FAST");
  const bool enable_fast = enable_fast_env == nullptr || (enable_fast_env[0] != 0 && enable_fast_env[0] != '0');
  const char* disable_fast_env = std::getenv("TRIS_DISABLE_PACKED4_FAST");
  const bool disable_fast = disable_fast_env != nullptr && disable_fast_env[0] != 0 && disable_fast_env[0] != '0';
  const Packed4FastConfig fast_config = (enable_fast && !disable_fast && num_groups > 0)
      ? resolve_packed4_fast_config(g_idx, cols, num_groups)
      : Packed4FastConfig{false, 0};
  const bool can_use_fast = fast_config.valid;
  if (can_use_fast) {
    const int groupsize = fast_config.groupsize;
    const int padded_cols = (cols + 1) & ~1;
    torch::Tensor fast_input = input;
    if (padded_cols != cols) {
      fast_input = torch::zeros({batch, padded_cols}, input.options());
      fast_input.slice(1, 0, cols).copy_(input);
    }
    auto output_fp = torch::zeros({batch, rows}, input.options().dtype(torch::kFloat));
    dim3 block(PACKED4_FAST_BLOCKWIDTH);
    dim3 grid((packed_cols + PACKED4_FAST_BLOCKHEIGHT - 1) / PACKED4_FAST_BLOCKHEIGHT,
              (rows + PACKED4_FAST_BLOCKWIDTH - 1) / PACKED4_FAST_BLOCKWIDTH,
              batch);
    Packed4LinearKernelFasterUniformGroups<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half2*>(fast_input.data_ptr<at::Half>()),
        qweight.data_ptr<int>(),
        qzeros.data_ptr<int>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        output_fp.data_ptr<float>(),
        batch,
        padded_cols / 2,
        packed_cols,
        rows,
        qzeros_cols,
        groupsize,
        num_groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int numel = batch * rows;
    const int cast_block = 256;
    const int cast_grid = (numel + cast_block - 1) / cast_block;
    Packed4LinearCastFloatToHalf<<<cast_grid, cast_block, 0, stream>>>(
        output_fp.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  dim3 block(PACKED4_BLOCK_ROWS);
  const int row_tiles = (rows + block.x - 1) / block.x;
  const int split_k_slices = resolve_packed4_split_k_slices(cols, batch, row_tiles, can_use_fast);
  if (split_k_slices > 1) {
    auto output_fp = torch::zeros({batch, rows}, input.options().dtype(torch::kFloat));
    dim3 splitk_grid(row_tiles, batch, static_cast<unsigned int>(split_k_slices));
    Packed4LinearKernelTiledLop3SplitK<<<splitk_grid, block, 0, stream>>>(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        qweight.data_ptr<int>(),
        qzeros.data_ptr<int>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        g_idx.data_ptr<int>(),
        output_fp.data_ptr<float>(),
        batch,
        rows,
        cols,
        qzeros_cols,
        split_k_slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int numel = batch * rows;
    const int cast_block = 256;
    const int cast_grid = (numel + cast_block - 1) / cast_block;
    Packed4LinearCastFloatToHalf<<<cast_grid, cast_block, 0, stream>>>(
        output_fp.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  dim3 grid(row_tiles, batch);
  Packed4LinearKernelTiledLop3<<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      qweight.data_ptr<int>(),
      qzeros.data_ptr<int>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
      g_idx.data_ptr<int>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      rows,
      cols,
      qzeros_cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}




torch::Tensor compact3_tris_precomputed_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor base_input,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor q_gmap_sparse,
    torch::Tensor q_gmap_sparse_prefix,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3,
    torch::Tensor output_scale,
    torch::Tensor bias,
    bool has_bias,
    torch::Tensor ns_c1_qweight,
    torch::Tensor ns_c1_qzeros,
    torch::Tensor ns_c1_scales,
    torch::Tensor ns_c1_g_idx,
    torch::Tensor ns_c2_qweight,
    torch::Tensor ns_c2_qzeros,
    torch::Tensor ns_c2_scales,
    torch::Tensor ns_c2_g_idx) {
  CHECK_CUDA(hidden); CHECK_CONTIGUOUS(hidden); CHECK_HALF(hidden);
  CHECK_CUDA(base_input); CHECK_CONTIGUOUS(base_input); CHECK_HALF(base_input);
  CHECK_CUDA(qweight1); CHECK_CONTIGUOUS(qweight1); CHECK_INT(qweight1);
  CHECK_CUDA(qweight2_sparse); CHECK_CONTIGUOUS(qweight2_sparse); CHECK_INT(qweight2_sparse);
  CHECK_CUDA(qweight2_sparse_prefix); CHECK_CONTIGUOUS(qweight2_sparse_prefix); CHECK_INT(qweight2_sparse_prefix);
  CHECK_CUDA(q_gmap_sparse); CHECK_CONTIGUOUS(q_gmap_sparse); CHECK_INT(q_gmap_sparse);
  CHECK_CUDA(q_gmap_sparse_prefix); CHECK_CONTIGUOUS(q_gmap_sparse_prefix); CHECK_INT(q_gmap_sparse_prefix);
  CHECK_CUDA(c_map); CHECK_CONTIGUOUS(c_map); CHECK_INT(c_map);
  CHECK_CUDA(g_idx); CHECK_CONTIGUOUS(g_idx); CHECK_INT(g_idx);
  CHECK_CUDA(col_pack3); CHECK_CONTIGUOUS(col_pack3); CHECK_LONG(col_pack3);
  CHECK_CUDA(row_pack3); CHECK_CONTIGUOUS(row_pack3); CHECK_LONG(row_pack3);
  CHECK_CUDA(output_scale); CHECK_CONTIGUOUS(output_scale); CHECK_HALF(output_scale);

  TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [batch, rank]");
  TORCH_CHECK(base_input.dim() == 2, "base_input must be 2D [batch, cols]");
  TORCH_CHECK(base_input.size(0) == hidden.size(0), "base_input batch mismatch");
  TORCH_CHECK(qweight1.dim() == 2 && qweight2_sparse.dim() == 2 && q_gmap_sparse.dim() == 2, "compact3 packed tensors must be 2D");
  TORCH_CHECK(qweight2_sparse_prefix.dim() == 1 && q_gmap_sparse_prefix.dim() == 1, "compact3 sparse prefix tensors must be 1D");
  TORCH_CHECK(c_map.dim() == 1, "c_map must be 1D");
  TORCH_CHECK(g_idx.dim() == 1, "g_idx must be 1D");
  TORCH_CHECK(col_pack3.dim() == 1, "col_pack3 must be 1D");
  TORCH_CHECK(row_pack3.dim() == 2, "row_pack3 must be 2D");

  const int batch = static_cast<int>(hidden.size(0));
  const int rank = static_cast<int>(hidden.size(1));
  const int packed_rows = (rank + 31) / 32;
  const int outfeatures = static_cast<int>(qweight1.size(1));
  const int num_groups = static_cast<int>(row_pack3.size(1));

  TORCH_CHECK(qweight1.size(0) == packed_rows, "qweight1 packed height mismatch");
  TORCH_CHECK(c_map.size(0) == packed_rows, "c_map packed length mismatch");
  TORCH_CHECK(g_idx.size(0) == rank, "g_idx rank mismatch");
  TORCH_CHECK(col_pack3.size(0) == rank, "col_pack3 rank mismatch");
  TORCH_CHECK(row_pack3.size(0) == outfeatures, "row_pack3 outfeature mismatch");
  TORCH_CHECK(qweight2_sparse_prefix.size(0) == packed_rows + 1, "qweight2_sparse_prefix shape mismatch");
  TORCH_CHECK(q_gmap_sparse_prefix.size(0) == packed_rows + 1, "q_gmap_sparse_prefix shape mismatch");
  check_half_vector(output_scale, "output_scale", outfeatures);
  if (has_bias) {
    check_half_vector(bias, "bias", outfeatures);
  }

  const bool has_ns_matrix = ns_c1_qweight.numel() > 0 && ns_c2_qweight.numel() > 0;
  int ns_rank = 0;
  int ns_c1_qzeros_cols = 0;
  auto ns_hidden = torch::zeros({batch, 0}, hidden.options());
  if (has_ns_matrix) {
    check_packed4_args(base_input, ns_c2_qweight, ns_c2_qzeros, ns_c2_scales, ns_c2_g_idx);
    ns_rank = static_cast<int>(ns_c2_qweight.size(1));
    check_packed4_weight_only(ns_c1_qweight, ns_c1_qzeros, ns_c1_scales, ns_c1_g_idx, outfeatures, ns_rank, "ns_c1");
    TORCH_CHECK(ns_c1_g_idx.size(0) == ns_rank, "ns_c1 g_idx length mismatch");
    ns_c1_qzeros_cols = static_cast<int>(ns_c1_qzeros.size(1));
    ns_hidden = packed4_linear_forward_cuda(base_input, ns_c2_qweight, ns_c2_qzeros, ns_c2_scales, ns_c2_g_idx);
  }

  auto output = torch::zeros({batch, outfeatures}, hidden.options());
  if (batch == 0 || rank == 0 || outfeatures == 0) {
    return output;
  }
  dim3 block(STRUCTURED_BLOCK_THREADS);
  dim3 grid((outfeatures + block.x - 1) / block.x, batch);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Compact3TriSMoEPrecomputedKernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
      qweight1.data_ptr<int>(),
      qweight2_sparse.data_ptr<int>(),
      qweight2_sparse_prefix.data_ptr<int>(),
      q_gmap_sparse.data_ptr<int>(),
      q_gmap_sparse_prefix.data_ptr<int>(),
      c_map.data_ptr<int>(),
      g_idx.data_ptr<int>(),
      col_pack3.data_ptr<int64_t>(),
      row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
      has_bias ? reinterpret_cast<half*>(bias.data_ptr<at::Half>()) : nullptr,
      ns_rank > 0 ? reinterpret_cast<half*>(ns_hidden.data_ptr<at::Half>()) : nullptr,
      has_ns_matrix ? ns_c1_qweight.data_ptr<int>() : nullptr,
      has_ns_matrix ? ns_c1_qzeros.data_ptr<int>() : nullptr,
      has_ns_matrix ? reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()) : nullptr,
      has_ns_matrix ? ns_c1_g_idx.data_ptr<int>() : nullptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      rank,
      outfeatures,
      num_groups,
      ns_rank,
      ns_c1_qzeros_cols,
      has_bias);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor grouped_ns_compensation_forward_cuda(
    torch::Tensor input,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles_value,
    torch::Tensor ns_rank_per_expert,
    torch::Tensor ns_c2_rank_offsets,
    torch::Tensor ns_c2_rank_word_offsets,
    torch::Tensor ns_c1_rank_word_offsets,
    torch::Tensor ns_c1_group_offsets,
    torch::Tensor ns_c2_qweight,
    torch::Tensor ns_c2_qzeros,
    torch::Tensor ns_c2_scales,
    torch::Tensor ns_c1_qweight,
    torch::Tensor ns_c1_qzeros,
    torch::Tensor ns_c1_scales,
    int64_t max_ns_rank_value,
    int64_t group_size_value,
    bool decode_phase) {
  check_grouped_ns_compensation_args(
      input,
      expert_ids,
      segment_offsets,
      problem_offsets,
      total_problem_tiles_value,
      ns_rank_per_expert,
      ns_c2_rank_offsets,
      ns_c2_rank_word_offsets,
      ns_c1_rank_word_offsets,
      ns_c1_group_offsets,
      ns_c2_qweight,
      ns_c2_qzeros,
      ns_c2_scales,
      ns_c1_qweight,
      ns_c1_qzeros,
      ns_c1_scales,
      max_ns_rank_value,
      group_size_value);

  const int batch = static_cast<int>(input.size(0));
  const int infeatures = static_cast<int>(input.size(1));
  const int num_experts = static_cast<int>(ns_rank_per_expert.size(0));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int total_problem_tiles = static_cast<int>(total_problem_tiles_value);
  const int max_ns_rank = static_cast<int>(max_ns_rank_value);
  const int group_size = static_cast<int>(group_size_value);
  const int total_rank = static_cast<int>(ns_c2_scales.size(1));
  const int total_rank_words = static_cast<int>(ns_c2_qzeros.size(1));
  const int group_count = static_cast<int>(ns_c2_qzeros.size(0));
  const int outfeatures = static_cast<int>(ns_c1_qweight.size(1));
  const int ns_c1_qzeros_cols = static_cast<int>(ns_c1_qzeros.size(1));

  auto output = torch::zeros({batch, outfeatures}, input.options());
  if (batch == 0 || infeatures == 0 || outfeatures == 0 || max_ns_rank == 0 || total_problem_tiles == 0) {
    return output;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block(STRUCTURED_BLOCK_OUT);
  auto hidden = torch::zeros({batch, max_ns_rank}, input.options());

  const bool use_decode_path = decode_phase || is_grouped_decode_problem(batch, num_segments, total_problem_tiles_value);
  const int hidden_row_tiles = (max_ns_rank + static_cast<int>(block.x) - 1) / static_cast<int>(block.x);
  const int output_row_tiles = (outfeatures + static_cast<int>(block.x) - 1) / static_cast<int>(block.x);
  const int hidden_split_k_slices = resolve_grouped_ns_split_k_slices(infeatures, total_problem_tiles, hidden_row_tiles, use_decode_path);
  const int output_split_k_slices = resolve_grouped_ns_split_k_slices(max_ns_rank, total_problem_tiles, output_row_tiles, use_decode_path);

  if (hidden_split_k_slices > 1) {
    auto hidden_accum = torch::zeros({batch, max_ns_rank}, input.options().dtype(torch::kFloat));
    dim3 hidden_grid(hidden_row_tiles, static_cast<unsigned int>(total_problem_tiles), static_cast<unsigned int>(hidden_split_k_slices));
    if (use_decode_path) {
      GroupedPacked4LinearOffsetDecodeC2SplitKKernel<<<hidden_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(input.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          ns_rank_per_expert.data_ptr<int>(),
          ns_c2_rank_offsets.data_ptr<int>(),
          ns_c2_rank_word_offsets.data_ptr<int>(),
          ns_c2_qweight.data_ptr<int>(),
          ns_c2_qzeros.data_ptr<int>(),
          reinterpret_cast<half*>(ns_c2_scales.data_ptr<at::Half>()),
          hidden_accum.data_ptr<float>(),
          num_experts,
          num_segments,
          infeatures,
          max_ns_rank,
          total_rank,
          total_rank_words,
          group_count,
          group_size,
          hidden_split_k_slices);
    } else {
      GroupedPacked4LinearOffsetSegmentC2SplitKKernel<<<hidden_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(input.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          ns_rank_per_expert.data_ptr<int>(),
          ns_c2_rank_offsets.data_ptr<int>(),
          ns_c2_rank_word_offsets.data_ptr<int>(),
          ns_c2_qweight.data_ptr<int>(),
          ns_c2_qzeros.data_ptr<int>(),
          reinterpret_cast<half*>(ns_c2_scales.data_ptr<at::Half>()),
          hidden_accum.data_ptr<float>(),
          num_experts,
          num_segments,
          infeatures,
          max_ns_rank,
          total_rank,
          total_rank_words,
          group_count,
          group_size,
          hidden_split_k_slices);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int hidden_numel = batch * max_ns_rank;
    const int cast_block = 256;
    const int cast_grid = (hidden_numel + cast_block - 1) / cast_block;
    Packed4LinearCastFloatToHalf<<<cast_grid, cast_block, 0, stream>>>(
        hidden_accum.data_ptr<float>(),
        reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
        hidden_numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    dim3 hidden_grid(hidden_row_tiles, static_cast<unsigned int>(total_problem_tiles));
    if (use_decode_path) {
      GroupedPacked4LinearOffsetDecodeC2Kernel<<<hidden_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(input.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          ns_rank_per_expert.data_ptr<int>(),
          ns_c2_rank_offsets.data_ptr<int>(),
          ns_c2_rank_word_offsets.data_ptr<int>(),
          ns_c2_qweight.data_ptr<int>(),
          ns_c2_qzeros.data_ptr<int>(),
          reinterpret_cast<half*>(ns_c2_scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          num_experts,
          num_segments,
          infeatures,
          max_ns_rank,
          total_rank,
          total_rank_words,
          group_count,
          group_size);
    } else {
      GroupedPacked4LinearOffsetSegmentC2Kernel<<<hidden_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(input.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          ns_rank_per_expert.data_ptr<int>(),
          ns_c2_rank_offsets.data_ptr<int>(),
          ns_c2_rank_word_offsets.data_ptr<int>(),
          ns_c2_qweight.data_ptr<int>(),
          ns_c2_qzeros.data_ptr<int>(),
          reinterpret_cast<half*>(ns_c2_scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          num_experts,
          num_segments,
          infeatures,
          max_ns_rank,
          total_rank,
          total_rank_words,
          group_count,
          group_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (output_split_k_slices > 1) {
    auto output_accum = torch::zeros({batch, outfeatures}, input.options().dtype(torch::kFloat));
    dim3 output_grid(output_row_tiles, static_cast<unsigned int>(total_problem_tiles), static_cast<unsigned int>(output_split_k_slices));
    if (use_decode_path) {
      GroupedPacked4LinearOffsetDecodeC1SplitKKernel<<<output_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          ns_rank_per_expert.data_ptr<int>(),
          ns_c1_rank_word_offsets.data_ptr<int>(),
          ns_c1_group_offsets.data_ptr<int>(),
          ns_c1_qweight.data_ptr<int>(),
          ns_c1_qzeros.data_ptr<int>(),
          reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()),
          output_accum.data_ptr<float>(),
          num_experts,
          num_segments,
          outfeatures,
          max_ns_rank,
          ns_c1_qzeros_cols,
          group_size,
          output_split_k_slices);
    } else {
      GroupedPacked4LinearOffsetSegmentC1SplitKKernel<<<output_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          ns_rank_per_expert.data_ptr<int>(),
          ns_c1_rank_word_offsets.data_ptr<int>(),
          ns_c1_group_offsets.data_ptr<int>(),
          ns_c1_qweight.data_ptr<int>(),
          ns_c1_qzeros.data_ptr<int>(),
          reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()),
          output_accum.data_ptr<float>(),
          num_experts,
          num_segments,
          outfeatures,
          max_ns_rank,
          ns_c1_qzeros_cols,
          group_size,
          output_split_k_slices);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int output_numel = batch * outfeatures;
    const int cast_block = 256;
    const int cast_grid = (output_numel + cast_block - 1) / cast_block;
    Packed4LinearCastFloatToHalf<<<cast_grid, cast_block, 0, stream>>>(
        output_accum.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        output_numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  dim3 output_grid(output_row_tiles, static_cast<unsigned int>(total_problem_tiles));
  if (use_decode_path) {
    GroupedPacked4LinearOffsetDecodeC1Kernel<<<output_grid, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        problem_offsets.data_ptr<int>(),
        ns_rank_per_expert.data_ptr<int>(),
        ns_c1_rank_word_offsets.data_ptr<int>(),
        ns_c1_group_offsets.data_ptr<int>(),
        ns_c1_qweight.data_ptr<int>(),
        ns_c1_qzeros.data_ptr<int>(),
        reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        num_experts,
        num_segments,
        outfeatures,
        max_ns_rank,
        ns_c1_qzeros_cols,
        group_size);
  } else {
    GroupedPacked4LinearOffsetSegmentC1Kernel<<<output_grid, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        problem_offsets.data_ptr<int>(),
        ns_rank_per_expert.data_ptr<int>(),
        ns_c1_rank_word_offsets.data_ptr<int>(),
        ns_c1_group_offsets.data_ptr<int>(),
        ns_c1_qweight.data_ptr<int>(),
        ns_c1_qzeros.data_ptr<int>(),
        reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        num_experts,
        num_segments,
        outfeatures,
        max_ns_rank,
        ns_c1_qzeros_cols,
        group_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor grouped_ns_c1_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles_value,
    torch::Tensor ns_rank_per_expert,
    torch::Tensor ns_c1_rank_word_offsets,
    torch::Tensor ns_c1_group_offsets,
    torch::Tensor ns_c1_qweight,
    torch::Tensor ns_c1_qzeros,
    torch::Tensor ns_c1_scales,
    int64_t max_ns_rank_value,
    int64_t group_size_value,
    bool decode_phase) {
  check_grouped_ns_c1_args(
      hidden,
      expert_ids,
      segment_offsets,
      problem_offsets,
      total_problem_tiles_value,
      ns_rank_per_expert,
      ns_c1_rank_word_offsets,
      ns_c1_group_offsets,
      ns_c1_qweight,
      ns_c1_qzeros,
      ns_c1_scales,
      max_ns_rank_value,
      group_size_value);

  const int batch = static_cast<int>(hidden.size(0));
  const int num_experts = static_cast<int>(ns_rank_per_expert.size(0));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int total_problem_tiles = static_cast<int>(total_problem_tiles_value);
  const int max_ns_rank = static_cast<int>(max_ns_rank_value);
  const int group_size = static_cast<int>(group_size_value);
  const int outfeatures = static_cast<int>(ns_c1_qweight.size(1));
  const int ns_c1_qzeros_cols = static_cast<int>(ns_c1_qzeros.size(1));

  auto output = torch::zeros({batch, outfeatures}, hidden.options());
  if (batch == 0 || outfeatures == 0 || max_ns_rank == 0 || total_problem_tiles == 0) {
    return output;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block(STRUCTURED_BLOCK_OUT);
  const bool use_decode_path = decode_phase || is_grouped_decode_problem(batch, num_segments, total_problem_tiles_value);
  const int output_row_tiles = (outfeatures + static_cast<int>(block.x) - 1) / static_cast<int>(block.x);
  const int output_split_k_slices = resolve_grouped_ns_split_k_slices(max_ns_rank, total_problem_tiles, output_row_tiles, use_decode_path);

  if (output_split_k_slices > 1) {
    auto output_accum = torch::zeros({batch, outfeatures}, hidden.options().dtype(torch::kFloat));
    dim3 output_grid(output_row_tiles, static_cast<unsigned int>(total_problem_tiles), static_cast<unsigned int>(output_split_k_slices));
    if (use_decode_path) {
      GroupedPacked4LinearOffsetDecodeC1SplitKKernel<<<output_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          ns_rank_per_expert.data_ptr<int>(),
          ns_c1_rank_word_offsets.data_ptr<int>(),
          ns_c1_group_offsets.data_ptr<int>(),
          ns_c1_qweight.data_ptr<int>(),
          ns_c1_qzeros.data_ptr<int>(),
          reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()),
          output_accum.data_ptr<float>(),
          num_experts,
          num_segments,
          outfeatures,
          max_ns_rank,
          ns_c1_qzeros_cols,
          group_size,
          output_split_k_slices);
    } else {
      GroupedPacked4LinearOffsetSegmentC1SplitKKernel<<<output_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          ns_rank_per_expert.data_ptr<int>(),
          ns_c1_rank_word_offsets.data_ptr<int>(),
          ns_c1_group_offsets.data_ptr<int>(),
          ns_c1_qweight.data_ptr<int>(),
          ns_c1_qzeros.data_ptr<int>(),
          reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()),
          output_accum.data_ptr<float>(),
          num_experts,
          num_segments,
          outfeatures,
          max_ns_rank,
          ns_c1_qzeros_cols,
          group_size,
          output_split_k_slices);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int output_numel = batch * outfeatures;
    const int cast_block = 256;
    const int cast_grid = (output_numel + cast_block - 1) / cast_block;
    Packed4LinearCastFloatToHalf<<<cast_grid, cast_block, 0, stream>>>(
        output_accum.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        output_numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  dim3 output_grid(output_row_tiles, static_cast<unsigned int>(total_problem_tiles));
  if (use_decode_path) {
    GroupedPacked4LinearOffsetDecodeC1Kernel<<<output_grid, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        problem_offsets.data_ptr<int>(),
        ns_rank_per_expert.data_ptr<int>(),
        ns_c1_rank_word_offsets.data_ptr<int>(),
        ns_c1_group_offsets.data_ptr<int>(),
        ns_c1_qweight.data_ptr<int>(),
        ns_c1_qzeros.data_ptr<int>(),
        reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        num_experts,
        num_segments,
        outfeatures,
        max_ns_rank,
        ns_c1_qzeros_cols,
        group_size);
  } else {
    GroupedPacked4LinearOffsetSegmentC1Kernel<<<output_grid, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        problem_offsets.data_ptr<int>(),
        ns_rank_per_expert.data_ptr<int>(),
        ns_c1_rank_word_offsets.data_ptr<int>(),
        ns_c1_group_offsets.data_ptr<int>(),
        ns_c1_qweight.data_ptr<int>(),
        ns_c1_qzeros.data_ptr<int>(),
        reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        num_experts,
        num_segments,
        outfeatures,
        max_ns_rank,
        ns_c1_qzeros_cols,
        group_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor grouped_ns_compensation_fused_decode_forward_cuda(
    torch::Tensor input,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles_value,
    torch::Tensor ns_rank_per_expert,
    torch::Tensor ns_c2_rank_offsets,
    torch::Tensor ns_c2_rank_word_offsets,
    torch::Tensor ns_c1_rank_word_offsets,
    torch::Tensor ns_c1_group_offsets,
    torch::Tensor ns_c2_qweight,
    torch::Tensor ns_c2_qzeros,
    torch::Tensor ns_c2_scales,
    torch::Tensor ns_c1_qweight,
    torch::Tensor ns_c1_qzeros,
    torch::Tensor ns_c1_scales,
    int64_t max_ns_rank_value,
    int64_t group_size_value) {
  check_grouped_ns_compensation_args(
      input,
      expert_ids,
      segment_offsets,
      problem_offsets,
      total_problem_tiles_value,
      ns_rank_per_expert,
      ns_c2_rank_offsets,
      ns_c2_rank_word_offsets,
      ns_c1_rank_word_offsets,
      ns_c1_group_offsets,
      ns_c2_qweight,
      ns_c2_qzeros,
      ns_c2_scales,
      ns_c1_qweight,
      ns_c1_qzeros,
      ns_c1_scales,
      max_ns_rank_value,
      group_size_value);

  const int batch = static_cast<int>(input.size(0));
  const int infeatures = static_cast<int>(input.size(1));
  const int num_experts = static_cast<int>(ns_rank_per_expert.size(0));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int total_problem_tiles = static_cast<int>(total_problem_tiles_value);
  const int max_ns_rank = static_cast<int>(max_ns_rank_value);
  const int group_size = static_cast<int>(group_size_value);
  const int total_rank = static_cast<int>(ns_c2_scales.size(1));
  const int total_rank_words = static_cast<int>(ns_c2_qzeros.size(1));
  const int group_count = static_cast<int>(ns_c2_qzeros.size(0));
  const int outfeatures = static_cast<int>(ns_c1_qweight.size(1));
  const int ns_c1_qzeros_cols = static_cast<int>(ns_c1_qzeros.size(1));

  auto output = torch::zeros({batch, outfeatures}, input.options());
  if (batch == 0 || infeatures == 0 || outfeatures == 0 || max_ns_rank == 0 || total_problem_tiles == 0) {
    return output;
  }

  auto output_accum = torch::zeros({batch, outfeatures}, input.options().dtype(torch::kFloat));
  const int rank_chunks = ceil_div_int(max_ns_rank, NS_FUSED_CHUNK);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block(GROUPED_NS_FUSED_BLOCK_THREADS);
  dim3 grid(static_cast<unsigned int>(total_problem_tiles), static_cast<unsigned int>(rank_chunks));
  GroupedPacked4LinearOffsetDecodeFusedKernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      expert_ids.data_ptr<int>(),
      segment_offsets.data_ptr<int>(),
      problem_offsets.data_ptr<int>(),
      ns_rank_per_expert.data_ptr<int>(),
      ns_c2_rank_offsets.data_ptr<int>(),
      ns_c2_rank_word_offsets.data_ptr<int>(),
      ns_c1_rank_word_offsets.data_ptr<int>(),
      ns_c1_group_offsets.data_ptr<int>(),
      ns_c2_qweight.data_ptr<int>(),
      ns_c2_qzeros.data_ptr<int>(),
      reinterpret_cast<half*>(ns_c2_scales.data_ptr<at::Half>()),
      ns_c1_qweight.data_ptr<int>(),
      ns_c1_qzeros.data_ptr<int>(),
      reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()),
      output_accum.data_ptr<float>(),
      num_experts,
      num_segments,
      infeatures,
      outfeatures,
      total_rank,
      total_rank_words,
      group_count,
      ns_c1_qzeros_cols,
      group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int output_numel = batch * outfeatures;
  const int cast_block = 256;
  const int cast_grid = (output_numel + cast_block - 1) / cast_block;
  Packed4LinearCastFloatToHalf<<<cast_grid, cast_block, 0, stream>>>(
      output_accum.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      output_numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor compact3_tris_precomputed_grouped_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor q_gmap_sparse,
    torch::Tensor q_gmap_sparse_prefix,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3,
    torch::Tensor output_scale,
    torch::Tensor bias,
    bool has_bias) {
  CHECK_CUDA(hidden); CHECK_CONTIGUOUS(hidden); CHECK_HALF(hidden);
  CHECK_CUDA(expert_ids); CHECK_CONTIGUOUS(expert_ids); CHECK_INT(expert_ids);
  CHECK_CUDA(segment_offsets); CHECK_CONTIGUOUS(segment_offsets); CHECK_INT(segment_offsets);
  CHECK_CUDA(problem_offsets); CHECK_CONTIGUOUS(problem_offsets); CHECK_INT(problem_offsets);
  CHECK_CUDA(qweight1); CHECK_CONTIGUOUS(qweight1); CHECK_INT(qweight1);
  CHECK_CUDA(qweight2_sparse); CHECK_CONTIGUOUS(qweight2_sparse); CHECK_INT(qweight2_sparse);
  CHECK_CUDA(qweight2_sparse_prefix); CHECK_CONTIGUOUS(qweight2_sparse_prefix); CHECK_INT(qweight2_sparse_prefix);
  CHECK_CUDA(q_gmap_sparse); CHECK_CONTIGUOUS(q_gmap_sparse); CHECK_INT(q_gmap_sparse);
  CHECK_CUDA(q_gmap_sparse_prefix); CHECK_CONTIGUOUS(q_gmap_sparse_prefix); CHECK_INT(q_gmap_sparse_prefix);
  CHECK_CUDA(c_map); CHECK_CONTIGUOUS(c_map); CHECK_INT(c_map);
  CHECK_CUDA(g_idx); CHECK_CONTIGUOUS(g_idx); CHECK_INT(g_idx);
  CHECK_CUDA(col_pack3); CHECK_CONTIGUOUS(col_pack3); CHECK_LONG(col_pack3);
  CHECK_CUDA(row_pack3); CHECK_CONTIGUOUS(row_pack3); CHECK_LONG(row_pack3);
  CHECK_CUDA(output_scale); CHECK_CONTIGUOUS(output_scale); CHECK_HALF(output_scale);

  TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [batch, rank]");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be 1D [segments]");
  TORCH_CHECK(segment_offsets.dim() == 1 && segment_offsets.size(0) == expert_ids.size(0) + 1, "segment_offsets must be 1D [segments + 1]");
  TORCH_CHECK(problem_offsets.dim() == 1 && problem_offsets.size(0) == expert_ids.size(0) + 1, "problem_offsets must be 1D [segments + 1]");
  TORCH_CHECK(qweight1.dim() == 3 && qweight2_sparse.dim() == 3 && q_gmap_sparse.dim() == 3, "compact3 grouped payload tensors must be 3D");
  TORCH_CHECK(qweight2_sparse_prefix.dim() == 2 && q_gmap_sparse_prefix.dim() == 2, "compact3 grouped sparse prefix tensors must be 2D");
  TORCH_CHECK(c_map.dim() == 2 && g_idx.dim() == 2, "compact3 grouped c_map/g_idx dims mismatch");
  TORCH_CHECK(col_pack3.dim() == 2, "compact3 grouped col_pack3 must be 2D");
  TORCH_CHECK(row_pack3.dim() == 3, "compact3 grouped row_pack3 must be 3D");
  TORCH_CHECK(output_scale.dim() == 2, "compact3 grouped output_scale must be 2D");

  const int batch = static_cast<int>(hidden.size(0));
  const int rank = static_cast<int>(hidden.size(1));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int num_experts = static_cast<int>(qweight1.size(0));
  const int packed_rows = (rank + 31) / 32;
  const int outfeatures = static_cast<int>(qweight1.size(2));
  const int qweight2_sparse_rows = static_cast<int>(qweight2_sparse.size(1));
  const int q_gmap_sparse_rows = static_cast<int>(q_gmap_sparse.size(1));
  const int num_groups = static_cast<int>(row_pack3.size(2));

  TORCH_CHECK(qweight1.size(1) == packed_rows, "compact3 grouped qweight1 packed height mismatch");
  TORCH_CHECK(qweight2_sparse.size(0) == num_experts && q_gmap_sparse.size(0) == num_experts, "compact3 grouped expert mismatch");
  TORCH_CHECK(qweight2_sparse.size(2) == outfeatures && q_gmap_sparse.size(2) == outfeatures, "compact3 grouped outfeature mismatch");
  TORCH_CHECK(qweight2_sparse_prefix.size(0) == num_experts && qweight2_sparse_prefix.size(1) == packed_rows + 1, "compact3 grouped qweight2 prefix mismatch");
  TORCH_CHECK(q_gmap_sparse_prefix.size(0) == num_experts && q_gmap_sparse_prefix.size(1) == packed_rows + 1, "compact3 grouped q_gmap prefix mismatch");
  TORCH_CHECK(c_map.size(0) == num_experts && c_map.size(1) == packed_rows, "compact3 grouped c_map mismatch");
  TORCH_CHECK(g_idx.size(0) == num_experts && g_idx.size(1) == rank, "compact3 grouped g_idx mismatch");
  TORCH_CHECK(col_pack3.size(0) == num_experts && col_pack3.size(1) == rank, "compact3 grouped col_pack3 mismatch");
  TORCH_CHECK(row_pack3.size(0) == num_experts && row_pack3.size(1) == outfeatures && row_pack3.size(2) == num_groups, "compact3 grouped row_pack3 mismatch");
  TORCH_CHECK(output_scale.size(0) == num_experts && output_scale.size(1) == outfeatures, "compact3 grouped output_scale mismatch");
  if (has_bias) {
    CHECK_CUDA(bias); CHECK_CONTIGUOUS(bias); CHECK_HALF(bias);
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == num_experts && bias.size(1) == outfeatures, "compact3 grouped bias mismatch");
  }

  auto output = torch::zeros({batch, outfeatures}, hidden.options());
  if (batch == 0 || rank == 0 || outfeatures == 0 || num_experts == 0 || num_segments == 0 || total_problem_tiles <= 0) {
    return output;
  }

  dim3 block(STRUCTURED_BLOCK_OUT);
  const int out_tiles = (outfeatures + block.x - 1) / block.x;
  const bool use_decode_path = is_grouped_decode_problem(batch, num_segments, total_problem_tiles);
  const int split_k_slices = resolve_grouped_split_k_slices(rank, static_cast<int>(total_problem_tiles), out_tiles, use_decode_path);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (split_k_slices > 1) {
    auto output_accum = torch::zeros({batch, outfeatures}, hidden.options().dtype(torch::kFloat32));
    dim3 splitk_grid(out_tiles, static_cast<unsigned int>(total_problem_tiles), static_cast<unsigned int>(split_k_slices));
    if (use_decode_path) {
      Compact3TriSMoEPrecomputedGroupedDecodeSplitKKernel<<<splitk_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_sparse.data_ptr<int>(),
          q_gmap_sparse_prefix.data_ptr<int>(),
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          output_accum.data_ptr<float>(),
          qweight2_sparse_rows,
          q_gmap_sparse_rows,
          num_segments,
          rank,
          outfeatures,
          num_groups,
          split_k_slices);
    } else {
      Compact3TriSMoEPrecomputedGroupedSegmentSplitKKernel<<<splitk_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_sparse.data_ptr<int>(),
          q_gmap_sparse_prefix.data_ptr<int>(),
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          output_accum.data_ptr<float>(),
          qweight2_sparse_rows,
          q_gmap_sparse_rows,
          num_segments,
          rank,
          outfeatures,
          num_groups,
          split_k_slices);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dim3 epilogue_block(32, 8);
    dim3 epilogue_grid((outfeatures + epilogue_block.x - 1) / epilogue_block.x, (batch + epilogue_block.y - 1) / epilogue_block.y);
    Compact3TriSMoEGroupedSplitKEpilogueKernel<<<epilogue_grid, epilogue_block, 0, stream>>>(
        output_accum.data_ptr<float>(),
        expert_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
        has_bias ? reinterpret_cast<half*>(bias.data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        batch,
        outfeatures,
        num_segments,
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  dim3 grid(out_tiles, static_cast<unsigned int>(total_problem_tiles));
  if (use_decode_path) {
    Compact3TriSMoEPrecomputedGroupedDecodeKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        problem_offsets.data_ptr<int>(),
        qweight1.data_ptr<int>(),
        qweight2_sparse.data_ptr<int>(),
        qweight2_sparse_prefix.data_ptr<int>(),
        q_gmap_sparse.data_ptr<int>(),
        q_gmap_sparse_prefix.data_ptr<int>(),
        c_map.data_ptr<int>(),
        g_idx.data_ptr<int>(),
        col_pack3.data_ptr<int64_t>(),
        row_pack3.data_ptr<int64_t>(),
        reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
        has_bias ? reinterpret_cast<half*>(bias.data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        qweight2_sparse_rows,
        q_gmap_sparse_rows,
        num_segments,
        rank,
        outfeatures,
        num_groups,
        has_bias);
  } else {
    Compact3TriSMoEPrecomputedGroupedSegmentKernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        problem_offsets.data_ptr<int>(),
        qweight1.data_ptr<int>(),
        qweight2_sparse.data_ptr<int>(),
        qweight2_sparse_prefix.data_ptr<int>(),
        q_gmap_sparse.data_ptr<int>(),
        q_gmap_sparse_prefix.data_ptr<int>(),
        c_map.data_ptr<int>(),
        g_idx.data_ptr<int>(),
        col_pack3.data_ptr<int64_t>(),
        row_pack3.data_ptr<int64_t>(),
        reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
        has_bias ? reinterpret_cast<half*>(bias.data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        qweight2_sparse_rows,
        q_gmap_sparse_rows,
        num_segments,
        rank,
        outfeatures,
        num_groups,
        has_bias);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor compact3_tris_precomputed_grouped_block_gr_predecode_cuda(
    torch::Tensor decoded_q_gmap_words,
    torch::Tensor expert_ids,
    torch::Tensor q_gmap_gr_words,
    torch::Tensor q_gmap_gr_offsets,
    torch::Tensor q_gmap_gr_meta,
    int64_t packed_rows,
    int64_t outfeatures) {
  CHECK_CUDA(decoded_q_gmap_words); CHECK_CONTIGUOUS(decoded_q_gmap_words); CHECK_INT(decoded_q_gmap_words);
  CHECK_CUDA(expert_ids); CHECK_CONTIGUOUS(expert_ids); CHECK_INT(expert_ids);
  CHECK_CUDA(q_gmap_gr_words); CHECK_CONTIGUOUS(q_gmap_gr_words); CHECK_INT(q_gmap_gr_words);
  CHECK_CUDA(q_gmap_gr_offsets); CHECK_CONTIGUOUS(q_gmap_gr_offsets); CHECK_INT(q_gmap_gr_offsets);
  CHECK_CUDA(q_gmap_gr_meta); CHECK_CONTIGUOUS(q_gmap_gr_meta); CHECK_INT(q_gmap_gr_meta);

  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be 1D [active_experts]");
  TORCH_CHECK(q_gmap_gr_words.dim() == 2 && q_gmap_gr_offsets.dim() == 2 && q_gmap_gr_meta.dim() == 2, "compact3 grouped block-GR tensors must be 2D [experts, ...]");
  TORCH_CHECK(packed_rows >= 0 && outfeatures >= 0, "packed_rows/outfeatures must be non-negative");

  const int active_count = static_cast<int>(expert_ids.size(0));
  const int packed_rows_i = static_cast<int>(packed_rows);
  const int outfeatures_i = static_cast<int>(outfeatures);
  const int gmap_block_count = ceil_div_int(packed_rows_i, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_i, BLOCK_GR_TILE_OUT);

  TORCH_CHECK(decoded_q_gmap_words.dim() == 3, "decoded_q_gmap_words must be 3D [active, blocks, words]");
  TORCH_CHECK(decoded_q_gmap_words.size(0) == active_count, "decoded_q_gmap_words active count mismatch");
  TORCH_CHECK(decoded_q_gmap_words.size(1) == gmap_block_count, "decoded_q_gmap_words block count mismatch");
  TORCH_CHECK(decoded_q_gmap_words.size(2) == BLOCK_GR_WORDS_PER_BLOCK, "decoded_q_gmap_words word count mismatch");
  TORCH_CHECK(q_gmap_gr_words.size(0) == q_gmap_gr_offsets.size(0) && q_gmap_gr_words.size(0) == q_gmap_gr_meta.size(0), "compact3 grouped block-GR expert mismatch");
  TORCH_CHECK(q_gmap_gr_offsets.size(1) == gmap_block_count + 1, "compact3 grouped q_gmap_gr_offsets shape mismatch");
  TORCH_CHECK(q_gmap_gr_meta.size(1) == gmap_block_count, "compact3 grouped q_gmap_gr_meta shape mismatch");

  if (active_count == 0 || gmap_block_count == 0) {
    return decoded_q_gmap_words;
  }

  const int q_gmap_gr_words_stride = static_cast<int>(q_gmap_gr_words.size(1));
  dim3 grid(gmap_block_count, static_cast<unsigned int>(active_count));
  dim3 block(BLOCK_GR_WORDS_PER_BLOCK);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Compact3TriSMoEGroupedBlockGRPredecodeKernel<<<grid, block, 0, stream>>>(
      expert_ids.data_ptr<int>(),
      q_gmap_gr_words.data_ptr<int>(),
      q_gmap_gr_offsets.data_ptr<int>(),
      q_gmap_gr_meta.data_ptr<int>(),
      q_gmap_gr_words_stride,
      decoded_q_gmap_words.data_ptr<int>(),
      gmap_block_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return decoded_q_gmap_words;
}

template <bool UniformK64Groups>
torch::Tensor compact3_tris_precomputed_grouped_forward_block_gr_predecoded_cuda_impl(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor decoded_q_gmap_words,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3,
    bool tail_only) {
  CHECK_CUDA(hidden); CHECK_CONTIGUOUS(hidden); CHECK_HALF(hidden);
  CHECK_CUDA(expert_ids); CHECK_CONTIGUOUS(expert_ids); CHECK_INT(expert_ids);
  CHECK_CUDA(segment_offsets); CHECK_CONTIGUOUS(segment_offsets); CHECK_INT(segment_offsets);
  CHECK_CUDA(problem_offsets); CHECK_CONTIGUOUS(problem_offsets); CHECK_INT(problem_offsets);
  CHECK_CUDA(decoded_q_gmap_words); CHECK_CONTIGUOUS(decoded_q_gmap_words); CHECK_INT(decoded_q_gmap_words);
  CHECK_CUDA(qweight1); CHECK_CONTIGUOUS(qweight1); CHECK_INT(qweight1);
  CHECK_CUDA(qweight2_sparse); CHECK_CONTIGUOUS(qweight2_sparse); CHECK_INT(qweight2_sparse);
  CHECK_CUDA(qweight2_sparse_prefix); CHECK_CONTIGUOUS(qweight2_sparse_prefix); CHECK_INT(qweight2_sparse_prefix);
  CHECK_CUDA(c_map); CHECK_CONTIGUOUS(c_map); CHECK_INT(c_map);
  CHECK_CUDA(g_idx); CHECK_CONTIGUOUS(g_idx); CHECK_INT(g_idx);
  CHECK_CUDA(col_pack3); CHECK_CONTIGUOUS(col_pack3); CHECK_LONG(col_pack3);
  CHECK_CUDA(row_pack3); CHECK_CONTIGUOUS(row_pack3); CHECK_LONG(row_pack3);

  TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [batch, rank]");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be 1D [segments]");
  TORCH_CHECK(segment_offsets.dim() == 1 && segment_offsets.size(0) == expert_ids.size(0) + 1, "segment_offsets must be 1D [segments + 1]");
  TORCH_CHECK(problem_offsets.dim() == 1 && problem_offsets.size(0) == expert_ids.size(0) + 1, "problem_offsets must be 1D [segments + 1]");
  TORCH_CHECK(qweight1.dim() == 3 && qweight2_sparse.dim() == 3, "compact3 grouped payload tensors must be 3D");
  TORCH_CHECK(qweight2_sparse_prefix.dim() == 2, "compact3 grouped qweight2 sparse prefix tensor must be 2D");
  TORCH_CHECK(c_map.dim() == 2 && g_idx.dim() == 2, "compact3 grouped c_map/g_idx dims mismatch");
  TORCH_CHECK(col_pack3.dim() == 2, "compact3 grouped col_pack3 must be 2D");
  TORCH_CHECK(row_pack3.dim() == 3, "compact3 grouped row_pack3 must be 3D");

  const int batch = static_cast<int>(hidden.size(0));
  const int rank = static_cast<int>(hidden.size(1));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int num_experts = static_cast<int>(qweight1.size(0));
  const int packed_rows = (rank + 31) / 32;
  const int outfeatures = static_cast<int>(qweight1.size(2));
  const int qweight2_sparse_rows = static_cast<int>(qweight2_sparse.size(1));
  const int num_groups = static_cast<int>(row_pack3.size(2));
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);

  TORCH_CHECK(qweight1.size(1) == packed_rows, "compact3 grouped qweight1 packed height mismatch");
  TORCH_CHECK(qweight2_sparse.size(0) == num_experts && qweight2_sparse.size(2) == outfeatures, "compact3 grouped qweight2 shape mismatch");
  TORCH_CHECK(qweight2_sparse_prefix.size(0) == num_experts && qweight2_sparse_prefix.size(1) == packed_rows + 1, "compact3 grouped qweight2 prefix mismatch");
  TORCH_CHECK(c_map.size(0) == num_experts && c_map.size(1) == packed_rows, "compact3 grouped c_map mismatch");
  TORCH_CHECK(g_idx.size(0) == num_experts && g_idx.size(1) == rank, "compact3 grouped g_idx mismatch");
  TORCH_CHECK(col_pack3.size(0) == num_experts && col_pack3.size(1) == rank, "compact3 grouped col_pack3 mismatch");
  TORCH_CHECK(row_pack3.size(0) == num_experts && row_pack3.size(1) == outfeatures && row_pack3.size(2) == num_groups, "compact3 grouped row_pack3 mismatch");
  TORCH_CHECK(decoded_q_gmap_words.dim() == 3, "decoded_q_gmap_words must be 3D [segments, blocks, words]");
  TORCH_CHECK(decoded_q_gmap_words.size(0) == num_segments, "decoded_q_gmap_words segment mismatch");
  TORCH_CHECK(decoded_q_gmap_words.size(1) == gmap_block_count, "decoded_q_gmap_words block mismatch");
  TORCH_CHECK(decoded_q_gmap_words.size(2) == BLOCK_GR_WORDS_PER_BLOCK, "decoded_q_gmap_words word mismatch");

  auto output = torch::zeros({batch, outfeatures}, hidden.options());
  if (batch == 0 || rank == 0 || outfeatures == 0 || num_experts == 0 || num_segments == 0 || total_problem_tiles <= 0) {
    return output;
  }

  dim3 block(STRUCTURED_BLOCK_OUT);
  dim3 grid(ceil_div_int(outfeatures, STRUCTURED_BLOCK_OUT), static_cast<unsigned int>(total_problem_tiles));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (tail_only) {
    Compact3TriSMoEPrecomputedGroupedSegmentNoScaleNoBiasBlockGRPredecodedKernel<true, UniformK64Groups><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        problem_offsets.data_ptr<int>(),
        decoded_q_gmap_words.data_ptr<int>(),
        qweight1.data_ptr<int>(),
        qweight2_sparse.data_ptr<int>(),
        qweight2_sparse_prefix.data_ptr<int>(),
        c_map.data_ptr<int>(),
        g_idx.data_ptr<int>(),
        col_pack3.data_ptr<int64_t>(),
        row_pack3.data_ptr<int64_t>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        qweight2_sparse_rows,
        num_segments,
        rank,
        outfeatures,
        num_groups,
        gmap_block_count);
  } else {
    Compact3TriSMoEPrecomputedGroupedSegmentNoScaleNoBiasBlockGRPredecodedKernel<false, UniformK64Groups><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
      expert_ids.data_ptr<int>(),
      segment_offsets.data_ptr<int>(),
      problem_offsets.data_ptr<int>(),
      decoded_q_gmap_words.data_ptr<int>(),
      qweight1.data_ptr<int>(),
      qweight2_sparse.data_ptr<int>(),
      qweight2_sparse_prefix.data_ptr<int>(),
      c_map.data_ptr<int>(),
      g_idx.data_ptr<int>(),
      col_pack3.data_ptr<int64_t>(),
      row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      qweight2_sparse_rows,
      num_segments,
      rank,
      outfeatures,
      num_groups,
      gmap_block_count);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor compact3_tris_precomputed_grouped_forward_block_gr_predecoded_cuda(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor decoded_q_gmap_words,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3,
    bool tail_only,
    bool use_uniform_k64_groups) {
  if (use_uniform_k64_groups) {
    return compact3_tris_precomputed_grouped_forward_block_gr_predecoded_cuda_impl<true>(
        hidden,
        expert_ids,
        segment_offsets,
        problem_offsets,
        total_problem_tiles,
        decoded_q_gmap_words,
        qweight1,
        qweight2_sparse,
        qweight2_sparse_prefix,
        c_map,
        g_idx,
        col_pack3,
        row_pack3,
        tail_only);
  }
  return compact3_tris_precomputed_grouped_forward_block_gr_predecoded_cuda_impl<false>(
      hidden,
      expert_ids,
      segment_offsets,
      problem_offsets,
      total_problem_tiles,
      decoded_q_gmap_words,
      qweight1,
      qweight2_sparse,
      qweight2_sparse_prefix,
      c_map,
      g_idx,
      col_pack3,
      row_pack3,
      tail_only);
}

template <bool UniformK64Groups>
torch::Tensor compact3_tris_precomputed_grouped_forward_block_gr_predecoded_hybrid_cuda_impl(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor full_problem_offsets,
    int64_t full_problem_tiles,
    torch::Tensor tail_problem_offsets,
    int64_t tail_problem_tiles,
    torch::Tensor decoded_q_gmap_words,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3) {
  torch::Tensor output = compact3_tris_precomputed_grouped_forward_block_gr_predecoded_cuda_impl<UniformK64Groups>(
      hidden,
      expert_ids,
      segment_offsets,
      tail_problem_offsets,
      tail_problem_tiles,
      decoded_q_gmap_words,
      qweight1,
      qweight2_sparse,
      qweight2_sparse_prefix,
      c_map,
      g_idx,
      col_pack3,
      row_pack3,
      true);
  CHECK_CUDA(full_problem_offsets); CHECK_CONTIGUOUS(full_problem_offsets); CHECK_INT(full_problem_offsets);
  TORCH_CHECK(full_problem_offsets.dim() == 1 && full_problem_offsets.size(0) == expert_ids.size(0) + 1,
              "full_problem_offsets must be 1D [segments + 1]");
  if (full_problem_tiles <= 0 || hidden.numel() == 0 || qweight1.numel() == 0) {
    return output;
  }
#if defined(USE_ROCM)
  TORCH_CHECK(false, "grouped compact3 M16 Tensor Core prefill is unavailable on ROCm");
#else
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props != nullptr && props->major >= 7, "grouped compact3 M16 Tensor Core prefill requires sm70 or newer");
  const int rank = static_cast<int>(hidden.size(1));
  const int outfeatures = static_cast<int>(qweight1.size(2));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int qweight2_sparse_rows = static_cast<int>(qweight2_sparse.size(1));
  const int num_groups = static_cast<int>(row_pack3.size(2));
  const int packed_rows = (rank + 31) / 32;
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  dim3 block(STRUCTURED_TC_WARPS * 32);
  dim3 grid(ceil_div_int(outfeatures, STRUCTURED_TC_N_TILE), static_cast<unsigned int>(full_problem_tiles));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Compact3TriSMoEPrecomputedGroupedM16TensorCoreBlockGRPredecodedKernel<UniformK64Groups><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
      expert_ids.data_ptr<int>(),
      segment_offsets.data_ptr<int>(),
      full_problem_offsets.data_ptr<int>(),
      decoded_q_gmap_words.data_ptr<int>(),
      qweight1.data_ptr<int>(),
      qweight2_sparse.data_ptr<int>(),
      qweight2_sparse_prefix.data_ptr<int>(),
      c_map.data_ptr<int>(),
      g_idx.data_ptr<int>(),
      col_pack3.data_ptr<int64_t>(),
      row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      qweight2_sparse_rows,
      num_segments,
      rank,
      outfeatures,
      num_groups,
      gmap_block_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
#endif
}

torch::Tensor compact3_tris_precomputed_grouped_forward_block_gr_predecoded_hybrid_cuda(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor full_problem_offsets,
    int64_t full_problem_tiles,
    torch::Tensor tail_problem_offsets,
    int64_t tail_problem_tiles,
    torch::Tensor decoded_q_gmap_words,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3,
    bool use_uniform_k64_groups) {
  if (use_uniform_k64_groups) {
    return compact3_tris_precomputed_grouped_forward_block_gr_predecoded_hybrid_cuda_impl<true>(
        hidden,
        expert_ids,
        segment_offsets,
        full_problem_offsets,
        full_problem_tiles,
        tail_problem_offsets,
        tail_problem_tiles,
        decoded_q_gmap_words,
        qweight1,
        qweight2_sparse,
        qweight2_sparse_prefix,
        c_map,
        g_idx,
        col_pack3,
        row_pack3);
  }
  return compact3_tris_precomputed_grouped_forward_block_gr_predecoded_hybrid_cuda_impl<false>(
      hidden,
      expert_ids,
      segment_offsets,
      full_problem_offsets,
      full_problem_tiles,
      tail_problem_offsets,
      tail_problem_tiles,
      decoded_q_gmap_words,
      qweight1,
      qweight2_sparse,
      qweight2_sparse_prefix,
      c_map,
      g_idx,
      col_pack3,
      row_pack3);
}

template <bool UniformK64Groups>
std::vector<torch::Tensor> compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_cuda_impl(
    torch::Tensor hidden_a,
    torch::Tensor hidden_b,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor decoded_q_gmap_words_a,
    torch::Tensor qweight1_a,
    torch::Tensor qweight2_sparse_a,
    torch::Tensor qweight2_sparse_prefix_a,
    torch::Tensor c_map_a,
    torch::Tensor g_idx_a,
    torch::Tensor col_pack3_a,
    torch::Tensor row_pack3_a,
    torch::Tensor decoded_q_gmap_words_b,
    torch::Tensor qweight1_b,
    torch::Tensor qweight2_sparse_b,
    torch::Tensor qweight2_sparse_prefix_b,
    torch::Tensor c_map_b,
    torch::Tensor g_idx_b,
    torch::Tensor col_pack3_b,
    torch::Tensor row_pack3_b,
    bool tail_only) {
  CHECK_CUDA(hidden_a); CHECK_CONTIGUOUS(hidden_a); CHECK_HALF(hidden_a);
  CHECK_CUDA(hidden_b); CHECK_CONTIGUOUS(hidden_b); CHECK_HALF(hidden_b);
  CHECK_CUDA(expert_ids); CHECK_CONTIGUOUS(expert_ids); CHECK_INT(expert_ids);
  CHECK_CUDA(segment_offsets); CHECK_CONTIGUOUS(segment_offsets); CHECK_INT(segment_offsets);
  CHECK_CUDA(problem_offsets); CHECK_CONTIGUOUS(problem_offsets); CHECK_INT(problem_offsets);
  CHECK_CUDA(decoded_q_gmap_words_a); CHECK_CONTIGUOUS(decoded_q_gmap_words_a); CHECK_INT(decoded_q_gmap_words_a);
  CHECK_CUDA(decoded_q_gmap_words_b); CHECK_CONTIGUOUS(decoded_q_gmap_words_b); CHECK_INT(decoded_q_gmap_words_b);

  TORCH_CHECK(hidden_a.dim() == 2 && hidden_b.dim() == 2, "hidden tensors must be 2D [batch, rank]");
  TORCH_CHECK(hidden_a.size(0) == hidden_b.size(0), "hidden_a/hidden_b batch mismatch");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be 1D [segments]");
  TORCH_CHECK(segment_offsets.dim() == 1 && segment_offsets.size(0) == expert_ids.size(0) + 1, "segment_offsets must be 1D [segments + 1]");
  TORCH_CHECK(problem_offsets.dim() == 1 && problem_offsets.size(0) == expert_ids.size(0) + 1, "problem_offsets must be 1D [segments + 1]");

  auto check_grouped_payload = [](torch::Tensor qweight1,
                                  torch::Tensor qweight2_sparse,
                                  torch::Tensor qweight2_sparse_prefix,
                                  torch::Tensor c_map,
                                  torch::Tensor g_idx,
                                  torch::Tensor col_pack3,
                                  torch::Tensor row_pack3,
                                  int rank,
                                  const char* label) {
    CHECK_CUDA(qweight1); CHECK_CUDA(qweight2_sparse); CHECK_CUDA(qweight2_sparse_prefix); CHECK_CUDA(c_map); CHECK_CUDA(g_idx); CHECK_CUDA(col_pack3); CHECK_CUDA(row_pack3);
    CHECK_CONTIGUOUS(qweight1); CHECK_CONTIGUOUS(qweight2_sparse); CHECK_CONTIGUOUS(qweight2_sparse_prefix); CHECK_CONTIGUOUS(c_map); CHECK_CONTIGUOUS(g_idx); CHECK_CONTIGUOUS(col_pack3); CHECK_CONTIGUOUS(row_pack3);
    CHECK_INT(qweight1); CHECK_INT(qweight2_sparse); CHECK_INT(qweight2_sparse_prefix); CHECK_INT(c_map); CHECK_INT(g_idx); CHECK_LONG(col_pack3); CHECK_LONG(row_pack3);
    TORCH_CHECK(qweight1.dim() == 3 && qweight2_sparse.dim() == 3, label, " qweight dims mismatch");
    TORCH_CHECK(qweight2_sparse_prefix.dim() == 2, label, " qweight2_sparse_prefix dim mismatch");
    TORCH_CHECK(c_map.dim() == 2 && g_idx.dim() == 2, label, " c_map/g_idx dims mismatch");
    TORCH_CHECK(col_pack3.dim() == 2, label, " col_pack3 dim mismatch");
    TORCH_CHECK(row_pack3.dim() == 3, label, " row_pack3 dim mismatch");
    const int num_experts = static_cast<int>(qweight1.size(0));
    const int packed_rows = (rank + 31) / 32;
    const int outfeatures = static_cast<int>(qweight1.size(2));
    const int num_groups = static_cast<int>(row_pack3.size(2));
    TORCH_CHECK(qweight1.size(1) == packed_rows, label, " qweight1 packed height mismatch");
    TORCH_CHECK(qweight2_sparse.size(0) == num_experts && qweight2_sparse.size(2) == outfeatures, label, " qweight2_sparse shape mismatch");
    TORCH_CHECK(qweight2_sparse_prefix.size(0) == num_experts && qweight2_sparse_prefix.size(1) == packed_rows + 1, label, " qweight2_sparse_prefix shape mismatch");
    TORCH_CHECK(c_map.size(0) == num_experts && c_map.size(1) == packed_rows, label, " c_map shape mismatch");
    TORCH_CHECK(g_idx.size(0) == num_experts && g_idx.size(1) == rank, label, " g_idx shape mismatch");
    TORCH_CHECK(col_pack3.size(0) == num_experts && col_pack3.size(1) == rank, label, " col_pack3 shape mismatch");
    TORCH_CHECK(row_pack3.size(0) == num_experts && row_pack3.size(1) == outfeatures && row_pack3.size(2) == num_groups, label, " row_pack3 shape mismatch");
  };

  const int batch = static_cast<int>(hidden_a.size(0));
  const int rank_a = static_cast<int>(hidden_a.size(1));
  const int rank_b = static_cast<int>(hidden_b.size(1));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  check_grouped_payload(qweight1_a, qweight2_sparse_a, qweight2_sparse_prefix_a, c_map_a, g_idx_a, col_pack3_a, row_pack3_a, rank_a, "payload_a");
  check_grouped_payload(qweight1_b, qweight2_sparse_b, qweight2_sparse_prefix_b, c_map_b, g_idx_b, col_pack3_b, row_pack3_b, rank_b, "payload_b");

  const int outfeatures_a = static_cast<int>(qweight1_a.size(2));
  const int outfeatures_b = static_cast<int>(qweight1_b.size(2));
  const int packed_rows_a = (rank_a + 31) / 32;
  const int packed_rows_b = (rank_b + 31) / 32;
  const int gmap_block_count_a = ceil_div_int(packed_rows_a, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
  const int gmap_block_count_b = ceil_div_int(packed_rows_b, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
  TORCH_CHECK(decoded_q_gmap_words_a.dim() == 3, "decoded_q_gmap_words_a must be 3D [segments, blocks, words]");
  TORCH_CHECK(decoded_q_gmap_words_b.dim() == 3, "decoded_q_gmap_words_b must be 3D [segments, blocks, words]");
  TORCH_CHECK(decoded_q_gmap_words_a.size(0) == num_segments && decoded_q_gmap_words_a.size(1) == gmap_block_count_a && decoded_q_gmap_words_a.size(2) == BLOCK_GR_WORDS_PER_BLOCK,
              "decoded_q_gmap_words_a shape mismatch");
  TORCH_CHECK(decoded_q_gmap_words_b.size(0) == num_segments && decoded_q_gmap_words_b.size(1) == gmap_block_count_b && decoded_q_gmap_words_b.size(2) == BLOCK_GR_WORDS_PER_BLOCK,
              "decoded_q_gmap_words_b shape mismatch");

  auto output_a = torch::zeros({batch, outfeatures_a}, hidden_a.options());
  auto output_b = torch::zeros({batch, outfeatures_b}, hidden_b.options());
  if (batch == 0 || num_segments == 0 || ((rank_a == 0 || outfeatures_a == 0) && (rank_b == 0 || outfeatures_b == 0)) || total_problem_tiles <= 0) {
    return {output_a, output_b};
  }

  const int qweight2_sparse_rows_a = static_cast<int>(qweight2_sparse_a.size(1));
  const int qweight2_sparse_rows_b = static_cast<int>(qweight2_sparse_b.size(1));
  const int num_groups_a = static_cast<int>(row_pack3_a.size(2));
  const int num_groups_b = static_cast<int>(row_pack3_b.size(2));
  const int out_tiles_max = ceil_div_int((outfeatures_a > outfeatures_b ? outfeatures_a : outfeatures_b), STRUCTURED_BLOCK_OUT);
  dim3 block(STRUCTURED_BLOCK_OUT);
  dim3 grid(out_tiles_max, static_cast<unsigned int>(total_problem_tiles));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (tail_only) {
    Compact3TriSMoEPrecomputedGroupedDualSegmentNoScaleNoBiasBlockGRPredecodedKernel<true, UniformK64Groups><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden_a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(hidden_b.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(),
        segment_offsets.data_ptr<int>(),
        problem_offsets.data_ptr<int>(),
        decoded_q_gmap_words_a.data_ptr<int>(),
        qweight1_a.data_ptr<int>(),
        qweight2_sparse_a.data_ptr<int>(),
        qweight2_sparse_prefix_a.data_ptr<int>(),
        c_map_a.data_ptr<int>(),
        g_idx_a.data_ptr<int>(),
        col_pack3_a.data_ptr<int64_t>(),
        row_pack3_a.data_ptr<int64_t>(),
        decoded_q_gmap_words_b.data_ptr<int>(),
        qweight1_b.data_ptr<int>(),
        qweight2_sparse_b.data_ptr<int>(),
        qweight2_sparse_prefix_b.data_ptr<int>(),
        c_map_b.data_ptr<int>(),
        g_idx_b.data_ptr<int>(),
        col_pack3_b.data_ptr<int64_t>(),
        row_pack3_b.data_ptr<int64_t>(),
        reinterpret_cast<half*>(output_a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output_b.data_ptr<at::Half>()),
        qweight2_sparse_rows_a,
        qweight2_sparse_rows_b,
        num_segments,
        rank_a,
        rank_b,
        outfeatures_a,
        outfeatures_b,
        num_groups_a,
        num_groups_b,
        gmap_block_count_a,
        gmap_block_count_b);
  } else {
    Compact3TriSMoEPrecomputedGroupedDualSegmentNoScaleNoBiasBlockGRPredecodedKernel<false, UniformK64Groups><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(hidden_a.data_ptr<at::Half>()),
      reinterpret_cast<half*>(hidden_b.data_ptr<at::Half>()),
      expert_ids.data_ptr<int>(),
      segment_offsets.data_ptr<int>(),
      problem_offsets.data_ptr<int>(),
      decoded_q_gmap_words_a.data_ptr<int>(),
      qweight1_a.data_ptr<int>(),
      qweight2_sparse_a.data_ptr<int>(),
      qweight2_sparse_prefix_a.data_ptr<int>(),
      c_map_a.data_ptr<int>(),
      g_idx_a.data_ptr<int>(),
      col_pack3_a.data_ptr<int64_t>(),
      row_pack3_a.data_ptr<int64_t>(),
      decoded_q_gmap_words_b.data_ptr<int>(),
      qweight1_b.data_ptr<int>(),
      qweight2_sparse_b.data_ptr<int>(),
      qweight2_sparse_prefix_b.data_ptr<int>(),
      c_map_b.data_ptr<int>(),
      g_idx_b.data_ptr<int>(),
      col_pack3_b.data_ptr<int64_t>(),
      row_pack3_b.data_ptr<int64_t>(),
      reinterpret_cast<half*>(output_a.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output_b.data_ptr<at::Half>()),
      qweight2_sparse_rows_a,
      qweight2_sparse_rows_b,
      num_segments,
      rank_a,
      rank_b,
      outfeatures_a,
      outfeatures_b,
      num_groups_a,
      num_groups_b,
      gmap_block_count_a,
      gmap_block_count_b);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_a, output_b};
}

std::vector<torch::Tensor> compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_cuda(
    torch::Tensor hidden_a,
    torch::Tensor hidden_b,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor decoded_q_gmap_words_a,
    torch::Tensor qweight1_a,
    torch::Tensor qweight2_sparse_a,
    torch::Tensor qweight2_sparse_prefix_a,
    torch::Tensor c_map_a,
    torch::Tensor g_idx_a,
    torch::Tensor col_pack3_a,
    torch::Tensor row_pack3_a,
    torch::Tensor decoded_q_gmap_words_b,
    torch::Tensor qweight1_b,
    torch::Tensor qweight2_sparse_b,
    torch::Tensor qweight2_sparse_prefix_b,
    torch::Tensor c_map_b,
    torch::Tensor g_idx_b,
    torch::Tensor col_pack3_b,
    torch::Tensor row_pack3_b,
    bool tail_only,
    bool use_uniform_k64_groups) {
  if (use_uniform_k64_groups) {
    return compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_cuda_impl<true>(
        hidden_a,
        hidden_b,
        expert_ids,
        segment_offsets,
        problem_offsets,
        total_problem_tiles,
        decoded_q_gmap_words_a,
        qweight1_a,
        qweight2_sparse_a,
        qweight2_sparse_prefix_a,
        c_map_a,
        g_idx_a,
        col_pack3_a,
        row_pack3_a,
        decoded_q_gmap_words_b,
        qweight1_b,
        qweight2_sparse_b,
        qweight2_sparse_prefix_b,
        c_map_b,
        g_idx_b,
        col_pack3_b,
        row_pack3_b,
        tail_only);
  }
  return compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_cuda_impl<false>(
      hidden_a,
      hidden_b,
      expert_ids,
      segment_offsets,
      problem_offsets,
      total_problem_tiles,
      decoded_q_gmap_words_a,
      qweight1_a,
      qweight2_sparse_a,
      qweight2_sparse_prefix_a,
      c_map_a,
      g_idx_a,
      col_pack3_a,
      row_pack3_a,
      decoded_q_gmap_words_b,
      qweight1_b,
      qweight2_sparse_b,
      qweight2_sparse_prefix_b,
      c_map_b,
      g_idx_b,
      col_pack3_b,
      row_pack3_b,
      tail_only);
}

template <bool UniformK64Groups>
std::vector<torch::Tensor> compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_hybrid_cuda_impl(
    torch::Tensor hidden_a,
    torch::Tensor hidden_b,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor full_problem_offsets,
    int64_t full_problem_tiles,
    torch::Tensor tail_problem_offsets,
    int64_t tail_problem_tiles,
    torch::Tensor decoded_q_gmap_words_a,
    torch::Tensor qweight1_a,
    torch::Tensor qweight2_sparse_a,
    torch::Tensor qweight2_sparse_prefix_a,
    torch::Tensor c_map_a,
    torch::Tensor g_idx_a,
    torch::Tensor col_pack3_a,
    torch::Tensor row_pack3_a,
    torch::Tensor decoded_q_gmap_words_b,
    torch::Tensor qweight1_b,
    torch::Tensor qweight2_sparse_b,
    torch::Tensor qweight2_sparse_prefix_b,
    torch::Tensor c_map_b,
    torch::Tensor g_idx_b,
    torch::Tensor col_pack3_b,
    torch::Tensor row_pack3_b) {
  std::vector<torch::Tensor> outputs = compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_cuda_impl<UniformK64Groups>(
      hidden_a,
      hidden_b,
      expert_ids,
      segment_offsets,
      tail_problem_offsets,
      tail_problem_tiles,
      decoded_q_gmap_words_a,
      qweight1_a,
      qweight2_sparse_a,
      qweight2_sparse_prefix_a,
      c_map_a,
      g_idx_a,
      col_pack3_a,
      row_pack3_a,
      decoded_q_gmap_words_b,
      qweight1_b,
      qweight2_sparse_b,
      qweight2_sparse_prefix_b,
      c_map_b,
      g_idx_b,
      col_pack3_b,
      row_pack3_b,
      true);
  CHECK_CUDA(full_problem_offsets); CHECK_CONTIGUOUS(full_problem_offsets); CHECK_INT(full_problem_offsets);
  TORCH_CHECK(full_problem_offsets.dim() == 1 && full_problem_offsets.size(0) == expert_ids.size(0) + 1,
              "full_problem_offsets must be 1D [segments + 1]");
  if (full_problem_tiles <= 0) {
    return outputs;
  }
#if defined(USE_ROCM)
  TORCH_CHECK(false, "grouped compact3 dual M16 Tensor Core prefill is unavailable on ROCm");
#else
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props != nullptr && props->major >= 7, "grouped compact3 dual M16 Tensor Core prefill requires sm70 or newer");
  const int num_segments = static_cast<int>(expert_ids.size(0));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block(STRUCTURED_TC_WARPS * 32);

  const int rank_a = static_cast<int>(hidden_a.size(1));
  const int outfeatures_a = static_cast<int>(qweight1_a.size(2));
  if (rank_a > 0 && outfeatures_a > 0) {
    const int packed_rows_a = (rank_a + 31) / 32;
    const int gmap_block_count_a = ceil_div_int(packed_rows_a, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
    dim3 grid_a(ceil_div_int(outfeatures_a, STRUCTURED_TC_N_TILE), static_cast<unsigned int>(full_problem_tiles));
    Compact3TriSMoEPrecomputedGroupedM16TensorCoreBlockGRPredecodedKernel<UniformK64Groups><<<grid_a, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden_a.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(), segment_offsets.data_ptr<int>(), full_problem_offsets.data_ptr<int>(),
        decoded_q_gmap_words_a.data_ptr<int>(), qweight1_a.data_ptr<int>(), qweight2_sparse_a.data_ptr<int>(),
        qweight2_sparse_prefix_a.data_ptr<int>(), c_map_a.data_ptr<int>(), g_idx_a.data_ptr<int>(),
        col_pack3_a.data_ptr<int64_t>(), row_pack3_a.data_ptr<int64_t>(),
        reinterpret_cast<half*>(outputs[0].data_ptr<at::Half>()),
        static_cast<int>(qweight2_sparse_a.size(1)), num_segments, rank_a, outfeatures_a,
        static_cast<int>(row_pack3_a.size(2)), gmap_block_count_a);
  }

  const int rank_b = static_cast<int>(hidden_b.size(1));
  const int outfeatures_b = static_cast<int>(qweight1_b.size(2));
  if (rank_b > 0 && outfeatures_b > 0) {
    const int packed_rows_b = (rank_b + 31) / 32;
    const int gmap_block_count_b = ceil_div_int(packed_rows_b, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
    dim3 grid_b(ceil_div_int(outfeatures_b, STRUCTURED_TC_N_TILE), static_cast<unsigned int>(full_problem_tiles));
    Compact3TriSMoEPrecomputedGroupedM16TensorCoreBlockGRPredecodedKernel<UniformK64Groups><<<grid_b, block, 0, stream>>>(
        reinterpret_cast<half*>(hidden_b.data_ptr<at::Half>()),
        expert_ids.data_ptr<int>(), segment_offsets.data_ptr<int>(), full_problem_offsets.data_ptr<int>(),
        decoded_q_gmap_words_b.data_ptr<int>(), qweight1_b.data_ptr<int>(), qweight2_sparse_b.data_ptr<int>(),
        qweight2_sparse_prefix_b.data_ptr<int>(), c_map_b.data_ptr<int>(), g_idx_b.data_ptr<int>(),
        col_pack3_b.data_ptr<int64_t>(), row_pack3_b.data_ptr<int64_t>(),
        reinterpret_cast<half*>(outputs[1].data_ptr<at::Half>()),
        static_cast<int>(qweight2_sparse_b.size(1)), num_segments, rank_b, outfeatures_b,
        static_cast<int>(row_pack3_b.size(2)), gmap_block_count_b);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
#endif
}

std::vector<torch::Tensor> compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_hybrid_cuda(
    torch::Tensor hidden_a,
    torch::Tensor hidden_b,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor full_problem_offsets,
    int64_t full_problem_tiles,
    torch::Tensor tail_problem_offsets,
    int64_t tail_problem_tiles,
    torch::Tensor decoded_q_gmap_words_a,
    torch::Tensor qweight1_a,
    torch::Tensor qweight2_sparse_a,
    torch::Tensor qweight2_sparse_prefix_a,
    torch::Tensor c_map_a,
    torch::Tensor g_idx_a,
    torch::Tensor col_pack3_a,
    torch::Tensor row_pack3_a,
    torch::Tensor decoded_q_gmap_words_b,
    torch::Tensor qweight1_b,
    torch::Tensor qweight2_sparse_b,
    torch::Tensor qweight2_sparse_prefix_b,
    torch::Tensor c_map_b,
    torch::Tensor g_idx_b,
    torch::Tensor col_pack3_b,
    torch::Tensor row_pack3_b,
    bool use_uniform_k64_groups) {
  if (use_uniform_k64_groups) {
    return compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_hybrid_cuda_impl<true>(
        hidden_a,
        hidden_b,
        expert_ids,
        segment_offsets,
        full_problem_offsets,
        full_problem_tiles,
        tail_problem_offsets,
        tail_problem_tiles,
        decoded_q_gmap_words_a,
        qweight1_a,
        qweight2_sparse_a,
        qweight2_sparse_prefix_a,
        c_map_a,
        g_idx_a,
        col_pack3_a,
        row_pack3_a,
        decoded_q_gmap_words_b,
        qweight1_b,
        qweight2_sparse_b,
        qweight2_sparse_prefix_b,
        c_map_b,
        g_idx_b,
        col_pack3_b,
        row_pack3_b);
  }
  return compact3_tris_precomputed_grouped_dual_forward_block_gr_predecoded_hybrid_cuda_impl<false>(
      hidden_a,
      hidden_b,
      expert_ids,
      segment_offsets,
      full_problem_offsets,
      full_problem_tiles,
      tail_problem_offsets,
      tail_problem_tiles,
      decoded_q_gmap_words_a,
      qweight1_a,
      qweight2_sparse_a,
      qweight2_sparse_prefix_a,
      c_map_a,
      g_idx_a,
      col_pack3_a,
      row_pack3_a,
      decoded_q_gmap_words_b,
      qweight1_b,
      qweight2_sparse_b,
      qweight2_sparse_prefix_b,
      c_map_b,
      g_idx_b,
      col_pack3_b,
      row_pack3_b);
}

template <bool UniformK64Groups>
torch::Tensor compact3_tris_precomputed_grouped_forward_block_gr_cuda_impl(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor q_gmap_gr_words,
    torch::Tensor q_gmap_gr_offsets,
    torch::Tensor q_gmap_gr_meta,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3,
    torch::Tensor output_scale,
    bool has_output_scale,
    torch::Tensor bias,
    bool has_bias) {
  CHECK_CUDA(hidden); CHECK_CONTIGUOUS(hidden); CHECK_HALF(hidden);
  CHECK_CUDA(expert_ids); CHECK_CONTIGUOUS(expert_ids); CHECK_INT(expert_ids);
  CHECK_CUDA(segment_offsets); CHECK_CONTIGUOUS(segment_offsets); CHECK_INT(segment_offsets);
  CHECK_CUDA(problem_offsets); CHECK_CONTIGUOUS(problem_offsets); CHECK_INT(problem_offsets);
  CHECK_CUDA(qweight1); CHECK_CONTIGUOUS(qweight1); CHECK_INT(qweight1);
  CHECK_CUDA(qweight2_sparse); CHECK_CONTIGUOUS(qweight2_sparse); CHECK_INT(qweight2_sparse);
  CHECK_CUDA(qweight2_sparse_prefix); CHECK_CONTIGUOUS(qweight2_sparse_prefix); CHECK_INT(qweight2_sparse_prefix);
  CHECK_CUDA(q_gmap_gr_words); CHECK_CONTIGUOUS(q_gmap_gr_words); CHECK_INT(q_gmap_gr_words);
  CHECK_CUDA(q_gmap_gr_offsets); CHECK_CONTIGUOUS(q_gmap_gr_offsets); CHECK_INT(q_gmap_gr_offsets);
  CHECK_CUDA(q_gmap_gr_meta); CHECK_CONTIGUOUS(q_gmap_gr_meta); CHECK_INT(q_gmap_gr_meta);
  CHECK_CUDA(c_map); CHECK_CONTIGUOUS(c_map); CHECK_INT(c_map);
  CHECK_CUDA(g_idx); CHECK_CONTIGUOUS(g_idx); CHECK_INT(g_idx);
  CHECK_CUDA(col_pack3); CHECK_CONTIGUOUS(col_pack3); CHECK_LONG(col_pack3);
  CHECK_CUDA(row_pack3); CHECK_CONTIGUOUS(row_pack3); CHECK_LONG(row_pack3);
  CHECK_CUDA(output_scale); CHECK_CONTIGUOUS(output_scale); CHECK_HALF(output_scale);

  TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [batch, rank]");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be 1D [segments]");
  TORCH_CHECK(segment_offsets.dim() == 1 && segment_offsets.size(0) == expert_ids.size(0) + 1, "segment_offsets must be 1D [segments + 1]");
  TORCH_CHECK(problem_offsets.dim() == 1 && problem_offsets.size(0) == expert_ids.size(0) + 1, "problem_offsets must be 1D [segments + 1]");
  TORCH_CHECK(qweight1.dim() == 3 && qweight2_sparse.dim() == 3, "compact3 grouped payload tensors must be 3D");
  TORCH_CHECK(qweight2_sparse_prefix.dim() == 2, "compact3 grouped qweight2 sparse prefix tensor must be 2D");
  TORCH_CHECK(q_gmap_gr_words.dim() == 2 && q_gmap_gr_offsets.dim() == 2 && q_gmap_gr_meta.dim() == 2, "compact3 grouped block-GR tensors must be 2D [experts, ...]");
  TORCH_CHECK(c_map.dim() == 2 && g_idx.dim() == 2, "compact3 grouped c_map/g_idx dims mismatch");
  TORCH_CHECK(col_pack3.dim() == 2, "compact3 grouped col_pack3 must be 2D");
  TORCH_CHECK(row_pack3.dim() == 3, "compact3 grouped row_pack3 must be 3D");
  TORCH_CHECK(output_scale.dim() == 2, "compact3 grouped output_scale must be 2D");

  const int batch = static_cast<int>(hidden.size(0));
  const int rank = static_cast<int>(hidden.size(1));
  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int num_experts = static_cast<int>(qweight1.size(0));
  const int packed_rows = (rank + 31) / 32;
  const int outfeatures = static_cast<int>(qweight1.size(2));
  const int qweight2_sparse_rows = static_cast<int>(qweight2_sparse.size(1));
  const int num_groups = static_cast<int>(row_pack3.size(2));
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);

  TORCH_CHECK(qweight1.size(1) == packed_rows, "compact3 grouped qweight1 packed height mismatch");
  TORCH_CHECK(qweight2_sparse.size(0) == num_experts, "compact3 grouped qweight2 expert mismatch");
  TORCH_CHECK(qweight2_sparse.size(2) == outfeatures, "compact3 grouped qweight2 outfeature mismatch");
  TORCH_CHECK(qweight2_sparse_prefix.size(0) == num_experts && qweight2_sparse_prefix.size(1) == packed_rows + 1, "compact3 grouped qweight2 prefix mismatch");
  TORCH_CHECK(q_gmap_gr_words.size(0) == num_experts && q_gmap_gr_offsets.size(0) == num_experts && q_gmap_gr_meta.size(0) == num_experts, "compact3 grouped block-GR expert mismatch");
  TORCH_CHECK(q_gmap_gr_offsets.size(1) == gmap_block_count + 1, "compact3 grouped q_gmap_gr_offsets shape mismatch");
  TORCH_CHECK(q_gmap_gr_meta.size(1) == gmap_block_count, "compact3 grouped q_gmap_gr_meta shape mismatch");
  TORCH_CHECK(c_map.size(0) == num_experts && c_map.size(1) == packed_rows, "compact3 grouped c_map mismatch");
  TORCH_CHECK(g_idx.size(0) == num_experts && g_idx.size(1) == rank, "compact3 grouped g_idx mismatch");
  TORCH_CHECK(col_pack3.size(0) == num_experts && col_pack3.size(1) == rank, "compact3 grouped col_pack3 mismatch");
  TORCH_CHECK(row_pack3.size(0) == num_experts && row_pack3.size(1) == outfeatures && row_pack3.size(2) == num_groups, "compact3 grouped row_pack3 mismatch");
  TORCH_CHECK(output_scale.size(0) == num_experts && output_scale.size(1) == outfeatures, "compact3 grouped output_scale mismatch");
  if (has_bias) {
    CHECK_CUDA(bias); CHECK_CONTIGUOUS(bias); CHECK_HALF(bias);
    TORCH_CHECK(bias.dim() == 2 && bias.size(0) == num_experts && bias.size(1) == outfeatures, "compact3 grouped bias mismatch");
  }
  auto output = torch::zeros({batch, outfeatures}, hidden.options());
  if (batch == 0 || rank == 0 || outfeatures == 0 || num_experts == 0 || num_segments == 0 || total_problem_tiles <= 0) {
    return output;
  }

  const int q_gmap_gr_words_stride = static_cast<int>(q_gmap_gr_words.size(1));
  dim3 segment_block(STRUCTURED_BLOCK_OUT);
  dim3 decode_block(BLOCK_GR_TILE_OUT, COMPACT3_SINGLE_BLOCK_GR_K_WORKERS);
  dim3 decode_splitk_block(BLOCK_GR_TILE_OUT);
  const int out_tiles = ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  const bool use_decode_path = is_grouped_decode_problem(batch, num_segments, total_problem_tiles);
  const int split_k_slices = resolve_grouped_block_gr_split_k_slices(rank, static_cast<int>(total_problem_tiles), out_tiles, use_decode_path);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (split_k_slices > 1) {
    auto output_accum = torch::zeros({batch, outfeatures}, hidden.options().dtype(torch::kFloat32));
    dim3 splitk_grid(out_tiles, static_cast<unsigned int>(total_problem_tiles), static_cast<unsigned int>(split_k_slices));
    if (use_decode_path) {
      Compact3TriSMoEPrecomputedGroupedDecodeSplitKBlockGRKernel<UniformK64Groups><<<splitk_grid, decode_splitk_block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_gr_words.data_ptr<int>(),
          q_gmap_gr_offsets.data_ptr<int>(),
          q_gmap_gr_meta.data_ptr<int>(),
          q_gmap_gr_words_stride,
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          output_accum.data_ptr<float>(),
          qweight2_sparse_rows,
          num_segments,
          rank,
          outfeatures,
          num_groups,
          split_k_slices);
    } else {
      Compact3TriSMoEPrecomputedGroupedSegmentSplitKBlockGRKernel<UniformK64Groups><<<splitk_grid, segment_block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_gr_words.data_ptr<int>(),
          q_gmap_gr_offsets.data_ptr<int>(),
          q_gmap_gr_meta.data_ptr<int>(),
          q_gmap_gr_words_stride,
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          output_accum.data_ptr<float>(),
          qweight2_sparse_rows,
          num_segments,
          rank,
          outfeatures,
          num_groups,
          split_k_slices);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!has_output_scale && !has_bias) {
      const int finalize_threads = 256;
      const int finalize_blocks = (batch * outfeatures + finalize_threads - 1) / finalize_threads;
      Compact3TriSMoEGroupedSplitKFinalizeHalfKernel<<<finalize_blocks, finalize_threads, 0, stream>>>(
          output_accum.data_ptr<float>(),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          batch * outfeatures);
    } else {
      dim3 epilogue_block(32, 8);
      dim3 epilogue_grid((outfeatures + epilogue_block.x - 1) / epilogue_block.x, (batch + epilogue_block.y - 1) / epilogue_block.y);
      Compact3TriSMoEGroupedSplitKEpilogueKernel<<<epilogue_grid, epilogue_block, 0, stream>>>(
          output_accum.data_ptr<float>(),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
          has_bias ? reinterpret_cast<half*>(bias.data_ptr<at::Half>()) : nullptr,
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          batch,
          outfeatures,
          num_segments,
          has_bias);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  dim3 grid(out_tiles, static_cast<unsigned int>(total_problem_tiles));
  if (use_decode_path) {
    if (has_output_scale) {
      if (has_bias) {
        Compact3TriSMoEPrecomputedGroupedDecodeBlockGRKernel<true, true, UniformK64Groups><<<grid, decode_block, 0, stream>>>(
            reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            segment_offsets.data_ptr<int>(),
            problem_offsets.data_ptr<int>(),
            qweight1.data_ptr<int>(),
            qweight2_sparse.data_ptr<int>(),
            qweight2_sparse_prefix.data_ptr<int>(),
            q_gmap_gr_words.data_ptr<int>(),
            q_gmap_gr_offsets.data_ptr<int>(),
            q_gmap_gr_meta.data_ptr<int>(),
            q_gmap_gr_words_stride,
            c_map.data_ptr<int>(),
            g_idx.data_ptr<int>(),
            col_pack3.data_ptr<int64_t>(),
            row_pack3.data_ptr<int64_t>(),
            reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
            reinterpret_cast<half*>(bias.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            qweight2_sparse_rows,
            num_segments,
            rank,
            outfeatures,
            num_groups);
      } else {
        Compact3TriSMoEPrecomputedGroupedDecodeBlockGRKernel<true, false, UniformK64Groups><<<grid, decode_block, 0, stream>>>(
            reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            segment_offsets.data_ptr<int>(),
            problem_offsets.data_ptr<int>(),
            qweight1.data_ptr<int>(),
            qweight2_sparse.data_ptr<int>(),
            qweight2_sparse_prefix.data_ptr<int>(),
            q_gmap_gr_words.data_ptr<int>(),
            q_gmap_gr_offsets.data_ptr<int>(),
            q_gmap_gr_meta.data_ptr<int>(),
            q_gmap_gr_words_stride,
            c_map.data_ptr<int>(),
            g_idx.data_ptr<int>(),
            col_pack3.data_ptr<int64_t>(),
            row_pack3.data_ptr<int64_t>(),
            reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
            nullptr,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            qweight2_sparse_rows,
            num_segments,
            rank,
            outfeatures,
            num_groups);
      }
    } else if (has_bias) {
      Compact3TriSMoEPrecomputedGroupedDecodeBlockGRKernel<false, true, UniformK64Groups><<<grid, decode_block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_gr_words.data_ptr<int>(),
          q_gmap_gr_offsets.data_ptr<int>(),
          q_gmap_gr_meta.data_ptr<int>(),
          q_gmap_gr_words_stride,
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
          reinterpret_cast<half*>(bias.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          qweight2_sparse_rows,
          num_segments,
          rank,
          outfeatures,
          num_groups);
    } else {
      Compact3TriSMoEPrecomputedGroupedDecodeBlockGRKernel<false, false, UniformK64Groups><<<grid, decode_block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_gr_words.data_ptr<int>(),
          q_gmap_gr_offsets.data_ptr<int>(),
          q_gmap_gr_meta.data_ptr<int>(),
          q_gmap_gr_words_stride,
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
          nullptr,
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          qweight2_sparse_rows,
          num_segments,
          rank,
          outfeatures,
          num_groups);
    }
  } else {
    if (has_output_scale) {
      if (has_bias) {
        Compact3TriSMoEPrecomputedGroupedSegmentBlockGRKernel<true, true, UniformK64Groups><<<grid, segment_block, 0, stream>>>(
            reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            segment_offsets.data_ptr<int>(),
            problem_offsets.data_ptr<int>(),
            qweight1.data_ptr<int>(),
            qweight2_sparse.data_ptr<int>(),
            qweight2_sparse_prefix.data_ptr<int>(),
            q_gmap_gr_words.data_ptr<int>(),
            q_gmap_gr_offsets.data_ptr<int>(),
            q_gmap_gr_meta.data_ptr<int>(),
            q_gmap_gr_words_stride,
            c_map.data_ptr<int>(),
            g_idx.data_ptr<int>(),
            col_pack3.data_ptr<int64_t>(),
            row_pack3.data_ptr<int64_t>(),
            reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
            reinterpret_cast<half*>(bias.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            qweight2_sparse_rows,
            num_segments,
            rank,
            outfeatures,
            num_groups);
      } else {
        Compact3TriSMoEPrecomputedGroupedSegmentBlockGRKernel<true, false, UniformK64Groups><<<grid, segment_block, 0, stream>>>(
            reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
            expert_ids.data_ptr<int>(),
            segment_offsets.data_ptr<int>(),
            problem_offsets.data_ptr<int>(),
            qweight1.data_ptr<int>(),
            qweight2_sparse.data_ptr<int>(),
            qweight2_sparse_prefix.data_ptr<int>(),
            q_gmap_gr_words.data_ptr<int>(),
            q_gmap_gr_offsets.data_ptr<int>(),
            q_gmap_gr_meta.data_ptr<int>(),
            q_gmap_gr_words_stride,
            c_map.data_ptr<int>(),
            g_idx.data_ptr<int>(),
            col_pack3.data_ptr<int64_t>(),
            row_pack3.data_ptr<int64_t>(),
            reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
            nullptr,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            qweight2_sparse_rows,
            num_segments,
            rank,
            outfeatures,
            num_groups);
      }
    } else if (has_bias) {
      Compact3TriSMoEPrecomputedGroupedSegmentBlockGRKernel<false, true, UniformK64Groups><<<grid, segment_block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_gr_words.data_ptr<int>(),
          q_gmap_gr_offsets.data_ptr<int>(),
          q_gmap_gr_meta.data_ptr<int>(),
          q_gmap_gr_words_stride,
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
          reinterpret_cast<half*>(bias.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          qweight2_sparse_rows,
          num_segments,
          rank,
          outfeatures,
          num_groups);
    } else {
      Compact3TriSMoEPrecomputedGroupedSegmentBlockGRKernel<false, false, UniformK64Groups><<<grid, segment_block, 0, stream>>>(
          reinterpret_cast<half*>(hidden.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_gr_words.data_ptr<int>(),
          q_gmap_gr_offsets.data_ptr<int>(),
          q_gmap_gr_meta.data_ptr<int>(),
          q_gmap_gr_words_stride,
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
          nullptr,
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          qweight2_sparse_rows,
          num_segments,
          rank,
          outfeatures,
          num_groups);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor compact3_tris_precomputed_grouped_forward_block_gr_cuda(
    torch::Tensor hidden,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor q_gmap_gr_words,
    torch::Tensor q_gmap_gr_offsets,
    torch::Tensor q_gmap_gr_meta,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3,
    torch::Tensor output_scale,
    bool has_output_scale,
    torch::Tensor bias,
    bool has_bias,
    bool use_uniform_k64_groups) {
  if (use_uniform_k64_groups) {
    return compact3_tris_precomputed_grouped_forward_block_gr_cuda_impl<true>(
        hidden,
        expert_ids,
        segment_offsets,
        problem_offsets,
        total_problem_tiles,
        qweight1,
        qweight2_sparse,
        qweight2_sparse_prefix,
        q_gmap_gr_words,
        q_gmap_gr_offsets,
        q_gmap_gr_meta,
        c_map,
        g_idx,
        col_pack3,
        row_pack3,
        output_scale,
        has_output_scale,
        bias,
        has_bias);
  }
  return compact3_tris_precomputed_grouped_forward_block_gr_cuda_impl<false>(
      hidden,
      expert_ids,
      segment_offsets,
      problem_offsets,
      total_problem_tiles,
      qweight1,
      qweight2_sparse,
      qweight2_sparse_prefix,
      q_gmap_gr_words,
      q_gmap_gr_offsets,
      q_gmap_gr_meta,
      c_map,
      g_idx,
      col_pack3,
      row_pack3,
      output_scale,
      has_output_scale,
      bias,
      has_bias);
}

template <bool UniformK64Groups>
std::vector<torch::Tensor> compact3_tris_precomputed_grouped_dual_forward_block_gr_cuda_impl(
    torch::Tensor hidden_a,
    torch::Tensor hidden_b,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor qweight1_a,
    torch::Tensor qweight2_sparse_a,
    torch::Tensor qweight2_sparse_prefix_a,
    torch::Tensor q_gmap_gr_words_a,
    torch::Tensor q_gmap_gr_offsets_a,
    torch::Tensor q_gmap_gr_meta_a,
    torch::Tensor c_map_a,
    torch::Tensor g_idx_a,
    torch::Tensor col_pack3_a,
    torch::Tensor row_pack3_a,
    torch::Tensor output_scale_a,
    bool has_output_scale_a,
    torch::Tensor bias_a,
    torch::Tensor qweight1_b,
    torch::Tensor qweight2_sparse_b,
    torch::Tensor qweight2_sparse_prefix_b,
    torch::Tensor q_gmap_gr_words_b,
    torch::Tensor q_gmap_gr_offsets_b,
    torch::Tensor q_gmap_gr_meta_b,
    torch::Tensor c_map_b,
    torch::Tensor g_idx_b,
    torch::Tensor col_pack3_b,
    torch::Tensor row_pack3_b,
    torch::Tensor output_scale_b,
    bool has_output_scale_b,
    torch::Tensor bias_b) {
  CHECK_CUDA(hidden_a); CHECK_CONTIGUOUS(hidden_a); CHECK_HALF(hidden_a);
  CHECK_CUDA(hidden_b); CHECK_CONTIGUOUS(hidden_b); CHECK_HALF(hidden_b);
  CHECK_CUDA(expert_ids); CHECK_CONTIGUOUS(expert_ids); CHECK_INT(expert_ids);
  CHECK_CUDA(segment_offsets); CHECK_CONTIGUOUS(segment_offsets); CHECK_INT(segment_offsets);
  CHECK_CUDA(problem_offsets); CHECK_CONTIGUOUS(problem_offsets); CHECK_INT(problem_offsets);
  TORCH_CHECK(hidden_a.dim() == 2 && hidden_b.dim() == 2, "hidden tensors must be 2D [batch, rank]");
  TORCH_CHECK(hidden_a.size(0) == hidden_b.size(0), "hidden_a/hidden_b batch mismatch");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be 1D [segments]");
  TORCH_CHECK(segment_offsets.dim() == 1 && segment_offsets.size(0) == expert_ids.size(0) + 1, "segment_offsets must be 1D [segments + 1]");
  TORCH_CHECK(problem_offsets.dim() == 1 && problem_offsets.size(0) == expert_ids.size(0) + 1, "problem_offsets must be 1D [segments + 1]");

  auto check_grouped_payload = [](torch::Tensor qweight1,
                                  torch::Tensor qweight2_sparse,
                                  torch::Tensor qweight2_sparse_prefix,
                                  torch::Tensor q_gmap_gr_words,
                                  torch::Tensor q_gmap_gr_offsets,
                                  torch::Tensor q_gmap_gr_meta,
                                  torch::Tensor c_map,
                                  torch::Tensor g_idx,
                                  torch::Tensor col_pack3,
                                  torch::Tensor row_pack3,
                                  torch::Tensor output_scale,
                                  torch::Tensor bias,
                                  bool has_bias,
                                  int rank) {
    CHECK_CUDA(qweight1); CHECK_CUDA(qweight2_sparse); CHECK_CUDA(qweight2_sparse_prefix); CHECK_CUDA(q_gmap_gr_words); CHECK_CUDA(q_gmap_gr_offsets); CHECK_CUDA(q_gmap_gr_meta);
    CHECK_CUDA(c_map); CHECK_CUDA(g_idx); CHECK_CUDA(col_pack3); CHECK_CUDA(row_pack3); CHECK_CUDA(output_scale);
    CHECK_CONTIGUOUS(qweight1); CHECK_CONTIGUOUS(qweight2_sparse); CHECK_CONTIGUOUS(qweight2_sparse_prefix); CHECK_CONTIGUOUS(q_gmap_gr_words); CHECK_CONTIGUOUS(q_gmap_gr_offsets); CHECK_CONTIGUOUS(q_gmap_gr_meta);
    CHECK_CONTIGUOUS(c_map); CHECK_CONTIGUOUS(g_idx); CHECK_CONTIGUOUS(col_pack3); CHECK_CONTIGUOUS(row_pack3); CHECK_CONTIGUOUS(output_scale);
    CHECK_INT(qweight1); CHECK_INT(qweight2_sparse); CHECK_INT(qweight2_sparse_prefix); CHECK_INT(q_gmap_gr_words); CHECK_INT(q_gmap_gr_offsets); CHECK_INT(q_gmap_gr_meta);
    CHECK_INT(c_map); CHECK_INT(g_idx); CHECK_LONG(col_pack3); CHECK_LONG(row_pack3); CHECK_HALF(output_scale);
    TORCH_CHECK(qweight1.dim() == 3 && qweight2_sparse.dim() == 3, "compact3 grouped payload tensors must be 3D");
    TORCH_CHECK(qweight2_sparse_prefix.dim() == 2, "compact3 grouped qweight2 sparse prefix tensor must be 2D");
    TORCH_CHECK(q_gmap_gr_words.dim() == 2 && q_gmap_gr_offsets.dim() == 2 && q_gmap_gr_meta.dim() == 2, "compact3 grouped block-GR tensors must be 2D [experts, ...]");
    TORCH_CHECK(c_map.dim() == 2 && g_idx.dim() == 2, "compact3 grouped c_map/g_idx dims mismatch");
    TORCH_CHECK(col_pack3.dim() == 2, "compact3 grouped col_pack3 must be 2D");
    TORCH_CHECK(row_pack3.dim() == 3, "compact3 grouped row_pack3 must be 3D");
    TORCH_CHECK(output_scale.dim() == 2, "compact3 grouped output_scale must be 2D");
    const int num_experts = static_cast<int>(qweight1.size(0));
    const int packed_rows = (rank + 31) / 32;
    const int outfeatures = static_cast<int>(qweight1.size(2));
    const int qweight2_sparse_rows = static_cast<int>(qweight2_sparse.size(1));
    const int num_groups = static_cast<int>(row_pack3.size(2));
    const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
    TORCH_CHECK(qweight1.size(1) == packed_rows, "compact3 grouped qweight1 packed height mismatch");
    TORCH_CHECK(qweight2_sparse.size(0) == num_experts, "compact3 grouped qweight2 expert mismatch");
    TORCH_CHECK(qweight2_sparse.size(2) == outfeatures, "compact3 grouped qweight2 outfeature mismatch");
    TORCH_CHECK(qweight2_sparse_prefix.size(0) == num_experts && qweight2_sparse_prefix.size(1) == packed_rows + 1, "compact3 grouped qweight2 prefix mismatch");
    TORCH_CHECK(q_gmap_gr_words.size(0) == num_experts && q_gmap_gr_offsets.size(0) == num_experts && q_gmap_gr_meta.size(0) == num_experts, "compact3 grouped block-GR expert mismatch");
    TORCH_CHECK(q_gmap_gr_offsets.size(1) == gmap_block_count + 1, "compact3 grouped q_gmap_gr_offsets shape mismatch");
    TORCH_CHECK(q_gmap_gr_meta.size(1) == gmap_block_count, "compact3 grouped q_gmap_gr_meta shape mismatch");
    TORCH_CHECK(c_map.size(0) == num_experts && c_map.size(1) == packed_rows, "compact3 grouped c_map mismatch");
    TORCH_CHECK(g_idx.size(0) == num_experts && g_idx.size(1) == rank, "compact3 grouped g_idx mismatch");
    TORCH_CHECK(col_pack3.size(0) == num_experts && col_pack3.size(1) == rank, "compact3 grouped col_pack3 mismatch");
    TORCH_CHECK(row_pack3.size(0) == num_experts && row_pack3.size(1) == outfeatures && row_pack3.size(2) == num_groups, "compact3 grouped row_pack3 mismatch");
    TORCH_CHECK(output_scale.size(0) == num_experts && output_scale.size(1) == outfeatures, "compact3 grouped output_scale mismatch");
    if (has_bias) {
      CHECK_CUDA(bias); CHECK_CONTIGUOUS(bias); CHECK_HALF(bias);
      TORCH_CHECK(bias.dim() == 2 && bias.size(0) == num_experts && bias.size(1) == outfeatures, "compact3 grouped bias mismatch");
    }
  };

  const int batch = static_cast<int>(hidden_a.size(0));
  const int rank_a = static_cast<int>(hidden_a.size(1));
  const int rank_b = static_cast<int>(hidden_b.size(1));
  check_grouped_payload(qweight1_a, qweight2_sparse_a, qweight2_sparse_prefix_a, q_gmap_gr_words_a, q_gmap_gr_offsets_a, q_gmap_gr_meta_a, c_map_a, g_idx_a, col_pack3_a, row_pack3_a, output_scale_a, bias_a, bias_a.numel() > 0, rank_a);
  check_grouped_payload(qweight1_b, qweight2_sparse_b, qweight2_sparse_prefix_b, q_gmap_gr_words_b, q_gmap_gr_offsets_b, q_gmap_gr_meta_b, c_map_b, g_idx_b, col_pack3_b, row_pack3_b, output_scale_b, bias_b, bias_b.numel() > 0, rank_b);

  const int num_segments = static_cast<int>(expert_ids.size(0));
  const int outfeatures_a = static_cast<int>(qweight1_a.size(2));
  const int outfeatures_b = static_cast<int>(qweight1_b.size(2));
  const int outfeatures_max = outfeatures_a > outfeatures_b ? outfeatures_a : outfeatures_b;
  const int num_groups_a = static_cast<int>(row_pack3_a.size(2));
  const int num_groups_b = static_cast<int>(row_pack3_b.size(2));
  const int qweight2_sparse_rows_a = static_cast<int>(qweight2_sparse_a.size(1));
  const int qweight2_sparse_rows_b = static_cast<int>(qweight2_sparse_b.size(1));
  const int q_gmap_gr_words_stride_a = static_cast<int>(q_gmap_gr_words_a.size(1));
  const int q_gmap_gr_words_stride_b = static_cast<int>(q_gmap_gr_words_b.size(1));
  const int packed_rows_a = (rank_a + 31) / 32;
  const int packed_rows_b = (rank_b + 31) / 32;
  const int gmap_block_count_a = ceil_div_int(packed_rows_a, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_a, BLOCK_GR_TILE_OUT);
  const int gmap_block_count_b = ceil_div_int(packed_rows_b, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures_b, BLOCK_GR_TILE_OUT);
  const bool has_bias_a = bias_a.numel() > 0;
  const bool has_bias_b = bias_b.numel() > 0;
  auto output_a = torch::zeros({batch, outfeatures_a}, hidden_a.options());
  auto output_b = torch::zeros({batch, outfeatures_b}, hidden_b.options());
  if (batch == 0 || outfeatures_max == 0 || num_segments == 0 || total_problem_tiles <= 0 || ((rank_a == 0 || outfeatures_a == 0) && (rank_b == 0 || outfeatures_b == 0))) {
    return {output_a, output_b};
  }

  const int out_tiles_a = (outfeatures_a + STRUCTURED_BLOCK_OUT - 1) / STRUCTURED_BLOCK_OUT;
  const int out_tiles_b = (outfeatures_b + STRUCTURED_BLOCK_OUT - 1) / STRUCTURED_BLOCK_OUT;
  const int out_tiles_max = out_tiles_a > out_tiles_b ? out_tiles_a : out_tiles_b;
  const bool use_decode_path = is_grouped_decode_problem(batch, num_segments, total_problem_tiles);
  const int split_k_a = resolve_grouped_block_gr_split_k_slices(rank_a, static_cast<int>(total_problem_tiles), out_tiles_max, use_decode_path);
  const int split_k_b = resolve_grouped_block_gr_split_k_slices(rank_b, static_cast<int>(total_problem_tiles), out_tiles_max, use_decode_path);
  const int split_k_slices = split_k_a > split_k_b ? split_k_a : split_k_b;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (split_k_slices > 1) {
    auto output_accum_a = torch::zeros({batch, outfeatures_a}, hidden_a.options().dtype(torch::kFloat32));
    auto output_accum_b = torch::zeros({batch, outfeatures_b}, hidden_b.options().dtype(torch::kFloat32));
    dim3 block(STRUCTURED_BLOCK_OUT);
    dim3 splitk_grid(out_tiles_max, static_cast<unsigned int>(total_problem_tiles), static_cast<unsigned int>(split_k_slices));
    if (use_decode_path) {
      Compact3TriSMoEPrecomputedGroupedDualDecodeSplitKBlockGRKernel<UniformK64Groups><<<splitk_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(hidden_b.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1_a.data_ptr<int>(),
          qweight2_sparse_a.data_ptr<int>(),
          qweight2_sparse_prefix_a.data_ptr<int>(),
          q_gmap_gr_words_a.data_ptr<int>(),
          q_gmap_gr_offsets_a.data_ptr<int>(),
          q_gmap_gr_meta_a.data_ptr<int>(),
          q_gmap_gr_words_stride_a,
          c_map_a.data_ptr<int>(),
          g_idx_a.data_ptr<int>(),
          col_pack3_a.data_ptr<int64_t>(),
          row_pack3_a.data_ptr<int64_t>(),
          qweight1_b.data_ptr<int>(),
          qweight2_sparse_b.data_ptr<int>(),
          qweight2_sparse_prefix_b.data_ptr<int>(),
          q_gmap_gr_words_b.data_ptr<int>(),
          q_gmap_gr_offsets_b.data_ptr<int>(),
          q_gmap_gr_meta_b.data_ptr<int>(),
          q_gmap_gr_words_stride_b,
          c_map_b.data_ptr<int>(),
          g_idx_b.data_ptr<int>(),
          col_pack3_b.data_ptr<int64_t>(),
          row_pack3_b.data_ptr<int64_t>(),
          output_accum_a.data_ptr<float>(),
          output_accum_b.data_ptr<float>(),
          qweight2_sparse_rows_a,
          qweight2_sparse_rows_b,
          num_segments,
          rank_a,
          rank_b,
          outfeatures_a,
          outfeatures_b,
          num_groups_a,
          num_groups_b,
          split_k_slices);
    } else {
      Compact3TriSMoEPrecomputedGroupedDualSegmentSplitKBlockGRKernel<UniformK64Groups><<<splitk_grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(hidden_b.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1_a.data_ptr<int>(),
          qweight2_sparse_a.data_ptr<int>(),
          qweight2_sparse_prefix_a.data_ptr<int>(),
          q_gmap_gr_words_a.data_ptr<int>(),
          q_gmap_gr_offsets_a.data_ptr<int>(),
          q_gmap_gr_meta_a.data_ptr<int>(),
          q_gmap_gr_words_stride_a,
          c_map_a.data_ptr<int>(),
          g_idx_a.data_ptr<int>(),
          col_pack3_a.data_ptr<int64_t>(),
          row_pack3_a.data_ptr<int64_t>(),
          qweight1_b.data_ptr<int>(),
          qweight2_sparse_b.data_ptr<int>(),
          qweight2_sparse_prefix_b.data_ptr<int>(),
          q_gmap_gr_words_b.data_ptr<int>(),
          q_gmap_gr_offsets_b.data_ptr<int>(),
          q_gmap_gr_meta_b.data_ptr<int>(),
          q_gmap_gr_words_stride_b,
          c_map_b.data_ptr<int>(),
          g_idx_b.data_ptr<int>(),
          col_pack3_b.data_ptr<int64_t>(),
          row_pack3_b.data_ptr<int64_t>(),
          output_accum_a.data_ptr<float>(),
          output_accum_b.data_ptr<float>(),
          qweight2_sparse_rows_a,
          qweight2_sparse_rows_b,
          num_segments,
          rank_a,
          rank_b,
          outfeatures_a,
          outfeatures_b,
          num_groups_a,
          num_groups_b,
          split_k_slices);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const bool all_no_scale_no_bias = !has_output_scale_a && !has_bias_a && !has_output_scale_b && !has_bias_b;
    if (all_no_scale_no_bias) {
      const int finalize_threads = 256;
      if (outfeatures_a > 0) {
        const int finalize_blocks_a = (batch * outfeatures_a + finalize_threads - 1) / finalize_threads;
        Compact3TriSMoEGroupedSplitKFinalizeHalfKernel<<<finalize_blocks_a, finalize_threads, 0, stream>>>(
            output_accum_a.data_ptr<float>(),
            reinterpret_cast<half*>(output_a.data_ptr<at::Half>()),
            batch * outfeatures_a);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      if (outfeatures_b > 0) {
        const int finalize_blocks_b = (batch * outfeatures_b + finalize_threads - 1) / finalize_threads;
        Compact3TriSMoEGroupedSplitKFinalizeHalfKernel<<<finalize_blocks_b, finalize_threads, 0, stream>>>(
            output_accum_b.data_ptr<float>(),
            reinterpret_cast<half*>(output_b.data_ptr<at::Half>()),
            batch * outfeatures_b);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      dim3 epilogue_block(32, 8);
      if (outfeatures_a > 0) {
        dim3 epilogue_grid_a((outfeatures_a + epilogue_block.x - 1) / epilogue_block.x, (batch + epilogue_block.y - 1) / epilogue_block.y);
        Compact3TriSMoEGroupedSplitKEpilogueKernel<<<epilogue_grid_a, epilogue_block, 0, stream>>>(
            output_accum_a.data_ptr<float>(),
            expert_ids.data_ptr<int>(),
            segment_offsets.data_ptr<int>(),
            reinterpret_cast<half*>(output_scale_a.data_ptr<at::Half>()),
            has_bias_a ? reinterpret_cast<half*>(bias_a.data_ptr<at::Half>()) : nullptr,
            reinterpret_cast<half*>(output_a.data_ptr<at::Half>()),
            batch,
            outfeatures_a,
            num_segments,
            has_bias_a);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      if (outfeatures_b > 0) {
        dim3 epilogue_grid_b((outfeatures_b + epilogue_block.x - 1) / epilogue_block.x, (batch + epilogue_block.y - 1) / epilogue_block.y);
        Compact3TriSMoEGroupedSplitKEpilogueKernel<<<epilogue_grid_b, epilogue_block, 0, stream>>>(
            output_accum_b.data_ptr<float>(),
            expert_ids.data_ptr<int>(),
            segment_offsets.data_ptr<int>(),
            reinterpret_cast<half*>(output_scale_b.data_ptr<at::Half>()),
            has_bias_b ? reinterpret_cast<half*>(bias_b.data_ptr<at::Half>()) : nullptr,
            reinterpret_cast<half*>(output_b.data_ptr<at::Half>()),
            batch,
            outfeatures_b,
            num_segments,
            has_bias_b);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
    return {output_a, output_b};
  }

  dim3 block(STRUCTURED_BLOCK_OUT);
  dim3 grid(out_tiles_max, static_cast<unsigned int>(total_problem_tiles));
  const bool all_no_scale_no_bias = !has_output_scale_a && !has_bias_a && !has_output_scale_b && !has_bias_b;
  if (use_decode_path) {
    if (all_no_scale_no_bias) {
      Compact3TriSMoEPrecomputedGroupedDualDecodeNoScaleNoBiasBlockGRKernel<UniformK64Groups><<<grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(hidden_b.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1_a.data_ptr<int>(),
          qweight2_sparse_a.data_ptr<int>(),
          qweight2_sparse_prefix_a.data_ptr<int>(),
          q_gmap_gr_words_a.data_ptr<int>(),
          q_gmap_gr_offsets_a.data_ptr<int>(),
          q_gmap_gr_meta_a.data_ptr<int>(),
          q_gmap_gr_words_stride_a,
          c_map_a.data_ptr<int>(),
          g_idx_a.data_ptr<int>(),
          col_pack3_a.data_ptr<int64_t>(),
          row_pack3_a.data_ptr<int64_t>(),
          qweight1_b.data_ptr<int>(),
          qweight2_sparse_b.data_ptr<int>(),
          qweight2_sparse_prefix_b.data_ptr<int>(),
          q_gmap_gr_words_b.data_ptr<int>(),
          q_gmap_gr_offsets_b.data_ptr<int>(),
          q_gmap_gr_meta_b.data_ptr<int>(),
          q_gmap_gr_words_stride_b,
          c_map_b.data_ptr<int>(),
          g_idx_b.data_ptr<int>(),
          col_pack3_b.data_ptr<int64_t>(),
          row_pack3_b.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output_b.data_ptr<at::Half>()),
          qweight2_sparse_rows_a,
          qweight2_sparse_rows_b,
          num_segments,
          rank_a,
          rank_b,
          outfeatures_a,
          outfeatures_b,
          num_groups_a,
          num_groups_b);
    } else {
      Compact3TriSMoEPrecomputedGroupedDualDecodeBlockGRKernel<UniformK64Groups><<<grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(hidden_b.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1_a.data_ptr<int>(),
          qweight2_sparse_a.data_ptr<int>(),
          qweight2_sparse_prefix_a.data_ptr<int>(),
          q_gmap_gr_words_a.data_ptr<int>(),
          q_gmap_gr_offsets_a.data_ptr<int>(),
          q_gmap_gr_meta_a.data_ptr<int>(),
          q_gmap_gr_words_stride_a,
          c_map_a.data_ptr<int>(),
          g_idx_a.data_ptr<int>(),
          col_pack3_a.data_ptr<int64_t>(),
          row_pack3_a.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_scale_a.data_ptr<at::Half>()),
          has_bias_a ? reinterpret_cast<half*>(bias_a.data_ptr<at::Half>()) : nullptr,
          qweight1_b.data_ptr<int>(),
          qweight2_sparse_b.data_ptr<int>(),
          qweight2_sparse_prefix_b.data_ptr<int>(),
          q_gmap_gr_words_b.data_ptr<int>(),
          q_gmap_gr_offsets_b.data_ptr<int>(),
          q_gmap_gr_meta_b.data_ptr<int>(),
          q_gmap_gr_words_stride_b,
          c_map_b.data_ptr<int>(),
          g_idx_b.data_ptr<int>(),
          col_pack3_b.data_ptr<int64_t>(),
          row_pack3_b.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_scale_b.data_ptr<at::Half>()),
          has_bias_b ? reinterpret_cast<half*>(bias_b.data_ptr<at::Half>()) : nullptr,
          reinterpret_cast<half*>(output_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output_b.data_ptr<at::Half>()),
          qweight2_sparse_rows_a,
          qweight2_sparse_rows_b,
          num_segments,
          rank_a,
          rank_b,
          outfeatures_a,
          outfeatures_b,
          num_groups_a,
          num_groups_b,
          has_bias_a,
          has_bias_b);
    }
  } else {
    if (all_no_scale_no_bias) {
      Compact3TriSMoEPrecomputedGroupedDualSegmentNoScaleNoBiasBlockGRKernel<UniformK64Groups><<<grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(hidden_b.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1_a.data_ptr<int>(),
          qweight2_sparse_a.data_ptr<int>(),
          qweight2_sparse_prefix_a.data_ptr<int>(),
          q_gmap_gr_words_a.data_ptr<int>(),
          q_gmap_gr_offsets_a.data_ptr<int>(),
          q_gmap_gr_meta_a.data_ptr<int>(),
          q_gmap_gr_words_stride_a,
          c_map_a.data_ptr<int>(),
          g_idx_a.data_ptr<int>(),
          col_pack3_a.data_ptr<int64_t>(),
          row_pack3_a.data_ptr<int64_t>(),
          qweight1_b.data_ptr<int>(),
          qweight2_sparse_b.data_ptr<int>(),
          qweight2_sparse_prefix_b.data_ptr<int>(),
          q_gmap_gr_words_b.data_ptr<int>(),
          q_gmap_gr_offsets_b.data_ptr<int>(),
          q_gmap_gr_meta_b.data_ptr<int>(),
          q_gmap_gr_words_stride_b,
          c_map_b.data_ptr<int>(),
          g_idx_b.data_ptr<int>(),
          col_pack3_b.data_ptr<int64_t>(),
          row_pack3_b.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output_b.data_ptr<at::Half>()),
          qweight2_sparse_rows_a,
          qweight2_sparse_rows_b,
          num_segments,
          rank_a,
          rank_b,
          outfeatures_a,
          outfeatures_b,
          num_groups_a,
          num_groups_b);
    } else {
      Compact3TriSMoEPrecomputedGroupedDualSegmentBlockGRKernel<UniformK64Groups><<<grid, block, 0, stream>>>(
          reinterpret_cast<half*>(hidden_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(hidden_b.data_ptr<at::Half>()),
          expert_ids.data_ptr<int>(),
          segment_offsets.data_ptr<int>(),
          problem_offsets.data_ptr<int>(),
          qweight1_a.data_ptr<int>(),
          qweight2_sparse_a.data_ptr<int>(),
          qweight2_sparse_prefix_a.data_ptr<int>(),
          q_gmap_gr_words_a.data_ptr<int>(),
          q_gmap_gr_offsets_a.data_ptr<int>(),
          q_gmap_gr_meta_a.data_ptr<int>(),
          q_gmap_gr_words_stride_a,
          c_map_a.data_ptr<int>(),
          g_idx_a.data_ptr<int>(),
          col_pack3_a.data_ptr<int64_t>(),
          row_pack3_a.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_scale_a.data_ptr<at::Half>()),
          has_bias_a ? reinterpret_cast<half*>(bias_a.data_ptr<at::Half>()) : nullptr,
          qweight1_b.data_ptr<int>(),
          qweight2_sparse_b.data_ptr<int>(),
          qweight2_sparse_prefix_b.data_ptr<int>(),
          q_gmap_gr_words_b.data_ptr<int>(),
          q_gmap_gr_offsets_b.data_ptr<int>(),
          q_gmap_gr_meta_b.data_ptr<int>(),
          q_gmap_gr_words_stride_b,
          c_map_b.data_ptr<int>(),
          g_idx_b.data_ptr<int>(),
          col_pack3_b.data_ptr<int64_t>(),
          row_pack3_b.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output_scale_b.data_ptr<at::Half>()),
          has_bias_b ? reinterpret_cast<half*>(bias_b.data_ptr<at::Half>()) : nullptr,
          reinterpret_cast<half*>(output_a.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output_b.data_ptr<at::Half>()),
          qweight2_sparse_rows_a,
          qweight2_sparse_rows_b,
          num_segments,
          rank_a,
          rank_b,
          outfeatures_a,
          outfeatures_b,
          num_groups_a,
          num_groups_b,
          has_bias_a,
          has_bias_b);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_a, output_b};
}

std::vector<torch::Tensor> compact3_tris_precomputed_grouped_dual_forward_block_gr_cuda(
    torch::Tensor hidden_a,
    torch::Tensor hidden_b,
    torch::Tensor expert_ids,
    torch::Tensor segment_offsets,
    torch::Tensor problem_offsets,
    int64_t total_problem_tiles,
    torch::Tensor qweight1_a,
    torch::Tensor qweight2_sparse_a,
    torch::Tensor qweight2_sparse_prefix_a,
    torch::Tensor q_gmap_gr_words_a,
    torch::Tensor q_gmap_gr_offsets_a,
    torch::Tensor q_gmap_gr_meta_a,
    torch::Tensor c_map_a,
    torch::Tensor g_idx_a,
    torch::Tensor col_pack3_a,
    torch::Tensor row_pack3_a,
    torch::Tensor output_scale_a,
    bool has_output_scale_a,
    torch::Tensor bias_a,
    torch::Tensor qweight1_b,
    torch::Tensor qweight2_sparse_b,
    torch::Tensor qweight2_sparse_prefix_b,
    torch::Tensor q_gmap_gr_words_b,
    torch::Tensor q_gmap_gr_offsets_b,
    torch::Tensor q_gmap_gr_meta_b,
    torch::Tensor c_map_b,
    torch::Tensor g_idx_b,
    torch::Tensor col_pack3_b,
    torch::Tensor row_pack3_b,
    torch::Tensor output_scale_b,
    bool has_output_scale_b,
    torch::Tensor bias_b,
    bool use_uniform_k64_groups) {
  if (use_uniform_k64_groups) {
    return compact3_tris_precomputed_grouped_dual_forward_block_gr_cuda_impl<true>(
        hidden_a,
        hidden_b,
        expert_ids,
        segment_offsets,
        problem_offsets,
        total_problem_tiles,
        qweight1_a,
        qweight2_sparse_a,
        qweight2_sparse_prefix_a,
        q_gmap_gr_words_a,
        q_gmap_gr_offsets_a,
        q_gmap_gr_meta_a,
        c_map_a,
        g_idx_a,
        col_pack3_a,
        row_pack3_a,
        output_scale_a,
        has_output_scale_a,
        bias_a,
        qweight1_b,
        qweight2_sparse_b,
        qweight2_sparse_prefix_b,
        q_gmap_gr_words_b,
        q_gmap_gr_offsets_b,
        q_gmap_gr_meta_b,
        c_map_b,
        g_idx_b,
        col_pack3_b,
        row_pack3_b,
        output_scale_b,
        has_output_scale_b,
        bias_b);
  }
  return compact3_tris_precomputed_grouped_dual_forward_block_gr_cuda_impl<false>(
      hidden_a,
      hidden_b,
      expert_ids,
      segment_offsets,
      problem_offsets,
      total_problem_tiles,
      qweight1_a,
      qweight2_sparse_a,
      qweight2_sparse_prefix_a,
      q_gmap_gr_words_a,
      q_gmap_gr_offsets_a,
      q_gmap_gr_meta_a,
      c_map_a,
      g_idx_a,
      col_pack3_a,
      row_pack3_a,
      output_scale_a,
      has_output_scale_a,
      bias_a,
      qweight1_b,
      qweight2_sparse_b,
      qweight2_sparse_prefix_b,
      q_gmap_gr_words_b,
      q_gmap_gr_offsets_b,
      q_gmap_gr_meta_b,
      c_map_b,
      g_idx_b,
      col_pack3_b,
      row_pack3_b,
      output_scale_b,
      has_output_scale_b,
      bias_b);
}

template <bool UniformK64Groups>
torch::Tensor compact3_linear_forward_cuda_impl(
    torch::Tensor input,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor q_gmap_sparse,
    torch::Tensor q_gmap_sparse_prefix,
    torch::Tensor q_gmap_gr_words,
    torch::Tensor q_gmap_gr_offsets,
    torch::Tensor q_gmap_gr_meta,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3,
    torch::Tensor input_scale,
    torch::Tensor output_scale,
    torch::Tensor bias,
    bool has_input_scale,
    bool has_output_scale,
    bool has_bias,
    torch::Tensor ns_c1_qweight,
    torch::Tensor ns_c1_qzeros,
    torch::Tensor ns_c1_scales,
    torch::Tensor ns_c1_g_idx,
    torch::Tensor ns_c2_qweight,
    torch::Tensor ns_c2_qzeros,
    torch::Tensor ns_c2_scales,
    torch::Tensor ns_c2_g_idx,
    bool has_ns_matrix,
    int split_k_slices,
    bool request_prefill_tensor_core) {
  CHECK_CUDA(input); CHECK_CONTIGUOUS(input); CHECK_HALF(input);
  CHECK_CUDA(qweight1); CHECK_CONTIGUOUS(qweight1); CHECK_INT(qweight1);
  CHECK_CUDA(qweight2_sparse); CHECK_CONTIGUOUS(qweight2_sparse); CHECK_INT(qweight2_sparse);
  CHECK_CUDA(qweight2_sparse_prefix); CHECK_CONTIGUOUS(qweight2_sparse_prefix); CHECK_INT(qweight2_sparse_prefix);
  CHECK_CUDA(q_gmap_sparse); CHECK_CONTIGUOUS(q_gmap_sparse); CHECK_INT(q_gmap_sparse);
  CHECK_CUDA(q_gmap_sparse_prefix); CHECK_CONTIGUOUS(q_gmap_sparse_prefix); CHECK_INT(q_gmap_sparse_prefix);
  CHECK_CUDA(q_gmap_gr_words); CHECK_CONTIGUOUS(q_gmap_gr_words); CHECK_INT(q_gmap_gr_words);
  CHECK_CUDA(q_gmap_gr_offsets); CHECK_CONTIGUOUS(q_gmap_gr_offsets); CHECK_INT(q_gmap_gr_offsets);
  CHECK_CUDA(q_gmap_gr_meta); CHECK_CONTIGUOUS(q_gmap_gr_meta); CHECK_INT(q_gmap_gr_meta);
  CHECK_CUDA(c_map); CHECK_CONTIGUOUS(c_map); CHECK_INT(c_map);
  CHECK_CUDA(g_idx); CHECK_CONTIGUOUS(g_idx); CHECK_INT(g_idx);
  CHECK_CUDA(col_pack3); CHECK_CONTIGUOUS(col_pack3); CHECK_LONG(col_pack3);
  CHECK_CUDA(row_pack3); CHECK_CONTIGUOUS(row_pack3); CHECK_LONG(row_pack3);
  TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, rank]");
  TORCH_CHECK(qweight1.dim() == 2 && qweight2_sparse.dim() == 2 && q_gmap_sparse.dim() == 2, "packed matrices must be 2D");
  TORCH_CHECK(q_gmap_gr_words.dim() == 1 && q_gmap_gr_offsets.dim() == 1 && q_gmap_gr_meta.dim() == 1, "block-GR gmap tensors must be 1D");
  TORCH_CHECK(c_map.dim() == 1, "c_map must be 1D");
  TORCH_CHECK(g_idx.dim() == 1, "g_idx must be 1D");
  TORCH_CHECK(col_pack3.dim() == 1, "col_pack3 must be 1D");
  TORCH_CHECK(row_pack3.dim() == 2, "row_pack3 must be 2D");
  const int batch = static_cast<int>(input.size(0));
  const int rank = static_cast<int>(input.size(1));
  const int packed_rows = (rank + 31) / 32;
  const int outfeatures = static_cast<int>(qweight1.size(1));
  const int num_groups = static_cast<int>(row_pack3.size(1));
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);
  TORCH_CHECK(qweight1.size(0) == packed_rows, "qweight1 packed height mismatch");
  TORCH_CHECK(c_map.size(0) == packed_rows, "c_map packed length mismatch");
  TORCH_CHECK(g_idx.size(0) == rank, "g_idx rank mismatch");
  TORCH_CHECK(col_pack3.size(0) == rank, "col_pack3 rank mismatch");
  TORCH_CHECK(row_pack3.size(0) == outfeatures, "row_pack3 outfeature mismatch");
  TORCH_CHECK(qweight2_sparse_prefix.dim() == 1 && qweight2_sparse_prefix.size(0) == packed_rows + 1, "qweight2_sparse_prefix shape mismatch");
  const bool use_block_gr = q_gmap_gr_offsets.size(0) == gmap_block_count + 1 && q_gmap_gr_meta.size(0) == gmap_block_count;
  if (!use_block_gr) {
    TORCH_CHECK(q_gmap_sparse_prefix.dim() == 1 && q_gmap_sparse_prefix.size(0) == packed_rows + 1, "q_gmap_sparse_prefix shape mismatch");
  }
  check_half_vector(input_scale, "input_scale", rank);
  check_half_vector(output_scale, "output_scale", outfeatures);
  if (has_bias) {
    check_half_vector(bias, "bias", outfeatures);
  }
  int ns_rank = 0;
  int ns_c1_qzeros_cols = 0;
  int ns_c2_qzeros_cols = 0;
  if (has_ns_matrix) {
    check_packed4_args(input, ns_c2_qweight, ns_c2_qzeros, ns_c2_scales, ns_c2_g_idx);
    ns_rank = static_cast<int>(ns_c2_qweight.size(1));
    check_packed4_weight_only(ns_c1_qweight, ns_c1_qzeros, ns_c1_scales, ns_c1_g_idx, outfeatures, ns_rank, "ns_c1");
    TORCH_CHECK(ns_c1_g_idx.size(0) == ns_rank, "ns_c1 g_idx length mismatch");
    ns_c1_qzeros_cols = static_cast<int>(ns_c1_qzeros.size(1));
    ns_c2_qzeros_cols = static_cast<int>(ns_c2_qzeros.size(1));
  }
  auto output = torch::zeros({batch, outfeatures}, input.options());
  if (batch == 0 || rank == 0 || outfeatures == 0) {
    return output;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool use_fast_no_scale_no_bias = !has_input_scale && !has_output_scale && !has_bias && !has_ns_matrix;
  const bool use_fast_no_scale_bias = !has_input_scale && !has_output_scale && has_bias && !has_ns_matrix;
  const bool use_fast_no_scale = use_fast_no_scale_no_bias || use_fast_no_scale_bias;
  if (use_block_gr) {
    const int max_split_k_slices = ceil_div_int(rank, STRUCTURED_TILE_K);
    if (split_k_slices < 1) {
      split_k_slices = 1;
    } else if (split_k_slices > max_split_k_slices) {
      split_k_slices = max_split_k_slices;
    }
    if (!use_fast_no_scale) {
      split_k_slices = 1;
    }

    if (split_k_slices > 1) {
      auto output_acc = torch::zeros({batch, outfeatures}, input.options().dtype(torch::kFloat));
      dim3 block(BLOCK_GR_TILE_OUT);
      dim3 grid((outfeatures + block.x - 1) / block.x, batch, split_k_slices);
      Compact3LinearKernelPackedFastNoScaleNoBiasSplitKBlockGR<UniformK64Groups><<<grid, block, 0, stream>>>(
          reinterpret_cast<half*>(input.data_ptr<at::Half>()),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_gr_words.data_ptr<int>(),
          q_gmap_gr_offsets.data_ptr<int>(),
          q_gmap_gr_meta.data_ptr<int>(),
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          output_acc.data_ptr<float>(),
          batch,
          rank,
          outfeatures,
          num_groups,
          split_k_slices);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      const int numel = batch * outfeatures;
      const int finalize_threads = 256;
      const int finalize_blocks = (numel + finalize_threads - 1) / finalize_threads;
      Compact3LinearFinalizeFloatKernel<<<finalize_blocks, finalize_threads, 0, stream>>>(
          output_acc.data_ptr<float>(),
          reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
          has_bias ? reinterpret_cast<half*>(bias.data_ptr<at::Half>()) : nullptr,
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          numel,
          outfeatures,
          has_output_scale,
          has_bias);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return output;
    }

    const bool use_prefill_tensor_core = use_fast_no_scale_no_bias
        && can_use_compact3_prefill_tensor_core(request_prefill_tensor_core, batch, split_k_slices);
    if (use_prefill_tensor_core) {
#ifndef USE_ROCM
      if (batch <= STRUCTURED_TC_M_TILE) {
        dim3 block(32);
        dim3 grid(
            ceil_div_int(outfeatures, STRUCTURED_TC_K_STEP),
            ceil_div_int(batch, STRUCTURED_TC_M_TILE));
        Compact3LinearPrefillTensorCoreBlockGRKernel<STRUCTURED_TC_K_STEP, 1, UniformK64Groups><<<grid, block, 0, stream>>>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()),
            qweight1.data_ptr<int>(),
            qweight2_sparse.data_ptr<int>(),
            qweight2_sparse_prefix.data_ptr<int>(),
            q_gmap_gr_words.data_ptr<int>(),
            q_gmap_gr_offsets.data_ptr<int>(),
            q_gmap_gr_meta.data_ptr<int>(),
            c_map.data_ptr<int>(),
            g_idx.data_ptr<int>(),
            col_pack3.data_ptr<int64_t>(),
            row_pack3.data_ptr<int64_t>(),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            batch,
            rank,
            outfeatures,
            num_groups);
      } else {
        dim3 block(STRUCTURED_TC_WARPS * 32);
        dim3 grid(
            ceil_div_int(outfeatures, STRUCTURED_TC_N_TILE),
            ceil_div_int(batch, STRUCTURED_TC_M_TILE));
        Compact3LinearPrefillTensorCoreBlockGRKernel<STRUCTURED_TC_N_TILE, STRUCTURED_TC_WARPS, UniformK64Groups><<<grid, block, 0, stream>>>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()),
            qweight1.data_ptr<int>(),
            qweight2_sparse.data_ptr<int>(),
            qweight2_sparse_prefix.data_ptr<int>(),
            q_gmap_gr_words.data_ptr<int>(),
            q_gmap_gr_offsets.data_ptr<int>(),
            q_gmap_gr_meta.data_ptr<int>(),
            c_map.data_ptr<int>(),
            g_idx.data_ptr<int>(),
            col_pack3.data_ptr<int64_t>(),
            row_pack3.data_ptr<int64_t>(),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            batch,
            rank,
            outfeatures,
            num_groups);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return output;
#endif
    }

    if (use_fast_no_scale) {
      dim3 block(BLOCK_GR_TILE_OUT, COMPACT3_SINGLE_BLOCK_GR_K_WORKERS);
      dim3 grid((outfeatures + block.x - 1) / block.x, batch);
      Compact3LinearKernelPackedFastNoScaleNoBiasBlockGR<UniformK64Groups><<<grid, block, 0, stream>>>(
          reinterpret_cast<half*>(input.data_ptr<at::Half>()),
          qweight1.data_ptr<int>(),
          qweight2_sparse.data_ptr<int>(),
          qweight2_sparse_prefix.data_ptr<int>(),
          q_gmap_gr_words.data_ptr<int>(),
          q_gmap_gr_offsets.data_ptr<int>(),
          q_gmap_gr_meta.data_ptr<int>(),
          c_map.data_ptr<int>(),
          g_idx.data_ptr<int>(),
          col_pack3.data_ptr<int64_t>(),
          row_pack3.data_ptr<int64_t>(),
          reinterpret_cast<half*>(output.data_ptr<at::Half>()),
          batch,
          rank,
          outfeatures,
          num_groups);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      if (use_fast_no_scale_bias) {
        const int numel = batch * outfeatures;
        const int bias_threads = 256;
        const int bias_blocks = (numel + bias_threads - 1) / bias_threads;
        AddBiasHalfKernel<<<bias_blocks, bias_threads, 0, stream>>>(
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            reinterpret_cast<half*>(bias.data_ptr<at::Half>()),
            numel,
            outfeatures);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      return output;
    }

    dim3 block(STRUCTURED_BLOCK_OUT);
    dim3 grid((outfeatures + block.x - 1) / block.x, batch);
    Compact3LinearKernelPackedBlockGR<UniformK64Groups><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        qweight1.data_ptr<int>(),
        qweight2_sparse.data_ptr<int>(),
        qweight2_sparse_prefix.data_ptr<int>(),
        q_gmap_gr_words.data_ptr<int>(),
        q_gmap_gr_offsets.data_ptr<int>(),
        q_gmap_gr_meta.data_ptr<int>(),
        c_map.data_ptr<int>(),
        g_idx.data_ptr<int>(),
        col_pack3.data_ptr<int64_t>(),
        row_pack3.data_ptr<int64_t>(),
        reinterpret_cast<half*>(input_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
        has_bias ? reinterpret_cast<half*>(bias.data_ptr<at::Half>()) : nullptr,
        has_ns_matrix ? ns_c1_qweight.data_ptr<int>() : nullptr,
        has_ns_matrix ? ns_c1_qzeros.data_ptr<int>() : nullptr,
        has_ns_matrix ? reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()) : nullptr,
        has_ns_matrix ? ns_c1_g_idx.data_ptr<int>() : nullptr,
        has_ns_matrix ? ns_c2_qweight.data_ptr<int>() : nullptr,
        has_ns_matrix ? ns_c2_qzeros.data_ptr<int>() : nullptr,
        has_ns_matrix ? reinterpret_cast<half*>(ns_c2_scales.data_ptr<at::Half>()) : nullptr,
        has_ns_matrix ? ns_c2_g_idx.data_ptr<int>() : nullptr,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        batch,
        rank,
        outfeatures,
        num_groups,
        ns_rank,
        ns_c1_qzeros_cols,
        ns_c2_qzeros_cols,
        has_input_scale,
        has_output_scale,
        has_bias,
        has_ns_matrix);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  dim3 block(STRUCTURED_BLOCK_OUT);
  dim3 grid((outfeatures + block.x - 1) / block.x, batch);
  Compact3LinearKernelPacked<<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      qweight1.data_ptr<int>(),
      qweight2_sparse.data_ptr<int>(),
      qweight2_sparse_prefix.data_ptr<int>(),
      q_gmap_sparse.data_ptr<int>(),
      q_gmap_sparse_prefix.data_ptr<int>(),
      c_map.data_ptr<int>(),
      g_idx.data_ptr<int>(),
      col_pack3.data_ptr<int64_t>(),
      row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(input_scale.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output_scale.data_ptr<at::Half>()),
      has_bias ? reinterpret_cast<half*>(bias.data_ptr<at::Half>()) : nullptr,
      has_ns_matrix ? ns_c1_qweight.data_ptr<int>() : nullptr,
      has_ns_matrix ? ns_c1_qzeros.data_ptr<int>() : nullptr,
      has_ns_matrix ? reinterpret_cast<half*>(ns_c1_scales.data_ptr<at::Half>()) : nullptr,
      has_ns_matrix ? ns_c1_g_idx.data_ptr<int>() : nullptr,
      has_ns_matrix ? ns_c2_qweight.data_ptr<int>() : nullptr,
      has_ns_matrix ? ns_c2_qzeros.data_ptr<int>() : nullptr,
      has_ns_matrix ? reinterpret_cast<half*>(ns_c2_scales.data_ptr<at::Half>()) : nullptr,
      has_ns_matrix ? ns_c2_g_idx.data_ptr<int>() : nullptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      rank,
      outfeatures,
      num_groups,
      ns_rank,
      ns_c1_qzeros_cols,
      ns_c2_qzeros_cols,
      has_input_scale,
      has_output_scale,
      has_bias,
      has_ns_matrix);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor compact3_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor qweight1,
    torch::Tensor qweight2_sparse,
    torch::Tensor qweight2_sparse_prefix,
    torch::Tensor q_gmap_sparse,
    torch::Tensor q_gmap_sparse_prefix,
    torch::Tensor q_gmap_gr_words,
    torch::Tensor q_gmap_gr_offsets,
    torch::Tensor q_gmap_gr_meta,
    torch::Tensor c_map,
    torch::Tensor g_idx,
    torch::Tensor col_pack3,
    torch::Tensor row_pack3,
    torch::Tensor input_scale,
    torch::Tensor output_scale,
    torch::Tensor bias,
    bool has_input_scale,
    bool has_output_scale,
    bool has_bias,
    torch::Tensor ns_c1_qweight,
    torch::Tensor ns_c1_qzeros,
    torch::Tensor ns_c1_scales,
    torch::Tensor ns_c1_g_idx,
    torch::Tensor ns_c2_qweight,
    torch::Tensor ns_c2_qzeros,
    torch::Tensor ns_c2_scales,
    torch::Tensor ns_c2_g_idx,
    bool has_ns_matrix,
    int split_k_slices,
    bool request_prefill_tensor_core,
    bool use_uniform_k64_groups) {
  if (use_uniform_k64_groups) {
    return compact3_linear_forward_cuda_impl<true>(
        input,
        qweight1,
        qweight2_sparse,
        qweight2_sparse_prefix,
        q_gmap_sparse,
        q_gmap_sparse_prefix,
        q_gmap_gr_words,
        q_gmap_gr_offsets,
        q_gmap_gr_meta,
        c_map,
        g_idx,
        col_pack3,
        row_pack3,
        input_scale,
        output_scale,
        bias,
        has_input_scale,
        has_output_scale,
        has_bias,
        ns_c1_qweight,
        ns_c1_qzeros,
        ns_c1_scales,
        ns_c1_g_idx,
        ns_c2_qweight,
        ns_c2_qzeros,
        ns_c2_scales,
        ns_c2_g_idx,
        has_ns_matrix,
        split_k_slices,
        request_prefill_tensor_core);
  }
  return compact3_linear_forward_cuda_impl<false>(
      input,
      qweight1,
      qweight2_sparse,
      qweight2_sparse_prefix,
      q_gmap_sparse,
      q_gmap_sparse_prefix,
      q_gmap_gr_words,
      q_gmap_gr_offsets,
      q_gmap_gr_meta,
      c_map,
      g_idx,
      col_pack3,
      row_pack3,
      input_scale,
      output_scale,
      bias,
      has_input_scale,
      has_output_scale,
      has_bias,
      ns_c1_qweight,
      ns_c1_qzeros,
      ns_c1_scales,
      ns_c1_g_idx,
      ns_c2_qweight,
      ns_c2_qzeros,
      ns_c2_scales,
      ns_c2_g_idx,
      has_ns_matrix,
      split_k_slices,
      request_prefill_tensor_core);
}

template <bool UniformK64Groups>
torch::Tensor compact3_dual_linear_forward_cuda_impl(
    torch::Tensor input,
    torch::Tensor a_qweight1,
    torch::Tensor a_qweight2_sparse,
    torch::Tensor a_qweight2_sparse_prefix,
    torch::Tensor a_gmap_gr_words,
    torch::Tensor a_gmap_gr_offsets,
    torch::Tensor a_gmap_gr_meta,
    torch::Tensor a_c_map,
    torch::Tensor a_g_idx,
    torch::Tensor a_col_pack3,
    torch::Tensor a_row_pack3,
    torch::Tensor b_qweight1,
    torch::Tensor b_qweight2_sparse,
    torch::Tensor b_qweight2_sparse_prefix,
    torch::Tensor b_gmap_gr_words,
    torch::Tensor b_gmap_gr_offsets,
    torch::Tensor b_gmap_gr_meta,
    torch::Tensor b_c_map,
    torch::Tensor b_g_idx,
    torch::Tensor b_col_pack3,
    torch::Tensor b_row_pack3,
    int split_k_slices) {
  CHECK_CUDA(input); CHECK_CONTIGUOUS(input); CHECK_HALF(input);
  TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, rank]");

  const int batch = static_cast<int>(input.size(0));
  const int rank = static_cast<int>(input.size(1));
  const int packed_rows = (rank + 31) / 32;
  const int outfeatures = static_cast<int>(a_qweight1.size(1));
  const int num_groups = static_cast<int>(a_row_pack3.size(1));
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(outfeatures, BLOCK_GR_TILE_OUT);

  auto check_projection = [&](const char* name,
                              torch::Tensor qweight1,
                              torch::Tensor qweight2_sparse,
                              torch::Tensor qweight2_sparse_prefix,
                              torch::Tensor gmap_gr_words,
                              torch::Tensor gmap_gr_offsets,
                              torch::Tensor gmap_gr_meta,
                              torch::Tensor c_map,
                              torch::Tensor g_idx,
                              torch::Tensor col_pack3,
                              torch::Tensor row_pack3) {
    CHECK_CUDA(qweight1); CHECK_CONTIGUOUS(qweight1); CHECK_INT(qweight1);
    CHECK_CUDA(qweight2_sparse); CHECK_CONTIGUOUS(qweight2_sparse); CHECK_INT(qweight2_sparse);
    CHECK_CUDA(qweight2_sparse_prefix); CHECK_CONTIGUOUS(qweight2_sparse_prefix); CHECK_INT(qweight2_sparse_prefix);
    CHECK_CUDA(gmap_gr_words); CHECK_CONTIGUOUS(gmap_gr_words); CHECK_INT(gmap_gr_words);
    CHECK_CUDA(gmap_gr_offsets); CHECK_CONTIGUOUS(gmap_gr_offsets); CHECK_INT(gmap_gr_offsets);
    CHECK_CUDA(gmap_gr_meta); CHECK_CONTIGUOUS(gmap_gr_meta); CHECK_INT(gmap_gr_meta);
    CHECK_CUDA(c_map); CHECK_CONTIGUOUS(c_map); CHECK_INT(c_map);
    CHECK_CUDA(g_idx); CHECK_CONTIGUOUS(g_idx); CHECK_INT(g_idx);
    CHECK_CUDA(col_pack3); CHECK_CONTIGUOUS(col_pack3); CHECK_LONG(col_pack3);
    CHECK_CUDA(row_pack3); CHECK_CONTIGUOUS(row_pack3); CHECK_LONG(row_pack3);

    TORCH_CHECK(qweight1.dim() == 2 && qweight2_sparse.dim() == 2, name, " packed matrices must be 2D");
    TORCH_CHECK(gmap_gr_words.dim() == 1 && gmap_gr_offsets.dim() == 1 && gmap_gr_meta.dim() == 1, name, " block-GR tensors must be 1D");
    TORCH_CHECK(c_map.dim() == 1, name, " c_map must be 1D");
    TORCH_CHECK(g_idx.dim() == 1, name, " g_idx must be 1D");
    TORCH_CHECK(col_pack3.dim() == 1, name, " col_pack3 must be 1D");
    TORCH_CHECK(row_pack3.dim() == 2, name, " row_pack3 must be 2D");
    TORCH_CHECK(qweight1.size(0) == packed_rows && qweight1.size(1) == outfeatures, name, " qweight1 shape mismatch");
    TORCH_CHECK(qweight2_sparse.size(1) == outfeatures, name, " qweight2_sparse outfeature mismatch");
    TORCH_CHECK(qweight2_sparse_prefix.dim() == 1 && qweight2_sparse_prefix.size(0) == packed_rows + 1, name, " qweight2_sparse_prefix shape mismatch");
    TORCH_CHECK(gmap_gr_offsets.size(0) == gmap_block_count + 1, name, " q_gmap_gr_offsets shape mismatch");
    TORCH_CHECK(gmap_gr_meta.size(0) == gmap_block_count, name, " q_gmap_gr_meta shape mismatch");
    TORCH_CHECK(c_map.size(0) == packed_rows, name, " c_map packed length mismatch");
    TORCH_CHECK(g_idx.size(0) == rank, name, " g_idx rank mismatch");
    TORCH_CHECK(col_pack3.size(0) == rank, name, " col_pack3 rank mismatch");
    TORCH_CHECK(row_pack3.size(0) == outfeatures && row_pack3.size(1) == num_groups, name, " row_pack3 shape mismatch");
  };

  check_projection(
      "dual_a",
      a_qweight1, a_qweight2_sparse, a_qweight2_sparse_prefix,
      a_gmap_gr_words, a_gmap_gr_offsets, a_gmap_gr_meta,
      a_c_map, a_g_idx, a_col_pack3, a_row_pack3);
  check_projection(
      "dual_b",
      b_qweight1, b_qweight2_sparse, b_qweight2_sparse_prefix,
      b_gmap_gr_words, b_gmap_gr_offsets, b_gmap_gr_meta,
      b_c_map, b_g_idx, b_col_pack3, b_row_pack3);

  auto output = torch::zeros({2, batch, outfeatures}, input.options());
  if (batch == 0 || rank == 0 || outfeatures == 0) {
    return output;
  }

  const int max_split_k_slices = ceil_div_int(rank, STRUCTURED_TILE_K);
  if (split_k_slices < 1) {
    split_k_slices = 1;
  } else if (split_k_slices > max_split_k_slices) {
    split_k_slices = max_split_k_slices;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (split_k_slices > 1) {
    auto output_acc = torch::empty({2, split_k_slices, batch, outfeatures}, input.options().dtype(torch::kFloat));
    dim3 block(BLOCK_GR_TILE_OUT);
    dim3 grid((outfeatures + block.x - 1) / block.x, batch, 2 * split_k_slices);
    Compact3DualLinearKernelPackedFastNoScaleNoBiasSplitKBlockGR<UniformK64Groups><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        a_qweight1.data_ptr<int>(),
        a_qweight2_sparse.data_ptr<int>(),
        a_qweight2_sparse_prefix.data_ptr<int>(),
        a_gmap_gr_words.data_ptr<int>(),
        a_gmap_gr_offsets.data_ptr<int>(),
        a_gmap_gr_meta.data_ptr<int>(),
        a_c_map.data_ptr<int>(),
        a_g_idx.data_ptr<int>(),
        a_col_pack3.data_ptr<int64_t>(),
        a_row_pack3.data_ptr<int64_t>(),
        b_qweight1.data_ptr<int>(),
        b_qweight2_sparse.data_ptr<int>(),
        b_qweight2_sparse_prefix.data_ptr<int>(),
        b_gmap_gr_words.data_ptr<int>(),
        b_gmap_gr_offsets.data_ptr<int>(),
        b_gmap_gr_meta.data_ptr<int>(),
        b_c_map.data_ptr<int>(),
        b_g_idx.data_ptr<int>(),
        b_col_pack3.data_ptr<int64_t>(),
        b_row_pack3.data_ptr<int64_t>(),
        output_acc.data_ptr<float>(),
        batch,
        rank,
        outfeatures,
        num_groups,
        split_k_slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int numel_per_projection = batch * outfeatures;
    const int total_numel = 2 * numel_per_projection;
    const int finalize_threads = 256;
    const int finalize_blocks = (total_numel + finalize_threads - 1) / finalize_threads;
    Compact3DualSplitKFinalizeHalfKernel<<<finalize_blocks, finalize_threads, 0, stream>>>(
        output_acc.data_ptr<float>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        numel_per_projection,
        split_k_slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  auto zero_bias = torch::zeros({outfeatures}, input.options());
  dim3 block(BLOCK_GR_TILE_OUT, COMPACT3_SINGLE_BLOCK_GR_K_WORKERS);
  dim3 grid((outfeatures + block.x - 1) / block.x, batch, 2);
  Compact3QKVLinearKernelPackedFastNoScaleBiasBlockGR<UniformK64Groups><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      a_qweight1.data_ptr<int>(),
      a_qweight2_sparse.data_ptr<int>(),
      a_qweight2_sparse_prefix.data_ptr<int>(),
      a_gmap_gr_words.data_ptr<int>(),
      a_gmap_gr_offsets.data_ptr<int>(),
      a_gmap_gr_meta.data_ptr<int>(),
      a_c_map.data_ptr<int>(),
      a_g_idx.data_ptr<int>(),
      a_col_pack3.data_ptr<int64_t>(),
      a_row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(zero_bias.data_ptr<at::Half>()),
      b_qweight1.data_ptr<int>(),
      b_qweight2_sparse.data_ptr<int>(),
      b_qweight2_sparse_prefix.data_ptr<int>(),
      b_gmap_gr_words.data_ptr<int>(),
      b_gmap_gr_offsets.data_ptr<int>(),
      b_gmap_gr_meta.data_ptr<int>(),
      b_c_map.data_ptr<int>(),
      b_g_idx.data_ptr<int>(),
      b_col_pack3.data_ptr<int64_t>(),
      b_row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(zero_bias.data_ptr<at::Half>()),
      a_qweight1.data_ptr<int>(),
      a_qweight2_sparse.data_ptr<int>(),
      a_qweight2_sparse_prefix.data_ptr<int>(),
      a_gmap_gr_words.data_ptr<int>(),
      a_gmap_gr_offsets.data_ptr<int>(),
      a_gmap_gr_meta.data_ptr<int>(),
      a_c_map.data_ptr<int>(),
      a_g_idx.data_ptr<int>(),
      a_col_pack3.data_ptr<int64_t>(),
      a_row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(zero_bias.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      rank,
      outfeatures,
      num_groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor compact3_dual_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor a_qweight1,
    torch::Tensor a_qweight2_sparse,
    torch::Tensor a_qweight2_sparse_prefix,
    torch::Tensor a_gmap_gr_words,
    torch::Tensor a_gmap_gr_offsets,
    torch::Tensor a_gmap_gr_meta,
    torch::Tensor a_c_map,
    torch::Tensor a_g_idx,
    torch::Tensor a_col_pack3,
    torch::Tensor a_row_pack3,
    torch::Tensor b_qweight1,
    torch::Tensor b_qweight2_sparse,
    torch::Tensor b_qweight2_sparse_prefix,
    torch::Tensor b_gmap_gr_words,
    torch::Tensor b_gmap_gr_offsets,
    torch::Tensor b_gmap_gr_meta,
    torch::Tensor b_c_map,
    torch::Tensor b_g_idx,
    torch::Tensor b_col_pack3,
    torch::Tensor b_row_pack3,
    int split_k_slices,
    bool use_uniform_k64_groups) {
  if (use_uniform_k64_groups) {
    return compact3_dual_linear_forward_cuda_impl<true>(
        input,
        a_qweight1,
        a_qweight2_sparse,
        a_qweight2_sparse_prefix,
        a_gmap_gr_words,
        a_gmap_gr_offsets,
        a_gmap_gr_meta,
        a_c_map,
        a_g_idx,
        a_col_pack3,
        a_row_pack3,
        b_qweight1,
        b_qweight2_sparse,
        b_qweight2_sparse_prefix,
        b_gmap_gr_words,
        b_gmap_gr_offsets,
        b_gmap_gr_meta,
        b_c_map,
        b_g_idx,
        b_col_pack3,
        b_row_pack3,
        split_k_slices);
  }
  return compact3_dual_linear_forward_cuda_impl<false>(
      input,
      a_qweight1,
      a_qweight2_sparse,
      a_qweight2_sparse_prefix,
      a_gmap_gr_words,
      a_gmap_gr_offsets,
      a_gmap_gr_meta,
      a_c_map,
      a_g_idx,
      a_col_pack3,
      a_row_pack3,
      b_qweight1,
      b_qweight2_sparse,
      b_qweight2_sparse_prefix,
      b_gmap_gr_words,
      b_gmap_gr_offsets,
      b_gmap_gr_meta,
      b_c_map,
      b_g_idx,
      b_col_pack3,
      b_row_pack3,
      split_k_slices);
}

template <bool UniformK64Groups>
torch::Tensor compact3_qkv_linear_forward_cuda_impl(
    torch::Tensor input,
    torch::Tensor q_qweight1,
    torch::Tensor q_qweight2_sparse,
    torch::Tensor q_qweight2_sparse_prefix,
    torch::Tensor q_gmap_gr_words,
    torch::Tensor q_gmap_gr_offsets,
    torch::Tensor q_gmap_gr_meta,
    torch::Tensor q_c_map,
    torch::Tensor q_g_idx,
    torch::Tensor q_col_pack3,
    torch::Tensor q_row_pack3,
    torch::Tensor q_bias,
    torch::Tensor k_qweight1,
    torch::Tensor k_qweight2_sparse,
    torch::Tensor k_qweight2_sparse_prefix,
    torch::Tensor k_gmap_gr_words,
    torch::Tensor k_gmap_gr_offsets,
    torch::Tensor k_gmap_gr_meta,
    torch::Tensor k_c_map,
    torch::Tensor k_g_idx,
    torch::Tensor k_col_pack3,
    torch::Tensor k_row_pack3,
    torch::Tensor k_bias,
    torch::Tensor v_qweight1,
    torch::Tensor v_qweight2_sparse,
    torch::Tensor v_qweight2_sparse_prefix,
    torch::Tensor v_gmap_gr_words,
    torch::Tensor v_gmap_gr_offsets,
    torch::Tensor v_gmap_gr_meta,
    torch::Tensor v_c_map,
    torch::Tensor v_g_idx,
    torch::Tensor v_col_pack3,
    torch::Tensor v_row_pack3,
    torch::Tensor v_bias,
    int split_k_slices) {
  CHECK_CUDA(input); CHECK_CONTIGUOUS(input); CHECK_HALF(input);
  TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, rank]");

  const int batch = static_cast<int>(input.size(0));
  const int rank = static_cast<int>(input.size(1));
  const int packed_rows = (rank + 31) / 32;
  const int q_outfeatures = static_cast<int>(q_qweight1.size(1));
  const int num_groups = static_cast<int>(q_row_pack3.size(1));
  const int gmap_block_count = ceil_div_int(packed_rows, BLOCK_GR_PACKED_ROWS) * ceil_div_int(q_outfeatures, BLOCK_GR_TILE_OUT);

  auto check_projection = [&](const char* name,
                              torch::Tensor qweight1,
                              torch::Tensor qweight2_sparse,
                              torch::Tensor qweight2_sparse_prefix,
                              torch::Tensor gmap_gr_words,
                              torch::Tensor gmap_gr_offsets,
                              torch::Tensor gmap_gr_meta,
                              torch::Tensor c_map,
                              torch::Tensor g_idx,
                              torch::Tensor col_pack3,
                              torch::Tensor row_pack3,
                              torch::Tensor bias) {
    CHECK_CUDA(qweight1); CHECK_CONTIGUOUS(qweight1); CHECK_INT(qweight1);
    CHECK_CUDA(qweight2_sparse); CHECK_CONTIGUOUS(qweight2_sparse); CHECK_INT(qweight2_sparse);
    CHECK_CUDA(qweight2_sparse_prefix); CHECK_CONTIGUOUS(qweight2_sparse_prefix); CHECK_INT(qweight2_sparse_prefix);
    CHECK_CUDA(gmap_gr_words); CHECK_CONTIGUOUS(gmap_gr_words); CHECK_INT(gmap_gr_words);
    CHECK_CUDA(gmap_gr_offsets); CHECK_CONTIGUOUS(gmap_gr_offsets); CHECK_INT(gmap_gr_offsets);
    CHECK_CUDA(gmap_gr_meta); CHECK_CONTIGUOUS(gmap_gr_meta); CHECK_INT(gmap_gr_meta);
    CHECK_CUDA(c_map); CHECK_CONTIGUOUS(c_map); CHECK_INT(c_map);
    CHECK_CUDA(g_idx); CHECK_CONTIGUOUS(g_idx); CHECK_INT(g_idx);
    CHECK_CUDA(col_pack3); CHECK_CONTIGUOUS(col_pack3); CHECK_LONG(col_pack3);
    CHECK_CUDA(row_pack3); CHECK_CONTIGUOUS(row_pack3); CHECK_LONG(row_pack3);
    check_half_vector(bias, name, q_outfeatures);

    TORCH_CHECK(qweight1.dim() == 2 && qweight2_sparse.dim() == 2, name, " packed matrices must be 2D");
    TORCH_CHECK(gmap_gr_words.dim() == 1 && gmap_gr_offsets.dim() == 1 && gmap_gr_meta.dim() == 1, name, " block-GR tensors must be 1D");
    TORCH_CHECK(c_map.dim() == 1, name, " c_map must be 1D");
    TORCH_CHECK(g_idx.dim() == 1, name, " g_idx must be 1D");
    TORCH_CHECK(col_pack3.dim() == 1, name, " col_pack3 must be 1D");
    TORCH_CHECK(row_pack3.dim() == 2, name, " row_pack3 must be 2D");
    TORCH_CHECK(qweight1.size(0) == packed_rows && qweight1.size(1) == q_outfeatures, name, " qweight1 shape mismatch");
    TORCH_CHECK(qweight2_sparse.size(1) == q_outfeatures, name, " qweight2_sparse outfeature mismatch");
    TORCH_CHECK(qweight2_sparse_prefix.dim() == 1 && qweight2_sparse_prefix.size(0) == packed_rows + 1, name, " qweight2_sparse_prefix shape mismatch");
    TORCH_CHECK(gmap_gr_offsets.size(0) == gmap_block_count + 1, name, " q_gmap_gr_offsets shape mismatch");
    TORCH_CHECK(gmap_gr_meta.size(0) == gmap_block_count, name, " q_gmap_gr_meta shape mismatch");
    TORCH_CHECK(c_map.size(0) == packed_rows, name, " c_map packed length mismatch");
    TORCH_CHECK(g_idx.size(0) == rank, name, " g_idx rank mismatch");
    TORCH_CHECK(col_pack3.size(0) == rank, name, " col_pack3 rank mismatch");
    TORCH_CHECK(row_pack3.size(0) == q_outfeatures && row_pack3.size(1) == num_groups, name, " row_pack3 shape mismatch");
  };

  check_projection(
      "q_bias",
      q_qweight1, q_qweight2_sparse, q_qweight2_sparse_prefix,
      q_gmap_gr_words, q_gmap_gr_offsets, q_gmap_gr_meta,
      q_c_map, q_g_idx, q_col_pack3, q_row_pack3, q_bias);
  check_projection(
      "k_bias",
      k_qweight1, k_qweight2_sparse, k_qweight2_sparse_prefix,
      k_gmap_gr_words, k_gmap_gr_offsets, k_gmap_gr_meta,
      k_c_map, k_g_idx, k_col_pack3, k_row_pack3, k_bias);
  check_projection(
      "v_bias",
      v_qweight1, v_qweight2_sparse, v_qweight2_sparse_prefix,
      v_gmap_gr_words, v_gmap_gr_offsets, v_gmap_gr_meta,
      v_c_map, v_g_idx, v_col_pack3, v_row_pack3, v_bias);

  auto output = torch::zeros({3, batch, q_outfeatures}, input.options());
  if (batch == 0 || rank == 0 || q_outfeatures == 0) {
    return output;
  }

  const int max_split_k_slices = ceil_div_int(rank, STRUCTURED_TILE_K);
  if (split_k_slices < 1) {
    split_k_slices = 1;
  } else if (split_k_slices > max_split_k_slices) {
    split_k_slices = max_split_k_slices;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (split_k_slices > 1) {
    auto output_acc = torch::zeros({3, batch, q_outfeatures}, input.options().dtype(torch::kFloat));
    dim3 block(BLOCK_GR_TILE_OUT);
    dim3 grid((q_outfeatures + block.x - 1) / block.x, batch, 3 * split_k_slices);
    Compact3QKVLinearKernelPackedFastNoScaleBiasSplitKBlockGR<UniformK64Groups><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        q_qweight1.data_ptr<int>(),
        q_qweight2_sparse.data_ptr<int>(),
        q_qweight2_sparse_prefix.data_ptr<int>(),
        q_gmap_gr_words.data_ptr<int>(),
        q_gmap_gr_offsets.data_ptr<int>(),
        q_gmap_gr_meta.data_ptr<int>(),
        q_c_map.data_ptr<int>(),
        q_g_idx.data_ptr<int>(),
        q_col_pack3.data_ptr<int64_t>(),
        q_row_pack3.data_ptr<int64_t>(),
        k_qweight1.data_ptr<int>(),
        k_qweight2_sparse.data_ptr<int>(),
        k_qweight2_sparse_prefix.data_ptr<int>(),
        k_gmap_gr_words.data_ptr<int>(),
        k_gmap_gr_offsets.data_ptr<int>(),
        k_gmap_gr_meta.data_ptr<int>(),
        k_c_map.data_ptr<int>(),
        k_g_idx.data_ptr<int>(),
        k_col_pack3.data_ptr<int64_t>(),
        k_row_pack3.data_ptr<int64_t>(),
        v_qweight1.data_ptr<int>(),
        v_qweight2_sparse.data_ptr<int>(),
        v_qweight2_sparse_prefix.data_ptr<int>(),
        v_gmap_gr_words.data_ptr<int>(),
        v_gmap_gr_offsets.data_ptr<int>(),
        v_gmap_gr_meta.data_ptr<int>(),
        v_c_map.data_ptr<int>(),
        v_g_idx.data_ptr<int>(),
        v_col_pack3.data_ptr<int64_t>(),
        v_row_pack3.data_ptr<int64_t>(),
        output_acc.data_ptr<float>(),
        batch,
        rank,
        q_outfeatures,
        num_groups,
        split_k_slices);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int numel_per_projection = batch * q_outfeatures;
    const int total_numel = 3 * numel_per_projection;
    const int finalize_threads = 256;
    const int finalize_blocks = (total_numel + finalize_threads - 1) / finalize_threads;
    Compact3QKVFinalizeFloatKernel<<<finalize_blocks, finalize_threads, 0, stream>>>(
        output_acc.data_ptr<float>(),
        reinterpret_cast<half*>(q_bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        numel_per_projection,
        q_outfeatures);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  dim3 block(BLOCK_GR_TILE_OUT, COMPACT3_SINGLE_BLOCK_GR_K_WORKERS);
  dim3 grid((q_outfeatures + block.x - 1) / block.x, batch, 3);
  Compact3QKVLinearKernelPackedFastNoScaleBiasBlockGR<UniformK64Groups><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(input.data_ptr<at::Half>()),
      q_qweight1.data_ptr<int>(),
      q_qweight2_sparse.data_ptr<int>(),
      q_qweight2_sparse_prefix.data_ptr<int>(),
      q_gmap_gr_words.data_ptr<int>(),
      q_gmap_gr_offsets.data_ptr<int>(),
      q_gmap_gr_meta.data_ptr<int>(),
      q_c_map.data_ptr<int>(),
      q_g_idx.data_ptr<int>(),
      q_col_pack3.data_ptr<int64_t>(),
      q_row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(q_bias.data_ptr<at::Half>()),
      k_qweight1.data_ptr<int>(),
      k_qweight2_sparse.data_ptr<int>(),
      k_qweight2_sparse_prefix.data_ptr<int>(),
      k_gmap_gr_words.data_ptr<int>(),
      k_gmap_gr_offsets.data_ptr<int>(),
      k_gmap_gr_meta.data_ptr<int>(),
      k_c_map.data_ptr<int>(),
      k_g_idx.data_ptr<int>(),
      k_col_pack3.data_ptr<int64_t>(),
      k_row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(k_bias.data_ptr<at::Half>()),
      v_qweight1.data_ptr<int>(),
      v_qweight2_sparse.data_ptr<int>(),
      v_qweight2_sparse_prefix.data_ptr<int>(),
      v_gmap_gr_words.data_ptr<int>(),
      v_gmap_gr_offsets.data_ptr<int>(),
      v_gmap_gr_meta.data_ptr<int>(),
      v_c_map.data_ptr<int>(),
      v_g_idx.data_ptr<int>(),
      v_col_pack3.data_ptr<int64_t>(),
      v_row_pack3.data_ptr<int64_t>(),
      reinterpret_cast<half*>(v_bias.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      rank,
      q_outfeatures,
      num_groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor compact3_qkv_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor q_qweight1,
    torch::Tensor q_qweight2_sparse,
    torch::Tensor q_qweight2_sparse_prefix,
    torch::Tensor q_gmap_gr_words,
    torch::Tensor q_gmap_gr_offsets,
    torch::Tensor q_gmap_gr_meta,
    torch::Tensor q_c_map,
    torch::Tensor q_g_idx,
    torch::Tensor q_col_pack3,
    torch::Tensor q_row_pack3,
    torch::Tensor q_bias,
    torch::Tensor k_qweight1,
    torch::Tensor k_qweight2_sparse,
    torch::Tensor k_qweight2_sparse_prefix,
    torch::Tensor k_gmap_gr_words,
    torch::Tensor k_gmap_gr_offsets,
    torch::Tensor k_gmap_gr_meta,
    torch::Tensor k_c_map,
    torch::Tensor k_g_idx,
    torch::Tensor k_col_pack3,
    torch::Tensor k_row_pack3,
    torch::Tensor k_bias,
    torch::Tensor v_qweight1,
    torch::Tensor v_qweight2_sparse,
    torch::Tensor v_qweight2_sparse_prefix,
    torch::Tensor v_gmap_gr_words,
    torch::Tensor v_gmap_gr_offsets,
    torch::Tensor v_gmap_gr_meta,
    torch::Tensor v_c_map,
    torch::Tensor v_g_idx,
    torch::Tensor v_col_pack3,
    torch::Tensor v_row_pack3,
    torch::Tensor v_bias,
    int split_k_slices,
    bool use_uniform_k64_groups) {
  if (use_uniform_k64_groups) {
    return compact3_qkv_linear_forward_cuda_impl<true>(
        input,
        q_qweight1,
        q_qweight2_sparse,
        q_qweight2_sparse_prefix,
        q_gmap_gr_words,
        q_gmap_gr_offsets,
        q_gmap_gr_meta,
        q_c_map,
        q_g_idx,
        q_col_pack3,
        q_row_pack3,
        q_bias,
        k_qweight1,
        k_qweight2_sparse,
        k_qweight2_sparse_prefix,
        k_gmap_gr_words,
        k_gmap_gr_offsets,
        k_gmap_gr_meta,
        k_c_map,
        k_g_idx,
        k_col_pack3,
        k_row_pack3,
        k_bias,
        v_qweight1,
        v_qweight2_sparse,
        v_qweight2_sparse_prefix,
        v_gmap_gr_words,
        v_gmap_gr_offsets,
        v_gmap_gr_meta,
        v_c_map,
        v_g_idx,
        v_col_pack3,
        v_row_pack3,
        v_bias,
        split_k_slices);
  }
  return compact3_qkv_linear_forward_cuda_impl<false>(
      input,
      q_qweight1,
      q_qweight2_sparse,
      q_qweight2_sparse_prefix,
      q_gmap_gr_words,
      q_gmap_gr_offsets,
      q_gmap_gr_meta,
      q_c_map,
      q_g_idx,
      q_col_pack3,
      q_row_pack3,
      q_bias,
      k_qweight1,
      k_qweight2_sparse,
      k_qweight2_sparse_prefix,
      k_gmap_gr_words,
      k_gmap_gr_offsets,
      k_gmap_gr_meta,
      k_c_map,
      k_g_idx,
      k_col_pack3,
      k_row_pack3,
      k_bias,
      v_qweight1,
      v_qweight2_sparse,
      v_qweight2_sparse_prefix,
      v_gmap_gr_words,
      v_gmap_gr_offsets,
      v_gmap_gr_meta,
      v_c_map,
      v_g_idx,
      v_col_pack3,
      v_row_pack3,
      v_bias,
      split_k_slices);
}
