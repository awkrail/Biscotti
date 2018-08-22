/*
 * Copyright 2018 Taichi Nishimura
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "biscotti/processor.h"

#include <algorithm>
#include <set>
#include <string.h>
#include <vector>

#include "biscotti/butteraugli_comparator.h"
#include "biscotti/comparator.h"
#include "biscotti/debug_print.h"
#include "biscotti/fast_log.h"
#include "biscotti/jpeg_data_decoder.h"
#include "biscotti/jpeg_data_encoder.h"
#include "biscotti/jpeg_data_reader.h"
#include "biscotti/jpeg_data_writer.h"
#include "biscotti/output_image.h"
#include "biscotti/quantize.h"
#include "biscotti/tensorflow_predictor.h"

namespace biscotti {

namespace {

static const size_t kBlockSize = 3 * kDCTBlockSize;

struct CoeffData {
  int idx;
  float block_err;
};
struct QuantData {
  int q[3][kDCTBlockSize];
  size_t jpg_size;
  bool dist_ok;
};
class Processor {
 public:
  bool ProcessJpegData(const Params& params, const JPEGData& jpg_in,
                       Comparator* comparator, GuetzliOutput* out,
                       ProcessStats* stats);

 private:
  void MaybeOutput(const std::string& encoded_jpg);
  void OutputJpeg(const JPEGData& in, std::string* out);
  void GetCSFPointFromBlock(const coeff_t block[kBlockSize], 
                            const coeff_t orig_block[kBlockSize],
                            const int comp,
                            std::vector<std::pair<int, float> >* input_order);
  void MultiplyProbabilityWithCoefficients(const JPEGData& jpg_in, 
                                           OutputImage* img, 
                                           const uint8_t comp_mask,
                                           std::vector<int>& y,
                                           std::vector<int>& cb,
                                           std::vector<int>& cr);

  Params params_;
  Comparator* comparator_;
  GuetzliOutput* final_output_;
  ProcessStats* stats_;
};

void RemoveOriginalQuantization(JPEGData* jpg, int q_in[3][kDCTBlockSize]) {
  for (int i = 0; i < 3; ++i) {
    JPEGComponent& c = jpg->components[i];
    const int* q = &jpg->quant[c.quant_idx].values[0];
    memcpy(&q_in[i][0], q, kDCTBlockSize * sizeof(q[0]));
    for (size_t j = 0; j < c.coeffs.size(); ++j) {
      c.coeffs[j] *= q[j % kDCTBlockSize];
    }
  }
  int q[3][kDCTBlockSize];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < kDCTBlockSize; ++j) q[i][j] = 1;
  SaveQuantTables(q, jpg);
}

bool CheckJpegSanity(const JPEGData& jpg) {
  const int kMaxComponent = 1 << 12;
  for (const JPEGComponent& comp : jpg.components) {
    const JPEGQuantTable& quant_table = jpg.quant[comp.quant_idx];
    for (int i = 0; i < comp.coeffs.size(); i++) {
      coeff_t coeff = comp.coeffs[i];
      int quant = quant_table.values[i % kDCTBlockSize];
      if (std::abs(static_cast<int64_t>(coeff) * quant) > kMaxComponent) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int GuetzliStringOut(void* data, const uint8_t* buf, size_t count) {
  std::string* sink =
      reinterpret_cast<std::string*>(data);
  sink->append(reinterpret_cast<const char*>(buf), count);
  return count;
}

void Processor::OutputJpeg(const JPEGData& jpg,
                           std::string* out) {
  out->clear();
  JPEGOutput output(GuetzliStringOut, out);
  if (!WriteJpeg(jpg, params_.clear_metadata, output)) {
    assert(0);
  }
}

void Processor::MaybeOutput(const std::string& encoded_jpg) {
  double score = comparator_->ScoreOutputSize(encoded_jpg.size());
  BISCOTTI_LOG(stats_, " Score[%.4f]", score);
  if (score < final_output_->score || final_output_->score < 0) {
    final_output_->jpeg_data = encoded_jpg;
    final_output_->score = score;
    BISCOTTI_LOG(stats_, " (*)");
  }
  BISCOTTI_LOG(stats_, "\n");
}

bool CompareQuantData(const QuantData& a, const QuantData& b) {
  if (a.dist_ok && !b.dist_ok) return true;
  if (!a.dist_ok && b.dist_ok) return false;
  return a.jpg_size < b.jpg_size;
}

// Compares a[0..kBlockSize) and b[0..kBlockSize) vectors, and returns
//   0 : if they are equal
//  -1 : if a is everywhere <= than b and in at least one coordinate <
//   1 : if a is everywhere >= than b and in at least one coordinate >
//   2 : if a and b are uncomparable (some coordinate smaller and some greater)
int CompareQuantMatrices(const int* a, const int* b) {
  int i = 0;
  while (i < kBlockSize && a[i] == b[i]) ++i;
  if (i == kBlockSize) {
    return 0;
  }
  if (a[i] < b[i]) {
    for (++i; i < kBlockSize; ++i) {
      if (a[i] > b[i]) return 2;
    }
    return -1;
  } else {
    for (++i; i < kBlockSize; ++i) {
      if (a[i] < b[i]) return 2;
    }
    return 1;
  }
}

double ContrastSensitivity(int k) {
  return 1.0 / (1.0 + kJPEGZigZagOrder[k] / 2.0);
}

double QuantMatrixHeuristicScore(const int q[3][kDCTBlockSize]) {
  double score = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < kDCTBlockSize; ++k) {
      score += 0.5 * (q[c][k] - 1.0) * ContrastSensitivity(k);
    }
  }
  return score;
}

namespace {

void UpdateACHistogram(const int weight,
                       const coeff_t* coeffs,
                       const int* q,
                       JpegHistogram* ac_histogram) {
  int r = 0;
  for (int k = 1; k < 64; ++k) {
    const int k_nat = kJPEGNaturalOrder[k];
    coeff_t coeff = coeffs[k_nat];
    if (coeff == 0) {
      r++;
      continue;
    }
    while (r > 15) {
      ac_histogram->Add(0xf0, weight);
      r -= 16;
    }
    int nbits = Log2FloorNonZero(std::abs(coeff / q[k_nat])) + 1;
    int symbol = (r << 4) + nbits;
    ac_histogram->Add(symbol, weight);
    r = 0;
  }
  if (r > 0) {
    ac_histogram->Add(0, weight);
  }
}

size_t ComputeEntropyCodes(const std::vector<JpegHistogram>& histograms,
                           std::vector<uint8_t>* depths) {
  std::vector<JpegHistogram> clustered = histograms;
  size_t num = histograms.size();
  std::vector<int> indexes(histograms.size());
  std::vector<uint8_t> clustered_depths(
      histograms.size() * JpegHistogram::kSize);
  ClusterHistograms(&clustered[0], &num, &indexes[0], &clustered_depths[0]);
  depths->resize(clustered_depths.size());
  for (size_t i = 0; i < histograms.size(); ++i) {
    memcpy(&(*depths)[i * JpegHistogram::kSize],
           &clustered_depths[indexes[i] * JpegHistogram::kSize],
           JpegHistogram::kSize);
  }
  size_t histogram_size = 0;
  for (size_t i = 0; i < num; ++i) {
    histogram_size += HistogramHeaderCost(clustered[i]) / 8;
  }
  return histogram_size;
}

size_t EntropyCodedDataSize(const std::vector<JpegHistogram>& histograms,
                            const std::vector<uint8_t>& depths) {
  size_t numbits = 0;
  for (size_t i = 0; i < histograms.size(); ++i) {
    numbits += HistogramEntropyCost(
        histograms[i], &depths[i * JpegHistogram::kSize]);
  }
  return (numbits + 7) / 8;
}

size_t EstimateDCSize(const JPEGData& jpg) {
  std::vector<JpegHistogram> histograms(jpg.components.size());
  BuildDCHistograms(jpg, &histograms[0]);
  size_t num = histograms.size();
  std::vector<int> indexes(num);
  std::vector<uint8_t> depths(num * JpegHistogram::kSize);
  return ClusterHistograms(&histograms[0], &num, &indexes[0], &depths[0]);
}

}  // namespace

bool IsGrayscale(const JPEGData& jpg) {
  for (int c = 1; c < 3; ++c) {
    const JPEGComponent& comp = jpg.components[c];
    for (size_t i = 0; i < comp.coeffs.size(); ++i) {
      if (comp.coeffs[i] != 0) return false;
    }
  }
  return true;
}

void Processor::GetCSFPointFromBlock(
  const coeff_t block[kBlockSize], const coeff_t orig_block[kBlockSize],
  const int comp, std::vector<std::pair<int, float> >* input_order) {
  static const uint8_t oldCsf[kDCTBlockSize] = {
      10, 10, 20, 40, 60, 70, 80, 90,
      10, 20, 30, 60, 70, 80, 90, 90,
      20, 30, 60, 70, 80, 90, 90, 90,
      40, 60, 70, 80, 90, 90, 90, 90,
      60, 70, 80, 90, 90, 90, 90, 90,
      70, 80, 90, 90, 90, 90, 90, 90,
      80, 90, 90, 90, 90, 90, 90, 90,
      90, 90, 90, 90, 90, 90, 90, 90,
  };
  static const double kWeight[3] = { 1.0, 0.22, 0.20 };
  #include "biscotti/order.inc"
  for(int k=1; k < kDCTBlockSize; ++k) {
    int idx = comp * kDCTBlockSize + k;
    if(block[idx] != 0) {
      float score;
      if(params_.new_zeroing_model) {
        score = std::abs(orig_block[idx]) * csf[idx] + bias[idx];
      } else {
        score = static_cast<float>((std::abs(orig_block[idx]) - kJPEGZigZagOrder[k] / 64.0) *
                kWeight[comp] / oldCsf[k]);
      }
      input_order->push_back(std::make_pair(idx, score));
    }
  }
}

void Processor::MultiplyProbabilityWithCoefficients(const JPEGData& jpg, 
                                                    OutputImage* img,
                                                    const uint8_t comp_mask,
                                                    std::vector<int>& y,
                                                    std::vector<int>& cb,
                                                    std::vector<int>& cr) {
  // TODO: YUV444, PNGデータへ対応
  std::vector<int> pred_y;
  std::vector<int> pred_cb;
  std::vector<int> pred_cr; 

  for(int c=0; c<3; ++c) {
    if((comp_mask & (1 << c)) == 1) {
      const int width = img->component(c).width() / img->component(c).factor_x();
      const int height = img->component(c).height() / img->component(c).factor_y();
      const int block_width = img->component(c).width_in_blocks();
      const int block_height = img->component(c).height_in_blocks();
      const int block_num = block_width * block_height;
      std::vector<int> binary_coeffs;

      if(c == 0) {
        binary_coeffs = y;
      } else if(c == 1) {
        // which?
        binary_coeffs = cr;
      } else {
        binary_coeffs = cb;
      }

      for(int block_y=0; block_y < block_height; ++block_y) {
        for(int block_x=0; block_x < block_width; ++block_x) {
          int predict[kDCTBlockSize] = { 0 };
          int block_ix = block_width * block_y + block_x;
          int block_row = block_ix / block_width;
          int block_col = block_ix % block_width;

          for(int row_y=0; row_y<8; ++row_y) {
            for(int col_x=0; col_x<8; ++col_x) {
              predict[row_y*8 + col_x] = binary_coeffs[kDCTBlockSize*block_width*block_row + block_col*8 + width*row_y + col_x];
            }
          }
          // 元データ : csfからの評価値を算出して利用してみる
          coeff_t block[kDCTBlockSize] = { 0 };
          coeff_t orig_block[kDCTBlockSize] = { 0 };
          img->component(c).GetCoeffBlock(block_x, block_y, block);
          std::vector<std::pair<int, float> > input_order;
          const JPEGComponent& comp = jpg.components[c];
          memcpy(&orig_block[0],
                &comp.coeffs[block_ix * kDCTBlockSize],
                kDCTBlockSize * sizeof(orig_block[0]));
          GetCSFPointFromBlock(block, orig_block, c, &input_order); // CSFの値を計算する

          for(int i=0; i<input_order.size(); ++i) {
            float score = input_order[i].second;
            if(score < 10) { // thresholdは考えもの => 5は小さすぎかも
              int idx = input_order[i].first;
              block[idx] *= predict[idx];
            }
          }
          img->component(c).SetCoeffBlock(block_x, block_y, block);
        }
      }
    }
  }

  std::string encoded_jpg;
  {
    JPEGData jpg_out = jpg;
    img->SaveToJpegData(&jpg_out);
    OutputJpeg(jpg_out, &encoded_jpg);
  }
  MaybeOutput(encoded_jpg);
}

bool Processor::ProcessJpegData(const Params& params, const JPEGData& jpg_in,
                                Comparator* comparator, GuetzliOutput* out,
                                ProcessStats* stats) {
  params_ = params;
  comparator_ = comparator;
  final_output_ = out;
  stats_ = stats;

  if (params.butteraugli_target > 2.0f) {
    fprintf(stderr,
            "Guetzli should be called with quality >= 84, otherwise the\n"
            "output will have noticeable artifacts. If you want to\n"
            "proceed anyway, please edit the source code.\n");
    return false;
  }
  if (jpg_in.components.size() != 3 || !HasYCbCrColorSpace(jpg_in)) {
    fprintf(stderr, "Only YUV color space input jpeg is supported\n");
    return false;
  }
  bool input_is_420;
  if (jpg_in.Is444()) {
    input_is_420 = false;
  } else if (jpg_in.Is420()) {
    input_is_420 = true;
  } else {
    fprintf(stderr, "Unsupported sampling factors:");
    for (size_t i = 0; i < jpg_in.components.size(); ++i) {
      fprintf(stderr, " %dx%d", jpg_in.components[i].h_samp_factor,
              jpg_in.components[i].v_samp_factor);
    }
    fprintf(stderr, "\n");
    return false;
  }
  int q_in[3][kDCTBlockSize];
  // Output the original image, in case we do not manage to create anything
  // with a good enough quality.
  std::string encoded_jpg;
  OutputJpeg(jpg_in, &encoded_jpg);
  final_output_->score = -1;
  BISCOTTI_LOG(stats, "Original Out[%7zd]", encoded_jpg.size());
  if (comparator_ == nullptr) {
    BISCOTTI_LOG(stats, " <image too small for Butteraugli>\n");
    final_output_->jpeg_data = encoded_jpg;
    final_output_->score = encoded_jpg.size();
    // Butteraugli doesn't work with images this small.
    return true;
  }
  {
    JPEGData jpg = jpg_in;
    RemoveOriginalQuantization(&jpg, q_in);
    OutputImage img(jpg.width, jpg.height);
    img.CopyFromJpegData(jpg);
    comparator_->Compare(img);
  }
  MaybeOutput(encoded_jpg);

  if(!input_is_420) {
    std::cout << "this image is not YUV420" << std::endl;
    return false;
  }

  JPEGData jpg = jpg_in;
  RemoveOriginalQuantization(&jpg, q_in);
  OutputImage img(jpg.width, jpg.height);
  img.CopyFromJpegData(jpg);
  int best_q[3][kDCTBlockSize];
  //memcpy(best_q, q_in, sizeof(best_q)); : 量子化テーブルを最適化する場合のみ

  for(int c=0; c<3; ++c) {
    for(int i=0; i<kDCTBlockSize; ++i) {
      best_q[c][i] = 3;
    }
  }
  best_q[0][0] = 1;
  img.CopyFromJpegData(jpg);
  img.ApplyGlobalQuantization(best_q);

  // TODO
  // Now Only I support YUV420, and 3dimension(only RGB, not GRAYSCALE, and not RGBA),
  // and only jpg image. I will support them in a few days.

  // MEMO
  // 1. transform coeffcients inference results into YUV420 forms.
  // 2. Multiply coefficients inference with blocks.
  // 3. call SetCoeffBlock
  // 4. call OutputJpeg

  // Deep Learning Process(Inference)
  std::vector<tensorflow::Tensor> outputs;
  int input_width = jpg.width;
  int input_height = jpg.height;

  biscotti::Predictor predictor(params.filename, params.model_path, 
                      input_width, input_height, "input_1", "biscotti_0", outputs); // input_1_1 => input_1
  bool dnn_ok = predictor.Process();

  // 変更後の画像サイズの反映
  input_width = predictor.GetWidth();
  input_height = predictor.GetHeight();

  if(!dnn_ok) {
    return false;
  }
  
  tensorflow::TTypes<float>::Flat result_flat = outputs[0].flat<float>();
  std::vector<int> y;
  std::vector<int> cb;
  std::vector<int> cr;

  if(input_is_420) {
    for(int i=0; i<outputs[0].NumElements(); ++i) {
      int pixel = i / 3;
      int row = pixel / input_width;
      int value = result_flat(i) >= 0.4 ? 1 : 0; // TODO : consider threshold
      if(i % 3 == 0) {
        y.push_back(value);
      } else if(i % 3 == 1) {
        if(pixel % 2 == 0 && row % 2 == 0) {
          cr.push_back(value);
        }
      } else {
        if(pixel % 2 == 0 && row % 2 == 0) {
          cb.push_back(value);
        }
      }
    }
  } else {
    for(int i=0; i<outputs[0].NumElements(); ++i) {
      int pixel = i / 3;
      int row = pixel / input_width;
      int value = result_flat(i) >= 0.4 ? 1 : 0; // TODO : consider threshold
      if(i % 3 == 0) {
        y.push_back(value);
      } else if(i % 3 == 1) {
        cr.push_back(value);
      } else {
        cb.push_back(value);
      }
    }
  }

  if(IsGrayscale(jpg)) {
    MultiplyProbabilityWithCoefficients(jpg, &img, 1, y, cb, cr);
  } else {
    MultiplyProbabilityWithCoefficients(jpg, &img, 7, y, cb, cr);
  }

  return true;
}

bool ProcessJpegData(const Params& params, const JPEGData& jpg_in,
                     Comparator* comparator, GuetzliOutput* out,
                     ProcessStats* stats) {
  Processor processor;
  return processor.ProcessJpegData(params, jpg_in, comparator, out, stats);
}

bool Process(const Params& params, ProcessStats* stats,
             const std::string& data,
             std::string* jpg_out) {
  JPEGData jpg;
  if (!ReadJpeg(data, JPEG_READ_ALL, &jpg)) {
    fprintf(stderr, "Can't read jpg data from input file\n");
    return false;
  }
  if (!CheckJpegSanity(jpg)) {
    fprintf(stderr, "Unsupported input JPEG (unexpectedly large coefficient "
            "values).\n");
    return false;
  }
  std::vector<uint8_t> rgb = DecodeJpegToRGB(jpg);
  if (rgb.empty()) {
    fprintf(stderr, "Unsupported input JPEG file (e.g. unsupported "
            "downsampling mode).\nPlease provide the input image as "
            "a PNG file.\n");
    return false;
  }
  GuetzliOutput out;
  ProcessStats dummy_stats;
  if (stats == nullptr) {
    stats = &dummy_stats;
  }
  std::unique_ptr<ButteraugliComparator> comparator;
  if (jpg.width >= 32 && jpg.height >= 32) {
    comparator.reset(
        new ButteraugliComparator(jpg.width, jpg.height, &rgb,
                                  params.butteraugli_target, stats));
  }
  bool ok = ProcessJpegData(params, jpg, comparator.get(), &out, stats);
  *jpg_out = out.jpeg_data;
  return ok;
}

bool Process(const Params& params, ProcessStats* stats,
             const std::vector<uint8_t>& rgb, int w, int h,
             std::string* jpg_out) {
  JPEGData jpg;
  if (!EncodeRGBToJpeg(rgb, w, h, &jpg)) {
    fprintf(stderr, "Could not create jpg data from rgb pixels\n");
    return false;
  }
  GuetzliOutput out;
  ProcessStats dummy_stats;
  if (stats == nullptr) {
    stats = &dummy_stats;
  }
  std::unique_ptr<ButteraugliComparator> comparator;
  if (jpg.width >= 32 && jpg.height >= 32) {
    comparator.reset(
        new ButteraugliComparator(jpg.width, jpg.height, &rgb,
                                  params.butteraugli_target, stats));
  }
  bool ok = ProcessJpegData(params, jpg, comparator.get(), &out, stats);
  *jpg_out = out.jpeg_data;
  return ok;
}

}  // namespace biscotti
