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

#ifndef BISCOTTI_TENSORFLOW_PREDICTOR_H_
#define BISCOTTI_TENSORFLOW_PREDICTOR_H_

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// TODO: Predictor
// 1. loading jpeg image method is duplicated with guetzli process.
// 2. Check YUV420, and RGB(not YUV444 image)
// 3. Check Image Size, width and height is divied by 16(write padding method)

namespace biscotti {

class Predictor {
  public:
    Predictor(const tensorflow::string image_path, const tensorflow::string graph_path,
              const tensorflow::int32 input_width, const tensorflow::int32 input_height,
              const tensorflow::string input_layer, const tensorflow::string output_layer,
              std::vector<tensorflow::Tensor>& outputs);
    bool Process();
    int predict_index(const int index) const;
    static tensorflow::Status ReadEntireFile(tensorflow::Env* env, const tensorflow::string& filename,
                                             tensorflow::Tensor* output);


    
  private:
    tensorflow::Status LoadGraph(const tensorflow::string& graph_path,
                                std::unique_ptr<tensorflow::Session>* session);
    tensorflow::Status ReadTensorFromImageFile(const tensorflow::string& file_name, const int input_height,
                                              const int input_width, const float input_mean, const float input_std,
                                              std::vector<tensorflow::Tensor>* out_tensors);
    tensorflow::string image_path;
    tensorflow::string graph_path;
    tensorflow::int32 input_width;
    tensorflow::int32 input_height;
    tensorflow::string input_layer;
    tensorflow::string output_layer;
    std::vector<tensorflow::Tensor>& outputs;
  };
}

#endif // BISCOTTI_TENSORFLOW_PREDICTOR_H_