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

#include <fstream>
#include <utility>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>

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

#include "biscotti/tensorflow_predictor.h"

namespace biscotti {

  Predictor::Predictor(const tensorflow::string image_path,  const tensorflow::string graph_path,
                       const tensorflow::int32 input_width,  const tensorflow::int32 input_height,
                       const tensorflow::string input_layer, const tensorflow::string output_layer,
                       std::vector<tensorflow::Tensor>& outputs)
                      : image_path(image_path),
                        graph_path(graph_path),
                        input_width(input_width),
                        input_height(input_height),
                        input_layer(input_layer),
                        output_layer(output_layer),
                        outputs(outputs) {}
  
  // this method is used for processing all units.
  // load graph, load image(but loading image is duplicated with guetzli process, so I will delete it),
  // and run session.
  // by this method, output property will be replaced.
  bool Predictor::Process() {
    // TODO List
    // global initialize
    // load and initialize model
    std::unique_ptr<tensorflow::Session> session;
    tensorflow::Status load_graph_status = LoadGraph(graph_path, &session);
    if(!load_graph_status.ok()) {
      LOG(ERROR) << load_graph_status;
      return false;
    }
  
    // Read Image
    // TODO : change code in order to get guetzli's rgb
    cv::Mat image;
    image = cv::imread(static_cast<std::string>(image_path));
    cv::cvtColor(image, image, CV_BGR2RGB);
    std::vector<float> image_vector;
    for(int y=0; y<image.rows; ++y) {
      for(int x=0; x<image.cols; ++x) {
        for(int c=0; c<image.channels(); ++c) {
          float value = image.data[y * image.step + x * image.elemSize() + c];
          image_vector.push_back(value / 255.0);
        }
      }
    }

    // 画像のサイズに変更すること。
    assert(input_width % 16 == 0 && input_height % 16 == 0);

    // もしかしたら, 本来入ってはならないサイズのものも入っている可能性が..?
    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, input_width, input_height, 3}));
    std::copy_n(image_vector.begin(), image_vector.size(), tensor.flat<float>().data());
    tensorflow::Status run_status = session->Run({{input_layer, tensor}},
                                                {output_layer}, {}, &outputs);
    if(!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return false;
    }

    return true;
  }

  tensorflow::Status Predictor::LoadGraph(const tensorflow::string& graph_file_name,
                                          std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if(!load_graph_status.ok()) {
      return tensorflow::errors::NotFound("Failed to load compute graph at ", graph_file_name);
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::Status session_create_status = (*session)->Create(graph_def);
    if(!session_create_status.ok()) {
      return session_create_status;
    }
  return tensorflow::Status::OK();
  }
}