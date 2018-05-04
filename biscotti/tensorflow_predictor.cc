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

  int Predictor::predict_index(const int index) const {
    // TODO : Stock result_flat
    tensorflow::TTypes<float>::Flat result_flat = outputs[0].flat<float>();
    if(result_flat(index) >= 0.5) {
      return 1;
    } else {
      return 0;
    }
  }
  
  // this method is used for processing all units.
  // load graph, load image(but loading image is duplicated with guetzli process, so I will delete it),
  // and run session.
  // by this method, output property will be replaced.
  bool Predictor::Process() {
    // TODO List
    // global initialize
    // tensorflow::port::InitMain <- これないから怪しそう...
    // load and initialize model
    std::unique_ptr<tensorflow::Session> session;
    tensorflow::Status load_graph_status = LoadGraph(graph_path, &session);
    if(!load_graph_status.ok()) {
      LOG(ERROR) << load_graph_status;
      return false;
    }
  
    // Read Image
    // TODO : change code in order to get guetzli's rgb
    std::vector<tensorflow::Tensor> tensors;
    float input_mean = 0;
    float input_std = 255;
    tensorflow::Status read_tensor_status =
        ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                                input_std, &tensors);
    if(!read_tensor_status.ok()) {
      LOG(ERROR) << read_tensor_status;
      return false;
    }

    // run session, and set output tensor to output property
    const tensorflow::Tensor& tensor = tensors[0];

    // std::vector<tensorflow::Tensor> results;
    tensorflow::Status run_status = session->Run({{input_layer, tensor}},
                                                {output_layer}, {}, &outputs);
    if(!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return false;
    }

    return true;
  }

  tensorflow::Status Predictor::ReadEntireFile(tensorflow::Env* env, const tensorflow::string& file_name,
                                               tensorflow::Tensor* output) {
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(file_name, &file_size));

    tensorflow::string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(file_name, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if(data.size() != file_size) {
      return tensorflow::errors::DataLoss("Truncated read of '", file_name,
                                        "' expected ", file_size, " got ",
                                        data.size());
    }
    output->scalar<tensorflow::string>()() = data.ToString();
    return tensorflow::Status::OK();
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

  tensorflow::Status Predictor::ReadTensorFromImageFile(const tensorflow::string& file_name, const int input_height,
                                                        const int input_width, const float input_mean, const float input_std,
                                                        std::vector<tensorflow::Tensor>* out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;
    tensorflow::string input_name = "file_reader";
    tensorflow::string output_name = "normalized";

    // Read filename into a tensor name input
    tensorflow::Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
      Predictor::ReadEntireFile(tensorflow::Env::Default(), file_name, &input));
    
    auto file_reader =
        Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);
    
    std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
        {"input", input},
    };

    // try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;

    if (tensorflow::StringPiece(file_name).ends_with(".png")) {
      image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                               DecodePng::Channels(wanted_channels));
    } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
      image_reader =
          Squeeze(root.WithOpName("squeeze_first_dim"),
                  DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else {
      // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
      image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    auto float_caster =
        Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root, float_caster, 0);
    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(
        root, dims_expander,
        Const(root.WithOpName("size"), {input_height, input_width}));
    // Subtract the mean and divide by the scale.
    Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
        {input_std});

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
    return tensorflow::Status::OK();
  }
}