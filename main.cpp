#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Kullanim: " << argv[0] << " <sayi1> <sayi2>\n";
    std::cerr << "  Iki sayi girin (0-10 arasi), toplamlarini tahmin eder.\n";
    return 1;
  }

  float a = std::atof(argv[1]);
  float b = std::atof(argv[2]);

  tensorflow::SavedModelBundle bundle;
  tensorflow::SessionOptions session_options;
  tensorflow::RunOptions run_options;

  tensorflow::Status status = tensorflow::LoadSavedModel(
      session_options, run_options, "saved_model", {"serve"}, &bundle);
  if (!status.ok()) {
    std::cerr << "Model yuklenemedi: " << status.ToString() << std::endl;
    return 1;
  }

  auto sig_it = bundle.GetSignatures().find("serving_default");
  if (sig_it == bundle.GetSignatures().end()) {
    std::cerr << "serving_default imzasi bulunamadi" << std::endl;
    return 1;
  }

  const auto& sig = sig_it->second;
  std::string input_name = sig.inputs().begin()->second.name();
  std::string output_name = sig.outputs().begin()->second.name();

  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                  tensorflow::TensorShape({1, 2}));
  auto input_map = input_tensor.matrix<float>();
  input_map(0, 0) = a;
  input_map(0, 1) = b;

  std::vector<tensorflow::Tensor> outputs;
  status = bundle.session->Run({{input_name, input_tensor}}, {output_name}, {},
                               &outputs);
  if (!status.ok()) {
    std::cerr << "Tahmin basarisiz: " << status.ToString() << std::endl;
    return 1;
  }

  float prediction = outputs[0].flat<float>()(0);
  float expected = a + b;
  float error = prediction - expected;
  float error_pct = (expected != 0.0f)
                        ? std::abs(error / expected) * 100.0f
                        : std::abs(error) * 100.0f;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << a << " + " << b << " = " << prediction << "\n";
  std::cout << "Beklenen: " << expected
            << ", Hata: " << std::setprecision(4) << error
            << " (%" << std::setprecision(2) << error_pct << ")" << std::endl;

  return 0;
}
