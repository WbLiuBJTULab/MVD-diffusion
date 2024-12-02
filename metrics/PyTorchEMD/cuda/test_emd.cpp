#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "emd.cpp"

#define CUDA_CHECK AT_CUDA_CHECK

at::Tensor TestApproxMatch() {
  at::Tensor xyz1 = torch::rand({16, 128, 3}).cuda();
  at::Tensor xyz2 = torch::rand({16, 256, 3}).cuda();

  return ApproxMatchForward(xyz1, xyz2);
}

at::Tensor TestMatchCost(const at::Tensor& match) {
  at::Tensor xyz1 = torch::rand({16, 128, 3}).cuda();
  at::Tensor xyz2 = torch::rand({16, 256, 3}).cuda();

  return MatchCostForward(xyz1, xyz2, match);
}

std::vector<at::Tensor> TestMatchCostBackward() {
  at::Tensor grad_cost = torch::ones({16}).cuda();
  at::Tensor match = torch::rand({16, 128, 256}).cuda();
  at::Tensor xyz1 = torch::rand({16, 128, 3}).cuda();
  at::Tensor xyz2 = torch::rand({16, 256, 3}).cuda();

  return MatchCostBackward(grad_cost, xyz1, xyz2, match);
}

int main() {
  if (!at::cuda::is_available()) {
    std::cerr << "CUDA is not available." << std::endl;
    return -1;
  }
  if (at::cuda::device_count() == 0) {
    std::cerr << "No GPU devices available." << std::endl;
    return -1;
  }
  CUDA_CHECK(cudaSetDevice(0));

  std::cout << "Testing ApproxMatch..." << std::endl;
  std::cout << TestApproxMatch() << std::endl;

  std::cout << "Testing MatchCost..." << std::endl;
  at::Tensor match = torch::rand({16, 128, 256}).cuda();
  std::cout << TestMatchCost(match) << std::endl;

  std::cout << "Testing MatchCostBackward..." << std::endl;
  std::vector<at::Tensor> grad = TestMatchCostBackward();
  for (const auto& g : grad) {
    std::cout << g << std::endl;
  }

  return 0;
}