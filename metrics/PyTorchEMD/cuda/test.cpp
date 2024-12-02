#include <iostream>
#include <torch/torch.h>    // libtorch头文件
#include <vector>
#include <ATen/cuda/CUDAContext.h>

using namespace torch;  // libtorch命名空间
using namespace std;

at::Tensor ApproxMatchForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2);

at::Tensor MatchCostForward(
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor match);

std::vector<at::Tensor> MatchCostBackward(
    const at::Tensor grad_cost,
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor match);

int main() {
    // 分别打印CUDA、cuDNN是否可用以及可用的CUDA设备个数
    // 可以发现函数名是和PyTorch里一样的
    cout << torch::cuda::is_available() << endl;
    cout << torch::cuda::cudnn_is_available() << endl;
    cout << torch::cuda::device_count() << endl;

    cout << "Hello" << std::endl;
    //at::Tensor xyz1 = torch::rand({16, 128, 3}).cuda();
    //cout << xyz1 << std::endl;

    cout << "Testing ApproxMatchForward..." << endl;
    cout << ApproxMatchForward(torch::rand({16, 128, 3}).cuda()
                               , torch::rand({16, 256, 3}).cuda()) << endl;
    cout << "Finish ApproxMatchForward " << endl;

    cout << "Testing MatchCostForward..." << endl;
    cout << MatchCostForward(torch::rand({16, 128, 3}).cuda()
                             , torch::rand({16, 256, 3}).cuda()
                             , torch::rand({16, 128, 256}).cuda()) << endl;
    cout << "Finish MatchCostForward " << endl;

    cout << "Testing MatchCostBackward..." << endl;
    cout << MatchCostBackward(torch::ones({16}).cuda()
                             , torch::rand({16, 128, 3}).cuda()
                             , torch::rand({16, 256, 3}).cuda()
                             , torch::rand({16, 128, 256}).cuda()) << endl;
    cout << "Finish MatchCostBackward " << endl;

    return 0;
}
