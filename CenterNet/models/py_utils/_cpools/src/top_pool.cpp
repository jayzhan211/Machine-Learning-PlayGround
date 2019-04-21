#include<torch/torch.h>
#include<vector>

std::vector<at::Tensor>top_pool_forward(at::Tensor input){
    at::Tensor output = at::zeros_like(input);
    int64_t height = input.size(2);

}