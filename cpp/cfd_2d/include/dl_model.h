#include <torch/torch.h>

struct DLModel : torch::nn::Module {
    DLModel() {
        int fil_num = 16;
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, fil_num, /*kernel_size=*/3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv6 = register_module("conv6", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv7 = register_module("conv7", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv8 = register_module("conv8", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv9 = register_module("conv9", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv10 = register_module("conv10", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv11 = register_module("conv11", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        conv12 = register_module("conv12", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, fil_num, /*kernel_size=*/3).padding(1)));
        reduce_channels = register_module("reduce_channels", torch::nn::Conv2d(torch::nn::Conv2dOptions(fil_num, 1, /*kernel_size=*/3).padding(1)));
        avgpool = register_module("avgpool", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2)));
        act = register_module("act", torch::nn::ReLU());
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x / (torch::max(x) + 1e-8);
        x = conv1->forward(x);
        auto la = act->forward(conv2->forward(x));
        auto lb = act->forward(conv3->forward(la));
        la = act->forward(conv4->forward(lb)) + la;
        lb = act->forward(conv5->forward(la));

        auto apa = avgpool->forward(lb);
        auto apb = act->forward(conv6->forward(apa));
        apa = act->forward(conv7->forward(apb)) + apa;
        apb = act->forward(conv8->forward(apa));
        apa = act->forward(conv9->forward(apb)) + apa;
        apb = act->forward(conv10->forward(apa));
        apa = act->forward(conv11->forward(apb)) + apa;
        apb = act->forward(conv12->forward(apa));
        apa = act->forward(conv11->forward(apb)) + apa;
        apb = act->forward(conv12->forward(apa));
        apa = act->forward(conv11->forward(apb)) + apa;

        auto upa = torch::nn::functional::interpolate(apa, torch::nn::functional::InterpolateFuncOptions().scale_factor(std::vector<double>{2}).mode(torch::kBicubic)) + lb;
        auto upb = act->forward(conv5->forward(upa));
        upa = act->forward(conv4->forward(upb)) + upa;
        upb = act->forward(conv3->forward(upa));
        upa = act->forward(conv2->forward(upb)) + upa;
        upb = act->forward(conv1->forward(reduce_channels->forward(upa)));
        upa = act->forward(conv2->forward(upb)) + upa;

        auto out = reduce_channels->forward(upa);

        // Uncomment if you need to set the boundary to 0
        // out.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0, torch::indexing::Slice()}, 0);
        // out.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), -1, torch::indexing::Slice()}, 0);
        // out.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}, 0);
        // out.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), -1}, 0);

        return out;
    }

    void copyWeightsFromScriptModule(const torch::jit::script::Module& scriptModule) {
        // Iterate through all parameters in the script module
        for (const auto& param : scriptModule.named_parameters()) {
            // Find the corresponding parameter in the current model and copy the data
            auto targetParam = this->named_parameters().find(param.name);
            if (targetParam != nullptr) {
                targetParam->copy_(param.value);
            }
        }

        // Iterate through all buffers in the script module
        for (const auto& buffer : scriptModule.named_buffers()) {
            // Find the corresponding buffer in the current model and copy the data
            auto targetBuffer = this->named_buffers().find(buffer.name);
            if (targetBuffer != nullptr) {
                targetBuffer->copy_(buffer.value);
            }
        }
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr}, conv6{nullptr}, conv7{nullptr}, conv8{nullptr}, conv9{nullptr}, conv10{nullptr}, conv11{nullptr}, conv12{nullptr}, reduce_channels{nullptr};
    torch::nn::AvgPool2d avgpool{nullptr};
    torch::nn::ReLU act{nullptr};
};
