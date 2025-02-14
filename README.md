# ComfyUI 插件节点 / ComfyUI Plugin Node

## 简介 / Introduction

## 打包环境 / Packaged Environment

**中文**
- 打包了目前5090、5090D、5080等50系显卡绘图在Stable-Diffusion-Webui所需的torch2.6+cuda12.8的跑图环境，[演示链接](https://b23.tv/xKnhB18),[夸克网盘](https://pan.quark.cn/s/f57aa79a1ecb)，提取码：7aCZ，已经在5090D在测试能运行。
- 50系显卡在comfyui所需的环境已经打包，[演示链接](https://b23.tv/MelKmVW),[夸克网盘](https://pan.quark.cn/s/f57aa79a1ecb)，提取码：7aCZ,已经测试5090可以使用Flux-Trainer训练Lora模型。

**English**
- Packaged the drawing environment of the 50 series GPU such as 5090, 5090D, 5080, etc. required by Stable-Diffusion-Webui with torch2.6+cuda12.8, [Demo link](https://b23.tv/xKnhB18), [Baidu Netdisk](https://pan.baidu.com/s/1rEJlhPeWLcKDHCeqctx60g?pwd=pxhd), and it has been tested on 5090D.
- The environment required by the 50 series GPU in comfyui has been packaged, [Demo link](https://b23.tv/MelKmVW), [Baidu Netdisk](https://pan.baidu.com/s/1rEJlhPeWLcKDHCeqctx60g?pwd=pxhd), and it has been tested that 5090 can use Flux-Trainer to train the Lora model.


**中文**
这是我个人使用的插件节点，在原有项目的基础上修改并扩展了部分功能。主要实现了以下内容：
- 在 ComfyUI 中重新复现 [Stable-Diffusion-Webui-Civitai-Helper](https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper) 项目中的部分功能
- 参考 [noob-wiki 数据集](https://huggingface.co/datasets/Laxhar/noob-wiki/tree/main) 自行实现了随机生成相关提示词的功能
- 实现了能够fp8剪枝的Checkpoint加载器，以减少用户一些场景下的显存占用
- 复制修改[deepseek](https://github.com/ziwang-com/comfyui-deepseek-r1)、[Janus-Pro](https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro)部分节点

**English**
This is a custom plugin node for ComfyUI that modifies and extends some features from existing projects. The main implementations include:
- Reproducing some features of the [Stable-Diffusion-Webui-Civitai-Helper](https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper) project within ComfyUI
- Implementing a feature to randomly generate related prompt words by referencing the [noob-wiki dataset](https://huggingface.co/datasets/Laxhar/noob-wiki/tree/main)
- Implementing a Checkpoint loader that can perform fp8 pruning to reduce memory usage in some user scenarios
- Copied and modified [deepseek](https://github.com/ziwang-com/comfyui-deepseek-r1) and [Janus-Pro](https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro) nodes

## 特点 / Features

**中文**
- **功能复现：** 在 ComfyUI 环境中成功复现了[Stable-Diffusion-Webui-Civitai-Helper](https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper)项目中的扫描模型下载预览图的功能，并修改实现了扫描时会扫描下载预览视频的功能
- **提示词生成：** 按照 [noob-wiki 数据集](https://huggingface.co/datasets/Laxhar/noob-wiki/tree/main) 实现了随机生成相关提示词的功能，随机添加画师串，并随机添加权重等功能，增强了用户体验
- **checkpoint加载器：** 实现了能够fp8剪枝的Checkpoint加载器，支持FP8精度英伟达的小显卡用户(6GB显存左右甚至更小的4G显存)也可以运行SDXL模型，避免出现显存不足的错误，虽然降低了精度，生成的图片质量会降低，但是在一些场景下可以接受,unet以及clip进行FP8剪枝对比如下：

**English**
- **Feature Reproduction:** Successfully reproduced the model preview image download feature from the [Stable-Diffusion-Webui-Civitai-Helper](https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper) project within the ComfyUI environment, and modified the implementation to download preview videos during scanning
- **Prompt Word Generation:** Implemented a feature to randomly generate related prompt words by referencing the [noob-wiki dataset](https://huggingface.co/datasets/Laxhar/noob-wiki/tree/main), randomly adding artist strings, and randomly adding weights to enhance the user experience
- **Checkpoint Loader:** Implemented a Checkpoint loader that can perform fp8 pruning, allowing users with small NVIDIA GPUs (around 6GB of VRAM or even smaller 4GB VRAM) to run the SDXL model without running out of memory errors. Although the accuracy is reduced and the image quality is lowered, it is acceptable in some scenarios. The comparison between unet and clip for FP8 pruning is as follows:

<table>
  <tr>
    <td align="center">
      <div>fp8_e5m2+fp8_e5m2</div>
      <img src="img/fp8_e5m2+fp8_e5m2_00001_.png" width="250" alt="fp8_e5m2+fp8_e5m2">
    </td>
    <td align="center">
      <div>fp8_e5m2+fp8_e4m3fn</div>
      <img src="img/fp8_e5m2+fp8_e4m3fn_00001_.png" width="250" alt="fp8_e5m2+fp8_e4m3fn">
    </td>
    <td align="center">
      <div>fp8_e5m2+fp16</div>
      <img src="img/fp8_e5m2+fp16_00001_.png" width="250" alt="fp8_e5m2+fp16">
    </td>
  </tr>
  <tr>
    <td align="center">
      <div>fp8_e4m3fn+fp8_e5m2</div>
      <img src="img/fp8_e4m3fn+fp8_e5m2_00001_.png" width="250" alt="fp8_e4m3fn+fp8_e5m2">
    </td>
    <td align="center">
      <div>fp8_e4m3fn+fp8_e4m3fn</div>
      <img src="img/fp8_e4m3fn+fp8_e4m3fn_00001_.png" width="250" alt="fp8_e4m3fn+fp8_e4m3fn">
    </td>
    <td align="center">
      <div>fp8_e4m3fn+fp16</div>
      <img src="img/fp8_e4m3fn+fp16_00001_.png" width="250" alt="fp8_e4m3fn+fp16">
    </td>
  </tr>
  <tr>
    <td align="center">
      <div>fp16+fp8_e5m2</div>
      <img src="img/fp16+fp8_e5m2_00001_.png" width="250" alt="fp16+fp8_e5m2">
    </td>
    <td align="center">
      <div>fp16+fp8_e4m3fn</div>
      <img src="img/fp16+fp8_e4m3fn_00001_.png" width="250" alt="fp16+fp8_e4m3fn">
    </td>
    <td align="center">
      <div>fp16+fp16</div>
      <img src="img/fp16+fp16_00001_.png" width="250" alt="fp16+fp16">
    </td>
  </tr>
</table>



## 安装 / Installation

**中文**
1. 克隆仓库到本地:
```bash
git clone https://github.com/hdfhssg/ComfyUI_pxtool.git
```
2. 根据项目需求安装依赖:
```bash
pip install -r requirements.txt
```
3. 按照 ComfyUI 插件安装指南，将插件节点放入指定目录

**English**
1. Clone the repository:
```bash
git clone https://github.com/hdfhssg/ComfyUI_pxtool.git
```
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Follow the ComfyUI plugin installation guide to place the plugin node in the designated directory

## 参考 / References

- [Stable-Diffusion-Webui-Civitai-Helper](https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper)
- [noob-wiki 数据集](https://huggingface.co/datasets/Laxhar/noob-wiki/tree/main)

## 贡献 / Contribution

**中文**
欢迎大家提出建议或贡献代码。如果您有任何问题或想法，请提交 issue 或 pull request

**English**
Contributions, suggestions, and improvements are welcome. If you have any questions or ideas, please submit an issue or a pull request

## 许可证 / License

**中文**
本项目采用 Apache-2.0 许可证，详情请参阅 [LICENSE](LICENSE) 文件。

**English**
This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.
