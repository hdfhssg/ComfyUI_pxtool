# ComfyUI 插件节点 / ComfyUI Plugin Node

## 简介 / Introduction

**中文**
这是我个人使用的插件节点，在原有项目的基础上修改并扩展了部分功能。主要实现了以下内容：
- 在 ComfyUI 中重新复现 [Stable-Diffusion-Webui-Civitai-Helper](https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper) 项目中的部分功能
- 参考 [noob-wiki 数据集](https://huggingface.co/datasets/Laxhar/noob-wiki/tree/main) 自行实现了随机生成相关提示词的功能

**English**
This is a custom plugin node for ComfyUI that modifies and extends some features from existing projects. The main implementations include:
- Reproducing some features of the [Stable-Diffusion-Webui-Civitai-Helper](https://github.com/zixaphir/Stable-Diffusion-Webui-Civitai-Helper) project within ComfyUI
- Implementing a feature to randomly generate related prompt words by referencing the [noob-wiki dataset](https://huggingface.co/datasets/Laxhar/noob-wiki/tree/main)

## 特点 / Features

**中文**
- **功能修改：** 对原有插件的部分功能进行了改进和优化
- **功能复现：** 在 ComfyUI 环境中成功复现了其他项目的部分功能
- **提示词生成：** 实现了随机生成相关提示词的功能，提升使用体验

**English**
- **Feature Modification:** Improved and optimized some functionalities of the original plugin
- **Feature Reproduction:** Successfully reproduced some features of other projects within the ComfyUI environment
- **Prompt Word Generation:** Implemented a feature to randomly generate related prompt words, enhancing the user experience

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
