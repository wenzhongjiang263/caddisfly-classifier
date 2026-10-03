# Caddisfly Image Classifier — 免费展示版

这是用于 Streamlit Community Cloud 的网站运行仓库。访客打开网址即可上传图片识别，无需安装软件。
模型为 ConvNeXt-Tiny，512 输入，25 个分类标签，包含 mottled 形态标签；使用两个检测器进行预处理。
原始数据、训练缓存、旧模型和本机虚拟环境不包含在此部署仓库中。

## 1. 创建 GitHub 仓库

登录 https://github.com ，点击右上角 `+` → `New repository`。
仓库名建议 `caddisfly-classifier`，选择 Public，点击 `Create repository`。
你的仓库链接将是 `https://github.com/你的用户名/caddisfly-classifier`。

把 `output/cloud_demo/repository` **里面的文件和目录**上传到仓库根目录。
不要上传它的父目录，不要上传完整研究项目，也不要把 release_assets 放进源码仓库。
保留 `.streamlit` 文件夹；使用 Git 或 GitHub Desktop 上传可避免遗漏隐藏文件。
入口 `cloud_app.py`、`requirements.txt`、`packages.txt` 应位于仓库根目录。

## 2. 上传模型 Release

在仓库页面点击 Releases → Create a new release，标签填写 `model-v1`。
上传 `output/cloud_demo/release_assets` 中的三个文件：

- last_deploy_calibrated.pt
- detector_yolo11n_512.pt
- detector_yolo11n_1024.pt

发布 Release。下载基地址是：
`https://github.com/你的用户名/你的仓库名/releases/download/model-v1`
Release 必须公开可下载；不要使用 Draft Release 或页面浏览地址 `/releases/tag/`。
网站使用 `deployment/model_assets.json` 中的大小和 SHA-256 验证三个文件。
无需访客下载，模型在云端首次进入识别页时自动下载。

## 3. 部署到 Streamlit Community Cloud

进入 https://share.streamlit.io ，登录并连接 GitHub。
点击 Create app，选择仓库、实际分支名和入口 **cloud_app.py**。
在 Advanced settings 中选择 **Python 3.12**。
在 Secrets 中粘贴下面内容，替换真实用户名和仓库名：

```toml
[deployment]
release_base_url = "https://github.com/你的用户名/你的仓库名/releases/download/model-v1"
```

点击 Deploy。成功后分享分配的 `.streamlit.app` 地址即可。
如果出现条款、账户授权或隐私设置，请由账户持有人核对并完成。

## 4. 上线验收

- 首页照片正常显示，开始识别能进入上传页。
- 上传真实照片，CPU 模型正常加载并返回结果。
- 低置信度时可补充同一标本的第二视角。
- 联合结果、参考图、地图和重新识别正常工作。
- 检查冷启动时间、识别时间及云端日志中的内存占用。

免费平台的速度、资源容量和休眠行为以实际运行和平台当时规则为准。
若报内存不足，先查看日志，不要删除检测器或改变模型来掩盖问题。
上传限制为 25 MB。首页使用项目指定原图，未压缩，较慢连接下加载可能较慢。
展示版隐藏反馈表单，因为平台本地磁盘不用于长期保存反馈。
本地研究项目仍保留原有反馈功能。

## 本地准备与验证状态

部署目录由 `tools/prepare_cloud_demo.py` 生成，模型权重与源码分开放置。
模型下载器有网络超时、文件大小及 SHA-256 校验，失败下载不会替换已有模型。
已在 Windows Python 3.12 环境验证 CPU 模型推理；Linux 安装与平台容量需上线后验证。

官方说明：
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
