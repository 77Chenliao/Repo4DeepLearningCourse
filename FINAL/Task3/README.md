由于文件过大且网络不稳定,github仍在上传中,目前可见网盘链接:
链接: https://pan.baidu.com/s/1CQInanI_E29iLxqkaspuAQ?pwd=nuby 提取码: nuby

数据处理(需要安装COLMAP)：
ns-process-data images --data /Users/jiaruinan/Library/Mobile Documents/com~apple~CloudDocs/Courses/24年春/神经网络与深度学习/nerf/images2 --output-dir /Users/jiaruinan/Library/Mobile Documents/com~apple~CloudDocs/Courses/24年春/神经网络与深度学习/nerf/processed_llfftest_2

处理好后的database.db文件过大无法上传github,见网盘链接task3同名文件

数据训练:
ns-train nerfacto --data /Users/jiaruinan/Library/Mobile Documents/com~apple~CloudDocs/Courses/24年春/神经网络与深度学习/nerf/processed_llfftest_2

/Users/jiaruinan/Documents/GitHub/Repo4DeepLearningCourse/FINAL/Task3/outputs/processed_llfftest_2/nerfacto/2024-06-29_184229/nerfstudio_models
下为ckpt文件,由于文件过大,见网盘链接task3

数据评估:
ns-eval --load-config=/Users/jiaruinan/Library/Mobile Documents/com~apple~CloudDocs/Courses/24年春/神经网络与深度学习/nerf/processed_llfftest_2/nerfacto/2024-06-29_184229/config.yml --output-path=output.json:

训练过程可视化:
tensorboard --logdir=/Users/jiaruinan/Library/Mobile Documents/com~apple~CloudDocs/Courses/24年春/神经网络与深度学习/nerf/outputs/processed_llfftest_2/nerfacto/2024-06-29_184229/events.out.tfevents.1719657767.featurize.52336.0

tensorboard文件过大无法上传github,见网盘链接task3同名文件


视频导出:
ns-viewer --load-config /Users/jiaruinan/Library/Mobile Documents/com~apple~CloudDocs/Courses/24年春/神经网络与深度学习/nerf/outputs/processed_llfftest_2/nerfacto/2024-06-29_184229/config.yml