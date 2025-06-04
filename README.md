本项目是对GRAPE: A multi-modal dataset of longitudinal follow-up visual field and fundus images for glaucoma management 论文的优化
所有的代码和数据都是在这篇论文的基础上修改的
在VF值进展估计中，展示了以resnet50，resnet50+transformer对CFP,ROI以及OD/OC数据以及对其进行styleGAN的分类，更过程性
在VF值估计中，只展示了最好结果的代码。两个python文件只有输出不同，其余都没变
（想要获取图片style GAN后的数据，可自行前往https://github.com/bethgelab/stylize-datasets.git 对数据进行处理）
