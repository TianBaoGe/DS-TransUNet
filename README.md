# DS-TransUNet
This repository contains the official code of DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation

## Requirements
* `Python==3.7.6`
* `Pytorch==1.8.0` && `CUDA 11.1`
* `timm==0.4.5`


## Experiments

### Kvasir-SEG
1. **Dataset**
	+ Downloading training dataset and move it into `./data`, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/17sUo2dLcwgPdO_fD4ySiS_4BVzc3wvwA/view?usp=sharing).
	+ Downloading testing dataset and move it into `./data` , which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1us5iOMWVh_4LAiACM-LQa73t1pLLPJ7l/view?usp=sharing).
2. **Testing**
	+ Downloading our trained DS-TransUNet-B from [Baidu Pan](https://pan.baidu.com/s/1EFZOX1C84mg1mVK6cAvpxg) (dd79), and move it into `./checkpoints`.
	+ run `test_kvasir.py`
	+ run `criteria.py` to get the DICE score, which uses [EvaluateSegmentation](https://github.com/Visceral-Project/EvaluateSegmentation). Or you can download our result images from [Baidu Pan](https://pan.baidu.com/s/1EFZOX1C84mg1mVK6cAvpxg) (dd79).
3. **Training**
	+ downloading `Swin-T` and `Swin-B` from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) or [Baidu Pan](https://pan.baidu.com/s/1CD52UXHnDp-oRhv0sHrLcw?pwd=ji2g) (ji2g) to `./checkpoints`.
	+ run `train_kvasir.py`


Code of other tasks will be comming soon.


## Reference
Some of the codes in this repo are borrowed from:
* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [PraNet](https://github.com/DengPingFan/PraNet)
* [TransFuse](https://github.com/Rayicer/TransFuse)


## Citation
Please consider citing us if you find this work helpful:

```bibtex
@article{lin2022ds,
  title={DS-TransUNet: Dual Swin Transformer U-Net for Medical Image Segmentation},
  author={Lin, Ailiang and Chen, Bingzhi and Xu, Jiayu and Zhang, Zheng and Lu, Guangming and Zhang, David},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2022},
  publisher={IEEE}
}
```

## Questions
Please drop an email to tianbaoge24@gmail.com

