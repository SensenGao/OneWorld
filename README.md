<div align="center">
<h1>
OneWorld: Taming Scene Generation with 3D Unified Representation Autoencoder</h1>

[Sensen Gao*](https://sensengao.github.io/), [Zhaoqing Wang*](https://derrickwang005.github.io/), [Qihang Cao](https://scholar.google.com/citations?user=oegbT6AAAAAJ&hl=zh-CN), [Dongdong Yu](https://scholar.google.com/citations?user=B2RmjSYAAAAJ&hl=zh-CN),  [Changhu Wang](https://scholar.google.com/citations?user=DsVZkjAAAAAJ&hl=en), [Tongliang Liu📧](https://tongliang-liu.github.io/), [Mingming Gong📧](https://mingming-gong.github.io/), [Jiawang Bian📧](https://jwbian.net/)

<a href="https://sensengao.github.io/OneWorld/"><img src="https://img.shields.io/badge/Project_Page-yellowgreen" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2603.16099"><img src="https://img.shields.io/badge/arXiv-2603.16099-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/datasets/Sensen02/NVS-Refined"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-NVS--Refined-ff9800" alt="Dataset"></a>
<!-- <a href="https://huggingface.co/JaceyH919/Gen3R"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a> -->

<p align="center">
  <a href="">
    <img src="./asserts/Teaser.png" alt="Logo" width="100%">
  </a>
</p>

<p align="left">
<strong>TL;DR</strong>: **We present OneWorld, the first diffusion framework for 3D scene generation that operates directly in the feature space of pretrained 3D foundation models**, enabling unified modeling of geometry, appearance, and semantics for superior cross-view consistency.
</p>

</div>


## 📦 Dataset — NVS-Refined (Released!)

We have released **[NVS-Refined](https://huggingface.co/datasets/Sensen02/NVS-Refined)** 🤗 — a curated, high-quality dataset for novel view synthesis, distilled from **RealEstate10K, ACID, DL3DV, and SpatialVid**: sharp, high-fidelity clips with large camera motion and high aesthetic quality (and noticeably-blurry clips restored via **Streaming FlashVSR** rather than discarded).

👉 **[huggingface.co/datasets/Sensen02/NVS-Refined](https://huggingface.co/datasets/Sensen02/NVS-Refined)**

> 🚀 **Our model will be released in a short time — stay tuned!**

## ✅ TODO
- [x] Release **[NVS-Refined dataset](https://huggingface.co/datasets/Sensen02/NVS-Refined)** 🤗
- [ ] Release weights and code — in a short time, stay tuned!

## 🎓 Citation
Please cite our paper if you find this repository useful:

```bibtex
@misc{gao2026oneworldtamingscenegeneration,
      title={OneWorld: Taming Scene Generation with 3D Unified Representation Autoencoder}, 
      author={Sensen Gao and Zhaoqing Wang and Qihang Cao and Dongdong Yu and Changhu Wang and Tongliang Liu and Mingming Gong and Jiawang Bian},
      year={2026},
      eprint={2603.16099},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.16099}, 
}
```
