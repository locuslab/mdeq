# Multiscale Deep Equilibrium Models

This repository contains the code for the multiscale deep equilibrium (MDEQ) model proposed in the paper [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656) by Shaojie Bai, Vladlen Koltun and J. Zico Kolter.

Is implicit deep learning relevant for general, large-scale pattern recognition tasks? We propose the multiscale deep equilibrium (MDEQ) model, which expands upon the DEQ formulation substantially to introduce simultaneous equilibrium modeling of multiple signal resolutions. Specifically, MDEQ solves for and backpropagates through synchronized equilibria of multiple feature representation streams. Such structure rectifies one of the major drawbacks of DEQ, and provide natural hierarchical interfaces for auxiliary losses and compound training procedures (e.g., pretraining and finetuning). Our experiment demonstrate for the first time that "shallow" implicit models can scale to and achieve near-SOTA results on practical computer vision tasks (e.g., megapixel images on Cityscapes segmentation).

We provide in this repo the implementation and the links to the pretrained classification & segmentation MDEQ models.

If you find thie repository useful for your research, please consider citing our work:

```
@inproceedings{bai2020multiscale,
    author    = {Shaojie Bai and Vladlen Koltun and J. Zico Kolter},
    title     = {Multiscale Deep Equilibrium Models},
    journal   = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2020},
}
```


### Overview

The structure of a multiscale deep equilibrium model (MDEQ) is shown below. All components of the model are shown in this figure (in practice, we use n=4).

<p align="center"/>
<img src='images/mdeq_structure.png' width="800px"/>
    
    
### Examples
    
Some examples of MDEQ segmentation results on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset.

<p align="center"/>
<img src='images/segmentation_example.png' width="800px"/>


### Requirements

PyTorch >=1.4.0, torchvision >= 0.4.0


### Datasets

- **CIFAR-10**: We download the CIFAR-10 dataset using PyTorch's torchvision package (included in this repo).
- **ImageNet** We follow the implementation from the [PyTorch ImageNet Training repo](https://github.com/pytorch/examples/tree/master/imagenet).
- **Cityscapes**: We download the Cityscapes dataset from its official website and process it according to [this repo](https://github.com/mcordts/cityscapesScripts). Cityscapes dataset additionally require a `list` folder that aligns each original image with its corresponding labeled segmented image. This `list` folder can be downloaded [here](https://drive.google.com/drive/folders/1nOOkjSaWQdEI4Q0ay3bnOQHWk7UIH_dV?usp=sharing).

All datasets should be downloaded, processed and put in the respective `data/[DATASET_NAME]` directory. The `data/` directory should look like the following:
```
data/
  cityscapes/
  imagenet/
  ...          (other datasets)
  list/        (see above)
```

### Usage

All experiment settings are provided in the `.yaml` files under the `experiments/` folder.

To train an MDEQ classification model on ImageNet/CIFAR-10, do

```sh
python tools/cls_train.py --cfg experiments/[DATASET_NAME]/[CONFIG_FILE_NAME].yaml
```

To train an MDEQ segmentation model on Cityscapes, do

```sh
python -m torch.distributed.launch --nproc_per_node=4 tools/seg_train.py --cfg experiments/[DATASET_NAME]/[CONFIG_FILE_NAME].yaml
```
Similarly, to test the model and generate segmentation results on Cityscapes, do

```sh
python tools/seg_test.py --cfg experiments/[DATASET_NAME]/[CONFIG_FILE_NAME].yaml
```
You can (and probably should) initiate the Cityscapes training with an ImageNet-pretrained MDEQ. You need to extract the state dict from the ImageNet checkpointed model, and set the `MODEL.PRETRAINED` entry in Cityscapes yaml file to this state dict on disk.

The model implementation and MDEQ's algorithmic components (e.g., L-Broyden's method) can be found in `lib/`.


### Pre-trained Models

We provide some reasonably good pre-trained weights here so that one can quickly play with DEQs without training from scratch.

| Description   | Task              | Dataset             | Model                   |
| ------------- | ----------------- | ------------------- | ----------------------- |
| MDEQ-XL | ImageNet Classification | ImageNet | [download (.pkl)](https://drive.google.com/file/d/1MBUFBOAAI8m2eccNbHePrukpOiAzPbji/view?usp=sharing) |
| MDEQ-XL | Cityscapes(val) Segmentation | Cityscapes | [download (.pkl)](https://drive.google.com/file/d/1Gu7pJLGvXBbU_sPxNfjiaROJwEwak2Z8/view?usp=sharing) |

**Example of how to use the pretrained ImageNet model to train on Cityscapes**:
  1. Download the pretrained ImageNet `.pkl` file.
  2. Put the model under `pretrained_models/` folder with some file name `[FILENAME]`.
  3. In the corresponding `experiments/cityscapes/cls_MDEQ_XL.yaml`, set `PRETRAINED` to `"pretrained_models/[FILENAME]"`. Make sure you **don't** make it the `MODEL_FILE`.
  4. Run the MDEQ segmentation training command (see the "Usage" section above).

(We'll soon update with the pretrained MDEQ-Large and MDEQ-Small ImageNet models!)


### Tips:

- To load the Cityscapes pretrained model, download the `.pkl` file below and specify the path in `config.[TRAIN/TEST].MODEL_FILE` (which is `''` by default) in the `.yaml` files.
- The difference between `[TRAIN/TEST].MODEL_FILE` and `MODEL.PRETRAINED` arguments in the yaml files: the former is used to load all of the model parameters; the latter is for compound training (e.g., when transferring from ImageNet to Cityscapes, we want to discard the final classifier FC layer). 
- The repo supports checkpointing of models at each epoch. One can resume from a previously saved checkpoint by turning on the `TRAIN.RESUME` argument in the yaml files.
- Just like DEQs, the MDEQ models can be slower than explicit deep networks, and even more so as the image size increases (because larger images typically require more Broyden iterations to converge well; see Figure 5 in the paper). But one can play with the forward and backward thresholds to adjust the runtime.


### Acknowledgement

Some utilization code (e.g., model summary and yaml processing) of this repo were modified from the [HRNet repo](https://github.com/HRNet/HRNet-Semantic-Segmentation) and the [DEQ repo](https://github.com/locuslab/deq).
