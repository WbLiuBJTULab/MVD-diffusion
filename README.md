# A Multi-View Dependency Diffusion Model for Point Cloud Completion


[Project]() | [Paper]() 

Implementation of Point Cloud Completion MVD-diffusion



## Requirements:

Make sure the following environments are installed.

```
python==3.6.13
pytorch==1.8.1
torchvision==0.9.1
cudatoolkit==11.1
matplotlib==2.2.5
tqdm==4.32.2
open3d==0.9.0
trimesh=3.7.14
scipy==1.5.4
```

Install PyTorchEMD by
```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```

The code was tested on Unbuntu with Titan RTX. 

## Data

For completion, we use ShapeNet rendering provided by [GenRe](https://github.com/xiumingzhang/GenRe-ShapeHD).
We provide script `convert_cam_params.py` to process the provided data.

For training the model on shape completion, we need camera parameters for each view
which are not directly available. To obtain these, simply run 
```bash
$ python convert_cam_params.py --dataroot DATA_DIR --mitsuba_xml_root XML_DIR
```
which will create `..._cam_params.npz` in each provided data folder for each view.

## Pretrained models


## Training:

```bash
$ python train_completion_20240925_1100.py --category SELECT_CATEGORY 
```

Please refer to the python file for optimal training parameters.

## Testing:

```bash
$ python test_completion_20240925_1100.py --category SELECT_CATEGORY --model MODEL_PATH
```

## Results

## Reference

## Acknowledgement
