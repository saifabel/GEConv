# GEConv: Geometric Edge Convolution for Rigid Transformation Invariant Features in 3D Point Clouds

# Datasets
Get the datasets and place them in the data folder.

Get the preprocessed ModelNet40 from [here (425MB)](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip). Move the uncompressed data folder to ```data/modelnet40_ply_hdf5_2048```

Get the preprocessed ScanObjectNN; [All preprocessed data (12GB)](https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip). Or the hardest variant only [PB_T50_RS here (300MB)](https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip). Move the uncompressed data folder to ```data/h5_files```


Get the preprocessed ShapeNetParts from [here (674MB)](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). Move the uncompressed data folder to ```data/shapenetcore_partanno_segmentation_benchmark_v0_normal```

# Environment

* Setup environment (optional)
``` 
conda create -n geconv python=3.7 -y
conda activate geconv
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y
pip install pyyaml==5.4.1 h5py scikit-learn==0.24.2 scipy==1.5.2 tqdm matplotlib==3.4.2 open3d==0.8.0.0
```
# Point Cloud Classification
## ModelNet40
* Run the training script:


``` 
python main.py
```

* Run the evaluation script after training finished:

``` 
python main.py --eval=True --model_path=checkpoints/exp/models/model.t7
```


* Run the evaluation script with our pretrained model:

``` 
python main.py --eval=True --model_path=pretrained/classification.t7
```
## ScanObjectNN
* Enter the ScanObjectNN directory with:

``` 
cd ScanObjectNN_
```
* Run the training script:

``` 
python main_scannet.py --dataset=PB_T50_RS
#dataset options ['OBJ_ONLY', 'OBJ_BG', 'PB_T50_RS']
```
* Run the evaluation after training
```
python main_scannet.py --dataset=PB_T50_RS --eval=True
```
* Run the evaluation on our pretrained models
```
python main_scannet.py --dataset=PB_T50_RS --eval=True --pretrained=True
#dataset options ['OBJ_ONLY', 'OBJ_BG', 'PB_T50_RS']
```

# Point Cloud Parts Segmentation
* Run the training script:

``` 
python main_partsegmentation.py
```
* Run the evaluation script with our pretrained models:

``` 
python main_partsegmentation.py --eval=True --model_path=pretrained/part_seg.pth
```

# Point Registration
* Run the model with:
``` 
#enter the directory
cd GEConvReg_

#test the clean model
python main_REG.py

#test with noise
python main_REG.py --gaussian_noise=True

#test unseen
python main_REG.py --unseen=True

```