<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="files/t45logo.png" width="100%">
</a>
</div>
<br/>

<!-- # CholecT45 
------------------------------------------------
An endoscopic video dataset for surgical action triplet recognition.

[![](https://img.shields.io/badge/UNDER-CONSTRUCTION-blue?style=for-the-badge)](https://hamzamohdzubair.github.io/redant/)
-->

------------------------------------------------


This folder includes: 
- CholecT45 dataset:
  - **data**: 45 cholecystectomy videos
  - **triplet**: triplet annotations on 45 videos
  - **instrument**: tool annotations on 45 videos
  - **verb**: action annotations on 45 videos
  - **target**: target annotations on 45 videos
  - **dict**: id-to-name mapping files
  - a LICENCE file
  - a README file


<details>
  <summary>  
  Expand this to visualize the dataset directory structure.
  </summary>
  
  ```
    ──CholecT45
        ├───data
        │   ├───VID01
        │   │   ├───000000.png
        │   │   ├───000001.png
        │   │   ├───000002.png
        │   │   ├───
        │   │   └───N.png
        │   ├───VID02
        │   │   ├───000000.png
        │   │   ├───000001.png
        │   │   ├───000002.png
        │   │   ├───
        │   │   └───N.png
        │   ├───
        │   ├───
        │   ├───
        │   |
        │   └───VIDN
        │       ├───000000.png
        │       ├───000001.png
        │       ├───000002.png
        │       ├───
        │       └───N.png
        |
        ├───triplet
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───instrument
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───verb
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───target
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───dict
        │   ├───triplet.txt
        │   ├───instrument.txt
        │   ├───verb.txt
        │   ├───target.txt
        │   └───maps.txt
        |
        ├───LICENSE
        └───README.md
   ```
</details>

<br>


The superset and a more complete version of the dataset, CholecT50, is now available [here](https://github.com/CAMMA-public/cholect50/blob/master/README.md).

# News and Updates:
- [ **29/04/2022** ]: Added PyTorch dataloader for the dataset.
- [ **02/05/2022** ]: Added TensorFlow v2 dataloader for the dataset.

<br>

---

------------------------------------------------
## Download Dataset

The CholecT45 dataset has been officially released for public use on April 12, 2022. If you wish to have access to this dataset, please kindly fill the request [form](https://forms.gle/jTdPJnZCmSe2Daw7A).

<br><br>

------------------------------------------------
DATASET DESCRIPTION
================================================

The CholecT45 dataset contains 45 videos of cholecystectomy procedures collected in Strasbourg, France. It is a subset of Cholec80 [1] dataset.
CholecT45 is an extension of CholecT40 [2] with additional videos and standardized annotations.
The images are extracted at 1 fps from the videos and annotated with triplet information about surgical actions in the format of <instrument, verb, target>. 
In total, there are 90489 frames and 127385 triplet instances in the dataset.
To ensure anonymity, frames corresponding to out-of-body views are entirely blacked (RGB 0 0 0) out.

<br>


------------------------------------------------
Triplet Annotations
================================================

Each triplet annotation file contains a table, consisting of 101 columns. 
Every row contains an annotation for an image in the video. 
The first column indicates the frame index of the annotated image in the video. The frame index is defined under a 0-based system. 
The other 100 columns are the binary labels for the triplets (0=not present; 1=present).
This last 100 columns sequentially correspond to the triplets IDs (0..99) and names as contained in the mapping file (dict/triplet.txt)

For simplicity, we also provide annotations for the various components of the triplets: instrument, verb and target.

<br>

------------------------------------------------
Instrument Annotations
================================================

Each instrument annotation file contains a table, consisting of 7 columns. 
Every row contains an annotation for an image in the video. 
The first column indicates the frame index of the annotated image in the video. The frame index is defined under a 0-based system. 
The other 6 columns are the binary labels for the instrument (0=not present; 1=present).
This last 6 columns sequentially correspond to the instrument IDs (0..5) and names as contained in the mapping file (dict/instrument.txt)

<br>


------------------------------------------------
Verb Annotations
================================================

Each verb annotation file contains a table, consisting of 11 columns. 
Every row contains an annotation for an image in the video. 
The first column indicates the frame index of the annotated image in the video. The frame index is defined under a 0-based system. 
The other 10 columns are the binary labels for the verb (0=not present; 1=present).
This last 10 columns sequentially correspond to the verb IDs (0..9) and names as contained in the mapping file (dict/verb.txt)

<br>



------------------------------------------------
Target Annotations
================================================

Each target annotation file contains a table, consisting of 16 columns. 
Every row contains an annotation for an image in the video. 
The first column indicates the frame index of the annotated image in the video. The frame index is defined under a 0-based system. 
The other 15 columns are the binary labels for the target (0=not present; 1=present).
This last 15 columns sequentially correspond to the target IDs (0..14) and names as contained in the mapping file (dict/target.txt)

<br>



------------------------------------------------
Dict
================================================

The dict folder contains mapping of the label ID to full name for various tasks viz-a-viz: triplet, instrument, verb, and target. 
Specifically, the maps.txt file contains a table, consisting of 6 columns for mapping triplet IDs to their component IDs.
This is useful for decomposing a triplet to its constituting components.
The first column indicates the triplet ID (that is instrument-verb-target paring IDs).
The second column indicates the instrument ID.
The third column indicates the verb IDs.
The fourth column indicates the target IDs.
The fifth column indicates the instrument-verb pairing IDs.
The sixth column indicates the instrument-target pairing IDs.

Example usage: 
The first row in the maps.txt shows:
1,0,2,0,2,0
This means that triplet iD 1 can be mapped to <0, 2, 0> which is {grasper, dissect, gallbladder}.



<br>



------------------------------------------------
License and References
================================================

This dataset could only be generated thanks to the continuous support from our surgical partners. In order to properly credit the authors and clinicians for their efforts, you are kindly requested to cite the work that led to the generation of this dataset:
- C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. Medical Image Analysis, 78 (2022) 102433.
   [![Journal Publication](https://img.shields.io/badge/Elsevier-Medical%20Image%20Analysis-orange)](https://doi.org/10.1016/j.media.2022.102433)
   [![Read on ArXiv](https://img.shields.io/badge/arxiv-2109.03223-red)](https://arxiv.org/abs/2109.03223) 
   [![GitHub](https://img.shields.io/badge/github-rendezvous-blue)](https://github.com/CAMMA-public/rendezvous)
   [![Result Demo](https://img.shields.io/youtube/views/d_yHdJtCa98?label=video%20demo&style=social)](https://www.youtube.com/watch?v=d_yHdJtCa98&t=61s)


The cholecT45 dataset is publicly released under the Creative Commons license [CC BY-NC-SA 4.0 LICENSE](https://creativecommons.org/licenses/by-nc-sa/4.0/). This implies that:
- the dataset cannot be used for commercial purposes,
- the dataset can be transformed (additional annotations, etc.),
- the dataset can be redistributed as long as it is redistributed under the same [license](LICENSE) with the obligation to cite the contributing work which led to the generation of the cholecT45 dataset (mentioned above).

By downloading and using this dataset, you agree on these terms and conditions.

<br>


------------------------------------------------
Dataset Splits and Baselines
================================================
The official splits of the dataset for deep learning models is provided in the paper:
- C.I. Nwoye, N. Padoy. Data Splits and Metrics for Benchmarking Methods on Surgical Action Triplet Datasets. arXiv PrePrint 2022.
[![Read on ArXiv](https://img.shields.io/badge/arxiv-2204.05235-red)](https://arxiv.org/abs/2204.05235) 

The paper provides extended experiments on the baseline methods using the official dataset splits.

![](files/cv.png)
**Fig. 1**: Cross-validation experiment schedule for CholecT50. For CholecT45, remove the last video in each fold.
<br>



------------------------------------------------
Data Loader
================================================
We provide data loader for the following frameworks:
- PyTorch :  [`dataloader_pth.py`](dataloader_pth.py)
- TensorFlow v1 & v2 :  [`dataloader_tf.py`](dataloader_tf.py)
- ...

... *if you use any part of this code, please cite the paper associated with the CholecT50 dataset.*

### Requirements:
- pillow
- torch & torchvision `for pyTorch users`
- tensorflow `for TensorFlow users`


### Usage

``` python
import ivtmetrics # install using: pip install ivtmetrics

# for PyTorch
import dataloader_pth as dataloader
from torch.utils.data import DataLoader

# for TensorFlow v1 & v2
import tensorflow as tf
import dataloader_tf as dataloader
```

<br>

**Initialize the metrics library**
```python    
metrics = ivtmetrics.Recognition(num_class=100)
```

<br>

**Build dataset pipeline**

Loading the cholect45 cross-validation variant with test set as fold 1 as follows:

```python
# initialize dataset: 
dataset = dataloader.CholecT50( 
          dataset_dir="/path/to/your/downloaded/dataset/cholect45/", 
          dataset_variant="cholect45-crossval",
          test_fold=1,
          augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
          )

# build dataset
train_dataset, val_dataset, test_dataset = dataset.build()
```

List of currently supported data augumentations:
 - use `dataset.list_augmentations()` to see the full list.
 - use `dataset.list_dataset_variants()` to see all the supported dataset variants

<br>

**Wrap as default data loader**

 - *PyTorch :*


 ```python
# train and val data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, prefetch_factor=3,)
val_dataloader   = DataLoader(val_dataset, batch_size=32, shuffle=True)

# test data set is built per video, so load differently
test_dataloaders = []
for video_dataset in test_dataset:
  test_dataloader = DataLoader(video_dataset, batch_size=32, shuffle=False)
  test_dataloaders.append(test_dataloader)
``` 

 - *TensorFlow v2 :*

```python
# train and val data loaders
train_dataloader = train_dataset.shuffle(20).batch(32).prefetch(5) # see tf.data.Dataset for more options
val_dataloader   = val_dataset.batch(32)

# test data set is built per video, so load differently
test_dataloaders = []
for video_dataset in test_dataset:
  test_dataloader = video_dataset.batch(32).prefetch(5)
  test_dataloaders.append(test_dataloader)
``` 

- *TensorFlow v1 :*

```python
# construct an iterator and train data loaders
train_dataset = train_dataset.shuffle(20).batch(32).prefetch(5) # see tf.data.Dataset for more options
iterator      = tf.data.Iterator.from_structure(output_types=train_dataset.output_types, output_shapes=train_dataset.output_shapes) 
init_train    = iterator.make_initializer(train_dataset) 

# using the same iterator, construct val data loaders
val_dataset = val_dataset.batch(32)
init_val    = iterator.make_initializer(val_dataset) 

# test data set is built per video, so load differently with the same iterator
init_tests = []
for video_dataset in test_dataset:
  video_dataset = video_dataset.batch(32)
  init          = iterator.make_initializer(video_dataset) 
  init_tests.append(init)

# outputs from iterator
tf_img, (tf_label_i, tf_label_v, tf_label_t, tf_label_ivt) = iterator.get_next()
```
<br>

**Reading the dataset during experiment (PyTorch and TensorFlow v2 only)**:
```python
total_epochs = 10
model = YourFantasticModel(...)
for epoch in range(total_epochs):
  # training
  for batch, (img, (label_i, label_v, label_t, label_ivt)) in enumerate(train_dataloader):
    pred_ivt = model(img)
    loss(label_ivt, pred_ivt)
      
  # validate
  for batch, (img, (label_i, label_v, label_t, label_ivt)) in enumerate(val_dataloader):
    pred_ivt = model(img)

# testing: test per video
for test_dataloader in test_dataloaders:
  for batch, (img, (label_i, label_v, label_t, label_ivt)) in enumerate(test_dataloader):
    pred_ivt = model(img)
    metrics.update(label_ivt, pred_ivt)
  metrics.video_end() # important for video-wise AP
```

<br>


**Reading the dataset during experiment (TensorFlow v1 only)**:

```python
total_epochs = 10
model = YourFantasticModel(...)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for epoch in range(total_epochs):
    # training
    sess.run([tf.local_variables_initializer(), init_train])
    while True:
      try:
        img, label_i, label_v, label_t, label_ivt = \
          sess.run([tf_img, tf_label_i, tf_label_v, tf_label_t, tf_label_ivt])
        pred_ivt = model(img)
        loss(label_ivt, pred_ivt)
      except tf.errors.OutOfRangeError: 
        # do what ever you want after an epoch train here  
        break      

    # validate
    sess.run([tf.local_variables_initializer(), init_val])
    while True:
      try:
        img, label_i, label_v, label_t, label_ivt = \
          sess.run([tf_img, tf_label_i, tf_label_v, tf_label_t, tf_label_ivt])
        pred_ivt = model(img)
        loss(label_ivt, pred_ivt)
      except tf.errors.OutOfRangeError: 
        # do what ever you want after an epoch val here
        break  

  # testing: test per video  
  for init_test in init_tests:
    sess.run([tf.local_variables_initializer(), init_test])
    while True:
      try:
        img, label_i, label_v, label_t, label_ivt = \
          sess.run([tf_img, tf_label_i, tf_label_v, tf_label_t, tf_label_ivt])
        pred_ivt = model(img)
        metrics.update(label_ivt, pred_ivt)
      except tf.errors.OutOfRangeError:
        metrics.video_end() # important for video-wise AP
        break    
```

<br>

**Obtain results**: 
```python
AP_i    = metrics.compute_video_AP("i")["AP"]
mAP_it  = metrics.compute_video_AP("it")["mAP"]
mAP_ivt = metrics.compute_video_AP("ivt")["mAP"]
```
<br> 

- For TensorFlow, we recommend the use of [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) for high-speed data loading.
- See [ivtmetrics](https://github.com/CAMMA-public/ivtmetrics) github for more details on metrics usage.

<br>



------------------------------------------------
Additional Information
================================================
The dataset has been used for MICCAI EndoVis challenge [CholecTriplet2021](https://cholectriplet2021.grand-challenge.org).
During the challenge, a lot of deep learning methods were presented on the dataset. 
The challenge report is published at:
- C.I. Nwoye, D. Alapatt, T. Yu, A. Vardazaryan, F. Xia, ... , C. Gonzalez, N. Padoy. CholecTriplet2021: a benchmark challenge for surgical action triplet recognition. arXiv PrePrint 2022. 
[![Read on ArXiv](https://img.shields.io/badge/arxiv-2204.04746-red)](https://arxiv.org/abs/2204.04746) 

<br>



------------------------------------------------
Acknowledgement
================================================

This work was supported by French state funds managed by BPI France (project CONDOR) and by the ANR (Labex CAMI, IHU Strasbourg, project DeepSurg, National AI Chair AI4ORSafety). We also thank the research teams of IHU and IRCAD for their help with the initial annotation of the dataset during the CONDOR project.

<br><br>
<img src="https://github.com/CAMMA-public/rendezvous/blob/main/files/ihu.png" width="6%" align="left" > 
<img src="https://github.com/CAMMA-public/rendezvous/blob/main/files/ANR-logo-2021-sigle.jpg" width="14%" align="left">
<img src="https://github.com/CAMMA-public/rendezvous/blob/main/files/condor.png" width="14%"  align="left">
<br>
<br><br>

------------------------------------------------
Contact
================================================

This dataset is maintained by the research group CAMMA: http://camma.u-strasbg.fr

Any updates regarding this dataset can be found here: http://camma.u-strasbg.fr/datasets

Any questions regarding the dataset can be sent to: camma.dataset@gmail.com




------------------------------------------------
References
================================================
* **[1]** A.P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, N. Padoy. EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. IEEE Trans. on Medical Imaging 2016.
  ```
  @article{twinanda2016endonet,
    title={Endonet: a deep architecture for recognition tasks on laparoscopic videos},
    author={Twinanda, Andru P and Shehata, Sherif and Mutter, Didier and Marescaux, Jacques and De Mathelin, Michel and Padoy, Nicolas},
    journal={IEEE transactions on medical imaging},
    volume={36},
    number={1},
    pages={86--97},
    year={2016}
  }
  ```

* **[2]** C.I. Nwoye, T. Yu, C. Gonzalez, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. Recognition of instrument-tissue interactions in endoscopic videos via action triplets.International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2020.
  ```
  @inproceedings{nwoye2020recognition,
     title={Recognition of instrument-tissue interactions in endoscopic videos via action triplets},
     author={Nwoye, Chinedu Innocent and Gonzalez, Cristians and Yu, Tong and Mascagni, Pietro and Mutter, Didier and Marescaux, Jacques and Padoy, Nicolas},
     booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
     pages={364--374},
     year={2020},
     organization={Springer}
  }
  ```

* **[3]** C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. Medical Image Analysis 2022.
  ```
  @article{nwoye2021rendezvous,
    title={Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos},
    author={Nwoye, Chinedu Innocent and Yu, Tong and Gonzalez, Cristians and Seeliger, Barbara and Mascagni, Pietro and Mutter, Didier and Marescaux, Jacques and Padoy, Nicolas},
    journal={Medical Image Analysis},
    volume={78},
    pages={102433},
    year={2022}
  }
  ```

* **[4]** C.I. Nwoye, D. Alapatt, T. Yu, A. Vardazaryan, F. Xia, ... , D. Mutter, N. Padoy. CholecTriplet2021: A benchmark challenge for surgical action triplet recognition. arXiv PrePrint arXiv:2204.04746. 2022.
  ```
  @article{nwoye2022cholectriplet2021,
    title={CholecTriplet2021: a benchmark challenge for surgical action triplet recognition},
    author={Nwoye, Chinedu Innocent and Alapatt, Deepak and Vardazaryan, Armine ... Gonzalez, Cristians and Padoy, Nicolas},
    journal={arXiv preprint arXiv:2204.04746},
    year={2022}
  }
  ```

* **[5]** C.I. Nwoye, N. Padoy. Data Splits and Metrics for Benchmarking Methods on Surgical Action Triplet Datasets. arXiv PrePrint arXiv:2204.05235. 2022.
  ```
  @article{nwoye2022data,
    title={Data Splits and Metrics for Benchmarking Methods on Surgical Action Triplet Datasets},
    author={Nwoye, Chinedu Innocent and Padoy, Nicolas},
    journal={arXiv preprint arXiv:2204.05235},
    year={2022}
  }
  ```
