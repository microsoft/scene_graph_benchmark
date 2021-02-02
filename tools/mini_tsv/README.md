# TSV Files

This folder shows a demo on how to generate and access TSV files. TSV denotes tab separated values. It is generic enough to represent a wide variety of datasets for image classification, object detection, and etc. The major advantage of using TSV file format is to allow only one big file to save all images or labels. This is very helpful for file transfer. 

## Data Format
Check [`tsv_demo.py`](tsv_demo.py) for a demo of how to generate and access tsv file. 

### train.tsv
For each row, there are two columns.
- **Image Key:** this is an unique key to identify the image. In the above demo, it is the image filename. This key is used to find the correspondence between different files of the same dataset. 
- **Encoded Image String:** we use base64 encoded string to represent the image. It can be easily decoded to the original image. 

### train.lineidx
Each file ended with .tsv is often coupled with a .lineidx file. The .lineidx file specifies the location of each row, which is useful to quickly access any row in the .tsv file using seek function. Check `maskrcnn_benchmark/structures/tsv_file.py` for details. 

### train.label.tsv
For each row, there are two columns. Image key and json string of a list of dictionary with label information. For object detection, each box is represented with at least two keys: "rect" in xyxy mode denoting the box coordinates and "class" denoting the class name.

### train.labelmap.tsv
This file consists of a list of class names. In .tsv file we use the class names for readability. During training, it will load the labelmap and convert the class name into a number in the same order as specified in labelmap for classifier training. 

### train.hw.tsv
This file consists of two columns: image key and image shape (height and width information). Each row looks like this: `image_key [{"width":100, "height": 200}]`. This file is useful to pre-group the images into landscape and portrait images to avoid unnecessary padding. It is also useful in box resizing. See function `generate_hw_file` in `maskrcnn_benchmark/structures/tsv_file_ops.py`.

### train.linelist.tsv
This file consists of a list of image line indexes. It is useful to select a subset of images for training. It is also useful to over-sample images by duplicating the indexes. See function `generate_linelist_file` in `maskrcnn_benchmark/structures/tsv_file_ops.py` for example of generating a linelist file of images with annotations.

### train.yaml
A yaml file consists of the above files for easy dataset setup. See `TSVYamlDatset` in `maskrcnn_benchmark/data/datasets/tsv_dataset.py` for details. 

