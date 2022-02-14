# TableNet: end-to-end Table detection and Tabular data extraction from Scanned Document Images
Tensorflow Implementation of the Tablenet research Paper. <br>
Blog Link : https://ashishsalaskar1.medium.com/tablenet-end-to-end-table-detection-and-tabular-data-extraction-from-scanned-document-images-13c846e8f8f5 <Br>
You can find a demo of the model here : https://youtu.be/wvREr71zPe4 <br>

## Introduction
The task here is of extracting tables from the image as a two-step process. First, we train a model which will try to detect the table regions i.e area of the image which contains a table. Once table regions are found, another model would then Recognize the structure of the table. We can split the process into two steps
1) Table Detection (Detect Table region)
2) Table Structure Segmentation ( Detect the rows and columns in the detected table)

In our model we try to integrate both these steps into a single end-to-end trainable Deep Learning model.


## Dataset
We will be using the following datasets:
1. Marmot Dataset
2. Extended Marmots Dataset.
The Marmot dataset has around 2000 pages which were collected and the corresponding ground-truths were extracted utilizing a semi-automatic ground-truthing tool “Marmot”. The dataset is composed of Chinese and English pages at a proportion of about 1:1. The English pages were crawled from the Citeseer website. The dataset contains images as well as an XML document containing the coordinates of the annotated table regions for each image.

## Dataset Preparation
### Extracting table masks from Marmots dataset
To extract the bounding boxes from the table mask in the Marmots dataset. First, we need to convert the hex numbers into integers first. Marmots dataset first defines a CropBox which represents the content area. The bounding boxes have attributes that are relative to this crop Box <br>
1. First, we extract the CropBox coordinates and then we scale the bounding box coordinates so as to fit the actual page coordinates.
2. We then transform the bounding box formats and fetch the masks
3. In the end, after generating the mask we create a data frame to save the paths of each image file so that we can access these files easily.

## TF Pipeline
Before we start working on the model, we will first create a Tensorflow Pipeline using a Dataloader for our dataset. This would be more efficient when compared to loading each image individually and processing them while training. Here are the preprocessing we do <br>
1. Read the image, table, and column masks from the paths. Resize each one of them to 800x800 and then normalize the images.
2. We can make use of TF data pipeline methods like prefetch and parallel_calls to make the process more efficient.

## Creating our Model
The model architecture consists of 3 main parts
Backbone / Encoder: Here we will use a pre-trained VGG-19 model as the encoder. The fully connected layers in the VGG-19 model (layers after Pool 5) are replaced with two (1x1) Conv layers. These act as input which is distributed to the table and column branches.
1. Table Prediction Branch: Predicts the table masks
2. Column Prediction Branch: Predicts the Column masks <br>

![](https://miro.medium.com/max/963/1*WQSKELXvDEl1SUzli7RrSg.png)

- Both column and table detection branches first have a 1x1 Conv layer followed by a series of fractionally slided convolution layers which are used to upscale the image to its original dimension. In the case of the column detection branch, we have an additional 1x1 Conv layer. The main purpose of using 1x1 Conv layers before upscaling is to reduce the dimensions.
- In each step, we will use skip-connections. The Pool3 and Pool4 layers will be combined with the layers in the two branches. These skip connections allow the model to take into consideration both high-level and low-level features which lead to better results
- In the end, both the table detection branch and column detection branch will output image masks that are of the same dimensions as that of the original image. We will then use the Training table and column masks to calculate the losses and train the model using backpropagation.

## Improve the predicted masks
Once we get the predicted masks from the model, we try to use a few OpenCV functions to smoothen out the boxes. First, Gaussian blur is applied to the masks and then we try to find the contours in the mask. Once contours are found for each contour we fit a rectangular bounding rectangle which gives significantly better masks. Also, we ignore regions that are very small which indicate irregular points/patches. Using these post-processing steps the masks obtained are significantly better.

## **Observations**

**Distribution of the scores**
- We can see that out distribution of IoU scores is skewed towards the left i.e most values fall on the right side. This is a good thing as we can see that most scores are > 0.4 and very few scores < 0.3. Generally, IoU score of 0.5 is considered fairly good.
- We categorize our samples into categories based on the IoU scores
      1.Best : IoU score > 0.7
      2.Average : IoU score between 0.4 and 0.7
      3.Worst : IoU score < 0.4

- We see that out of 467 samples 87% of the images belonged to Best category, 12% in the average and only 1% in the worst category. This seems like a fairly good distribution in our case.

**Looking at images in the categories**
1. **BEST**
  - Most of these images have only 1 table
  - The tables are larger in size and occupy a mmajor part of the iamge size.
  - Also, when we look at the actual image as the size of the tables are large, there is very less text in the image exclusing the tables.

2. **AVERAGE**
  - We have multiple tables in a single image
  - Compared to BEST cases, the table sizes are significatly smaller.
  - Our model fails in cases where tables are adjacent to once another. In such cases we see that our model combines the masks of adjacent tables into one single table.
  - In some cases whitespace surrounding a table is also predicted as mask, because of which the predicted mask is slightly larger than the ground truth masks.

3. **WORST CASES**
  - We see that if table sizes are very small i.e small height, the our models fails to predict the masks properly.

4. **IoU = 0 or NaN**

  - As we assumed before that if IoU score is NaN or 0 its because there were no tables in the image. When we plot the images with IoU as NaN we see the same.
  - For some images we see that even though there were no tables, some tiny dots were predicted becuase of which IoU was 0.0
  
  
## Extract Text from the Tables
Once the table masks are predicted we use the PyTesseract library in order to extract the text from the Tables and save them into CSV files. We follow these steps
First, take the input image and get the predicted masks from the model. Apply post-processing on these masks.
Using these masks, get the regions of interest from the original image. We mask non-table regions with black color.
Use OpenCV functions, in order to extract these individual tables. For each table apply some processing like sharpening filters, reshape and thresholding.
Once each table is processed, save that table image separately. We then pass the Table region to extract the text from the table and save it into a CSV file.
In the end, we would get images of individual tables as well as the CSV files for each table.

![](https://miro.medium.com/max/2000/1*CTq5ej_BUWadf_k2TqnrvA.jpeg)
