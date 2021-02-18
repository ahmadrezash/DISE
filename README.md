
# Welcome to DISE!

This is DISE project.
Deep Image Search Engine.


# Files

```
lib
│
│   settings
│   tests
│   Trainer
│   utils
│
└───Data
    │   Dataset
    │   FeatureExtractor
    │   PerceptualLoss
    │   Transforms


```

`lib` folder is some reusable structured code like "Dataset Class" or more.

`settings` file is contain some basic config like "torch device" and model PATHs is loaded in other functions.
They are for loading universally in everywhere their needed.

`test` has some test for modules like dataset-loader or featureExtractor or etc

`Trainer` a learner class. You can define your meta parameters like epoch, learning rate or ... and train your model.

`utils` contain some function for visualizing data or more activity.

`Data` directory:
-   `Dataset` is wrapper class on Dataset. Set default path for dataset files dir in settings
-   `FeatureExtractor` is a torch model class get your image in `PIL.Image` object and return feature vector. If set a PCA in fe object defining it's applied if don't return complete fe vector of VGG16. (25k len remaining).

    

