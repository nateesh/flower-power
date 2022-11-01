
*A machine learning assignment by Nina and Nathan*

For this assignment we explored transfer learning using Tensorflow/Keras *without* fine-tuning, that is, freezing base model layers. The transfer learning process was then sped up using the headless base model, `F()`, as a 'feature extractor preprocessor', see `task_9` and `task_10` in `model.py`. This involved creating an auxiliary dataset of activations from the (headless) base model, `F(X)`, and the original labels, `y`. The new model's input layer matches the output shape of `F()`'s last layer.

### ~ Some observations ~
There are immense savings in computation resources using this technique because activations for frozen layers don't need to be recomputed. This limits training to layers used for classification at the end of the model. For transfer learning *with* fine-tuning, some variation of this process may be possible depending of which layers are un-frozen.

### ~ Free Time in the future? ~
- [ ] Try out fine-tuning by unfreezing model layers

### ~ Downloading the dataset ~

```
FLOWERS_DIR = small_flower_dataset/

def download_images():
    """If the images aren't already downloaded, save them to FLOWERS_DIR."""
    if not os.path.exists(FLOWERS_DIR):
        DOWNLOAD_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
        print('Downloading flower images from %s...' % DOWNLOAD_URL)
        urllib.request.urlretrieve(DOWNLOAD_URL, 'flower_photos.tgz')
        !tar xfz flower_photos.tgz
    print('Flower photos are located in %s' % FLOWERS_DIR)

download_images()
  ```
