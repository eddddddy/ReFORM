# ReFORM
**R**ecommendation **E**ngine **For** **M**yself (though it can be for anyone if the model is trained on different data)

This is a content-based recommendation system for music that is tailored to my (or anyone's) specific tastes. At the heart of the system is a deep convolutional auto-encoder that learns a compressed representation of any song (converted into a spectrogram). It then uses a similarity metric in the embedded space to carry out the actual recommendation.
## Method
### Data Generation
WAV files of songs were compiled, and for each of these a mel-spectrogram was computed for every channel with a frame size of 2048 samples, a hop size of 512 samples, and 128 frequency bins. These spectrograms were stored in their entirety in a file on disk
(which have not added to the repository due to size limitations).

### Pretraining
The architecture of the auto-encoder is shown in model_pretrain.png. Each layer of the encoder is followed by a decoder of the same size and shape as that layer and all preceding layers. The outputs of all decoders except the last are used to compute auxiliary losses, which are then scaled and added to the loss of the last 'main' decoder. This is to ensure that the model learns meaningful features at all layers during the encoding.

For every pass-through of the training data, a random 640-frame window is selected from the mel-spectrogram as input to the auto-encoder. The loss is given as the mean-squared-error of the input and the decoder reconstruction. ADAM optimizer was used with a learning rate starting at 0.1, which is dropped by a factor of 10 once the loss plateaued. Training was stopped after the third plateau.

### Clustering
This training phase is needed to ensure that samples from the same song would cluster closely together in the embedded space.

The architecture of the clustering auto-encoder is shown in model_cluster.png. The auxiliary decoders have been removed, and a new output was added to the final encoder layer. The loss for this output is given as the mean-squared-error of the embedded representation and the batch mean of the embedded representation, where batching for this phase is done by random selecting the required amount of 640-frame windows from a single song. (Because of this, all batch normalization layers are frozen from this point onwards.) This loss is scaled and added to the reconstruction loss.

Aside from these differences, training was carried out in the same way as pretraining. ADAM optimizer was used with an initial learning rate of 0.1, then dropped up to 3 times once the loss plateaued.
