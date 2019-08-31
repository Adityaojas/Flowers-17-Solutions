# Flowers-17-Solutions
Approaches to solve Flowers 17 classification problem by University of Oxford

Download the dataset from the link: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
Extract the 'jpg' folder from the zip file downloaded.

[IMPORTANT]
1) Make a Flower_17 Directory inside the repo after cloning or downloading it.
2) Make 17 subdirectories with names {1-17} inside the 'jpg' folder. Move the images inside 'jpg' to the sub directories such    that every consecutive 80 images lie in each sub directory, the paths would be like:
   jpg/1/{image_0001.jpg - image_0080}, jpg/2/{image_0081.jpg - image_0160}, ...., jpg/17/{image_1281.jpg - image_1360}
3) Keep the 'jpg' directory inside 'Flower_17'

'minivggnet_wo_aug_flower17.py' solves the problem by applying simple minivggnet architecture without any data augmentation or learning rate scheduling. The accuracy comes out to be 65% after 100 epochs and saving the best model. High overfitting can be seen from the loss'accuracy plot. Overfitting is not a surprise since the data is really less, 80 images per class, and its an extremely fine - grained classification, the classes are not highly discriminant.

'minivggnet_aug_lrs_flower17.py' solves the problem by applying both learning rate scheduling and data augmentation. The accuracy can be seen to increase from 65% to over 76%. Overfitting has significantly diminished. Data Augmentation and Learning rate scheduling clearly improved the classification given the architecture.

Next up is the tranfer learning approach.

[IMPORTANT]
Using Flower17_extract_features_VGG16.py, I extract the features of the Flower 17 dataset using the pretrained imagenet weights of VGG16 architecture available in the keras.applications module. I chop off the architecture at the head (final Fully-Connected layers, just after the last Pooling Layer) to get a flattened feature set of dimensions (N, 512*7*7) {N is the number of datapoints, (7, 7, 512) is the shape of the output of each datapoint after the network is chopped off. The heavy feature dataset is stored in .hdf5 format to help it fit in the RAM. h5py module is used for this purpose. The final .hdf5 feature set and labels are stored in the directory: Flower_17/transfer_learning_VGG16/

Flower17_transfer_learning_VGG16_logistic_reg.py uses the extracted features and applies simple logistic regression and tunes the hyperparameters to obtain about 91% of classification accuracy which is huge increment from our last approach.

Flower_17 finetuning is a network surgery approach the would use the body of VGG16 architecture, and a custom head network which eventually would consist of a 2 fully connected Dense layers. Thw warming up accuracy (trained on 25 epochs) using the RMSprop optimizer comes out to be about 87% and the final accuracy after training 100 epochs post warm up comes out to be about 92%.
