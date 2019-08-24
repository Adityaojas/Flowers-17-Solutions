# Flowers-17-Solutions
Approaches to solve Flowers 17 classification problem by University of Oxford

Download the dataset from the link: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
Extract the 'jpg' folder from the zip file downloaded.

Make 17 subdirectories with names {1-17} inside the 'jpg' directory. Move the images inside the jpg director to the sub directories such that every consecutive 80 images lie in each sub directory, the paths would be like:
jpg/1/{image_0001.jpg - image_0080}, jpg/2/{image_0081.jpg - image_0160}, ...., jpg/17/{image_1281.jpg - image_1360}

'minivggnet_wo_aug_flower17.py' solves the problem by applying simple minivggnet architecture without any data augmentation or learning rate scheduling. The accuracy comes out to be 65% after 100 epochs and saving the best model. High overfitting can be seen from the loss'accuracy plot. Overfitting is not a surprise since the data is really less, 80 images per class, and its an extremely fine - grained classification, the classes are not highly discriminant.

'minivggnet_aug_lrs_flower17.py' solves the problem by applying both learning rate scheduling and data augmentation. The accuracy can be seen to increase from 65% to over 76%. Overfitting has significantly diminished. Data Augmentation and Learning rate scheduling clearly improved the classification given the architecture.
