# Flowers-17-Solutions
Approaches to solve Flowers 17 classification problem by University of Oxford

Download the dataset from the link: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
Extract the 'jpg' folder from the zip file downloaded.

make 17 subdirectories with name {1-17} inside the jpg directory. Move the images inside the jpg director to the sub directories such that every consecutive 80 images lie in each sub directory, the paths would be like:
jpg/1/{image_0001.jpg - image_0080}, jpg/2/{image_0081.jpg - image_0160}, ...., jpg/17/{image_1281.jpg - image_1360}
