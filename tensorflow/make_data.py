import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from scipy.misc import imresize

import models

def predict(model_data_path, image_folder):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)
        
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                directory = root + '/'
                if 'jpg' in file and file[0] is not '.':
                    image_path = directory + file
                    # Read image
                    img = Image.open(image_path)
                    img = img.resize([width,height], Image.ANTIALIAS)
                    img = np.array(img).astype('float32')
                    img = np.expand_dims(np.asarray(img), axis = 0)

                    # Use to load from npy file
                    #net.load(model_data_path, sess) 

                    # Evalute the network for the given image
                    pred = sess.run(net.get_output(), feed_dict={input_node: img})
                    pred_depth = pred[0,:,:,0]
                    output_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_directory = os.path.dirname(image_path)
                    plt.imsave(os.path.join(output_directory, "{}_depth.png".format(output_name)), pred_depth, cmap='plasma')
                    np.save(os.path.join(output_directory, "{}_depth.npy".format(output_name)), pred_depth)
                    print('finished exporting {}'.format(os.path.join(output_directory, output_name)))
        
        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_folder', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_folder)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



