import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse



class Predict_:
    def __init__(self,):
        self.image_size=128
        self.num_channels =3
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('saved_model/cnn_model.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./saved_model/'))
        self.graph = tf.get_default_graph()
    
    def get_batch(self,file):
        images =[]
        image = cv2.imread(file)
        # Resizing the image to our desired size and preprocessing will be done exactly as done during training
        image = cv2.resize(image, (self.image_size, self.image_size),0,0, cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)
        x_batch = images.reshape(1, self.image_size,self.image_size,self.num_channels)
        return x_batch
    
    def predict(self,file):
        y_pred = self.graph.get_tensor_by_name("y_pred:0")

        ## Let's feed the images to the input placeholders
        x= self.graph.get_tensor_by_name("x:0") 
        y_true = self.graph.get_tensor_by_name("y_true:0") 
        classes = os.listdir('flowers_small')
        y_test_images = np.zeros((1, len(classes))) 
        x_batch = self.get_batch(file)
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result=self.sess.run(y_pred, feed_dict=feed_dict_testing)
        return classes[np.argmax(result)],result,classes
        
        
        
        
        
        
        
        
        
        
        