'''
Done by:
Vinay HN - n10180893
Rahul Rajendran - n10145281
Karthikeyan Jayavel - n10124004
'''
import tensorflow
from tensorflow import keras as keras
from tensorflow.python.keras import datasets
import random
import numpy as np

from tensorflow.python.keras import backend as K
from sklearn import model_selection
import matplotlib.pyplot as plt



def data_process():

    #Load the Data

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    test_cond = [0,1,8,9]
    test_train_cond = [2,3,4,5,6,7]
    
    test_only_lab = np.concatenate((y_test[np.in1d(y_test, test_cond)], \
                       y_train[np.in1d(y_train, test_cond)]))
    
    test_train_lab = np.concatenate((y_test[np.in1d(y_test, test_train_cond)], \
                       y_train[np.in1d(y_train, test_train_cond)]))
    
    test_only = np.concatenate((x_test[np.in1d(y_test, test_cond)], \
                       x_train[np.in1d(y_train, test_cond)]))
    
    test_train = np.concatenate((x_test[np.in1d(y_test, test_train_cond)], \
                       x_train[np.in1d(y_train, test_train_cond)]))
    
    return test_only_lab, test_train_lab, test_only, test_train

#------------------------------------------------------------------------------

def euclidean_distance(vectors):
    #Compute the Euclidian Distance between the 2 vectors
    
    a, b = vectors
    return K.sqrt(K.maximum(K.sum(K.square(a - b), axis=1, keepdims=True), 
                            K.epsilon()))


#------------------------------------------------------------------------------
def contrastive_loss(y_true, y_pred):

    '''
      y_true : true label 1 for positive pair, 0 for negative pair
      y_pred : distance output of the Siamese network    
    '''
    margin = 1

    # if positive pair, y_true == 1, if negative pair, y_true == 0,
 
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

#------------------------------------------------------------------------------
def compute_accuracy(predictions, labels):

    n = labels.shape[0]

	# count the True Positive
    acc =  (labels[predictions.ravel() < 0.5].sum() +  

               (1-labels[predictions.ravel() >= 0.5]).sum() ) / n  # Count the True Negative

    return acc

#------------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    #Compute classification Accuracy
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

#------------------------------------------------------------------------------
def create_pairs(x, digit_indices,digit_num):

    '''
       This function creates positive and negative pairs in an
       alternative fashion.
       @param
         digit_indices : list of lists
            digit_indices[k] is the list of indices of occurences digit k in 
            the dataset
       @return
         np.array(pairs), np.array(labels) 
         where np.array(pairs) is an array of pairs and np.array(labels) is
         an array of labels
         np.array(labels)[i] ==1 if np.array(pairs)[i] is a positive pair
         np.array(labels)[i] ==0 if np.array(pairs)[i] is a negative pair
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(digit_num)]) - 1

    for d in range(digit_num):

        for i in range(n):

            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]

            # z1 and z2 form the positive pair

            inc = random.randrange(1, digit_num)
            dn = (d + inc) % digit_num
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]

            # z1 and z2 form the negative pair
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)

#------------------------------------------------------------------------------

def create_Siamese_network(input_dim):
    seq = keras.models.Sequential()

    seq.add(keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu', 
                                input_shape=input_dim))

    seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    seq.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))

    seq.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))         

    seq.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        
     #4 layers of Convolutional Network. 
  
    seq.add(keras.layers.Dropout(0.25))
    
    seq.add(keras.layers.Flatten())

    seq.add(keras.layers.Dense(576, activation='relu'))

    seq.add(keras.layers.Dropout(0.25))

    seq.add(keras.layers.Dense(128, activation='relu'))
        
    return seq
     
#------------------------------------------------------------------------------

def train_test_evaluate(epochs):

    '''
    Train Siamese Network to predict wheather the 2 images are true positive or false negative. 
    '''
    # Load the Dataset and slipt the Data into training and testing with the ratio of 0.8 as training. 

    test_only_lab, test_train_lab, test_only, test_train  = data_process()
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
                                test_train, test_train_lab, test_size=0.2)

    all_img = np.concatenate((test_only,x_test), axis=0)
    all_lab = np.concatenate((test_only_lab,y_test), axis=0)

    img_row, img_col = x_train.shape[1:3]


    all_img = all_img.astype('float32')
    test_only = test_only.astype('float32')
    test_train = test_train.astype('float32')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalise the input image with the value 0 to 1
    all_img /= 255
    test_only /= 255
    test_train /= 255 
    x_train /= 255 
    x_test /= 255    

    
    #Reshape the Data
    all_img = all_img.reshape(all_img.shape[0], img_row, img_col, 1)
    test_train = test_train.reshape(test_train.shape[0], img_row, img_col, 1)
    test_only = test_only.reshape(test_only.shape[0], img_row, img_col, 1)
    x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)
    x_test = x_test.reshape(x_test.shape[0], img_row, img_col, 1)
    input_dim = (img_row, img_col, 1)
    
    
    # create the training and testing pairs. 
    test_cond = [0,1,8,9]
    test_train_cond = [2,3,4,5,6,7]
    all_cond = [0,1,2,3,4,5,6,7,8,9]
    
    digit_indices = [np.where(y_train == i)[0] for i in test_train_cond]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices,len(test_train_cond))

    digit_indices = [np.where(y_test == i)[0] for i in test_train_cond]
    te_pairs, te_y = create_pairs(x_test, digit_indices, len(test_train_cond)) 
    
    digit_indices = [np.where(test_only_lab == i)[0] for i in test_cond]
    a0189_pairs, a0189_y = create_pairs(test_only, digit_indices, len(test_cond))
    
    digit_indices = [np.where(all_lab == i)[0] for i in all_cond]
    all_pairs, all_y = create_pairs(all_img, digit_indices, len(all_cond))

    # network definition
    base_network = create_Siamese_network(input_dim)

    input_a = keras.layers.Input(shape=input_dim)
    input_b = keras.layers.Input(shape=input_dim)

    # The weights of the network will be shared across the two branches because we re use the same instance base Network. 

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)  

    # Compute the Distance between two vectors.

    distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])

    # input_a and input_b are the pair of images and output the Euclidian distance

    model = keras.models.Model([input_a, input_b], distance)
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics = [accuracy])

    siamese=model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          verbose=0,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
    
    
    #Compute the Accuracy
    
    pred = model.predict([a0189_pairs[:, 0], a0189_pairs[:, 1]])
    a0189_acc = compute_accuracy(pred, a0189_y)

    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    a234567_acc = compute_accuracy(pred, te_y)
    
    pred = model.predict([all_pairs[:, 0], all_pairs[:, 1]])
    all_acc = compute_accuracy(pred, all_y)
    
  
    print('Epoch :  {}'.format(epochs))
    print('* Accuracy of [234567]: %0.2f%%' % (100 * a234567_acc))
    print('* Accuracy of[0123456789] : %0.2f%%' % (100 * all_acc))
    print('* Accuracy of [0189]: %0.2f%%' % (100 * a0189_acc))
    
    # Validation error vs Time
    
    loss = siamese.history['loss']
    val_loss = siamese.history['val_loss']
    acc = siamese.history['accuracy']
    val_acc = siamese.history['val_accuracy']
    epp = range(len(acc))
    
    plt.plot(epp, loss, '-', color='red', label='Training loss')
    plt.plot(epp, val_loss, '-', color='blue', label='Validation loss')
    plt.title('Training and Validation Loss vs Time')
    plt.legend()
    plt.show()
    
    plt.plot(epp, acc, '-', color='teal', label='Training Accuracy')
    plt.plot(epp, val_acc, '-', color='green', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy vs Time')
    plt.legend()
    plt.show()
    
    return a0189_acc

    
def test_different_epochs():
    mylist=[5,10,15,20]
    acc0189=[]
    for i in mylist:
        acc0189.append(100*train_test_evaluate(epochs=i))
        print('==============================================================')
        
    plt.plot(mylist, acc0189, '-', color='gray',
             label='prediction(0189) accuracy')
    plt.title('Prediction [0189] Accuracy vs Epochs')
    plt.legend()
    plt.show()
#------------------------------------------------------------------------------        

if __name__=='__main__':

    test_different_epochs()
