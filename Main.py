import numpy as np
from Network import Network
import doodler_forall
from matplotlib import pyplot as plt
import configparser

# get the configuration from the config file
config = configparser.ConfigParser()
config.read('config_1.ini')

"""
:param count: number of images
:param n: lentgh and width of the image
:param _doodle_image_types_: types of images
:param train_size: size of the train dataset
:param validation_size: size of the validation dataset
:returns: train, validation and test datasets and their size
"""
def generator(count,n,_doodle_image_types_,train_size=0.7, validation_size=0.2):

    # generate the train cases
    train_cases_count=int(count*train_size)
    train_cases=doodler_forall.gen_standard_cases(train_cases_count,rows=n,cols=n,types=_doodle_image_types_,wr=[0.2,0.5],hr=[0.2,0.4],
    noise=0.01, cent=False, show=False, flat=True,
    fc=(1,1),auto=False,mono=True,one_hots=True,multi=False)

    # generate the validation cases
    validation_cases_count=int(count*validation_size)
    validation_cases=doodler_forall.gen_standard_cases(validation_cases_count,rows=n,cols=n,types=_doodle_image_types_,wr=[0.2,0.5],hr=[0.2,0.4],
    noise=0.01, cent=False, show=False, flat=True,
    fc=(1,1),auto=False,mono=True,one_hots=True,multi=False)

    # generate the tests cases
    test_cases_count=int(count*(1-train_size-validation_size))
    test_cases=doodler_forall.gen_standard_cases(test_cases_count,rows=n,cols=n,types=_doodle_image_types_,wr=[0.2,0.5],hr=[0.2,0.4],
    noise=0.01, cent=False, show=False, flat=True,
    fc=(1,1),auto=False,mono=True,one_hots=True,multi=False)

    return train_cases,validation_cases,test_cases,train_cases_count,validation_cases_count,test_cases_count

"""
:param number_of_batches: number of batches
:param number_of_epoch: number of epoch
:param train_cases: train cases
:param train_size: size of the train dataset
:param validation_cases: validation cases
:param validation_size: size of the validation dataset
:param test_cases: train cases
:param test_size: size of the test dataset
:returns: the fitted neural network
"""
def fit(neural_network,number_of_batches,number_of_epoch,train_cases,train_size,validation_cases,validation_size,test_cases,test_size):
    # compute the batch length
    train_batche_lentgh=train_size//number_of_batches

    train_losses=[]
    validation_losses=[]
    # epoches
    for j in range(number_of_epoch):
        print('\nEPOCH n°'+str(j+1)+'/'+str(number_of_epoch)+'\n---------------------------------\n')
        # batches
        for i in range(number_of_batches):
            print('\n----- batch n°'+str(i+1)+'/'+str(number_of_batches)+' -----\n')
            train_loss=neural_network.forward_pass(train_batche_lentgh,train_cases[0][train_batche_lentgh*i:train_batche_lentgh*(i+1)],train_cases[1][train_batche_lentgh*i:train_batche_lentgh*(i+1)])
            train_losses.append(train_loss)
            neural_network.backward_pass(train_cases[1][train_batche_lentgh*(i+1)-1])
            validation_loss=neural_network.forward_pass(validation_size,validation_cases[0],validation_cases[1])
            validation_losses.append(validation_loss)

    # test cases
    test_loss=neural_network.forward_pass(test_size,test_cases[0],test_cases[1])
    test_losses=np.ones(test_size)*test_loss

    x1 = np.linspace(0, number_of_batches*number_of_epoch-1, number_of_batches*number_of_epoch)
    x2 = np.linspace(0, number_of_batches*number_of_epoch-1, number_of_batches*number_of_epoch)
    x3 = np.linspace(number_of_batches*number_of_epoch, number_of_batches*number_of_epoch+10, test_size)

    plt.plot(x1, train_losses , color='blue')
    plt.plot(x2, validation_losses , color='yellow')
    plt.plot(x3, test_losses , color='red')

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    plt.show()

    return neural_network



_doodle_image_types_=['ring','frame','triangle','bar']

n=int(config['GLOBALS']['n'])
count=int(config['GLOBALS']['count'])

train_cases,validation_cases,test_cases,train_cases_count,validation_cases_count,test_cases_count=generator(count=count, n=n,_doodle_image_types_=_doodle_image_types_)

number_of_batches=int(config['GLOBALS']['number_of_batches'])
test_batche_lentgh=len(test_cases)//number_of_batches
number_of_epoch=int(config['GLOBALS']['number_of_epoch'])

# set the neural network
N=Network(config)

fit(N,number_of_batches,number_of_epoch,train_cases,train_cases_count,validation_cases,validation_cases_count,test_cases,test_cases_count)