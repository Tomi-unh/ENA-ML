#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 19:57:53 2023

@author: tadewuyi
"""

import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os 
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import pickle
import gc




class TempModels:
  def __init__(self,
               station: str,
               path: str = '../data/Model/',
               data_path: str = '../data/Prepped_data/',
               temp_name: str ='temp_dict.pkl',
               region_name: str = 'region_data.pkl',
               target_name: str = 'target_data.pkl',
               hidden_layers: tuple =(1000,100, 10),
               array_dim: tuple =(6,),
               input_dim: tuple = (6,160,160,1),
               learning_rate: float =0.0000001,
               mlt_dim: tuple = (1,),
               momentum: float = 0.9,
               test_size: float = 0.2,
               metrics: tuple =['mean_squared_error', 'mean_absolute_error'],
               loss: str = 'mean_squared_error',
               optimizer: str = 'adam'):
    '''
    Initializes the TempModels class with configuration parameters.
  
    Parameters:
      ----------
        -station (str): This is the string of the 3 character name of the station (Case sensitive).
        - path (str): The path to the data directory.
        -data_path (str): The path to the datasets used for this analysis
        - temp_name (str): Name of the temperature data pickle file.
        - region_name (str): Name of the region data pickle file.
        - target_name (str): Name of the target data pickle file.
        - hidden_layers (tuple): Tuple of integers representing hidden layer sizes in the neural network.
        - array_dim (tuple): Tuple representing the shape of the array input.
        - input_dim (tuple): Tuple representing the shape of the image input.
        - learning_rate (float): Learning rate for the optimizer.
        - momentum (float): Momentum for the optimizer.
        - metrics (list): List of metric names for model evaluation.
        - loss (str): Loss function for model training.
        - optimizer (str): Optimizer for model training. Default AdamW. Options of AdamW, Adam, or SGD.
        - test (float): Proportion of data to use for testing.
    '''
    self.station = station.upper()
    self.path = path
    self.mlt_dim = mlt_dim
    self.data_path = data_path
    self.temp_name = temp_name
    self.region_name = region_name
    self.target_name = target_name
    self.hidden_layers = hidden_layers
    self.array_dim = array_dim
    self.input_dim = input_dim
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.metrics = metrics
    self.loss = loss
    self.test_size = test_size
    self.optimizer = optimizer
  
  
  def load_data(self, filename):
    
    with open(filename, 'rb') as file:
      return pickle.load(file)
    
    
  def prep_tempdata(self):
    '''
    This function takes in the temperature data dictionary and returns an image array of all the temperature maps 
    stacked up in order for the neural network models.
    
    Parameters:
      ----------
      -data_pat: Path to the location of the stored temperature data dictionary. This is a pickle file.
      -temp_name: Name of the pickled temperature data file.
      
      Return:
        --------
        -img_arr: Array of the temperature maps. 
    '''
   
    filename = os.path.join(self.data_path,f'{self.station}_temp_dict.pkl')
    
    return (data for data in self.load_data(filename).values())
  
  
  
  def equalize_list(self,lst: list, fill_value: np.ndarray) -> list:
    '''
    Takes in a list of lists of mlt values and eualizes them by making the sublists have the same amount of 
    elements. Ex. 
          my_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]]
          equalize_list(my_list, fill_value=0)
          
          my_list = [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9], [10, 0, 0, 0]]
          
          The initial list was taken and parsed into the function to return a list with equal amount of
          sublist elements. This is done by filling in zeros where appropriate.
          
    Parameters:
      ----------
      -lst: List of lists to be processed.
      -fill_value: Value of the fill being inserted into the list. Can also be of any np.ndarray shape. E.g (160,160).
      
      Yield:
        -------
        -lst: The new list with the appended values.
    '''
    max_length = max(len(sublist) for sublist in lst)
    
    for sublist in lst:
        while len(sublist) < max_length:
            sublist.append(fill_value)
  

    return lst
            
  
  def prep_region_data(self):
    '''
    Prepare the datasets. This opens the isolated regions dictionary along with the 4 quadrants images
    and converts the values into a suitable format for a Neural Network model and returns the cleaned up arrays. 
    
    Parameters:
      ----------
      NONE
      
      Return:
        --------
        img_array, mlt_array, quadrant_imgs, qudrant_mean: This returns the isolated region image arrays 
        accompanied by their respective averaged mlt values. The qudrant images are also returned along with a list 
        of its mean values.
    '''
    #import the data 
    filename = os.path.join(self.data_path,f'{self.station}_region_data.pkl')
    
    with open(filename, 'rb') as file:
      data = pickle.load(file)
    
    #create list for storing the processed data
    mlt_ls = []
    image_ls = []
    
    #extracts value of the sub-dictionaries. 
    for key, subdict in data.items():
      
      if 'mlt' in subdict:
        mlt_ls.append(subdict['mlt'])
        
      if 'image' in subdict:
        image_ls.append(subdict['image'])
      
    #euqalize the datasets to make sure the inputs are all of the same shape and size and convert to an array.
    mlt_ls = self.equalize_list(mlt_ls, fill_value = 0)
    
    fill_img = np.zeros((160,160), dtype = float)
    
    img_ls = self.equalize_list(image_ls, fill_value = fill_img)
    
    #convert the data lists to arrays.
    img_array = np.array(img_ls, dtype = float)
    
    del img_ls, fill_img
    
#    img_array = img_array.reshape((img_array.shape[0],img_array.shape[1], img_array.shape[2], 1))
    
    mlt_array = np.array(mlt_ls, dtype = float)
    del mlt_ls
    
    #define the empty list for storing the mean values and the images
    img_mean = []
    quadrant_imgs = []
    
    #Load the 4 quadrant images for clean up
    filename_imgs = os.path.join(self.data_path, f'{self.station}_pieslice_data.pkl')
    
    with open(filename_imgs, 'rb') as file_imgs:
      data_imgs = pickle.load(file_imgs)
    
    
    for key, subdict in data_imgs.items():
      if 'mean' in subdict:
        img_mean.append(subdict['mean'])
      
      if 'image' in subdict:
        quadrant_imgs.append(subdict['image'])
        
    
    img_mean_array = np.array(img_mean, dtype = float)
    quadrant_img_array = np.array(quadrant_imgs, dytpe = float)
    
    return img_array, mlt_array, img_mean_array, quadrant_img_array
  
  
  
    
  def region_model(self):
    """
    Build and compile a Keras neural network model for regression with concatenation of an array input,
    Conv2D, and MaxPooling layers. This model takes in the temperature maps along with

    Parameters:
      ---------- 
    Returns:
      --------
        keras.models.Model: Compiled Keras model.
    """
    # Define input layers for image and array inputs
    image_inputs = Input(shape = self.input_dim)
    array_input = Input(shape=self.array_dim)
    mlt_array = Input(shape = self.mlt_dim)

    # Convolutional Layer
    conv_outputs = []
    for i in range(self.input_dim[0]):
      
      x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(image_inputs[:,i,:,:,:])
      x = BatchNormalization()(x)  # Add BatchNormalization
      x = MaxPooling2D((2, 2))(x)
      
      
      #Add another layer of conv2d, maxpooling and batchnormalization
      x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
      x = BatchNormalization()(x)  
      x = MaxPooling2D((2, 2))(x)
  
      # Flatten the convolutional output
      x = Flatten()(x)
      conv_outputs.append(x)
    
    #concatenate the flattened images together 
    conv_concat = Concatenate()(conv_outputs)
    
    # Concatenate the flattened images and array inputs
    concatenated_inputs = Concatenate()([conv_concat, array_input, mlt_array])

    # Create the model architecture
    for units in self.hidden_layers:
        concatenated_inputs = Dense(units, activation='relu')(concatenated_inputs)

    # Output layer for regression
    output_layer = Dense(1, activation='linear')(concatenated_inputs)

    # Create the final model with inputs and output
    final_model = Model(inputs=[image_inputs, array_input], outputs=output_layer)

    # Choose the Optimizer of choice
    if self.optimizer == 'adam':
        opt = Adam(learning_rate = self.learning_rate)
        
    elif self.optimizer == 'sgd':
        opt = SGD(learning_rate = self.learning_rate, momentum = self.momentum)
        
    else:
        raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
    #Compile the model.    
    final_model.compile(optimizer=opt, loss= self.loss, metrics= self.metrics)

    return final_model
  
  
  
  def Four_Quadrants(self):
    '''
    This model takes in four quadrants of the temperature maps as inputs instead of just one image like the temp_model below.
    Using these images, a prediction of db/dt is made.
    '''
    
    # Define input layers for image and array inputs
    quad_imgs = Input(shape = (4,160,160,1))
    quad_mean = Input(shape=(4,))
    
    log_array = np.log(quad_mean)
    
    squared_array = quad_mean**2
    
    

    # Convolutional Layer
    conv_outputs = []
    for i in range(self.input_dim[0]):
      
      x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(quad_imgs[:,i,:,:,:])
      x = BatchNormalization()(x)  # Add BatchNormalization
      x = MaxPooling2D((2, 2))(x)
      
      
      #Add another layer of conv2d, maxpooling and batchnormalization
      x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
      x = BatchNormalization()(x)  
      x = MaxPooling2D((2, 2))(x)
  
      # Flatten the convolutional output
      x = Flatten()(x)
      conv_outputs.append(x)
    
    #concatenate the flattened images together 
    conv_concat = Concatenate()(conv_outputs)
    
    # Concatenate the flattened images and array inputs
    concatenated_inputs = Concatenate()([conv_concat, quad_mean, log_array, squared_array])

    # Create the model architecture
    for units in self.hidden_layers:
        concatenated_inputs = Dense(units, activation='relu')(concatenated_inputs)

    # Output layer for regression
    output_layer = Dense(1, activation='linear')(concatenated_inputs)

    # Create the final model with inputs and output
    final_model = Model(inputs=[quad_imgs, quad_mean], outputs=output_layer)

    # Choose the Optimizer of choice
    if self.optimizer == 'adam':
        opt = Adam(learning_rate = self.learning_rate)
        
    elif self.optimizer == 'sgd':
        opt = SGD(learning_rate = self.learning_rate, momentum = self.momentum)
        
    else:
        raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
    #Compile the model.    
    final_model.compile(optimizer=opt, loss= self.loss, metrics= self.metrics)

    return final_model
  
  
  
   
  def temp_model(self):
    '''
    This function builds and compiles a simple Nueral Network model that takes in images as 
    input and predicts a single value as the output ( regression).
    '''
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape= (160,160,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    # Choose the Optimizer of choice
    if self.optimizer == 'adam':
        opt = Adam(learning_rate = self.learning_rate)
        
    elif self.optimizer == 'sgd':
        opt = SGD(learning_rate = self.learning_rate, momentum = self.momentum)
        
    else:
        raise ValueError(f"Unknown optimizer: {self.optimizer}")
    
    # compile model   
    model.compile(optimizer= opt, loss=self.loss, metrics=self.metrics)
    return model
  
  
  
  def train_test(self):
    '''
    This function takes the dataset and splits them into a train and test split. The test split is 
    approximately 0.2 of the original data. I'm aware of the fact that there is a built in function for 
    train test splits, but I have three sets of inputs for two different models. One model takes in two 
    inputs while the other one takes in the remaining input. For all these inputs and models, there is only
    one target data, and I wanted consistency between all the train test splits. Once I figure out if the 
    '''
    
    #get isolated region data along with the respective mlt values
    iso_img_arr, mlt_arr, quadrant_mean, quadrant_img = self.prep_region_data()
    
    
    N = len(mlt_arr)  # length of sample used
  
    # Calculate the number of samples for testing
    test_samples = int(N * self.test_size)
    
    # Create an array of indices from 0 to N-1
    all_indices = np.arange(N)
    
    # Randomly shuffle the indices
    np.random.shuffle(all_indices)
    
    # Select the first 'test_samples' indices for testing
    test_indices = all_indices[:test_samples]
    
    # The remaining indices will be used for training
    train_indices = all_indices[test_samples:]
    
    
    
    #Process the dataset and delete redundant data
    train_mlt = mlt_arr[train_indices]
    train_iso = iso_img_arr[train_indices]
    
    test_mlt = mlt_arr[test_indices]
    test_iso = iso_img_arr[test_indices]
    
    train_quad_mean = quadrant_mean[train_indices]
    test_quad_mean = quadrant_mean[test_indices]
    
    train_quad_img = quadrant_img[train_indices]
    test_quad_img = quadrant_img[test_indices]
    
    
    del iso_img_arr, mlt_arr, quadrant_mean, quadrant_img
    
    #get temperature map data
    temp_arr = np.array((list(self.prep_tempdata())), dtype = float)
    
    #Proces the temp_arr and divide into training and testing splits
    train_temp = temp_arr[train_indices]
    test_temp = temp_arr[test_indices]
    
    del temp_arr
    
    
    #get target data  
    filename_target = os.path.join(self.data_path, f'{self.station}_target_data.pkl')
    
    
    with open(filename_target, 'rb') as file: #Open the pickle file.
      target_data = pickle.load(file)
    
    #convert into an array 
    target_arr = np.stack(target_data)
    
    target_float = target_arr[:,1].astype(float) #Choose the target data which is dbht and convert to float
    
    station_mlt = target_arr[:,2].astype(float) #Choose the mlt values and convert to float values
    
    del target_arr
    #Process the target data and get the training and testing split
    
    train_target = target_float[train_indices]
    test_target = target_float[test_indices]
    
    del target_float
    
    train_station_mlt = station_mlt[train_indices] 
    test_station_mlt  = station_mlt[test_indices]
    
    del station_mlt
    #store the split data into a dictionary
    
    processed_data = {
    'train_temp': train_temp,
    'test_temp': test_temp,
    'train_mlt': train_mlt,
    'test_mlt': test_mlt,
    'train_iso': train_iso,
    'test_iso': test_iso,
    'train_quad': train_quad_img,
    'test_quad': test_quad_img,
    'train_quad_mean': train_quad_mean,
    'test_quad_mean': test_quad_mean,
    'train_station_mlt': train_station_mlt,
    'test_station_mlt': test_station_mlt,
    'train_target': train_target,
    'test_target': test_target
    }
        
    #save the dictionary into a .pickle file
    
    filename = os.path.join(self.path, f'{self.station}_split_data.pkl')
    
    
    with open(filename, 'wb') as file: #Open the pickle file.
      pickle.dump(processed_data,file)
    

  
  
  def run_models(self, 
                 train_iso, train_mlt, train_temp, 
                 train_station_mlt,train_quad, 
                 train_quad_mean, train_target,
                 region_model: bool = True,
                 temp_model: bool = True, 
                 Four_Quadrants: bool = True
                 ):
    
    '''
    Run the Models. 
    
    Parameters:
      ----------
      -train_iso: Array of the train dataset for the isolated region.
      -train_mlt: Array of the train dataset for the mlt values that corresponds with the isolated regions.
      -train_temp: Array of the train dataset for the temperature maps only model.
      -train_target: Array of the train dataset for the target data, which is dbht
      
      Return:
        --------
        None 
    '''
    

    
    early_stopping = EarlyStopping(
                      monitor='val_loss',  # Metric to monitor (usually validation loss)
                      patience=10,          # Number of epochs with no improvement after which training will stop
                      verbose=2,            # Print messages about early stopping
                      restore_best_weights=True) # Restore the best model weights when training stops
    
    if region_model: 
      #call the model
      model_iso = self.region_model()

      #fit the models 
      model_iso.fit([train_iso, train_mlt, train_station_mlt], train_target, 
                    verbose = 2, validation_split = 0.2, 
                    batch_size = 16, shuffle = True,
                    callbacks = [early_stopping])
      
      #save the model 
      model_iso.save(os.path.join(self.path, f'{self.station}_iso_model.h5'))
      
      
    if temp_model:
      
      #call the model 
      model_temp = self.temp_model()
      
      
      model_temp.fit(train_temp, train_target, 
                     verbose = 2, validation_split = 0.2, 
                     batch_size = 16, shuffle = True,
                     callbacks = [early_stopping])
    
      #Save the model
      model_temp.save(os.path.join(self.path, f'{self.station}_temp_model.h5'))
      
    if Four_Quadrants:
      
      #call the model 
      model_fourQuadrants = self.Four_Qudrants()
      
      
      model_fourQuadrants.fit([train_quad, train_quad_mean], train_target, 
                     verbose = 2, validation_split = 0.2, 
                     batch_size = 16, shuffle = True,
                     callbacks = [early_stopping])
    
      #Save the model
      model_fourQuadrants.save(os.path.join(self.path, f'{self.station}_4_Quadrants.h5'))


        
  def model_eval(self, test_iso, 
                 test_mlt, test_temp, 
                 test_station_mlt, test_quad, 
                 test_quad_mean, test_target, 
                 region_model: bool = True,
                 temp_model: bool = True, 
                 Four_Quadrants: bool = True):
    '''
    Evaluate the models and compare with the actual test values. 
    
    Parameters:
      ----------
      -test_iso: Array of the test dataset for the isolated region.
      -test_mlt: Array of the test dataset for the mlt values that corresponds with the isolated regions.
      -test_temp: Array of the test dataset for the temperature maps only model.
      -test_target: Array of the test dataset for the target data, which is dbht
      -test_station_mlt: Array of the station mlt values.
      
      Return:
        --------
        None 
    '''
    
    
    #load models
    iso_model =load_model(os.path.join(self.path, f'{self.station}_iso_model.h5'))

    
    if region_model:
      iso_model =load_model(os.path.join(self.path, f'{self.station}_iso_model.h5'))
      
      #make predictions 
      predicted_iso = iso_model.predict([test_iso, test_mlt])
      
      rmse_iso = np.sqrt(mean_squared_error(test_target, predicted_iso))
      mae_iso = mean_absolute_error(test_target, predicted_iso)
      
      
      fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))  # Change the subplot layout to (2, 1)
      # Plot for the second model evaluation
      ax1.plot(test_target, label='Actual Values', color='blue')
      ax1.plot(predicted_iso, label='Predicted Values', color='green')
      ax1.set_title('Model 2: Actual vs. Predicted for ENA and MLT Based Model')
      ax1.set_xlabel('Data Points')
      ax1.set_ylabel('DB/DT')
      ax1.grid(True)
      ax1.margins(x=0)
      ax1.legend(loc='upper left')
      
      legend_x, legend_y, _, _ = ax1.get_legend().get_bbox_to_anchor().bounds
      ax1.text(0.8, 0.95, f'RMSE: {rmse_iso:.2f}', transform=ax1.transAxes, color='red')
      ax1.text(0.8, 0.9, f'MAE: {mae_iso:.2f}', transform=ax1.transAxes, color='red')
      
      # Save the figure
      plt.savefig(os.path.join(self.path, f'{self.station}_iso_model.png'))
      
      
      
    if temp_model:
      #load model
      temp_model = load_model(os.path.join(self.path, f'{self.station}_temp_model.h5'))
      
      #make predictions
      predicted_temp = temp_model.predict(test_temp)
      
      # Calculate evaluation metrics for the second model
      rmse_temp = np.sqrt(mean_squared_error(test_target, predicted_temp))
      mae_temp = mean_absolute_error(test_target, predicted_temp)
         
      fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))  # Change the subplot layout to (2, 1)
    
      # Plot for the first model evaluation
      ax1.plot(test_target, label='Actual Values', color='blue')
      ax1.plot(predicted_temp, label='Predicted Values', color='green')
      ax1.set_title('Model 1: Actual vs. Predicted For ENA based Model')
      ax1.set_xlabel('Data Points')
      ax1.set_ylabel('DB/DT')
      ax1.grid(True)
      ax1.margins(x=0)
      ax1.legend(loc='upper left')
      
      legend_x, legend_y, _, _ = ax1.get_legend().get_bbox_to_anchor().bounds
      ax1.text(0.8, 0.95, f'RMSE: {rmse_temp:.2f}', transform=ax1.transAxes, color='red')
      ax1.text(0.8, 0.9, f'MAE: {mae_temp:.2f}', transform=ax1.transAxes, color='red')
      
      # Save the figure
      plt.savefig(os.path.join(self.path, f'{self.station}_temp_model.png'))
      
    if Four_Quadrants:
      
      #load model
      quad_model = load_model(os.path.join(self.path, f'{self.station}_4_Quadrants.h5'))
      
      #make predictions
      
      predicted_quad = quad_model.predict([test_quad, test_quad_mean])
      
      rmse_quad = np.sqrt(mean_squared_error(test_target, predicted_quad))
      mae_quad = mean_absolute_error(test_target, predicted_quad)
      
      fig, ax1 = plt.subplots(1,1, figsize = (15,15))
      # Plot for the first model evaluation
      ax1.plot(test_target, label='Actual Values', color='blue')
      ax1.plot(predicted_quad, label='Predicted Values', color='green')
      ax1.set_title('Model 3: Actual vs. Predicted For ENA Quadrants Model')
      ax1.set_xlabel('Data Points')
      ax1.set_ylabel('DB/DT')
      ax1.grid(True)
      ax1.margins(x=0)
      ax1.legend(loc='upper left')
      
      legend_x, legend_y, _, _ = ax1.get_legend().get_bbox_to_anchor().bounds
      ax1.text(0.8, 0.95, f'RMSE: {rmse_quad:.2f}', transform=ax1.transAxes, color='red')
      ax1.text(0.8, 0.9, f'MAE: {mae_quad:.2f}', transform=ax1.transAxes, color='red')

      
      # Save the figure
      plt.savefig(os.path.join(self.path, f'{self.station}_quad_model.png'))
    

        
    
if __name__ == "__main__":
    
  
    # Create an instance of the TempModels class with your desired configuration
    temp_models = TempModels('NEW')
    
    #check if file with the prepped dataset already exists
    path = '../data/Model/'
    filename = os.path.join(path, f'{temp_models.station}_split_data.pkl')
    
    
    if os.path.isfile(filename):
      with open(filename, 'rb') as file:
        data = pickle.load(file)
  

        
      train_iso = data['train_iso']
      train_temp = data['train_temp']
      train_mlt = data['train_mlt']
      train_target = data['train_target']
      train_station_mlt = data['train_station_mlt']
      train_quad = data['train_quad']
      train_quad_mean = data['train_quad_mean']
      
  

      test_iso = data['test_iso']
      test_temp = data['test_temp']
      test_mlt = data['test_mlt']
      test_target = data['test_target']
      test_station_mlt = data['test_station_mlt']
      test_quad = data['test_quad']
      test_quad_mean = data['test_quad_mean']
      

        
      # Train models
      temp_models.run_models(train_iso, train_mlt, train_temp, train_quad, train_quad_mean, train_target)
  
  
      # Evaluate models
      temp_models.model_eval(test_iso, test_mlt, test_temp, test_quad, test_quad_mean, test_target)
          
      del train_iso, train_temp, train_mlt, train_target, train_quad, train_quad_mean
      del test_iso, test_temp, test_mlt, test_target, test_quad, test_quad_mean
      
      gc.collect()
        
    else:   
      # Perform data preparation
      temp_models.train_test()
      
      data = temp_models.load_data(filename)
 
      train_iso = data['train_iso']
      train_temp = data['train_temp']
      train_mlt = data['train_mlt']
      train_target = data['train_target']
      train_station_mlt = data['train_station_mlt']
      train_quad = data['train_quad']
      train_quad_mean = data['train_quad_mean']
      
  

      test_iso = data['test_iso']
      test_temp = data['test_temp']
      test_mlt = data['test_mlt']
      test_target = data['test_target']
      test_station_mlt = data['test_station_mlt']
      test_quad = data['test_quad']
      test_quad_mean = data['test_quad_mean']
      

        
      # Train models
      temp_models.run_models(train_iso, train_mlt, train_temp, train_quad, train_quad_mean, train_target)
  
  
      # Evaluate models
      temp_models.model_eval(test_iso, test_mlt, test_temp, test_quad, test_quad_mean, test_target)
          
      del train_iso, train_temp, train_mlt, train_target, train_quad, train_quad_mean
      del test_iso, test_temp, test_mlt, test_target, test_quad, test_quad_mean
      
      gc.collect() 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
  
  