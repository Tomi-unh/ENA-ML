#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 00:40:58 2023

@author: tadewuyi


This is a prep file for the TWINS ENA dataset. The TWINS ENA dataset is a set of images of the temperature
of the magnetotial at a given time. The temperature of the energectic neutral atoms (ENAs) in the magnetotail
is used as a proxy for observing the dynamics of the magnetotail without having to rely on in-situ measurements
fro various spacecrafts. 
"""

import sys 

sys.path.append('../Cannonical-Correlation/')

from Region_Extraction import pieslice

import spacepy.pycdf as cdf
import pandas as pd 
import numpy as np
import os
import bisect
import cv2
import pickle
from math import pi, atan2
from collections import defaultdict
import datetime as dt
from tqdm import tqdm
import re



class DataProcessor:
  def __init__(self, 
               station,
               path: str = '../data/Prepped_data/',
               temp_path: str ='/data/twins/CDF_files',
               SM_path: str = '../../../data/supermag/',
               img_center: tuple = (40,0),
               threshold: int = 3,
               min_size: int = 100,
               max_size: int = 10000):
    '''
    Initialize a DataProcessor object.

    Parameters:
      ----------
    - station: Name of the station being used for the target data without the file extension. This should be in a .feather format.
    -path: Path to the location for the prepped data to be save.
    - temp_path: Path to the temperature data location. Defaults to '../data/TWINS/'.
    - SM_path: Path to the SuperMag dataset. Defaults to '../data/SuperMag/'.
    - img_center: A tuple representing the image center coordinates (x, y). Defaults to (40, 0).
    - threshold: The threshold value for region extraction. Defaults to 3.
    - min_size: Minimum size of isolated regions. Defaults to 100.
    - max_size: Maximum size of isolated regions. Defaults to 10000.
    '''
    self.station = station.upper() #Capitalize the station name.
    self.path = path
    self.temp_path = temp_path
    self.SM_path = SM_path
    self.threshold = threshold
    self.min_size = min_size
    self.max_size = max_size
    self.img_center = img_center
    
    
  def natural_sort_key(self, string):
    '''
    Key function for sorting files naturally.
    Parameters:
        - s: File name.
    Returns:
        - Key for natural sorting.
    '''
    
    # Split the input string into text and numeric parts
    parts = re.split(r'(\d+)', string)
  
    # Convert numeric parts to integers for proper numeric sorting
    parts[1::2] = map(int, parts[1::2])
  
    return parts
  
    
  def prep_data(self) -> None:
    '''
    This function takes in .feather files for  the supermag data and the Energetic Neutral Atoms (ENA) files and 
    creates a dictionary output. The ENA files are filled with 10 minutes cadence images and the images are stored
    in the dictionary with the key being the corresponding time. The supermag station file is used to figure out 
    the appropriate time step to keep based on the availability of the specific station. 
    For example, 'NEW' station might not have data at time T, therefore temperature maps from time T isn't kept 
    as a part of the training dataset.
    
    Parameter:
      ---------
      -station: Name of the station being used for the target data. This should be in a .feather format.
      -temp_path: This is the path to the temperature data location.
      -SM_path: Path to the supermag dataset. 
      
      Return:
        -------
        None 
    '''
    
    
    image = [img for img in os.listdir(self.temp_path)]
    
    image = sorted(image, key = self.natural_sort_key)
    
    #Create empty dictionary and list to append the returned values 
    temp_dict = {}
    target_data = []
    
    for image_file in tqdm(image, desc = 'Creating ENA Temperature Maps dictionary...'):
    
      data = cdf.CDF(os.path.join(self.temp_path,image_file))
      
      epoch = data['Epoch'][:]
      
      temp_arr = data['Ion_Temperature'][:]
      

      for i in range(len(epoch)):
        if i == 0:
          temp_data = temp_arr[i]
          
          temp_data[temp_data == -1] = 0 #set all the -1 in the temperature maps into 0. 
          norm_temp = temp_data/np.max(temp_data) #normalized temperature map. 
          
          #load the SuperMAG station data
          df = pd.read_feather(os.path.join(self.SM_path, f'{self.station}.feather'))

          #Find the time index that corresponds to the ith time index from the TWINS data.
          time_index = bisect.bisect_left(df['Date_UTC'], (epoch[i] + dt.timedelta(minutes = 10)))
          
          
          SM_time = df['Date_UTC'][time_index] #define the SuperMAG time
          time_diff = (SM_time - epoch[i]).total_seconds() #Define the time difference between SM and TWINS
          
          dbht = df['dbht'][time_index] #Define the datapoint of interest
          mlt = df['MLT'][time_index] #Get the MLT value as well. 
          
          
          #condition to make sure the time diffrence isn't huge, and a measurement was taken at that time.
          if not np.isnan(dbht) and time_diff < 900:
            
            
            #Keep the first map is all is met, and append its corresponding target data to the list.
            temp_dict[SM_time] = norm_temp
            target_data.append([SM_time, dbht, mlt])
          
        else:
          diff = abs((epoch[i] - epoch[i-1]).total_seconds())
          if diff > 120:
            
            '''
            Same step as the one above, but for i != 0. The if statement makes sure that 
            the maps being loaded are
            '''
            temp_data = temp_arr[i]
            
            temp_data[temp_data == -1] = 0 #set all the -1 in the temperature maps into 0. 
            norm_temp = temp_data/np.max(temp_data) #normalized temperature map. 
            
            df = pd.read_feather(os.path.join(self.SM_path, f'{self.station}.feather'))
            

            time_index = bisect.bisect_left(df['Date_UTC'], (epoch[i] + dt.timedelta(minutes = 10)))
            
    
            SM_time = df['Date_UTC'][time_index]
            time_diff = (SM_time - epoch[i]).total_seconds()
            
            dbht = df['dbht'][time_index]
            mlt = df['MLT'][time_index]
            
            if not np.isnan(dbht) and time_diff < 900:
        
              temp_dict[SM_time] = norm_temp
              target_data.append([SM_time, dbht, mlt])
        
      print(f'Length of temp_dict: {len(temp_dict)}, Length of target_data: {len(target_data)}')
        
    with open(os.path.join(self.path, f'{self.station}_temp_dict.pkl'), 'wb') as f:
      pickle.dump(temp_dict, f)
      
    with open(os.path.join(self.path, f'{self.station}_target_data.pkl'), 'wb') as f:
      pickle.dump(target_data, f)
    
  
  
  def extract_features(self,image_data: np.ndarray) -> np.ndarray:
    '''
    This function takes in an np.ndarray image data and extracts out regions in the data that are a certain threshold
    greater than the background pixels. The threshold is a constant value multiplied by the standard deviation of the 
    image data. These regions are filtered based on the size (# of pixels in the region).
    A list of isolated regions of images is returned.
    
    Parameters:
      ----------
      -image_data: Image array for the region extraction.
      
      Return:
        -------
        isolated_regions: List of all the isolated regions that meets the requirement.
    '''
    
    # Calculate the standard deviation of the image
    std_dev = np.std(image_data)
    
    # Create a binary mask where values above the threshold are set to 1
    mask = np.where(image_data > (self.threshold)*std_dev, 1, 0).astype(np.uint8)
    
    # Use connected components labeling to identify isolated regions
    _, labels = cv2.connectedComponents(mask)
    
    isolated_regions = []
    
    for label in range(1, labels.max() + 1):
        region = (labels == label)
        region_size = np.sum(region)
    
        if self.min_size <= region_size <= self.max_size:
            # Extract the region from the original image_data
            isolated_region = image_data.copy()
            isolated_region[~region] = 0
            isolated_regions.append(isolated_region)
    
    return isolated_regions 
  

  
  def quad_imgs(self, image):
    '''
    Takes in temperature maps and splits them into 4 quadrants. Returns a list of images.
    Also returns the average of each slice generated along with other engineered features desired. 
    '''
    mean, four_imgs = pieslice(image, angle_steps = 4)
    
    return mean, four_imgs
    
    
  def MLT(self, coordinates: tuple)-> float:
    '''
    This function takes in the coordinates of a pixel and returns the MLT value of said pixel relative to the 
    center of the image or a defined point. 
    
    Parameter:
      ----------
      -coordinates: tuple of pixel coordinates in the form (x,y).
      
      Return:
        --------
        mlt: MLT value of  the pixel.
    '''
    x,y = coordinates
    
    x_center,y_center = self.img_center
    

    
    # Calculate the differences in x and y coordinates
    dx = x - x_center
    dy = y - y_center
    
    # Calculate the angle in radians using atan2
    angle_rad = atan2(dy, dx)

    
    mlt = ((angle_rad + pi)*12)/pi
    
    return mlt
  
  
  
    
  def Process_data(self):
    '''
    Put everything all together. This function calls the prep_data function if the prepped data doesn't already 
    exists and then parses it into the extract_features function to get the isolated features required for analysis.
    If the extracted feature file already exists, then the function aborts. 
    '''
   
    #Load the temperature maps dictionary.
    dict_path = os.path.join(self.path, f'{self.station}_temp_dict.pkl') #Path to the temperature dictionary need in this function.
    
    if os.path.isfile(dict_path):
      
      with open(dict_path, 'rb') as file:
        temperature_data = pickle.load(file) #Open the temp dictionary if it already exists.
        
    else:
      self.prep_data()
      
      with open(dict_path, 'rb') as file:
        temperature_data = pickle.load(file)
    
    
    
    '''
    Get the isolated regions from the temperature maps. 
    Temperature map data dictionary is opened and each image value in the dictionary 
    is put through into the extract_features function. This returns a dictionary with a key
    and a subkey. The main key is the timestamp of the temperature maps, while the sub keys is 'MLT'
    and 'images'.
    '''
    #define the path for the isolated regions images
    region_path = os.path.join(self.path, f'{self.station}_region_data.pkl')
    
    if os.path.isfile(region_path):
      print('Region file already exists!')#This stops the function if the file already exists. 
      
    else:
      
      extract_dict = defaultdict(dict)
      

          
      for key, value in tqdm(temperature_data.items(), desc = 'Getting MLT value for Isolated regions...'):
        
        extracted_region = self.extract_features(image_data = value) #Exctract the regions in the image above a certain multiple of std
        
        #Calculate the mlt for each non zero pixel in the image
        avg_mlt = []
        for image in extracted_region:
          mlt = []
          pixel_coords = np.where(image !=0)
          
          x,y = pixel_coords
          coords = list(zip(x,y))
          
          for coord in coords:  
            pixel_mlt = self.MLT(coord)  
            mlt.append([pixel_mlt])
          
          #Get the average value of the mlts
          mlt_mean= np.mean(mlt)
          avg_mlt.append(mlt_mean)
        
        
        extract_dict[key]['mlt'] = avg_mlt
        extract_dict[key]['image'] = extracted_region
        
      with open(os.path.join(self.path, f'{self.station}_region_data.pkl'), 'wb') as f:
        pickle.dump(extract_dict, f)
      
      
    '''
    Create a list of four images from the temperature maps. Each image in the list represents one part of 
    the temperature map quadrants. Along with the images, the pieslilce function also returns the average value
    of each image in the four images returned. Store the data generated in a dictionary
    '''
    #define the path for the pieslice images
    pieslice_path = os.path.join(self.path, f'{self.station}_pieslice_data.pkl')
    
    if os.path.isfile(pieslice_path):
      print('Pie Slice file already exists!')#This stops the function if the file already exists. 
      
    else:
      
      pieslice_dict = defaultdict(dict)#initialize the dictionary to store the images
      
      for key, value in tqdm(temperature_data.items(), desc = 'Getting the four quadrant images...'):
        mean, imgs = self.quad_imgs(value) #Get the images and mean values from the pieslice function
        pieslice_dict[key]['mean'] = mean #store the mean in a subkey in the dictionary
        pieslice_dict[key]['image'] = imgs #store the images in a subkey in the dictionary
       
      with open(os.path.join(self.path, f'{self.station}_pieslice_data.pkl'), 'wb') as f:
        pickle.dump(pieslice_dict, f)
      
        

if __name__ == '__main__':
  
  process = DataProcessor('NEW')
  process.Process_data()
  

  
  
  
  
  
  
  
  