#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 00:40:58 2023

@author: tadewuyi
"""
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


class DataProcessor:
  def __init__(self, 
               station,
               path: str = '../data/Prepped_data/',
               temp_path: str ='../data/TWINS/',
               SM_path: str = '../data/SuperMag/',
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
    
    
    
    
  def prep_data(self) -> None:
    '''
    This function takes in .feather files for  the supermag data and the Energetic Neutral Atoms (ENA) files and 
    creates a dictionary output. The ENA files are filled with 10 minutes cadence images and the images are stored
    in the dictionary with the key being the corresponding time.
    
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
    
    image.sort()
    
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
          
          temp_data[temp_data == -1] = 0
          norm_temp = temp_data/np.max(temp_data) #normalized temperature map. 
          
          df = pd.read_feather(os.path.join(self.SM_path, f'{self.station}.feather'))
#          df_cleaned = df.dropna()
          
#          del df
          
          time_index = bisect.bisect_left(df['Date_UTC'], (epoch[i] + dt.timedelta(minutes = 10)))
          
  
          SM_time = df['Date_UTC'][time_index]
          time_diff = (SM_time - epoch[i]).total_seconds()
          
          dbht = df['dbht'][time_index]
          mlt = df['MLT'][time_index]
          
          if not np.isnan(dbht) and time_diff < 900:
      
            temp_dict[SM_time] = norm_temp
            target_data.append([SM_time,dbht, mlt])
          
        else:
          diff = abs((epoch[i] - epoch[i-1]).total_seconds())
          if diff > 120:
            temp_data = temp_arr[i]
            
            temp_data[temp_data == -1] = 0
            norm_temp = temp_data/np.max(temp_data) #normalized temperature map. 
            
            df = pd.read_feather(os.path.join(self.SM_path, f'{self.station}.feather'))
            
#            df_cleaned = df.dropna() #Drop the rows that contains nan values 
            
#            del df #delete the unneeded dataframe
            
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
    image data. These regions are filtered based on the size (# of pixels in the region). A list of isolated regions of images 
    is returned.
    
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
    
    for label in tqdm(range(1, labels.max() + 1), desc = 'Working on Isolated regions...'):
        region = (labels == label)
        region_size = np.sum(region)
    
        if self.min_size <= region_size <= self.max_size:
            # Extract the region from the original image_data
            isolated_region = image_data.copy()
            isolated_region[~region] = 0
            isolated_regions.append(isolated_region)
    
    return isolated_regions 
    
    
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
    
    region_path = os.path.join(self.path, f'{self.station}_region_data.pkl')
    
    if os.path.isfile(region_path):
      raise ValueError('The file already exists!')#This stops the class if the output file already exists. 
      
    else:
      
      extract_dict = defaultdict(dict)
      
      dict_path = os.path.join(self.path, f'{self.station}_temp_dict.pkl') #Path to the temperature dictionary need in this function.
      
      if os.path.isfile(dict_path):
        
        with open(dict_path, 'rb') as file:
          temperature_data = pickle.load(file) #Open the temp dictionary if it already exists.
          
      else:
        self.prep_data()
        
        with open(dict_path, 'rb') as file:
          temperature_data = pickle.load(file)
          
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
  

if __name__ == '__main__':
  
  process = DataProcessor('NEW')
  process.Process_data()
  

  
  
  
  
  
  
  
  