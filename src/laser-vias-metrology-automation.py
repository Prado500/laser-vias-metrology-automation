# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 09:02:10 2025

@author: David Alejandro De los reyes Ostos
@matrikel_nr: 12435240

This is the .py file containing the script utilized to perform all of the tasks requested in the
final CDAM project, regarding a scientific analysis of lasser-vias axes and diameter measures.


"""


# from scipy.stats import skew, kurtosis # used to calculate the kurtosis and skewness values, to better understand if the raw dataset is a gaussian-fitting type (I experienced underfitting Issues)
import numpy as np #Vital for using np arrays, masking and histogramming.
import os # Used in the process of data capturing making sure that each file read was a .csv file.
import pandas as pd # Used to organize the outputs in a readable way to save them in the output text file requested.
import matplotlib.pyplot as plt # Used to plot the histograms.
from scipy.optimize import curve_fit # Used to perform the Gaussian Fit across top and bottom diameters histrograms.




"-------------------------------------------------------------------------------------------------------"

"""
section 1; data reading and storing. (First evaluation step)

This section contains all the code-wise steps addresed to retrieve the major and minor
axes contained in the .csv process files. 

"""

"-------------------------------------------------------------------------------------------------------"


# In here, I capture the data for every single process file (8 in total, 160 measures each.),
# I storage it inside an numpy array importing, offcourse, numpy and os
# for internal-file-path making purposes, as explained further.

directory_to_processes = r"C:\Users\Alejandro\Documents\2024-B\CDAM\final_proyect\Data files"

all_measurements = [] # List to store the meassures from all of the process files

# To further have the full path to each process, I captured the name of each file in the following list (usiong os):
csv_processes_file_names = [file for file in os.listdir(directory_to_processes) if file.endswith(".csv")] # I know since I put all the data in a clean folder, it's not necessary to use list comprehension to ensure each file is a .csv file, but I tought It would be a fun way to practice a Pythonical concept :) 

# Here I reach and read each process file storing the measures per procces in "measurements", which is then storaged inside the list "all_measurements". (nested lists were used to discriminate each measure data group).
for process_number in csv_processes_file_names:
    
    path_to_process = os.path.join(directory_to_processes, process_number)  # Full path to every process file (using os)
    
    measurements = np.loadtxt(path_to_process, delimiter=",", skiprows=1, usecols=(0, 1, 2)) # Loading data from process n1..n2...n8 skipping the first row as it is the headline
    
    all_measurements.append(measurements) # Append to the main list

# Here I convert the nested list to a single NumPy array as it is more efficient to handle listed data and allows for slicing.
all_measurements = np.array(all_measurements) # Since we used np.loadtext() to iteratively retreive text information from 8 different .csv files, each of them containing 80 rows and 3 columns, the resulting np array storaging all of the measures has an 8 x 80 x 3 shape.

print("Shape of the dataset (num_files, num_rows, num_columns):", all_measurements.shape) # Display the shape of "all_measurements" to make sure all of the information was correctly captured and storaged.



"-------------------------------------------------------------------------------------------------------"

"""
section 2; discrimination of the retreived data into major and minor axes per process. (Evaluation step 1, part 2)

This section contains all the code-wise steps addresed to split the data into 
major and minor axes, for each of the 8 processes, to later calculate the averaged top and bottom diameters per process.

To do so, I first separated the data as a whole, major and minor axes measures.
Afterwards, I used slicing to storage the major and minor axes measures per process using numpy arrays.

"""

"-------------------------------------------------------------------------------------------------------"


# Extracting major and minor axes measures from the all_measurements np array; 2 new np arrays created: "major_axes" and "minor_axes". Theese arrays will have a dimention 8 x 80, as we did not choose any values from the minor axis colimn (in the case of the major axis array) nor from the minor axis column (in the case of the minor axes array)
major_axes = all_measurements[:, :, 1]  # Major Axes 
minor_axes = all_measurements[:, :, 2]  # Minor Axes

# Print verification 
print("Major Axes (shape):", major_axes.shape)
print("Minor Axes (shape):", minor_axes.shape)
print("\n")

# Separation of major and minor axes by process using slicing
major_axes_process_1 = major_axes[0, ::]
minor_axes_process_1 = minor_axes[0, ::]

# Print verification 
print("Major Axes (Process 1):", major_axes_process_1)
print("Minor Axes (Process 1):", minor_axes_process_1)
print("\n")

# the same but for the remaining processes:
major_axes_process_2 = major_axes[1, ::]
minor_axes_process_2 = minor_axes[1, ::]

# process 3

major_axes_process_3 = major_axes[2, ::]
minor_axes_process_3 = minor_axes[2, ::]

# process 4

major_axes_process_4 = major_axes[3, ::]
minor_axes_process_4 = minor_axes[3, ::]

# process 5

major_axes_process_5 = major_axes[4, ::]
minor_axes_process_5 = minor_axes[4, ::]

# process 6

major_axes_process_6 = major_axes[5, ::]
minor_axes_process_6 = minor_axes[5, ::]

# process 7

major_axes_process_7 = major_axes[6, ::]
minor_axes_process_7 = minor_axes[6, ::]

# process 8

major_axes_process_8 = major_axes[7, ::]
minor_axes_process_8 = minor_axes[7, ::]



"-------------------------------------------------------------------------------------------------------"

"""
section 3; Calculation of the average diameter per process. (Evaluation step 1, part 2)

This section contains all the code-wise steps addresed to calculate the average diameter for each axis-measure-pair in the 8 processes.

To do so, I used the zip() function to iterate over the 16 previously generated numpy arrays containing the axis values for each of the processes,
and then computed the averaged diameter using the simplest approach I could think. Afterwads, every computed diameter was consecutively
stored in the list "diameters_processes", which I converted into a numpy array to easly handle the data.

I must say I'm aware it was not necessary to separate the top and bottom diameters per process, but I thought that doing it would
allow to practice pythonical concepts discussed in class, as the zip function, so I decided to do it for my practice and as an 
adition to he given homeworks.
"""

"-------------------------------------------------------------------------------------------------------"



diameters_processes = []  # List to store the diameters for each process

for major, minor in zip(
    [major_axes_process_1, major_axes_process_2, major_axes_process_3, major_axes_process_4,
     major_axes_process_5, major_axes_process_6, major_axes_process_7, major_axes_process_8],
    [minor_axes_process_1, minor_axes_process_2, minor_axes_process_3, minor_axes_process_4,
     minor_axes_process_5, minor_axes_process_6, minor_axes_process_7, minor_axes_process_8]
):
    diameters = (major + minor) / 2  # Compute average diameter
    diameters_processes.append(diameters)

diameters_processes = np.array(diameters_processes) # As every process will have 80 averaged diameter measures, the resulting shape of the diameters_processes array will be 8 x 80 (each row will have a computed value).



"-------------------------------------------------------------------------------------------------------"

"""
section 4; discrimination of the diameters in top and bottom diameters per process using boolean masking.
(Evaluation step 2)

This section contains all the code-wise steps addresed to diferentiate between top and bottom diameters,
departuring fromm the averaged diameters per process. 

To separate the data I iterated over all of the 80 diameter values per process storaged in 
the diameters_processes array and compared each diameter against a 30 µm treshold to create 
2 boolean masks.

I first created the boolean mask corresponding to the top diameters and then utilized the
complement numpy function "~" to generate an inverted version of the top diameters maks,
which I used as the bottom diameters mask.

I used slicing with the masks to index, crop and storage the values abouve 30 µm
in the top_diameters_masked array. The values below 30 µm were stored in the bottom_diameters_masked
array. 

NOTE: I also separated the values using a different approach (discriminating by comparing
every pair of consecutive diameter values, and assingning the largest value for ecah pair 
as the top diameter, assuming both diameters belong to the same lasser-via).

I compared the results and those were the same. I did it to practice, and I left 
this secondary approach code commented as an appendix, so that it can be used if needed.

"""

"-------------------------------------------------------------------------------------------------------"



# Placeholder arrays for top and bottom diameters using masking
top_diameters_masked = np.zeros((8, 40))  # Placeholder for Top Diameters (8 x 40)
bottom_diameters_masked = np.zeros((8, 40))  # Placeholder for Bottom Diameters (8 x 40)

# Threshold for separating top and bottom diameters (UNITS: µm)
threshold = 30.0

# Each mask exists only for the iteration of a specific process.
for process_idx in range(8):  # Loop over 8 processes
    
    top_mask = diameters_processes[process_idx, :] > threshold # Creation of boolean mask for the bottom diameters.    
    bottom_mask = ~top_mask  # Inverse of top_mask
    
    # Store top and bottom diameters based on masks.
    top_diameters_masked[process_idx, :] = diameters_processes[process_idx, top_mask]
    bottom_diameters_masked[process_idx, :] = diameters_processes[process_idx, bottom_mask]

# Print verification of masked results
print("Top Diameters (Masking) - Process 1:", top_diameters_masked[0, :])
print("Bottom Diameters (Masking) - Process 1:", bottom_diameters_masked[0, :])
print("\n")

print("Top Diameters (Masking) - Process 2:", top_diameters_masked[1, :])
print("Bottom Diameters (Masking) - Process 2:", bottom_diameters_masked[1, :])
print("\n")

print("Top Diameters (Masking) - Process 3:", top_diameters_masked[2, :])
print("Bottom Diameters (Masking) - Process 3:", bottom_diameters_masked[2, :])
print("\n")

print("Top Diameters (Masking) - Process 4:", top_diameters_masked[3, :])
print("Bottom Diameters (Masking) - Process 4:", bottom_diameters_masked[3, :])
print("\n")

print("Top Diameters (Masking) - Process 5:", top_diameters_masked[4, :])
print("Bottom Diameters (Masking) - Process 5:", bottom_diameters_masked[4, :])
print("\n")

print("Top Diameters (Masking) - Process 6:", top_diameters_masked[5, :])
print("Bottom Diameters (Masking) - Process 6:", bottom_diameters_masked[5, :])
print("\n")

print("Top Diameters (Masking) - Process 7:", top_diameters_masked[6, :])
print("Bottom Diameters (Masking) - Process 7:", bottom_diameters_masked[6, :])
print("\n")

print("Top Diameters (Masking) - Process 8:", top_diameters_masked[7, :])
print("Bottom Diameters (Masking) - Process 8:", bottom_diameters_masked[7, :])
print("\n")



"-------------------------------------------------------------------------------------------------------"

"""
Appendix; Beta version of section 4. (Evaluation step 2) (Not utilized to perform any further calculations)

This section contains an alternative approach to distinguish top from bottom diameters per 
process. It considers every 2 consecutive pair of averaged-diameter values corresond to the 
same lasser-via.

To do so, I consider that bottom diameters are usually smaller than top diameters due to 
tapering. I found the largest diameter value for each diameter-measure pair and I stored it
in a numpy array (Top diameters array). I utilized another numpy array to storage the smaller
values (bottom diameters). 

I did not chose this approach over the masking one, as a doble for loop is utilized to
iterate over the data and store it. 

"""

"-------------------------------------------------------------------------------------------------------"


# top_diameters = np.zeros((8, 40))  # Placeholder for Top Diameters (8 x 40)
# bottom_diameters = np.zeros((8, 40))  # Placeholder for Bottom Diameters (8 x 40)

# # Iterate over each process and assign the larger value as the top and smaller as the bottom
# for process_idx in range(8):  # Loop over 8 processes
#     for via_idx in range(40):  # Since I compare 2 values per iteration, I iterate 40 times.
#         diameter_value_1 = diameters_processes[process_idx, via_idx * 2]  # First diameter value of the via pair
#         diameter_value_2= diameters_processes[process_idx, via_idx * 2 + 1]  # Second diameter value of the via pair
        
#         # Assign Top and Bottom diameters based on which value is larger
#         top_diameters[process_idx, via_idx] = max(diameter_value_1,  diameter_value_2)
#         bottom_diameters[process_idx, via_idx] = min(diameter_value_1, diameter_value_2)

# # Print verification
# print("Top Diameters (Process 1):", top_diameters[0, ::])
# print("Bottom Diameters (Process 1):", bottom_diameters[0, ::])


# print("Top Diameters (Process 2):", top_diameters[1, ::])
# print("Bottom Diameters (Process 2):", bottom_diameters[1, ::])


# print("Top Diameters (Process 3):", top_diameters[2, ::])
# print("Bottom Diameters (Process 3):", bottom_diameters[2, ::])


# print("Top Diameters (Process 4):", top_diameters[3, ::])
# print("Bottom Diameters (Process 4):", bottom_diameters[3, ::])


# print("Top Diameters (Process 5):", top_diameters[4, ::])
# print("Bottom Diameters (Process 5):", bottom_diameters[4, ::])

# print("Top Diameters (Process 6):", top_diameters[5, ::])
# print("Bottom Diameters (Process 6):", bottom_diameters[5, ::])


# print("Top Diameters (Process 7):", top_diameters[6, ::])
# print("Bottom Diameters (Process 7):", bottom_diameters[6, ::])


# print("Top Diameters (Process 8):", top_diameters[7, ::])
# print("Bottom Diameters (Process 8):", bottom_diameters[7, ::])



"-------------------------------------------------------------------------------------------------------"

"""
section 5; plotting and applying a Gaussian fit to histograms for the top and bottom diameters per process. 
(Evaluation steps 3 and 4).

This section contains all the code-wise steps addresed to plot and fit a series of 16 
histograms, arranged among 8 figures ( one figure per process, each holding a histogram for 
the top and bottom diameters), departuring from the separated diameter values
stored in top_diameters and bottom_diameters.

I also capture the Amplitude values for each Gaussian fit, to use them in the next section 
when calculating the roundness parameter.

NOTE: I decided to plot 2 histograms in the same figure to practice the use of subplots and
reduce the final ammount of visual representations after executing the script.

"""

"-------------------------------------------------------------------------------------------------------"



peak_top_diameters = [] # List containing the peak position (μ) in relation to the horizontal axis of all the Gaussian fits for the top diameter histograms per process. As the horizontal axis corresponds to the Diameter values, μ represents the average Diameter for each top-diameter dataset per process. Theese values are captured to perform the roundness calculation, in next section.
peak_bottom_diameters = [] # List containing the peak position (μ) in relation to the horizontal axis of all the Gaussian fits for the bottom diameter histograms per process. As the horizontal axis corresponds to the Diameter values, μ represents the average Diameter for each bottom-diameter dataset per process. Theese values are captured to perform the roundness calculation, in next section.

std_top_diameters = [] # List containing the standard deviation (σ) values of all the Gaussian fits for the top diameter histograms per process. Theese values are captured to be displayed as part of the important parameters requested, in next section.
std_bottom_diameters =[] # List containing the standard deviation (σ) values of all the Gaussian fits for the bottom diameter histograms per process. Theese values are captured to be displayed as part of the important parameters requested, in next section.

# Statements to clear up the peak values (μ) and the standard deviation values (σ) to avoid data contamination.
peak_top_diameters.clear() 
peak_bottom_diameters.clear()

std_top_diameters.clear() 
std_bottom_diameters.clear()



# Gaussian function definition.
def gaussian(x, A, mu, sigma):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Function to calculate R-squared value for the Gaussian fit.
def calculate_r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Function to plot histograms with Gaussian fits and adjustable ticks.
def plot_histograms_with_fit(top_data, bottom_data, process_idx, bin_width=0.5, x_range=None, tick_interval=2):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), sharey=True) # The shape is 1 x 2 because I will have 1 row with 2 axes.

    # Definition of the overall range for the x-axis (for tunning purposes, although tunning it may result in a wrong displayal of the histograms).
    if x_range is None:
        x_min = min(np.min(top_data), np.min(bottom_data)) # contains the largest and smallest value for the x axis of the inner plot corresponding to the bottom diameters.
        x_max = max(np.max(top_data), np.max(bottom_data)) # contains the largest and smallest value for the x axis of the inner plot corresponding to the top diameters.
    else:
        x_min, x_max = x_range

    bin_edges = np.arange(x_min, x_max + bin_width, bin_width) # adding x_max + bin width so that the final bin edge is included in the bin_edges array 
    bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2 # I calculate the center of each bin

    # Parameters needed to plot each fit of the Top Diameter Histograms.
    top_counts, _ = np.histogram(top_data, bins=bin_edges)
    top_popt, _ = curve_fit(gaussian, bin_centers, top_counts, p0=[np.max(top_counts), np.mean(top_data), np.std(top_data)]) # This line (and the curve_fit() function) is sponsored by scipy.optimize. top_popt is just an array containing the optimized A, µ and sigma parameters to use them as inputs for the Gaussian function. We ignore the covariance matrix it also privedes. - np.max gives the amplitud, np.mean gives the mean data and np.std gives the standard deviation. 
    peak_top_diameter = top_popt[1] # The μ value of each top diameter is captured. It is returned as an optimized parameter by the curve_fit function, and it is used in the calculation of the roundness as D (average diameter value).
    std_top_diameter = top_popt[2] # In here, the standard deviation (σ) values resulting of applying the Gaussian distribution to the top diameters are captured.
    top_y_fit = gaussian(bin_centers, *top_popt) #stores the fitted Gaussian curve values. Special thanks too the *fucntion. It really saves time.
    top_r_squared = calculate_r_squared(top_counts, top_y_fit)# Calculation of the adjustment by least-squares.
    # top_residuals = top_counts - top_y_fit #Added to check residuals, as I was facing underfitting.

    #Plotting the histogram for the top diameters as a bar chart in the first sublpot of the figure ax[0].        

    ax[0].bar(bin_centers, top_counts, width=bin_width, color="blue", edgecolor="black", alpha=0.6, label="Top Diameters")
    ax[0].plot(bin_centers, top_y_fit, color="red", linewidth=2, label=f"Fit: $y(x) = \\frac{{{top_popt[0]:.2f}}}{{\\sigma \\sqrt{{2\\pi}}}} e^{{-\\frac{{1}}{{2}}\\left(\\frac{{x-{top_popt[1]:.2f}}}{{{top_popt[2]:.2f}}}\\right)^2}}$\n$R^2 = {top_r_squared:.2f}$") 
    ax[0].set_xlabel("Diameter (µm)")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title(f"Top Diameters - Process {process_idx + 1}")
    ax[0].legend()
    ax[0].set_xticks(np.arange(x_min, x_max + tick_interval, tick_interval))  # Adjust tick interval

    #Plotting residuals for the top diameters, to understand better the quiality of my fit.   

    # ax[1, 0].bar(bin_centers, top_residuals, width=bin_width, color="blue", edgecolor="black", alpha=0.6)
    # ax[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    # ax[1, 0].set_xlabel("Diameter (µm)")
    # ax[1, 0].set_ylabel("Residuals")

    # Plot Bottom Diameters Histogram
    bottom_counts, _ = np.histogram(bottom_data, bins=bin_edges)
    bottom_popt, _ = curve_fit(gaussian, bin_centers, bottom_counts, p0=[np.max(bottom_counts), np.mean(bottom_data), np.std(bottom_data)])
    peak_bottom_diameter = bottom_popt[1] # The μ value of each bottom diameter is captured. It is returned as an optimized parameter by the curve_fit function, and it is used in the calculation of the roundness as D (average diameter value).
    std_bottom_diameter = top_popt[2]  # In here, the standard deviation (σ) values resulting of applying the gaussian distribution to the bottom diameters are captured.
    bottom_y_fit = gaussian(bin_centers, *bottom_popt)
    bottom_r_squared = calculate_r_squared(bottom_counts, bottom_y_fit)
    # bottom_residuals = bottom_counts - bottom_y_fit #Added to check residuals, as I was facing underfitting.

    #Plotting the histogram for the bottom diameters as a bar chart in the second sublpot of the figure ax[1] 

    ax[1].bar(bin_centers, bottom_counts, width=bin_width, color="yellow", edgecolor="black", alpha=0.6, label="Bottom Diameters")
    ax[1].plot(bin_centers, bottom_y_fit, color="orange", linewidth=2, label=f"Fit: $y(x) = \\frac{{{bottom_popt[0]:.2f}}}{{\\sigma \\sqrt{{2\\pi}}}} e^{{-\\frac{{1}}{{2}}\\left(\\frac{{x-{bottom_popt[1]:.2f}}}{{{bottom_popt[2]:.2f}}}\\right)^2}}$\n$R^2 = {bottom_r_squared:.2f}$")
    ax[1].set_xlabel("Diameter (µm)")
    ax[1].set_title(f"Bottom Diameters - Process {process_idx + 1}")
    ax[1].legend()
    ax[1].set_xticks(np.arange(x_min, x_max + tick_interval, tick_interval))  # Adjust tick interval

    #Plotting residuals for the bottom diameters, to understand better the quiality of my fit.   

    # ax[1, 1].bar(bin_centers, bottom_residuals, width=bin_width, color="yellow", edgecolor="black", alpha=0.6)
    # ax[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    # ax[1, 1].set_xlabel("Diameter (µm)")
    # ax[1, 1].set_ylabel("Residuals")

    # Formatting
    ax[0].set_xlim(x_min, x_max)
    ax[1].set_xlim(x_min, x_max)
    fig.suptitle(f"Histogram with Gaussian Fit - Process {process_idx + 1}", fontsize=16)
    fig.tight_layout(pad=3)
    plt.show()
    return peak_top_diameter, peak_bottom_diameter, std_top_diameter, std_bottom_diameter

# Plotting the top and bottom diameters histogram per process

for printing in range (8):
   peak_top, peak_bottom, std_top, std_bottom = plot_histograms_with_fit( # peak_top_diameters and peak_bottom_diameters are 2 lists containing the 8 peak values from the gaussian fit for the top, and bottom values, respectively. I will use them later on to perform the roundness calculation.
        top_diameters_masked[printing, :],
        bottom_diameters_masked[printing, :],
        process_idx=printing, # The process index present in the title of each figure
        bin_width=0.4,  # Adjust bin width here
        x_range=(20, 36),  # Adjust x-axis range here if needed
        tick_interval=1  # Adjust tick interval here
    )
   peak_top_diameters.append(peak_top) # The top diameter peak values are added to a list for the calculation of the roundness values.
   peak_bottom_diameters.append(peak_bottom) # The bottom diameter peak values are added to a list for the calculation of the roundness values.
   
   std_top_diameters.append(std_top) # The top diameter standard deviation values are added to a list to be displayed as important parameters in next section.
   std_bottom_diameters.append(std_bottom) # The bottom diameter standard deviation values are added to a list to be displayed as important parameters in next section.
   
   
"-------------------------------------------------------------------------------------------------------"

"""
section 6; Calculating and plotting the important parameters (mean value, standard deviation, roundness, tapper angle) for the top and bottom diameters per process.

This section contains all the code-wise steps addresed to calculate and display the important parameters
in a series of tables, where they can be easily read.

There are 2 mean and 2 standard deviation values displayed for the top and bottom diameters, across all processes 
(2 obtained directly from each Gaussian-fitted histogram, and 2 calculated using the actual data).

There are 2 roundnes values calculated and displayed, using both the averaged diameter values
retrieved after performing the Gaussean fit (μ), and the averaged diameter values calculated 
using the actual data, for all of the processes.

To calculate each roundness value (4 per process, 32 in total), the average major and minor 
axes values of each diameter group was calculated out of the actual data. 

The angle value was calculated concidering that the tangent of alfa equals the averaged top diameter
minus the averaged bottom diameter, all of that divided by 40 µm. Then, by applying the arc_tan to the previously
calculated value, we fing alpha. The diameter values used came from the Gaussian ditributions applied for the top
and bottom diameters per process.

Note:The taper angle will be included as a subplot table when plotting the axes, as there is only 
one value per process (and that doesn't match the grid structure for a 6 x 3 DataFrame). 
I'm currently searching for a way to merge 2 cells using Pandas, but for now I will proceed as told.

"""

"-------------------------------------------------------------------------------------------------------"


Pi = np.pi # Pi value used in the calculation of the tapper angle.

Layer_thickness = 20 #[µm]  Value given to perform the calculation of the taper angle.

dataframes = []  # List that will contain the generated DataFrame objects.

for process_idx in range(8):
    
    # Iterative assignation of the average top and bottom diameters (μ) and standard deviation (σ) values per process (Directly retrieved from the graphs):
    mean_top_gaussian = (peak_top_diameters[process_idx]) # mean_top_gaussian is an 8 x 1-shaped numpy array. Contains the peak (μ) values for each histogram of the top diameters per process, to use them in the calculation of the roundness as the average diameter D. 
    std_top_gaussian = (std_top_diameters[process_idx]) # std_top_gaussian is an 8 x 1-shaped numpy array. Contains all of the 8 standard deviation (σ) values obtained from each of the Gaussian-fitted top diameter histograms per process.
    
    mean_bottom_gaussian = (peak_bottom_diameters[process_idx]) # mean_bottom_gaussian is an 8 x 1-shaped numpy array. Contains the peak (μ) values for each histogram of the bottom diameters per process, to use them in the calculation of the roundness as the average diameter D.  
    std_bottom_gaussian = (std_top_diameters[process_idx]) # std_bottom_gaussian is pretty similar to std_top_gaussian, but the retrieval process is performed for the bottom diameters.  
    
    # Iterative calculation of the important parameters per process (std, and mean values) calculated out of the actual data:
    mean_top = np.mean(top_diameters_masked[process_idx, :]) # mean_top is an 8 x 1-shaped numpy array. contains all of the 8 mean top diameter values for each of the processes (each of them containing 40 top diameter measures). The values were calculated using the information retrieved from the .csv files.
    std_top = np.std(top_diameters_masked[process_idx, :]) # std_top is an 8 x 1-shaped numpy array. contains all of the 8 standard deviation values for each of the processes. The values were calculated using the information retrieved from the .csv files. 
    
    mean_bottom = np.mean(bottom_diameters_masked[process_idx, :]) # mean_bottom is pretty similar to mean_top, but the calculation is performed for the bottom diameters. 
    std_bottom = np.std(bottom_diameters_masked[process_idx, :]) # std_bottom is pretty similar to std_top, but the calculation is performed for the bottom diameters.  

    # Computing of the average major and minor axes, to perform the roundness calculation.
    avg_major_axis = np.mean(major_axes[process_idx, :])
    avg_minor_axis = np.mean(minor_axes[process_idx, :])
    
    peak_top_diameter = peak_top_diameters[process_idx] 
    peak_bottom_diameter = peak_bottom_diameters[process_idx] 
    
    # Calculating the roundness when using the actual average values for top and bottom diameters as D.
    roundness_top_actual = (1 - ((avg_major_axis - avg_minor_axis) / mean_top)) * 100
    roundness_bottom_actual = (1 - ((avg_major_axis - avg_minor_axis) / mean_bottom)) * 100
    
    # Calculating the roundness when using the amplitude value of the gaussian fit for top and bottom diameters (Average diameter value according the Gaussian fit performed) 
    roundness_top_peak = (1 - ((avg_major_axis - avg_minor_axis) / peak_top_diameter)) * 100
    roundness_bottom_peak = (1 - ((avg_major_axis - avg_minor_axis) / peak_bottom_diameter)) * 100
    
    # Calculating the taper angle
    tan_alfa =  (peak_top_diameter -  peak_bottom_diameter) / Layer_thickness * 2
    taper_angle = np.arctan(tan_alfa) * (180 / Pi )
    
    # df is the DataFrame object containing the results of calculating the important parameters per process
    df = pd.DataFrame({ 
        "Metric": ["Gaussian Mean (μ)", "Gaussian Std Dev (σ)", "Roundness (μ From Gaussian Fit) [%] ", " Actual Mean", "Actual Std Dev", "Actual Roundness [%]"],
        "Top Diameters (µm)": [mean_top_gaussian, std_top_gaussian, roundness_top_peak,  mean_top,  std_top, roundness_top_actual],
        "Bottom Diameters (µm)": [ mean_bottom_gaussian, std_bottom_gaussian, roundness_bottom_peak, mean_bottom, std_bottom, roundness_bottom_actual ]
    })

    # df_taper is the DataFrame object containing the taper angle computed. 
    df_taper = pd.DataFrame({
        "Taper Angle [°]" : [f"{taper_angle:.2f}"] 
    })
    
    dataframes.append((df, df_taper))
    
    # The dataframes including the first 3 important parameters are converted to axes to show them as figures. To do that, a figure with the matching dimentions for the table is created using plt.subplots().
    fig, axes = plt.subplots(nrows = 2, figsize=(8, 4))

    # Turn each df into axes (tables); for the STD, Mean, and roundness. The taper angle will be included as a subplot table when plotting the axes, as there is only one value per process (and that doesn't match the grid structure for a 6 x 3 DataFrame). I'm currently searching for a way to merge 2 cells using Pandas, but for now I will proceed as told.
    table = axes[0].table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center",
                     colColours=["#4CAF50", "#A5D6A7", "#A5D6A7"])  # Color scheme for headings.

    # Styling for the first table (STD, mean, and roundness)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.3)  # Adjust size

    # Iterating over each cell and give a specific color schema
    for (row, col), cell in table.get_celld().items(): # get_celld().items() returns a dictionary whose keys are tuples (row and column cordinates for each cell), and whose values are matplotlib objects containing the details and content of each cell. Because of the nature of the keys, tuple unpacking was employed.
        if row == 0:  # Each cell belonging to the header (first row) will have a different color schema.
            cell.set_text_props(weight="bold", color="white")  # Texto en negrita y blanco
            cell.set_facecolor("#4CAF50")  # Green for the headline. 
        else:  # Color schema for the cells that dont belong to the header.
            cell.set_facecolor("#F1F8E9")  # Light green for the cells.
            cell.set_edgecolor("black")  # Black borders
     
    # Axis removal
    axes[0].axis("off")

    # Title setting
    axes[0].set_title(f"Process {process_idx + 1} - Important Parameters", fontsize=12, fontweight="bold", pad=10)
    
    # Turning the taper angle df into an ax (table)
    table_taper = axes[1].table(cellText=df_taper.values, colLabels=df_taper.columns, cellLoc="center", loc="center",
                                colColours=["#4CAF50", "#A5D6A7", "#A5D6A7"])
    
    # Styling for the second table (taper angle)                     colColours=["#A5D6A7", "#A5D6A7"])
    table_taper.auto_set_font_size(False)
    table_taper.set_fontsize(10)

    # Iterating over each cell and give a specific color schema
    for (row, col), cell in table_taper.get_celld().items(): # get_celld().items() returns a dictionary whose keys are tuples (row and column cordinates for each cell), and whose values are matplotlib objects containing the details and content of each cell. Because of the nature of the keys, tuple unpacking was employed.
        if row == 0:  # Each cell belonging to the header (first row) will have a different color schema.
            cell.set_text_props(weight="bold", color="white")  # Texto en negrita y blanco
            cell.set_facecolor("#4CAF50")  # Green for the headline. 
        else:  # Color schema for the cells that dont belong to the header.
            cell.set_facecolor("#F1F8E9")  # Light green for the cells.
            cell.set_edgecolor("black")  # Black borders
     
    # Axis removal
    axes[1].axis("off")


    plt.tight_layout()
    plt.show()

# Displaying the tables of the calculated metrics in the output section
for index, (df, df_taper) in enumerate(dataframes):
    print(f"Process {index + 1} Data")
    print(df)
    print(df_taper.to_string(index=False, col_space=10, justify='right'))
    print("\n")


"-------------------------------------------------------------------------------------------------------"

"""
section 7; Saving the output results in a .txt file (for all of the important parameters 
requested, and the top and bottom diameter values per process).

This section contains the code-wise steps addressed to export to a readable nice looking .txt file
the results of separating the top and bottom diameters across every process, and the
prevously calculated and ploted important metrics.

I utilized a for loop to iterate over the separated top and bottom diameters per process, and the 
previously created dataframes for both the important parameters and the taper angle per process.



"""

"-------------------------------------------------------------------------------------------------------"


# Specify the directory were the file will be saved
user_directory = input("Enter a specific directory path or hit Enter to store the file in the default location").strip()

# Use current directory if none provided
if not user_directory:
    user_directory = os.getcwd()

# Create the directory if it doesn’t exist
if not os.path.exists(user_directory):
    os.makedirs(user_directory)

# Define the filename and path
output_filename = "Laser_Vias_Analysis_Results.txt"
output_path = os.path.join(user_directory, output_filename)

# Save the results to the .txt file
with open(output_path, "w") as file:
    # **(1) Report Header**
    file.write("=" * 100 + "\n")
    file.write(" Laser-Drilled Vias Analysis Report - DAVID ALEJANDRO DE LOS REYES OSTOS - Matrikel nr: 12435240\n")
    file.write("=" * 100 + "\n")
    file.write("This report contains the analysis of laser-drilled vias across 8 processes.\n")
    file.write("It includes top and bottom diameters, Gaussian fit parameters, and all of the important metrics requested.\n")
    file.write("\n\n")
    
    for process_idx, (df, df_taper) in enumerate(dataframes):
        file.write(f"========== Process {process_idx + 1} ==========\n\n")
        
        # **(2) Display Top and Bottom Diameters in Table Format**
        file.write("Top and Bottom Diameters per Process (µm):\n\n")
        
        # Format the table header
        file.write(f"{'Index':<8}{'Top Diameter (µm)':<20}{'Bottom Diameter (µm)':<20}\n") # < creates a space and left aligns the value to be printed (Index in this case).
        file.write("=" * 50 + "\n")

        # Write the diameter values row by row
        top_diameters_list = top_diameters_masked[process_idx, :]
        bottom_diameters_list = bottom_diameters_masked[process_idx, :]

        for i in range(len(top_diameters_list)):
            file.write(f"{i+1:<8}{top_diameters_list[i]:<20.2f}{bottom_diameters_list[i]:<20.2f}\n")
            
        file.write("\n")
      
        # **(5) Save the Important Metrics Table**
        file.write("Important Parameters:\n")
        file.write(df.to_string(index=False) + "\n\n")

        # **(6) Save the Taper Angle Table**
        file.write(df_taper.to_string(index=False) + "\n\n")

        file.write("-" * 100 + "\n\n")        

    file.write("\n" + "=" * 100 + "\n")
    file.write(" END OF REPORT\n")
    file.write("=" * 100 + "\n")

print(f"Results successfully saved to: {output_path}")






