#without truncating sequence length
import pickle
from torch.autograd import Variable
import numpy as np
import torch
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#save data
datarows = 80
#seq_length = 17500 ##major
num_sensors=98
save_file = False
plot_local = False
plot_folder = '../plots/'
pickle_path = "../pickles/"

#load sensor header details
sensor_headers = pd.read_excel("../EMPIR_Data/"+"ScopeDataWithHeadings.xlsx").columns[1:]

#now handle inputs loading
input_data_path = "../EMPIR_Data/Data/AFRC Radial Forge - Historical Data/"
file_names = os.listdir(input_data_path)

if os.path.isfile(pickle_path+"strath_full_inputs.p") == False:
    data_inputs_list = []
    for i in range(len(file_names)):
        file_csv = pd.read_csv(input_data_path+ file_names[i], encoding='cp1252')
        file_csv= file_csv.drop('Timer Tick [ms]', axis=1)
        np_data = file_csv.values[:,:]
        data_inputs_list.append(np_data)    
    pickle.dump(data_inputs_list, open( pickle_path+"strath_full_inputs.p", "wb" ) )
else:
    data_inputs_list = pickle.load( open( pickle_path+"strath_full_inputs.p", "rb" ) )

#now handle output loading
output_data_path = "../EMPIR_Data/Data/"
if os.path.isfile(pickle_path+"strath_full_outputs.p") == False:
    output_pd = pd.read_excel(output_data_path+"CMMData.xlsx")
    pickle.dump(output_pd, open( "../pickles/strath_full_outputs.p", "wb" ) )
else:
    output_pd = pickle.load( open( pickle_path+"strath_full_outputs.p", "rb" ) )

#extract necessary output values
output_headers = output_pd.columns
base_val = output_pd.values[0,:]

output_val = output_pd.values[3:,:]
output_val = output_val[:-8,:]
output_val = Variable(torch.from_numpy(output_val).float())

np_data_outputs = output_val.cpu().detach().numpy()

#extract error from expected base values
for output in range(np_data_outputs.shape[1]):
    np_data_outputs[:,output] -=base_val[output]

#investigate timesteps of input sensors
np_data_timesteps = np.array([data_inputs_list[i].shape[0] for i in range(len(data_inputs_list))])
np_data_inputs_list = np.array(data_inputs_list)
print(stats.describe(np_data_timesteps))
plt.title("Histogram of Timesteps Distribution")
plt.hist(np_data_timesteps)
plt.xlabel("Timesteps")
plt.ylabel("Frequency")
plt.savefig(plot_folder+'histogram_vs_timestep.png', bbox_inches="tight")
plt.show()

plt.title("Graph of Timesteps vs Part Number")
plt.plot(np.arange(len(np_data_timesteps)),np_data_timesteps)
plt.xlabel("Part Number")
plt.ylabel("Timesteps")
plt.savefig(plot_folder+'timesteps_vs_partnumber.png', bbox_inches="tight")
plt.show()
#from plot, seems that there are two categories of ranges: ~18000 and ~23000

#lets have full plots of both ranges
#separate data into two ranges
index_18k =np.arange(datarows)[np_data_timesteps<=20000]
index_23k =np.arange(datarows)[np_data_timesteps>20000]
np_data_inputs_18k = np_data_inputs_list[index_18k]
np_data_inputs_23k = np_data_inputs_list[index_23k]


#plot in local or save into file
if save_file or plot_local:
    for sensor_index in range(num_sensors):
    #sensor_index = 1
        sensor_detail = sensor_headers[sensor_index]
        num_parts = len(data_inputs_list)
        color_plots = plt.cm.viridis(np.linspace(0,1,num_parts))
        color_plots[:,3] = 0.8
          
        #begin plotting, loop for every part data with different color  
        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
        
        #prepare plot data for 18k 
        num_parts = np_data_inputs_18k.shape[0]
        color_plots = plt.cm.spring(np.linspace(0,1,num_parts))
        color_plots[:,3] = 0.8
        
        for parts_row in range(num_parts):
            sensor_data_row = np_data_inputs_18k[parts_row][:,sensor_index]    
            ax1.plot(sensor_data_row,color=color_plots[parts_row])
        
        #prepare plot data for 23k 
        num_parts = np_data_inputs_23k.shape[0]
        color_plots = plt.cm.spring(np.linspace(0,1,num_parts))
        color_plots[:,3] = 0.8
        
        for parts_row in range(num_parts):
            sensor_data_row = np_data_inputs_23k[parts_row][:,sensor_index]
            ax2.plot(sensor_data_row,color=color_plots[parts_row])
        
        #set axes and titles
        fig.suptitle(sensor_detail, fontsize=16)
        ax1.set_title("18k Category")
        ax2.set_title("23k Category")
        ax2.set_xlabel("Timesteps")
        
        #save plots
        if(save_file):
            plt.savefig(plot_folder+'sensors/'+'sensor_'+str(sensor_index)+'.png', bbox_inches="tight")
        if(plot_local):
            plt.show()
        plt.clf()
    
#now lets compare outputs of 18k vs 23k
np_data_outputs_18k = np_data_outputs[index_18k]
np_data_outputs_23k = np_data_outputs[index_23k]

for output in range(np_data_outputs.shape[1]):
    plt.title("Histogram: "+output_headers[output])
    plt.hist(np_data_outputs_18k[:,output],density=True)
    plt.hist(np_data_outputs_23k[:,output],density=True)
    plt.xlabel("Error from baseline")
    plt.ylabel("Probability Density")
    plt.legend(labels=['18k','23k'],loc='best')
    #plt.savefig(plot_folder+'outputs/'+'output_histogram_'+str(output)+'.png', bbox_inches="tight")
    plt.show()
    plt.clf()
   
#print statistics for outputs
df_data_outputs_18k = pd.DataFrame(np_data_outputs_18k,columns=output_headers)
df_data_outputs_23k = pd.DataFrame(np_data_outputs_23k,columns=output_headers)

print(df_data_outputs_18k.describe())
print(df_data_outputs_23k.describe())

#significance test
std_error = np.sqrt(df_data_outputs_18k.std()**2/df_data_outputs_18k.count()[0]+df_data_outputs_23k.std()**2/df_data_outputs_23k.count()[0])
t_stats = (df_data_outputs_18k.mean()-df_data_outputs_23k.mean())/std_error

#check if difference is statistically significant with 95% confidence
print(np.abs(t_stats)>1.98)
#NOTE: there are some dimensions which show statistically significant differences

#let's investigate more on the input side's features and its relationship with the output dimensions
#convert into features, drop useless columns
#sensor_number =0
if 'df_features_inputs' in globals():
    del df_features_inputs
for sensor_number in range(num_sensors):
    for part_number in range(datarows):
        desc = stats.describe(np_data_inputs_list[part_number][:,sensor_number])
        desc_dict={'means_'+str(sensor_number):desc.mean.flatten(),'std_'+str(sensor_number):np.sqrt(desc.variance.flatten()),'skewness_'+str(sensor_number):np.array([desc.skewness]),'kurtosis_'+str(sensor_number):np.array([desc.kurtosis])}
        
        if 'df_features_inputs_perSensor' not in globals():
            df_features_inputs_perSensor =pd.DataFrame(desc_dict)
        else:   
            df_features_inputs_perSensor_temp =pd.DataFrame(desc_dict)
            df_features_inputs_perSensor=df_features_inputs_perSensor.append(df_features_inputs_perSensor_temp,ignore_index=True)
    if 'df_features_inputs' not in globals():
        df_features_inputs =df_features_inputs_perSensor
    else:
        df_features_inputs = df_features_inputs.join(df_features_inputs_perSensor,rsuffix=str(sensor_number))
    if 'df_features_inputs_perSensor' in globals():
        del df_features_inputs_perSensor

timesteps_count= pd.DataFrame({'timesteps_count':np_data_timesteps})
df_features_inputs.join(timesteps_count,rsuffix=str(sensor_number))

#create dataFrame of output dimensions
df_outputs = pd.DataFrame(np_data_outputs)
df_outputs.columns = ['output_'+str(i) for i in range(df_outputs.shape[1])]

#final df of inputs and outputs
df_final_combine = df_features_inputs.join(df_outputs,rsuffix=str(sensor_number))

#remove useless inputs
useless_columns = []
for col in df_final_combine:
    if len(df_final_combine[col].unique()) == 1:
        useless_columns.append(col)

count_useless_sensors = []
for col in useless_columns:
    if 'means' in col:
        count_useless_sensors.append(col)
print("Useless sensors: ")
print(count_useless_sensors)
print("Count: "+ str(len(count_useless_sensors)))


#save into csv
df_final_combine_dropped = df_final_combine.drop(useless_columns,axis=1)
df_final_combine_dropped.to_csv("../data_massaged/"+'data_massaged_v1.csv')

pickle.dump(df_final_combine_dropped, open( "../pickles/strath_preproc_data.p", "wb" ) )
strath_preproc_data = pickle.load( open( pickle_path+"strath_preproc_data.p", "rb" ) )

















