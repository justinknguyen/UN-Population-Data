# 592-project.py
# Justin Nguyen & Nathan Tham
# Group 21 - The Batt Boys

# Using the provided UN Population Datasets, the user will be able to choose a Region/Country/Area and one of the two datasets provided to view the data
# and aggregate stats calculated from the data. An excel of the final combined datframe will be exported for viewing (6308 rows and 11 columns).
# Additionally, a plot of the value increase per year of each series within the chosen Region/Country/Area will be provided.

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
import matplotlib.pyplot as plt

def create_dataframe(un_codes, un_pop_ind, un_pop_val):
    '''
    Creates dataframe from input .xlxs files. Combines all data into one large dataframe. 
    Arguments:
        un_codes - All Regions/Countries/Areas along with their associated type
        un_pop_ind - Country code and Region/Country/Area data from input files
        un_pop_val - All data from datasets 
    Returns the combined dataframe 
    '''
    # ***************** Stage 2: DataFrame Creation ***************** #
    # 3 level row index (Type -> Region/Country/Area -> Code)
    # 1 level column index (Index, Year, Series, Capital City, Value, Dataset, + created 5 columns)
    # Type will indicate if it is a UN Region, UN Sub-Region, or Country.

    row_df1 = pd.DataFrame({'Region/Country/Area'   : un_pop_ind['Region/Country/Area'],
                            'Code'                  : un_pop_ind['Code']                })

    row_df2 = pd.DataFrame({'Type'                  : un_codes['Type'],
                            'Region/Country/Area'   : un_codes['Region/Country/Area']})

    # ''' vvvv REQUIREMENT: first .merge() '''                                                  # Merge the two dataframes together based on Region/Country/Area matches 
    row_df = pd.merge(row_df2, row_df1, on='Region/Country/Area', how='right').fillna('-')      # "how=" prevents removal of non-matches when merging.

    index = pd.MultiIndex.from_frame(row_df)                                                    # Turn row_df into multi-index
    columns = list(un_pop_val.columns)                                                          # Get list of column names

    # ''' vvvv REQUIREMENT: multi-index Pandas dataframe '''
    df = pd.DataFrame(un_pop_val.values, index=index, columns=columns).fillna('-')              # Create dataframe using multi-index and columns
    df.insert(0, 'Index', range(len(un_pop_ind)))                                               # Add Index column for better readability 
    df.to_excel('UN Population Datasets/combined_df.xlsx')                                      # Export as an excel file

    return df 

def user_prompt(df, un_pop_ind):
    '''
    Intakes user input for Region/Country/Area and dataset. Function then filters dataframe to return only the requested dataset. 
    Arguments:
        df - Conbined dataframe
        un_pop_ind - Country code and Region/Country/Area data from input files
    Returns filtered dataframe as well as user inputs for Region/Country/Area and dataset #
    '''
     # ***************** Stage 3: User Entry ***************** #

    code_list = list(un_pop_ind.drop_duplicates(subset='Code')['Code'])                                 # Get list of codes
    name_list = list(un_pop_ind.drop_duplicates(subset='Region/Country/Area')['Region/Country/Area'])   # Get list of names
    name_code_map = dict(zip(name_list, code_list))                                                     # Dictionary (key: name, value: code)
    code_name_map = dict(zip(code_list, name_list))                                                     # Dictionary (key: code, value: name)
    
    # ''' vvvv REQUIREMENT: first user input searches multi-index '''
    while 1:                                                                                            # Loops until valid input found
        check1 = input("\nPlease enter a Region/Country/Area name or code that you wish to calculate statistics for: ")
        try:
            if check1 in name_code_map:                                                                 # if input is in the list of names 
                break
            elif int(check1) in code_name_map:                                                          # if input is in the list of codes
                break
            else:
                raise ValueError
        except ValueError:
            print("\n    Error: You must enter a valid Region/Country/Area name or code. Enter 'help' to see name and code list.")
            if check1 == 'help':                                                                        # print name/code if user inputs 'help'
                print("")
                for name in name_code_map:
                    print("    " + name + " : " + str(name_code_map[name]))
    
    idx = pd.IndexSlice
    df_sorted = df.sort_values(['Series','Year'], ascending=[True,True])                                # Sort entire dataframe by year (ascending) and series
    try:
        filtered = df_sorted.loc[idx[:, :, int(check1)], idx[:]]                                        # Filter df based on code
    except:
        filtered = df_sorted.loc[idx[:, check1], idx[:]]                                                # Filter df based on name

    # ''' vvvv REQUIREMENT: create pivot table and export Matplotlib plot '''
    filtered.pivot_table('Value', index='Year', columns='Series').plot()
    plt.ylabel('Values per Year')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.50), ncol= 2)                                # Place legend outside of graph 
    plt.savefig('figure.png', bbox_inches = 'tight')                                                    # Save graph as png file in project dir

    # ''' vvvv REQUIREMENT: second user input chooses dataset '''
    while 1:
        check2 = input("\nPlease select the Dataset that you wish to calculate statistics for (1 or 2): ")
        try:
            if int(check2) == 1:                                                                        # if input is 1 
                print("\n*** Requested Region/Country/Area Dataset ***\n")
                print(code_name_map[int(check1)], ", Code:", int(check1), ", Dataset:", int(check2))
                break
            elif int(check2) == 2:                                                                      # if input is 2
                print("\n*** Requested Region/Country/Area Dataset ***\n")
                print(code_name_map[int(check1)], ", Code:", int(check1), ", Dataset:", int(check2))
                break
            else:
                raise ValueError
        except ValueError:
            print("\n    Error: You must enter either 1 or 2. (1: Population characteristics 2: Population data)")
    
    print("Filtered Dataset:")
    # ''' vvvv REQUIREMENT: masking operation '''
    mask = filtered['Dataset'] == int(check2)                                                           # Filter df again based on dataset chosen
    filtered = filtered[mask] 
    print(filtered)

    return filtered, check1, check2

def calculations(df, filtered):
    '''
    Calculates the mean, standard deviation, minimum, maximum, and sum for all the requested dataset
    Arguments:
        df - Combined dataframe
        filtered - Filtered dataframe containing only requested country from user_prompt()
    Returns dataframe filled with statistical information for requested country to caller
    '''
    # ***************** Stage 4: Analysis and Calculations ***************** #

    print("\n*** Calculating Aggregate Stats for Selected Dataset ***\n")

    # MEAN
    # ''' vvvv REQUIREMENT: .groupby() '''
    mean = filtered.groupby('Series')['Value'].mean().rename('Mean')                                    # Group by Series then calculate on Value column
    mean_df = filtered.reset_index()                                                                    # Turn multi-index back to columns                         
    mean_df = pd.merge(mean_df, mean, on='Series').set_index(['Type', 'Region/Country/Area', 'Code'])   # Place back as multi-index after merging
    # ''' ^^^^ REQUIREMENT: second .merge() '''
    # STD
    std = filtered.groupby('Series')['Value'].std().rename('STD')                                       # Group by Series then calculate on Value column
    std_df = mean_df.reset_index()                                                                      # Turn multi-index back to columns
    std_df = pd.merge(std_df, std, on='Series').set_index(['Type', 'Region/Country/Area', 'Code'])      # Place back as multi-index after merging
    # MIN
    min = filtered.groupby('Series')['Value'].min().rename('Minimum')                                   # Group by Series then calculate on Value column
    min_df = std_df.reset_index()                                                                       # Turn multi-index back to columns
    min_df = pd.merge(min_df, min, on='Series').set_index(['Type', 'Region/Country/Area', 'Code'])      # Place back as multi-index after merging
    # MAX
    max = filtered.groupby('Series')['Value'].max().rename('Maximum')                                   # Group by Series then calculate on Value column
    max_df = min_df.reset_index()                                                                       # Turn multi-index back to columns
    max_df = pd.merge(max_df, max, on='Series').set_index(['Type', 'Region/Country/Area', 'Code'])      # Place back as multi-index after merging
    # SUM
    sum = filtered.groupby('Series')['Value'].sum().rename('Sum')                                       # Group by Series then calculate on Value column
    calc_df = max_df.reset_index()                                                                      # Turn multi-index back to columns
    calc_df = pd.merge(calc_df, sum, on='Series').set_index(['Type', 'Region/Country/Area', 'Code'])    # Place back as multi-index after merging

    print(calc_df)
    print("")

    # ''' vvvv REQUIREMENT: pivot table '''
    table = pd.pivot_table(calc_df, values=['Value'], 
                                    index=['Type', 'Region/Country/Area', 'Code'], 
                                    aggfunc={'Value':[np.mean, np.std, np.min, np.max, np.sum]})
    
    print(table)

    print("\n*** Calculation Completed! Uploading to Dataset ***\n")
    
    calc_df = calc_df.reset_index()                                                                     # Turn multi-index back to columns                                                                                          
    df = df.reset_index()
    filt_calc_df = calc_df.drop(['Type','Region/Country/Area','Code','Year','Series','Capital City','Value','Dataset'], axis=1)

    # ''' vvvv REQUIREMENT: added 5 columns to combined dataframe '''
    final_df = pd.merge(df, filt_calc_df, on='Index', how='left').set_index(['Type', 'Region/Country/Area', 'Code']).fillna(0)  # Place back as multi-index after merging
    final_df = final_df.sort_values(['Code','Series','Year'], ascending=[True,True,True])                                       # Sort entire dataframe by year (ascending) and series

    return final_df

def calculate_all(df, requested_df, un_pop_ind, requested_code, requested_ds):
    '''
    Calculates the mean, standard deviation, minimum, maximum, and sum for all the entire dataset
    Arguments: 
        df - Combined dataframe
        requested_df - Dataframe with statistic columns. Already contains statistical data from calculations() 
        un_pop_ind - Country code and Region/Country/Area data from input files
        requested_code - Code that was requested by user
        requested_ds - Dataset that was requested by user
    Returns final dataframe to caller 
    '''
    code_list = list(un_pop_ind.drop_duplicates(subset='Code')['Code'])                                     # Get list of codes

    for dataset in range(1,3):
        for code in code_list:
            idx = pd.IndexSlice
            df_sorted = df.sort_values(['Series','Year'], ascending=[True,True])                            # Sort entire dataframe by year (ascending) and series
            try:
                filtered = df_sorted.loc[idx[:, :, int(code)], idx[:]]                                      # Filter df based on code
            except:
                filtered = df_sorted.loc[idx[:, code], idx[:]]                                              # Filter df based on name
            mask = filtered['Dataset'] == int(dataset)                                                      # Filter df again based on dataset chosen
            filtered = filtered[mask] 
            
            # MEAN
            mean = filtered.groupby('Series')['Value'].mean().rename('Mean')                                                # Group by Series then calculate on Value column
            mean_df = filtered.reset_index()                                                                                # Turn multi-index back to columns                     
            mean_df = pd.merge(mean_df, mean, on='Series', how='left').set_index(['Type', 'Region/Country/Area', 'Code'])   # Place back as multi-index after merging
            # STD
            std = filtered.groupby('Series')['Value'].std().rename('STD')                                                   # Group by Series then calculate on Value column
            std_df = mean_df.reset_index()                                                                                  # Turn multi-index back to columns
            std_df = pd.merge(std_df, std, on='Series').set_index(['Type', 'Region/Country/Area', 'Code'])                  # Place back as multi-index after merging
            # MIN
            min = filtered.groupby('Series')['Value'].min().rename('Minimum')                                               # Group by Series then calculate on Value column
            min_df = std_df.reset_index()                                                                                   # Turn multi-index back to columns
            min_df = pd.merge(min_df, min, on='Series').set_index(['Type', 'Region/Country/Area', 'Code'])                  # Place back as multi-index after merging
            # MAX
            max = filtered.groupby('Series')['Value'].max().rename('Maximum')                                               # Group by Series then calculate on Value column
            max_df = min_df.reset_index()                                                                                   # Turn multi-index back to columns
            max_df = pd.merge(max_df, max, on='Series').set_index(['Type', 'Region/Country/Area', 'Code'])                  # Place back as multi-index after merging
            # SUM
            sum = filtered.groupby('Series')['Value'].sum().rename('Sum')                                                   # Group by Series then calculate on Value column
            calc_df = max_df.reset_index()                                                                                  # Turn multi-index back to columns
            calc_df = pd.merge(calc_df, sum, on='Series').set_index(['Type', 'Region/Country/Area', 'Code'])                # Place back as multi-index after merging

            if int(code) != int(requested_code) or int(dataset) != int(requested_ds):
                calc_df = calc_df.reset_index()
                requested_df = requested_df.reset_index()
                filt_calc_df = calc_df.drop(['Type','Region/Country/Area','Code','Year','Series','Capital City','Value','Dataset'], axis=1)
                requested_df = pd.merge(requested_df, filt_calc_df, on='Index', how='left').set_index(['Type', 'Region/Country/Area', 'Code']).fillna(0)    # Place back as multi-index after merging                                                                 # Do not recalculate stats if it was already calculated
                requested_df['Mean'] = requested_df['Mean_x'] + requested_df['Mean_y']
                requested_df['STD'] = requested_df['STD_x'] + requested_df['STD_y']
                requested_df['Minimum'] = requested_df['Minimum_x'] + requested_df['Minimum_y']
                requested_df['Maximum'] = requested_df['Maximum_x'] + requested_df['Maximum_y']
                requested_df['Sum'] = requested_df['Sum_x'] + requested_df['Sum_y']
                requested_df = requested_df.drop(['Mean_x', 'Mean_y', 'STD_x', 'STD_y', 'Minimum_x', 'Minimum_y', 'Maximum_x', 'Maximum_y', 'Sum_x', 'Sum_y'], axis=1)
                final_df = requested_df.sort_values(['Code','Series','Year'], ascending=[True,True,True])   # Sort entire dataframe by year (ascending) and series
    return final_df

def main():
    '''
    main(), implements all function calls. Used for loading data and print statement.
    '''
    print("\nENSF 592 Final Project")

    # ***************** Stage 1: Dataset Selection ***************** #

    # ''' vvvv REQUIREMENT: 3 separate Excel sheets '''
    un_codes     = pd.read_excel('UN Population Datasets/UN Codes (modified).xlsx')                 
    un_pop1_ind  = pd.read_excel('UN Population Datasets/UN Population Dataset 1.xlsx', usecols='A:B')      # Country code and Region/Country/Area data from columns
    un_pop2_ind  = pd.read_excel('UN Population Datasets/UN Population Dataset 2.xlsx', usecols='A:B')      # Country code and Region/Country/Area data from columns
    un_pop1_val  = pd.read_excel('UN Population Datasets/UN Population Dataset 1.xlsx')
    un_pop2_val  = pd.read_excel('UN Population Datasets/UN Population Dataset 2.xlsx')

    files = [un_pop2_ind, un_pop1_ind]
    un_pop_ind = pd.concat(files)                                                                           # Concat population datasets into one dataframe
    un_pop_ind = un_pop_ind.sort_values('Code', ascending=True)

    files = [un_pop2_val, un_pop1_val]
    un_pop_val = pd.concat(files)                                                                           # Concat population datasets into one dataframe
    un_pop_val = un_pop_val.sort_values('Code', ascending=True)                                             # Sort dataframe by ascending code order
    un_pop_val = un_pop_val.drop(['Code', 'Region/Country/Area'], axis=1)

    df = create_dataframe(un_codes, un_pop_ind, un_pop_val)
    print("\nCombined Dataset:")  
    print(df)

    print("\n*** Aggregate Stats for Entire Dataset ***\n")
    # ''' vvvv REQUIREMENT: describe method '''
    print(df.dropna().describe()) 
    
    filtered, requested_code, requested_ds = user_prompt(df, un_pop_ind)  
    requested_df = calculations(df, filtered)                                                               # Calculate the requested statistics     
    requested_df.to_excel('UN Population Datasets/final_df.xlsx')                                           # export as an excel file                

    # ***************** Stage 5: Export and Matplotlib ***************** #

    # ''' vvvv REQUIREMENT: final combined dataset has 6000+ rows and 11 columns '''
    print("*** Calculating Remaining Statistics for Entire Dataset... ***\n")
    final_df = calculate_all(df, requested_df, un_pop_ind, requested_code, requested_ds)                    # Calculate statistics across entire dataframe
    final_df.to_excel('UN Population Datasets/final_df.xlsx')                                               # export as an excel file
    print("*** Upload Complete! Please view 'final_df.xlsx' under UN Population Datasets. ***\n")


if __name__ == '__main__':
    main()