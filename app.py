from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import os
import numpy as np

app = Flask(__name__)

# Some parameters
ref_date = "2025-01"
version = "_vtest"

allFundNames = ["RRF", "CP", "HE21", "HE23", "HE25", "EuroHPC", "EIC", "IHI", "KDT", "SNS", "DEP", "CEF2"]

mainfold = os.getcwd()
outputfold = os.path.join(mainfold, "output")
datafold = os.path.join(mainfold, "data")

targetOrder = [
    "Basic digital skills", "ICT specialists",
    "Gigabit network coverage", "5G coverage",
    "Semiconductors", "Edge nodes", "Quantum computing", "Cloud computing services", "Data analytics", "Artificial intelligence",
    "Digital late adopters", "Unicorns",
    "Digital public services", "Electronic health records", "e-ID"
]

# Ensure the mainfold directory exists
os.makedirs(mainfold, exist_ok=True)

# Define file paths
EIC_path = os.path.join(mainfold, "data", "_HE", "_EIC", 'INPUT_HE_EIC_tables_NOT UPDATED_20240403.xlsx')
KDT_path = os.path.join(mainfold, "data", "_HE", "_JUs", "KDT", 'INPUT_Chips_20251303.xlsx')
IHI_path = os.path.join(mainfold, "data", "_HE", "_JUs", "IHI", 'INPUT_IHI_20250313.xlsx')
EuroHPC_path = os.path.join(mainfold, "data", "_HE", "_JUs", "EuroHPC", 'INPUT_EuroHPC_20250313.xlsx')
SNS_path = os.path.join(mainfold, "data", "_HE", "_JUs", "SNS", 'INPUT_SNS_20240313.xlsx')
HE21_path = os.path.join(mainfold, "data", "_HE", "HE21-22", 'HE21_22_target_allocation_rawdata.xlsx')
HE23_path = os.path.join(mainfold, "data", "_HE", "HE23-24", 'HE23_24_target_allocation_rawdata.xlsx')
HE25_path = os.path.join(mainfold, "data", "_HE", "HE25-27", 'HE25_27_target_allocation_rawdata.xlsx')
CEF2_path = os.path.join(mainfold, "data", "_CEF2", 'INPUT_CEF2_20250313.xlsx')
DEP25_path = os.path.join(mainfold, "data", "_DEP", 'INPUT_DEP25-27_20240430.xlsx')
RRF_path = os.path.join(mainfold, "data", "_RRF", 'FENIX_Extraction 10.03.25.xlsx')
CP_path = os.path.join(mainfold, "data", "_CP", 'CP21_27_final_tables.xlsx')
DEP25_27_path = os.path.join(mainfold, "data", '_DEP', 'INPUT_DEP25_TO-BE-CONFIRMED.xlsx')
DEP_path = os.path.join(mainfold, "data", '_DEP', 'INPUT_DEP21-24_tables_NO YEAR_20240403_corrected_v2.xlsx')

# Read data into dataframes
fundList = [EIC_path, KDT_path, IHI_path, EuroHPC_path, SNS_path, HE21_path, HE23_path, HE25_path, CEF2_path, DEP25_path, CP_path]
listnames = ["EIC", "KDT", "IHI", "EuroHPC", "SNS", "HE21", "HE23", "HE25", "CEF2", "DEP25", "CP"]

df_dict = {}
for i, file_path in enumerate(fundList):
    df_dict[listnames[i]] = pd.read_excel(file_path)

# Create individual dataframe variables in the global namespace
globals().update(df_dict)

# DEP cleaning
DEP = pd.read_excel(DEP_path, sheet_name="DEP DD WP23-24 shares")
DEP2 = pd.read_excel(DEP_path, sheet_name="DEP DD WP21-22&EDIH21-23 shares")
DEP3 = pd.read_excel(DEP25_27_path)

DEP_main = pd.concat([DEP, DEP2, DEP3], ignore_index=True)


RRF_allocations = "mapping_interventionFields_targets.xlsx"
RRF_targets = pd.read_excel(os.path.join(datafold, "_RRF", RRF_allocations))

RI_name = "2 - 009bis - Investment in digital-related R&I activities (including excellence research centres, industrial research, experimental development, feasibility studies, acquisition of fixed or intangible assets for digital related R&I activities)"
quater_name = "6 - 021quater - Investment in advanced technologies such as: High-Performance Computing and Quantum computing capacities/Quantum communication capacities (including quantum encryption); in microelectronics design, production and system-integration; next generation of European data, cloud and edge capacities (infrastructures, platforms and services); virtual and augmented reality, DeepTech and other digital advanced technologies. Investment in securing the digital supply chain."

RRF = pd.read_excel(RRF_path)
conditions = RRF['Digital Objective Intervention field'].isin([RI_name, quater_name])
RRF['match_key'] = RRF.apply(lambda x: f"{x['Digital Objective Intervention field']} - {x['Measure Reference']} - {x['Measure Name']}" 
                            if x['Digital Objective Intervention field'] in [RI_name, quater_name]
                            else x['Digital Objective Intervention field'], axis=1)

RRF['Cost'] = RRF['Cost']/(10**6)

RRF_targets_clean = RRF_targets.copy()
RRF_targets_clean['match_key'] = RRF_targets_clean.apply(
    lambda x: f"{x['intervention_field']} - {x['measure']}" 
    if x['intervention_field'] in [RI_name, quater_name]
    else x['intervention_field'], 
    axis=1)

RRF_measures_merged = pd.merge(RRF, RRF_targets_clean, on='match_key', how='left')

RRF_measures_merged['Digital Tag'] = pd.to_numeric(RRF_measures_merged['Digital Tag'], errors='coerce')
RRF_measures_merged['dd-relevant budget'] = RRF_measures_merged['Cost'] * RRF_measures_merged['dd_share'] * RRF_measures_merged['Digital Tag']
RRF_measures_merged['Total budget'] = RRF_measures_merged['Cost']
RRF_measures_merged = RRF_measures_merged.drop(columns=['Digital Objective Intervention field', 'match_key', 'intervention_field'])

RRF = RRF_measures_merged.copy()

df_dict['DEP_main'] = DEP_main
df_dict['RRF'] = RRF
# Alter DEP25
DEP25 = DEP25[(DEP25['year'] != 2025) & (DEP25['fund_part'] != "Main Work Programme")]
# Update the dictionary with the filtered dataframe
df_dict["DEP25"] = DEP25

HE21['Total budget'] = HE21['Budget']
HE23['Total budget'] = HE23['Budget']
HE25['Total budget'] = HE25['Budget']

# Update the dictionary with the filtered dataframe
df_dict["HE21"] = HE21
df_dict["HE23"] = HE23
df_dict["HE25"] = HE25

def share2budg(df):
    for target in targetOrder:
        if target in df.columns:
            df[target] = df[target] * df['dd_budget']
    return df

def search_keywords_in_df(df, fund_name, keywords):
    keywords = [keyword.lower() for keyword in keywords]
    
    if fund_name == 'RRF':
        str_columns = df.select_dtypes(include=['object']).columns
        matches = pd.Series(False, index=df.index)
        for col in str_columns:
            matches |= df[col].astype(str).str.lower().apply(lambda x: any(keyword in x for keyword in keywords))
        
        if matches.any():
            temp_df = pd.DataFrame()
            temp_df['call_topic_measure'] = df['Measure Reference'][matches] + " - " + df['Measure Name'][matches]
            temp_df['Fund'] = fund_name
            
            # Find the 'dd-relevant budget' column
            budget_col = next((col for col in df.columns if 'dd-relevant' in str(col).lower() and 'budget' in str(col).lower() and 'share' not in str(col).lower()), None)
            if budget_col:
                temp_df['dd_budget'] = df[budget_col][matches]
            
            # Find the 'total budget' column (flexible naming)
            total_budget_col = next((col for col in df.columns if 'total' in str(col).lower() and 'budget' in str(col).lower() and 'digital' not in str(col).lower()), None)
            if total_budget_col:
                temp_df['total_budget'] = df[total_budget_col][matches]
                       
            # Add digital priority columns if they exist
            for target in targetOrder:
                if target in df.columns:
                    temp_df[target] = df[target][matches]
            
            return temp_df
    else:
        str_columns = df.select_dtypes(include=['object']).columns
        matches = pd.Series(False, index=df.index)
        for col in str_columns:
            matches |= df[col].astype(str).str.lower().apply(lambda x: any(keyword in x for keyword in keywords))
        
        if matches.any():
            temp_df = pd.DataFrame()
            for idx in df.index[matches]:
                for col in str_columns:
                    if any(keyword in str(df.loc[idx, col]).lower() for keyword in keywords):
                        temp_df.loc[idx, 'call_topic_measure'] = df.loc[idx, col]
                        break
            
            temp_df['Fund'] = fund_name
            
            # Find the 'dd-relevant budget' column
            budget_col = next((col for col in df.columns if 'dd' in str(col).lower() and 'budget' in str(col).lower() and 'share' not in str(col).lower()), None)
            if budget_col:
                temp_df['dd_budget'] = df[budget_col][matches]
            
            # Find the 'total budget' column (flexible naming)
            total_budget_col = next((col for col in df.columns if 'total' in str(col).lower() and 'budget' in str(col).lower() and 'digital' not in str(col).lower()), None)
            if total_budget_col:
                temp_df['total_budget'] = df[total_budget_col][matches]
            
            # Add digital priority columns if they exist
            for target in targetOrder:
                if target in df.columns:
                    temp_df[target] = df[target][matches]
            
            return temp_df
    
    return pd.DataFrame()

def perform_keyword_search(keywords):
    all_results = pd.DataFrame()

    for fund_name, df in df_dict.items():
        results = search_keywords_in_df(df, fund_name, keywords)
        if not results.empty:
            all_results = pd.concat([all_results, results])

    all_results = share2budg(all_results)
    # Change name of funds
    all_results['Fund'] = all_results['Fund'].replace({
        'DEP25': 'DEP',
        'DEP_main': 'DEP',
        'HE25': 'HE',
        'HE23': 'HE',
        'HE21': 'HE'
    })

    all_results['Fund'] = np.where(all_results['call_topic_measure'] == "Joint Undertaking - European Partnership for High Performance Computing (EuroHPC)",
                                   "EuroHPC", all_results['Fund'])

    df = all_results.copy()

    # Include 'total_budget' in the columns to convert and summarize
    cols_to_convert = ['dd_budget', 'total_budget'] + targetOrder
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    # Summarize by Fund and include 'total_budget'
    summary_df = df.groupby('Fund')[cols_to_convert].sum()
    summary_df.loc['Total'] = summary_df.sum()

    return summary_df, all_results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        keywords = request.form.get('keywords').split(',')
        keywords = [kw.strip() for kw in keywords]
        summary_df, all_results = perform_keyword_search(keywords)  # Get both dataframes

        output_folder = os.path.join(mainfold, "code_jtj", "Keyword_search_results")
        os.makedirs(output_folder, exist_ok=True)

        # Save summary_df to Excel
        summary_file = os.path.join(output_folder, "Keyword_Summary.xlsx")
        summary_df.to_excel(summary_file, index=True)

        # Save all_results to Excel
        all_results_file = os.path.join(output_folder, "All_Results.xlsx")
        all_results.to_excel(all_results_file, index=False)

        # Clean up the DataFrame to ensure no leading/trailing whitespace
        summary_df.columns = summary_df.columns.str.strip()
        summary_df = summary_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Format the DataFrame to display two decimal places
        formatted_table = summary_df.to_html(classes='data', float_format="{:.2f}".format, index=True)

        # Pass the keywords to the template
        return render_template('index.html', tables=[formatted_table], titles=summary_df.columns.values, keywords=keywords)
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    output_folder = os.path.join(mainfold, "code_jtj", "Keyword_search_results")
    return send_from_directory(output_folder, filename)

if __name__ == '__main__':
    app.run()
