import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dictionary for teams and their functionalities
team_rules_templates = {
    'Risk Team': {
        'Basic Financial Data Checks': {
            'RSI': (0, 100),
            'implied_volatility': (0, 1),
            'Sharpe Ratio': (-200, 200),
            'theta': (-1000, 1000),
        },
        'Stock Price Data Checks': {
            'Close_lag_1': (0, 10000),
            'Close_lag_2': (0, 10000),
            'Close_lag_3': (0, 10000),
            'MA_20': (0, 10000),
            'MA_50': (0, 10000),
        }
    },
    'Options Team': {
        'Option Data Checks': {
            'Stock_price': (0, 10000),  # Stock price range
            'Option_price': (0, 10000),  # Option price range
            'Strike_Price': (0, 10000),  # Strike price range
            'implied_volatility': (0, 1),  # Implied volatility percentage
            'Vega': (0, 100),  # Vega range
            'delta': (-1, 1),  # Delta range (-1 to 1 for puts and calls)
            'gamma': (0, 10),  # Gamma range
            'theta': (-100, 0),  # Theta, typically negative
            'rho': (-1, 1),  # Rho range
            'sharpe_ratio': (-200, 200),  # Sharpe ratio range
            'Rolling_Std': (0, 100),  # Rolling standard deviation
        }
    },
    'Crypto Team': {
        'Basic Bitcoin Data Checks': {
            'btc_market_price': (0, 1000000),  # Bitcoin market price range
            'btc_total_bitcoins': (0, 21000000),  # Total bitcoins, max of 21 million
            'btc_market_cap': (0, 2000000000000),  # Market cap in USD
            'btc_trade_volume': (0, 1000000000),  # Trading volume in USD
        },
        'Block and Transaction Data Checks': {
            'btc_blocks_size': (0, 1000000),  # Total block size in bytes
            'btc_avg_block_size': (0, 2000),  # Average block size in bytes
            'btc_n_transactions_per_block': (0, 5000),  # Transactions per block
            'btc_median_confirmation_time': (0, 6000),  # Confirmation time in seconds
            'btc_hash_rate': (0, 300000000),  # Hash rate in hashes per second
        },
        'Revenue and Cost Data Checks': {
            'btc_miners_revenue': (0, 1000000000),  # Miners revenue in USD
            'btc_transaction_fees': (0, 100000),  # Transaction fees in BTC
            'btc_cost_per_transaction_percent': (0, 100),  # Cost per transaction in percentage
            'btc_cost_per_transaction': (0, 5000),  # Cost per transaction in USD
        }
    },
    'Vanilla model Team': {
        'Stock Market Data Checks': {
            'Open_price': (0, 10000),  # Opening price of the stock
            'Close_price': (0, 10000),  # Closing price of the stock
            'High_price': (0, 10000),  # Highest price during the trading day
            'Low_price': (0, 10000),  # Lowest price during the trading day
            'Volume': (0, 1000000000),  # Volume of stocks traded
            'Market_cap': (0, 1000000000000),  # Market capitalization in USD
            'PE_ratio': (0, 1000),  # Price-to-Earnings ratio
            'EPS': (-100, 1000),  # Earnings per share
        }
    },
    '3D shapes Team': {
        '3D Geometric Data Checks': {
            'shape_volume': (0, 10000),  # Volume of 3D shapes
            'shape_surface_area': (0, 5000),  # Surface area of 3D shapes
            'shape_vertices': (3, 1000),  # Number of vertices (minimum 3)
            'shape_edges': (3, 1000),  # Number of edges
            'shape_faces': (1, 1000),  # Number of faces
        }
    }
}

# Function to apply rules for any team
def apply_team_rules(df, rules):
    original_row_count = len(df)
    filtered_df = df.copy()
    removed_rows = pd.DataFrame()  # To store rows that are removed
    removal_reasons = []  # To store reasons for removal
    removed_indices = []  # To track row indices that were removed

    violation_summary = {}  # Dictionary to count violations per rule

    for column, (low, high) in rules.items():
        if column in filtered_df.columns:
            # Find rows that violate the rule
            invalid_rows = filtered_df[(filtered_df[column] < low) | (filtered_df[column] > high)]
            if not invalid_rows.empty:
                # Log the reason for removal and store removed rows with their indices
                removed_rows = pd.concat([removed_rows, invalid_rows])
                removal_reasons.extend([f"{column} out of range ({low}, {high})"] * len(invalid_rows))
                removed_indices.extend(invalid_rows.index.tolist())

                # Count the number of violations for each rule
                violation_summary[column] = violation_summary.get(column, 0) + len(invalid_rows)
            
            # Keep only valid rows
            filtered_df = filtered_df[(filtered_df[column] >= low) & (filtered_df[column] <= high)]
    
    # Reset index for consistency
    removed_rows.reset_index(drop=True, inplace=True)
    filtered_row_count = len(filtered_df)
    rows_affected = original_row_count - filtered_row_count
    
    return filtered_df, rows_affected, removed_rows, removal_reasons, removed_indices, violation_summary

# Streamlit app
st.title("Enhanced Data Integrity Verification System")

# Team selection
team_selection = st.selectbox("Select a Team", list(team_rules_templates.keys()))

# If a team is selected with rules
if team_selection in team_rules_templates and team_rules_templates[team_selection]:
    st.write(f"### {team_selection} Data Integrity Verification")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.write(df)

        # Sidebar for selecting rule template or defining custom rules
        st.sidebar.title(f"Define Validation Rules for {team_selection}")
        
        # Select pre-defined rule template
        selected_template = st.sidebar.selectbox("Choose a Rule Template", ["None"] + list(team_rules_templates[team_selection].keys()))

        custom_rules = {}
        
        # Display template rules if selected
        if selected_template != "None":
            st.sidebar.write(f"Selected Template: {selected_template}")
            template_rules = team_rules_templates[team_selection][selected_template]
            for col, (low, high) in template_rules.items():
                st.sidebar.write(f"{col}: {low} to {high}")
            custom_rules.update(template_rules)

        st.sidebar.write("### OR Define Custom Rules")
        
        # Allow user to define custom rules using a dropdown for columns
        if 'df' in locals():
            custom_column = st.sidebar.selectbox("Select Column Name", df.columns.tolist())
            low = st.sidebar.number_input(f"Lower Bound for {custom_column}", value=0.0)
            high = st.sidebar.number_input(f"Upper Bound for {custom_column}", value=100.0)
            if st.sidebar.button("Add Custom Rule"):
                custom_rules[custom_column] = (low, high)
                st.sidebar.write(f"Custom Rule Added: {custom_column} between {low} and {high}")
        
        # Apply rules button
        if st.button(f"Apply Rules for {team_selection}"):
            if custom_rules:
                cleaned_data, rows_affected, removed_data, removal_reasons, removed_indices, violations = apply_team_rules(df, custom_rules)
                st.write(f"{rows_affected} rows affected")
                st.write("### Cleaned Data:")
                st.write(cleaned_data)

                if len(removed_data) > 0:
                    st.write("### Removed Data:")
                    removed_data['Removal Reason'] = removal_reasons
                    st.write(removed_data)

                st.write("### Violation Summary:")
                for rule, count in violations.items():
                    st.write(f"{rule}: {count} violations")
                
                # Show a simple bar chart of original vs cleaned data
                st.write("### Before and After Data Comparison")
                fig, ax = plt.subplots()
                data_counts = [len(df), len(cleaned_data)]
                ax.bar(["Original Data", "Cleaned Data"], data_counts, color=["blue", "green"])
                ax.set_title("Data Row Count Before and After Cleaning")
                ax.set_ylabel("Number of Rows")
                st.pyplot(fig)

            else:
                st.write("No rules defined!")
        
        # Download cleaned data button
        if 'cleaned_data' in locals() and not cleaned_data.empty:
            st.download_button(
                label="Download Cleaned Data as CSV",
                data=cleaned_data.to_csv(index=False).encode('utf-8'),
                file_name="cleaned_data.csv",
                mime="text/csv",
            )
else:
    st.write(f"No specific rules available for {team_selection} yet.")
