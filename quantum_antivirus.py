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
    'Options Team': None,  # Placeholder for Team 1 rules
    'Vanilla model Team': None,  # Placeholder for Team 2 rules
    'Crybto Team': None,  # Placeholder for Team 3 rules
    '3D shapes Team': None   # Placeholder for Team 4 rules
}

# Function to apply rules for the Risk Team
def apply_risk_team_rules(df, rules):
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

# If the Risk team is selected
if team_selection == 'Risk Team':
    st.write("### Risk Team Financial Data Integrity Verification")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.write(df)

        # Sidebar for selecting rule template or defining custom rules
        st.sidebar.title("Define Validation Rules for Risk Team")
        
        # Select pre-defined rule template
        selected_template = st.sidebar.selectbox("Choose a Rule Template", ["None"] + list(team_rules_templates['Risk Team'].keys()))

        custom_rules = {}
        
        # Display template rules if selected
        if selected_template != "None":
            st.sidebar.write(f"Selected Template: {selected_template}")
            template_rules = team_rules_templates['Risk Team'][selected_template]
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
        if st.button("Apply Risk Team Validation Rules"):
            if custom_rules:
                st.write("### Applying the following rules:")
                st.write(custom_rules)
                cleaned_data, rows_affected, removed_rows, removal_reasons, removed_indices, violation_summary = apply_risk_team_rules(df, custom_rules)
                st.write("### Cleaned Data")
                st.write(cleaned_data)

                # Show message if any rows were affected
                if rows_affected > 0:
                    st.write(f"⚠️ {rows_affected} row(s) were removed due to the applied rules.")
                    
                    # Show rows that were removed, the reason, and the specific row index
                    reasons_df = pd.DataFrame({
                        "Removed Row Index": removed_indices,
                        "Reason for Removal": removal_reasons
                    })
                    st.write("### Rows Affected and Reasons:")
                    st.write(reasons_df)

                    # Violation summary - shows how many times each rule was violated
                    st.write("### Rule Violation Summary:")
                    violation_summary_df = pd.DataFrame({
                        "Rule": violation_summary.keys(),
                        "Violations": violation_summary.values()
                    })
                    st.write(violation_summary_df)

                    # --- Visualization: Violation Distribution ---
                    st.write("### Violation Distribution")
                    fig, ax = plt.subplots()
                    sns.barplot(x=list(violation_summary.keys()), y=list(violation_summary.values()), ax=ax)
                    ax.set_title("Number of Violations per Rule")
                    ax.set_xlabel("Rules")
                    ax.set_ylabel("Violations")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    # --- Visualization: Removed Rows Heatmap ---
                    st.write("### Heatmap of Removed Rows")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    heatmap_data = removed_rows.apply(lambda x: x.notnull().astype(int))
                    sns.heatmap(heatmap_data, cmap="coolwarm", cbar=False, ax=ax)
                    ax.set_title("Rows Affected by Rule Violations")
                    st.pyplot(fig)

                else:
                    st.write("✅ No rows were affected by the applied rules.")
            else:
                st.write("No rules defined yet.")

        # --- Visualization: Before and After Comparison ---
        if 'cleaned_data' in locals():
            st.write("### Data Integrity Before and After")
            fig, ax = plt.subplots()
            before_after_data = pd.DataFrame({
                'State': ['Before Cleaning', 'After Cleaning'],
                'Row Count': [len(df), len(cleaned_data)]
            })
            sns.barplot(x='State', y='Row Count', data=before_after_data, ax=ax)
            ax.set_title("Rows Before and After Data Cleaning")
            st.pyplot(fig)

            # Download cleaned data
            csv = cleaned_data.to_csv(index=False)
            st.download_button("Download Cleaned Data", csv, "cleaned_data.csv", "text/csv")

# If other teams are selected, placeholders for future rule creation
else:
    st.write(f"### {team_selection} Data Integrity Verification")
    st.write("Rules for this team are not defined yet. Please come back later as we define custom rules for this team.")
