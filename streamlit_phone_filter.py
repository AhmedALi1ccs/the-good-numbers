import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import io

# Set page configuration
st.set_page_config(
    page_title="Phone Data Filter",
    page_icon="📞",
    layout="wide"
)

def filter_phone_data(input_file, output_file=None, reorder_phones=True):
    """
    Original filter_phone_data function with modifications for Streamlit
    """
    # Define tags to keep
    tags_to_keep = {"KindSkip A", "KindSkip E", "A", "E", "Direct-Skip A", 
                    "Direct-Skip E", "Realeflow A", "Realeflow E", "Direct-Skip A-PR",
                    "Direct-Skip E-PR", "Direct-Skip A", "Direct-Skip E", 
                    "Kind-Skip A", "Kind-Skip B", "Kind-Skip C", "E1", "E2", "E3"}
    
    # Read CSV file (modified for Streamlit)
    try:
        if isinstance(input_file, str):
            st.info(f"Reading CSV file: {input_file}")
            df = pd.read_csv(input_file, low_memory=False)
        else:
            # Handle uploaded file object
            df = pd.read_csv(input_file, low_memory=False)
        
        st.success(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    # Identify phone-related columns
    phone_cols, tag_cols, type_cols, status_cols = [], [], [], []
    for i in range(1, 31):
        if f'Phone {i}' in df.columns:
            phone_cols.append(f'Phone {i}')
            tag_cols.append(f'Phone Tags {i}')
            type_cols.append(f'Phone Type {i}')
            status_cols.append(f'Phone Status {i}')

    st.info(f"Found {len(phone_cols)} phone columns")
    
    # Precompute valid tags mask
    def check_tag(tag_value):
        if pd.isna(tag_value):
            return False
        s = str(tag_value).strip()
        if s in tags_to_keep:
            return True
        parts = re.split(r'[,;|/]+', s)
        return any(part.strip() in tags_to_keep for part in parts)
    
    # Progress bar for processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Vectorized tag validation
    status_text.text("Validating tags...")
    progress_bar.progress(25)
    
    if tag_cols:
        # Convert to string for safe processing
        tag_df = df[tag_cols].astype(str)
        # Apply vectorized tag checking
        valid_mask = tag_df.applymap(check_tag)
        row_has_valid_tag = valid_mask.any(axis=1)
    else:
        row_has_valid_tag = np.zeros(len(df), dtype=bool)

    # Process rows with valid tags
    status_text.text("Processing rows with valid tags...")
    progress_bar.progress(50)
    
    for cols in zip(phone_cols, tag_cols, type_cols, status_cols):
        pc, tc, tyc, sc = cols
        col_idx = tag_cols.index(tc)
        # Create mask for invalid phones in valid rows
        invalid_mask = row_has_valid_tag & ~valid_mask[tc]
        df.loc[invalid_mask, [pc, tc, tyc, sc]] = np.nan

    # Process rows without valid tags (keep first 3 phones)
    status_text.text("Processing rows without valid tags...")
    progress_bar.progress(75)
    
    if not row_has_valid_tag.all() and phone_cols:
        invalid_rows = df.index[~row_has_valid_tag]
        
        # Create mask for first 3 non-null phones in invalid rows
        phone_mask = df.loc[invalid_rows, phone_cols].notna()
        keep_mask = (phone_mask.cumsum(axis=1) <= 3) & phone_mask
        
        # Apply mask to all phone-related columns
        for col_group in [phone_cols, tag_cols, type_cols, status_cols]:
            # Get values as array for efficient masking
            vals = df.loc[invalid_rows, col_group].values
            mask_vals = keep_mask.values
            # Apply mask directly to numpy array
            vals[~mask_vals] = np.nan
            # Assign back to DataFrame
            df.loc[invalid_rows, col_group] = vals

    # Reorder phones to remove gaps (vectorized version)
    if reorder_phones and phone_cols:
        status_text.text("Reordering phone numbers to remove gaps...")
        progress_bar.progress(90)
        
        # Combine all phone-related data
        all_cols = phone_cols + tag_cols + type_cols + status_cols
        sorted_cols = sorted(all_cols, key=lambda x: int(re.search(r'\d+', x).group()))
        
        # Extract values as numpy array for efficient processing
        phone_data = df[phone_cols].values
        tag_data = df[tag_cols].values
        type_data = df[type_cols].values
        status_data = df[status_cols].values
        
        # Create masks for non-null values
        phone_mask = ~pd.isnull(phone_data)
        
        # Preallocate new arrays
        new_phone = np.full_like(phone_data, np.nan, dtype=object)
        new_tags = np.full_like(tag_data, np.nan, dtype=object)
        new_types = np.full_like(type_data, np.nan, dtype=object)
        new_status = np.full_like(status_data, np.nan, dtype=object)
        
        # Vectorized reordering
        for i in range(phone_data.shape[0]):
            # Get non-null indices for this row
            non_null = phone_mask[i]
            # Count non-null values
            count = np.sum(non_null)
            if count == 0:
                continue
                
            # Get the actual values in order
            new_phone[i, :count] = phone_data[i, non_null]
            new_tags[i, :count] = tag_data[i, non_null]
            new_types[i, :count] = type_data[i, non_null]
            new_status[i, :count] = status_data[i, non_null]
        
        # Assign back to DataFrame
        df[phone_cols] = new_phone
        df[tag_cols] = new_tags
        df[type_cols] = new_types
        df[status_cols] = new_status

    # Complete processing
    status_text.text("Processing complete!")
    progress_bar.progress(100)
    
    return df

def main():
    """
    Main Streamlit app function
    """
    st.title("📞 Phone Data Filter Application")
    st.markdown("Upload a CSV file to filter phone data based on specified tags and reorder phone columns.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    reorder_phones = st.sidebar.checkbox("Reorder phones to remove gaps", value=True)
    
    # Display the tags that will be kept
    st.sidebar.subheader("Tags to Keep:")
    tags_to_keep = ["KindSkip A", "KindSkip E", "A", "E", "Direct-Skip A", 
                    "Direct-Skip E", "Realeflow A", "Realeflow E", "Direct-Skip A-PR",
                    "Direct-Skip E-PR", "Kind-Skip A", "Kind-Skip B", "Kind-Skip C", 
                    "E1", "E2", "E3"]
    
    for tag in tags_to_keep:
        st.sidebar.text(f"• {tag}")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Show file details
        st.subheader("File Information")
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size} bytes"
        }
        st.json(file_details)
        
        # Process button
        if st.button("Process File", type="primary"):
            with st.spinner("Processing your file..."):
                # Process the file
                filtered_df = filter_phone_data(uploaded_file, reorder_phones=reorder_phones)
                
                if filtered_df is not None:
                    st.success("File processed successfully!")
                    
                    # Display results summary
                    st.subheader("Processing Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Rows", len(filtered_df))
                    
                    with col2:
                        st.metric("Total Columns", len(filtered_df.columns))
                    
                    with col3:
                        phone_cols = [col for col in filtered_df.columns if col.startswith('Phone ') and not any(x in col for x in ['Tags', 'Type', 'Status'])]
                        st.metric("Phone Columns", len(phone_cols))
                    
                    # Show preview of processed data
                    st.subheader("Data Preview")
                    st.dataframe(filtered_df.head(10), use_container_width=True)
                    
                    # Download button
                    csv_buffer = io.StringIO()
                    filtered_df.to_csv(csv_buffer, index=False)
                    csv_string = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="Download Filtered CSV",
                        data=csv_string,
                        file_name=f"filtered_{uploaded_file.name}",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    # Show statistics
                    st.subheader("Phone Data Statistics")
                    phone_stats = {}
                    
                    for i in range(1, 31):
                        phone_col = f'Phone {i}'
                        if phone_col in filtered_df.columns:
                            non_null_count = filtered_df[phone_col].notna().sum()
                            if non_null_count > 0:
                                phone_stats[phone_col] = non_null_count
                    
                    if phone_stats:
                        stats_df = pd.DataFrame(list(phone_stats.items()), 
                                              columns=['Phone Column', 'Non-null Count'])
                        st.dataframe(stats_df, use_container_width=True)
    
    # Instructions
    st.subheader("How to Use")
    st.markdown("""
    1. **Upload your CSV file** using the file uploader above
    2. **Configure settings** in the sidebar (optional)
    3. **Click 'Process File'** to filter the phone data
    4. **Review the results** and download the filtered CSV
    
    **What this app does:**
    - Filters phone data based on predefined tags
    - Keeps only phones with valid tags or first 3 phones for rows without valid tags
    - Optionally reorders phone columns to remove gaps
    - Provides detailed statistics about the processed data
    """)

if __name__ == "__main__":
    main()
