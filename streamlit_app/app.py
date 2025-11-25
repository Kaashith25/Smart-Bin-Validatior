import streamlit as st
import sys
import os
import pandas as pd
import json
from PIL import Image

# -------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------
current_app_folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_app_folder) 
src_folder = os.path.join(project_root, "src")

if src_folder not in sys.path:
    sys.path.append(src_folder)

try:
    from inference import analyze_image
except ImportError as e:
    st.error(f"CRITICAL ERROR: {e}")
    st.stop()

# -------------------------------------------------------
# LOAD DATABASE
# -------------------------------------------------------
@st.cache_data
def load_descriptions():
    json_path = os.path.join(project_root, "processed", "asin_text.json")
    small_json_path = os.path.join(project_root, "processed", "asin_text_small.json")
    target_file = json_path if os.path.exists(json_path) else small_json_path
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return sorted(list(set(data.values())))
    except:
        return []

all_descriptions = load_descriptions()

# -------------------------------------------------------
# APP CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="SmartBin Validator", layout="wide")

st.markdown("""
<style>
    .pass-box { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;}
    .fail-box { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.title("SmartBin: Order Validation")
st.markdown("**System:** `Zero-Shot Verification` | **Status:** `Active`")
st.markdown("---")

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.header("Create Order")

if 'order_list' not in st.session_state:
    st.session_state.order_list = {}

input_method = st.sidebar.radio("Input Method:", ["Search Database", "Manual Entry"])

with st.sidebar.form("add_item_form"):
    if input_method == "Search Database":
        if all_descriptions:
            selected_desc = st.selectbox("Search Item", all_descriptions)
            final_desc = selected_desc
        else:
            final_desc = None
    else:
        manual_text = st.text_input("Enter Description", placeholder="e.g. Red Nike Shoes")
        final_desc = manual_text

    qty = st.number_input("Quantity", min_value=1, value=1)
    
    if st.form_submit_button("Add Item to Order") and final_desc:
        st.session_state.order_list[final_desc] = qty
        st.success(f"Added: {final_desc[:20]}...")

st.sidebar.markdown("---")
st.sidebar.subheader("Current Manifest")

if st.session_state.order_list:
    df_order = pd.DataFrame(list(st.session_state.order_list.items()), columns=["Item", "Qty"])
    st.sidebar.dataframe(df_order, hide_index=True)
    if st.sidebar.button("Clear Manifest"):
        st.session_state.order_list = {}
        st.rerun()
else:
    st.sidebar.info("Order list is empty.")

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Bin Feed")
    uploaded_file = st.file_uploader("Upload Bin Image", type=['jpg', 'png', 'jpeg'])
    
    img_path = None
    if uploaded_file:
        img_path = os.path.join(project_root, "temp_app_upload.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(img_path, caption="Input Image", use_column_width=True)

with col2:
    st.subheader("2. Validation Analysis")
    
    if st.button("Validate Order", type="primary"):
        if not uploaded_file or not st.session_state.order_list:
            st.error("Please provide both an image and an order.")
        else:
            with st.spinner("Verifying Inventory..."):
                target_items = list(st.session_state.order_list.keys())
                detected_counts = analyze_image(img_path, target_descriptions=target_items)
                
                results_data = []
                all_verified = True
                
                for item, expected_qty in st.session_state.order_list.items():
                    raw_found = detected_counts.get(item, 0)
                    
                    # STRICT LOGIC:
                    # If found >= expected -> Match
                    # Else -> Mismatch (No partials)
                    if raw_found >= expected_qty:
                        status = "✅ MATCH"
                    else:
                        status = "❌ MISMATCH"
                        all_verified = False
                        
                    results_data.append({
                        "Item": item,
                        "Quantity": expected_qty, 
                        "Status": status
                    })
                
                st.write("")
                # Display clean table (No "Found" column)
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)