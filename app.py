import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.title("TV Delivery Optimizer")

st.markdown("""
**A company wishes to minimise its costs of delivering televisions from three depots (D1, D2 and D3) to three stores (Store 1, Store 2, and Store 3). The cost of delivering one TV is £5 per mile. [Source](https://www.accaglobal.com/uk/en/student/exam-support-resources/professional-exams-study-resources/strategic-business-leader/technical-articles/big-data-sbl.html)**
""")

st.markdown("## Problem Setup")

# Depot and Store Data
depot_labels = ["D1", "D2", "D3"]
depot_supply = [2500, 3100, 1250]
store_labels = ["Store 1", "Store 2", "Store 3"]
store_caps = [2000, 3000, 2000]

# Depot table (centered with heading)
st.markdown("### 📦 TVs Available at Each Depot")
depot_df = pd.DataFrame({"Depot": depot_labels, "TVs Available": depot_supply})
st.markdown(depot_df.style.set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center')]},
    {'selector': 'td', 'props': [('text-align', 'center')]}
]).to_html(), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Store table (centered with heading)
st.markdown("### 🏬 Store Capacity Constraints")
store_df = pd.DataFrame({"Store": store_labels, "Capacity": store_caps})
st.markdown(store_df.style.set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center')]},
    {'selector': 'td', 'props': [('text-align', 'center')]}
]).to_html(), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Distance Matrix
distances = np.array([
    [22, 33, 40],  # D1 to Stores 1–3
    [27, 30, 22],  # D2 to Stores 1–3
    [36, 20, 25],  # D3 to Stores 1–3
])

st.markdown("### 🗺️ Distance Matrix (miles)")
distance_df = pd.DataFrame(distances, index=depot_labels, columns=store_labels)
st.markdown(distance_df.style.set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center')]},
    {'selector': 'td', 'props': [('text-align', 'center')]}
]).to_html(), unsafe_allow_html=True)

# Cost per mile
cost_per_mile = 5
c = (distances * cost_per_mile).flatten()

# Constraints for stores (upper bounds)
A_store = np.zeros((3, 9))
for j in range(3):
    for i in range(3):
        A_store[j, 3*i + j] = 1
b_store = store_caps

# Constraints for depots (equality)
A_depot = np.zeros((3, 9))
for i in range(3):
    A_depot[i, 3*i : 3*i+3] = 1
b_depot = depot_supply

# Bounds (x >= 0)
bounds = [(0, None) for _ in range(9)]

# Solve using linprog
res = linprog(
    c=c,
    A_ub=A_store,
    b_ub=b_store,
    A_eq=A_depot,
    b_eq=b_depot,
    bounds=bounds,
    method="highs"
)

st.markdown("## Optimization Results")

if res.success:
    x = np.round(res.x).astype(int).reshape(3, 3)
    shipment_df = pd.DataFrame(x, index=depot_labels, columns=store_labels)

    st.markdown("### ✅ Optimized TV Shipment Plan")
    st.markdown(shipment_df.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ]).to_html(), unsafe_allow_html=True)

    total_cost = res.fun
    st.write(f"### 💰 Total Delivery Cost: £{total_cost:,.2f}")

else:
    st.error("❌ Optimization failed: " + res.message)
