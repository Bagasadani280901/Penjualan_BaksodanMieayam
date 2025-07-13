import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value

st.set_page_config(page_title="Aplikasi Optimasi Produksi", layout="centered")

st.title("📈 Aplikasi Optimasi Produksi")
st.markdown("Model Linear Programming untuk memaksimalkan keuntungan produksi.")

# Input data produk
num_products = st.number_input("Jumlah Produk", min_value=2, step=1)

product_names = []
profits = []
for i in range(num_products):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input(f"Nama Produk {i+1}", value=f"P{i+1}")
    with col2:
        profit = st.number_input(f"Keuntungan per unit {name}", value=0.0)
    product_names.append(name)
    profits.append(profit)

# Input jumlah batasan sumber daya
num_constraints = st.number_input("Jumlah Batasan Sumber Daya", min_value=1, step=1)

constraints = []
limits = []

st.subheader("🔧 Batasan Produksi")
for i in range(num_constraints):
    st.markdown(f"**Batasan {i+1}**")
    row = []
    for j in range(num_products):
        coef = st.number_input(f"Koefisien {product_names[j]} untuk Batasan {i+1}", key=f"c_{i}_{j}", value=0.0)
        row.append(coef)
    limit = st.number_input(f"Nilai Maksimum untuk Batasan {i+1}", key=f"l_{i}", value=0.0)
    constraints.append(row)
    limits.append(limit)

if st.button("🚀 Optimalkan Produksi"):
    # Buat model LP
    model = LpProblem(name="optimasi-produksi", sense=LpMaximize)
    x = [LpVariable(name=f"x{i}", lowBound=0) for i in range(num_products)]

    # Fungsi objektif
    model += lpSum([profits[i] * x[i] for i in range(num_products)]), "Total_Keuntungan"

    # Tambah kendala
    for i in range(num_constraints):
        model += lpSum([constraints[i][j] * x[j] for j in range(num_products)]) <= limits[i], f"Kendala_{i+1}"

    # Selesaikan model
    status = model.solve()

    # Output hasil
    st.subheader("📊 Hasil Optimasi")
    for i in range(num_products):
        st.write(f"{product_names[i]}: {x[i].value():.2f} unit")
    st.success(f"Total Keuntungan Maksimum: Rp {value(model.objective):,.2f}")

    # Visualisasi hanya untuk dua variabel
    if num_products == 2:
        x_vals = np.linspace(0, max(limits)*2, 400)
        fig, ax = plt.subplots()

        for i in range(num_constraints):
            a, b = constraints[i]
            y_vals = (limits[i] - a * x_vals) / b if b != 0 else np.full_like(x_vals, np.nan)
            ax.plot(x_vals, y_vals, label=f"Kendala {i+1}")

        # Area feasible
        for x1 in x_vals:
            feasible_y = []
            for x2 in np.linspace(0, max(limits)*2, 400):
                valid = all(
                    constraints[i][0] * x1 + constraints[i][1] * x2 <= limits[i]
                    for i in range(num_constraints)
                )
                if valid:
                    feasible_y.append(x2)
            if feasible_y:
                ax.fill_between([x1], min(feasible_y), max(feasible_y), color="lightblue", alpha=0.3)

        ax.plot(x[0].value(), x[1].value(), 'ro', label="Solusi Optimal")
        ax.set_xlabel(product_names[0])
        ax.set_ylabel(product_names[1])
        ax.set_title("Area Feasible dan Solusi Optimal")
        ax.legend()
        st.pyplot(fig)
