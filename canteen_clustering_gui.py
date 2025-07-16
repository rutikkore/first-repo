import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import messagebox, ttk

# Simulated canteen food data
data = {
    'Food_Item': [
        'Vada Pav', 'Samosa', 'Maggie', 'Pizza', 'Burger',
        'Biryani', 'Poha', 'Sandwich', 'Noodles', 'Paratha'
    ],
    'Popularity': [120, 100, 80, 60, 70, 90, 110, 95, 85, 105],
    'Price': [15, 20, 40, 120, 100, 90, 25, 45, 50, 30],
    'Avg_Consumption_Time': [5, 5, 8, 15, 12, 20, 6, 7, 10, 9]
}
df = pd.DataFrame(data)

# Clustering function
def run_clustering():
    global df
    features = ['Popularity', 'Price', 'Avg_Consumption_Time']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    df.to_csv("canteen_clustered_menu.csv", index=False)
    messagebox.showinfo("Clustering", "Clustering complete and saved as CSV!")

# Plotting function
def show_plot():
    if 'Cluster' not in df.columns:
        messagebox.showwarning("Run Clustering", "Please run clustering first.")
        return
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Price', y='Popularity',
        hue='Cluster',
        size='Avg_Consumption_Time',
        sizes=(40, 400),
        data=df,
        palette='viridis',
        legend='full'
    )
    for i in range(len(df)):
        plt.text(df['Price'][i]+1, df['Popularity'][i], df['Food_Item'][i], fontsize=9)
    plt.title("Clustering of Canteen Food Items")
    plt.xlabel("Price (Rs.)")
    plt.ylabel("Popularity (weekly sales)")
    plt.tight_layout()
    plt.savefig("canteen_clusters_plot.png")
    plt.show()

# View Data in a new window
def view_data():
    if 'Cluster' not in df.columns:
        messagebox.showwarning("Run Clustering", "Please run clustering first.")
        return
    view = tk.Toplevel(root)
    view.title("Clustered Data")

    tree = ttk.Treeview(view)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    for _, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))
    tree.pack(expand=True, fill='both')

# GUI Setup
root = tk.Tk()
root.title("Canteen Food Clustering")
root.geometry("400x300")

label = tk.Label(root, text="Canteen Food Clustering Tool", font=("Arial", 14))
label.pack(pady=10)

btn1 = tk.Button(root, text="Run Clustering", command=run_clustering, width=25)
btn1.pack(pady=5)

btn2 = tk.Button(root, text="Show Cluster Plot", command=show_plot, width=25)
btn2.pack(pady=5)

btn3 = tk.Button(root, text="View Clustered Data", command=view_data, width=25)
btn3.pack(pady=5)

btn4 = tk.Button(root, text="Exit", command=root.destroy, width=25, bg="red", fg="white")
btn4.pack(pady=20)

root.mainloop()
