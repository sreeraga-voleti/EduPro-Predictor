import matplotlib
matplotlib.use('Agg')  # ✅ Disable any GUI backend
import matplotlib.pyplot as plt
import pandas as pd
import os


def generate_graphs(csv_path):
    df = pd.read_csv(csv_path)  # Ensure file exists

    # Ensure static folder exists
    if not os.path.exists("static"):
        os.makedirs("static")

    # Graph 1: Placement Eligibility Count
    plt.figure()
    df['Eligible for placement'].value_counts().plot(kind='bar')
    plt.title("Placement Eligibility Distribution")
    plt.xlabel("Placement Eligibility")
    plt.ylabel("Number of Students")
    plt.savefig("static/graph1.png")
    plt.close()

    # Graph 2: Attendance vs Placement
    plt.figure()
    df.groupby('Eligible for placement')['Attendance'].mean().plot(kind='bar')
    plt.title("Average Attendance vs Placement Result")
    plt.xlabel("Placement Eligibility")
    plt.ylabel("Average Attendance")
    plt.savefig("static/graph2.png")
    plt.close()

    # Graph 3: Internal Marks vs Placement
    plt.figure()
    df.groupby('Eligible for placement')['Internal marks (out of 20)'].mean().plot(kind='bar')
    plt.title("Internal Marks vs Placement Result")
    plt.xlabel("Placement Eligibility")
    plt.ylabel("Average Internal Marks")
    plt.savefig("static/graph3.png")
    plt.close()

    # ✅ Graph 4: Age vs Placement Eligibility
    plt.figure()
    df.groupby('Eligible for placement')['Age'].mean().plot(kind='bar')
    plt.title("Average Age vs Placement Result")
    plt.xlabel("Placement Eligibility")
    plt.ylabel("Average Age")
    plt.savefig("static/graph4.png")
    plt.close()

        # ✅ Graph 5: How Backlogs Affect Placement Eligibility
    plt.figure()
    df.groupby('backlogs')['Eligible for placement'].mean().plot(kind='bar')
    plt.title("Effect of Backlogs on Placement Eligibility")
    plt.xlabel("Number of Backlogs")
    plt.ylabel("Placement Eligibility Rate")
    plt.ylim(0, 1)  # Scale from 0 to 100% placement chance
    plt.savefig("static/graph5.png")
    plt.close()


    print("✅ All graphs saved successfully!")
