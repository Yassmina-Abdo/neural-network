import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('IrisData.txt')

def draw_x1_x2():
    sns.set_style("whitegrid")
    sns.FacetGrid(data, hue="Class",
                  height=5).map(plt.scatter,
                                'X1',
                                'X2').add_legend()
    plt.show()

def draw_x1_x3():
    sns.set_style("whitegrid")
    sns.FacetGrid(data, hue="Class",
                  height=5).map(plt.scatter,
                                'X1',
                                'X3').add_legend()
    plt.show()

def draw_x1_x4():
    sns.set_style("whitegrid")
    sns.FacetGrid(data, hue="Class",
                  height=5).map(plt.scatter,
                                'X1',
                                'X4').add_legend()
    plt.show()

def draw_x2_x3():
    sns.set_style("whitegrid")
    sns.FacetGrid(data, hue="Class",
                  height=5).map(plt.scatter,
                                'X2',
                                'X3').add_legend()
    plt.show()

def draw_x2_x4():
    sns.set_style("whitegrid")
    sns.FacetGrid(data, hue="Class",
                  height=5).map(plt.scatter,
                                'X2',
                                'X4').add_legend()
    plt.show()

def draw_x3_x4():
    sns.set_style("whitegrid")
    sns.FacetGrid(data, hue="Class",
                  height=5).map(plt.scatter,
                                'X3',
                                'X4').add_legend()
    plt.show()


