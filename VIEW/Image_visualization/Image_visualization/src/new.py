import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置页面标题
st.title("选择模式")

# 创建一个下拉菜单供用户选择
option = st.selectbox("请选择模式", ["lab", "res", "data_visualization", "plot_customization"])

# 根据用户的选择执行不同的操作
if option == "lab":
    st.write("你选择了 lab 模式。这里可以添加 lab 模式下的具体功能代码。")
elif option == "res":
    st.write("你选择了 res 模式。这里可以添加 res 模式下的具体功能代码。")
elif option == "data_visualization":
    st.subheader("数据可视化")
    # 生成一些示例数据
    data = np.random.randn(100, 2)
    df = pd.DataFrame(data, columns=['x', 'y'])

    # 显示数据框
    st.dataframe(df)

    # 绘制散点图
    st.subheader("散点图")
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'])
    st.pyplot(fig)

    # 绘制直方图
    st.subheader("直方图")
    fig, ax = plt.subplots()
    ax.hist(df['x'], bins=20)
    st.pyplot(fig)

elif option == "plot_customization":
    st.subheader("绘图定制")
    # 生成一些示例数据
    data = np.random.randn(100)

    # 让用户选择绘图类型
    plot_type = st.selectbox("选择绘图类型", ["直方图", "箱线图", "密度图"])

    if plot_type == "直方图":
        fig, ax = plt.subplots()
        bins = st.slider("选择直方图的箱数", 5, 50, 20)
        ax.hist(data, bins=bins)
        st.pyplot(fig)
    elif plot_type == "箱线图":
        fig, ax = plt.subplots()
        ax.boxplot(data)
        st.pyplot(fig)
    elif plot_type == "密度图":
        fig, ax = plt.subplots()
        sns.kdeplot(data, ax=ax)
        st.pyplot(fig)