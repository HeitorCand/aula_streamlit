import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título do dashboard
st.title("Dashboard com Streamlit")

# Instruções iniciais
st.write("Você pode carregar um arquivo CSV para visualizar os dados e gerar gráficos interativos.")

# Carregar o arquivo CSV
uploaded_file = st.file_uploader("Carregue um arquivo CSV", type="csv")

if uploaded_file is not None:
    # Ler o arquivo CSV
    df = pd.read_csv(uploaded_file)
    
    # Mostrar os dados em uma tabela
    st.write("Dados carregados:")
    st.dataframe(df)

    # Exibir opções de colunas numéricas para o gráfico
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_x = st.selectbox("Selecione a coluna para o eixo X:", numeric_columns)
    selected_y = st.selectbox("Selecione a coluna para o eixo Y:", numeric_columns)

    # Gerar o gráfico
    if st.button("Gerar Gráfico"):
        fig, ax = plt.subplots()
        ax.scatter(df[selected_x], df[selected_y])
        ax.set_xlabel(selected_x)
        ax.set_ylabel(selected_y)
        ax.set_title(f'Gráfico de {selected_x} vs {selected_y}')
        st.pyplot(fig)

else:
    st.write("Por favor, carregue um arquivo CSV para começar.")

# Rodapé com observações
st.write("Essa aplicação foi criada como um exemplo para introduzir conceitos básicos de Streamlit.")
