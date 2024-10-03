import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Função para carregar os dados
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # 50% dos dados para evitar problemas de memória
    df = df.sample(frac=0.1, random_state=42)
    return df

# Função para tratar as colunas categóricas e garantir que os dados sejam numéricos
def preprocess_data(df, features):
    df = pd.get_dummies(df, columns=features, drop_first=True)
    return df

# Função para treinar o modelo e fazer a predição
def train_and_predict(model_type, X_train, X_test, y_train, y_test):
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=10)  # Reduzir o número de estimadores para evitar consumo excessivo
    elif model_type == 'Árvore de Decisão':
        model = DecisionTreeRegressor()
    
    # Treinando o modelo
    model.fit(X_train, y_train)

    # Fazendo a predição
    y_pred = model.predict(X_test)

    # Calculando o erro
    error = mean_squared_error(y_test, y_pred)
    
    return y_pred, error, model

# Interface com Streamlit
st.title("Treinamento e Predição de Gasto do Cliente")

# Upload do arquivo
uploaded_file = st.file_uploader("Envie o arquivo CSV com os dados", type=["csv"])

if uploaded_file is not None:
    # Carregar dados
    data = load_data(uploaded_file)
    
    # Exibir a tabela completa dos dados originais logo após o upload
    st.write("Dados carregados com sucesso! Aqui estão os dados originais:")
    st.dataframe(data)  # Exibir a tabela original, sem relação com as seleções futuras

    # Pergunta se há colunas de data
    has_date_column = st.checkbox("O conjunto de dados contém uma coluna de data?")

    date_column = None
    if has_date_column:
        # Se sim, permitir que o usuário selecione a coluna de data
        date_column = st.selectbox("Selecione a coluna de data", options=data.columns.tolist())
        data[date_column] = pd.to_datetime(data[date_column])  # Converter para formato datetime

    # Selecionar a coluna que identifica o cliente
    cliente_coluna = st.selectbox("Selecione a coluna que identifica o cliente", options=data.columns.tolist())

    # Selecionar as features e o target
    st.write("Selecione as colunas de entrada (exceto a coluna de data e a coluna do cliente) e a coluna alvo (gasto do cliente)")
    features = st.multiselect("Escolha as colunas de entrada", options=[col for col in data.columns if col != cliente_coluna and col != date_column], help="A coluna de data foi removida automaticamente se você a selecionou antes.")
    target = st.selectbox("Escolha a coluna alvo", options=[col for col in data.columns if col != cliente_coluna and col != date_column])

    # Botão para confirmar a seleção das features e target antes de escolher o modelo
    if st.button("Confirmar seleção de features e target"):
        st.session_state['features'] = features
        st.session_state['target'] = target
        st.session_state['data'] = data  # Armazenar o DataFrame processado

        X = data[features]
        y = data[target]

        # Preprocessar os dados para garantir que são numéricos
        X = preprocess_data(X, features)

        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Armazenar os dados no session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

        # Armazenar a coluna de data para o gráfico
        if date_column:
            st.session_state['date_column'] = data.loc[X_test.index, date_column]

    # Verificar se as features e target já foram selecionadas
    if 'features' in st.session_state and 'target' in st.session_state:
        # Selecionar o modelo
        model_type = st.selectbox("Selecione o modelo de aprendizado de máquina", 
                                    ["Random Forest", "Árvore de Decisão"])

        # Botão para treinar e fazer predições
        if st.button("Treinar e Fazer Predições"):
            
            # Loading
            with st.spinner("Treinando e fazendo predições..."):
                y_pred, error, model = train_and_predict(model_type, st.session_state['X_train'], st.session_state['X_test'], st.session_state['y_train'], st.session_state['y_test'])

                # Armazenar os resultados no session_state para evitar perda ao mudar a interface
                st.session_state['y_pred'] = y_pred
                st.session_state['error'] = error

    # Verificar se já há previsões no session_state
    if 'y_pred' in st.session_state:
        y_pred = st.session_state['y_pred']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        error = st.session_state['error']
        data = st.session_state['data']  # Carregar o DataFrame armazenado
        date_column_data = st.session_state.get('date_column', None)

        # Adicionando coluna de previsões ao DataFrame de teste
        X_test['Previsão'] = y_pred
        X_test['Real'] = y_test.values
        X_test[cliente_coluna] = data.loc[X_test.index, cliente_coluna]

        if date_column_data is not None:
            X_test['Data'] = date_column_data  # Adicionar coluna de data

        st.write(f"Erro Médio Quadrático (MSE): {error}")
        st.write("Resultados com previsões")
        st.write(X_test)

        # Selecionar o cliente para visualizar a predição
        cliente_selecionado = st.selectbox("Selecione um cliente para visualizar as previsões", options=X_test[cliente_coluna].unique())

        if cliente_selecionado:
            cliente_data = X_test[X_test[cliente_coluna] == cliente_selecionado]

            # Verificar se a coluna de data existe
            if date_column_data is not None:
                # Ordenar os resultados pela coluna de data e plotar o gráfico por data
                cliente_data = cliente_data.sort_values(by='Data')
                st.write(f"Previsões vs Real para o cliente: {cliente_selecionado}")
                
                fig, ax = plt.subplots()
                ax.plot(cliente_data['Data'], cliente_data['Real'], label='Real', marker='o')
                ax.plot(cliente_data['Data'], cliente_data['Previsão'], label='Previsão', marker='x')
                ax.set_title(f'Previsão vs Real para o Cliente {cliente_selecionado}')
                ax.set_xlabel('Data')
                ax.set_ylabel('Valor do Gasto')
                ax.legend()

            else:
                # Plotar o gráfico por índice caso não tenha a coluna de data
                st.write(f"Previsões vs Real para o cliente: {cliente_selecionado}")
                
                fig, ax = plt.subplots()
                ax.plot(cliente_data.index, cliente_data['Real'], label='Real', marker='o')
                ax.plot(cliente_data.index, cliente_data['Previsão'], label='Previsão', marker='x')
                ax.set_title(f'Previsão vs Real para o Cliente {cliente_selecionado}')
                ax.set_xlabel('Índice')
                ax.set_ylabel('Valor do Gasto')
                ax.legend()

            st.pyplot(fig)
