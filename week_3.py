import pandas as pd

def calc():

    # Load Data
    data = pd.read_csv('desafio1.csv', index_col = 'RowNumber')

    # Group by state
    group = data[['estado_residencia','pontuacao_credito']].groupby('estado_residencia')['pontuacao_credito']

    # Calculate statistics
    ans = pd.DataFrame({
        # Moda
        'moda': group.agg(lambda x: x.value_counts().index[0]),
        # Media
        'media': group.mean(),
        # Mediana
        'mediana': group.median(),
        # Desvio padrao
        'desvio_padrao': group.std()
    }, index = data['estado_residencia'].unique())

    # Store as JSON
    ans.transpose().to_json('submission.json')


if __name__ == '__main__':

    calc()