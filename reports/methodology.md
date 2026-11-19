# Metodologia e limitações

## Estrutura da base

A base contém 30 observações e duas variáveis:

- `YearsExperience`
- `Salary`

Não há valores ausentes nem duplicatas completas.

## Estratégia analítica

O projeto combina:

1. análise exploratória da relação entre experiência e salário;
2. correlação de Pearson;
3. regressão linear simples;
4. divisão treino/teste em 80/20;
5. avaliação por R², MAE e RMSE;
6. validação cruzada com 5 folds;
7. análise visual dos resíduos.

## Interpretação

O coeficiente da regressão descreve uma associação linear estimada na amostra.
Ele não deve ser interpretado como efeito causal da experiência sobre o salário.

## Limitações

- a amostra possui apenas 30 registros;
- há somente uma variável explicativa;
- não há informação sobre país, moeda, cargo, setor, formação, localização, empresa ou senioridade;
- a faixa observada de experiência vai de 1,1 a 10,5 anos;
- previsões fora dessa faixa seriam extrapolações;
- o projeto é didático e não deve ser usado para definir salários reais.
