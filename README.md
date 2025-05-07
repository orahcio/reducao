# Redução de dados fotométricos

O servidor pode ser iniciado por exemplo

```shell
gunicorn -b 0.0.0.0:5000 app:app --timeout 0 -w 4
```

de acordo com [essa refeência](https://dev.to/chand1012/how-to-host-a-flask-server-with-gunicorn-and-https-942). A flag `--timeout` adicionei por conta que para redução em que mais imagens são carregadas é necessário um tempo maior para carregar o aplicativo com todas as imagens.

Acrescentei as flags `--timeout 0` para desligar o tempo de 30s de requisição (algumas figuras são grandes para serem enviadas para o usuário irá demorar), e `-w 4` ou `2` para deixar um número de _workers_ mais padronizado conforme documentção do _gunicorn_. Não vi muito para que serve o número de _workers_.

## Instalação

Tentativa para fazer funcionar no _python 3.11.6_ e nas bibliotecas atualizadas para essa versão.

Último teste bem sucedido foi com o _python 3.6.15_ e as bibliotecas podem ser instaladas com
```bash
pip install bokeh colorcet flask numpy werkzeug astropy astroquery photutils statsmodels pandas openpyxl gunicorn xlrd
```
ou
```bash
pip install -r requirements.txt
```

Preciso mexer para que o código seja adequado a versões mais novas, que tenha suporte ainda pela comunidade.

## Como usar

![Tela principal do aplicativo, três colunas: a primeira tem controle de contraste da imagem, regulador do tamanhos do raio para abertura fotométrica e seletor de tipo de objeto; a segunda coluna tem a imagem e três botões acima dela para limpar as seleções de fontes na imagem, copiar as coordenadas da imagem de referência e salvar tabela; a terceira coluna possui intruções para fazer solução de placa com o nova.astrometry.net e um seletor para escolher a imagem de referência.](screenshot.jpeg)

A tela inicial é um formulário muito simples para selecionar as imagens nos diferentes filtros _B_, _V_ e _R_. É possível selecionar mais de uma imagem `.fits` ou `.fit` para cada filtro. Ao submeter o formulário a tela acima será o ambiente para escolher as fontes de luz.

Uma pequena instrução já se encontra na tela principal para fazer a correção de placa usando a API dos [Astrometry.net](nova.astrometry.net). Caso queira proceder usando uma imagem já corrigida, basta substituir a imagem de referência por uma já corrigida pelo [Astrometry.net](nova.astrometry.net).

### Objetos estudados

Recomendo primeiramente selecionar os objetos a serem estudados nas imagens mantendo a opção _obj_selecionada na primeira coluna e clicando na imagem com o botão esquerdo no centro do objeto (um algoritmo de busca de centróide roda ao proceder com o clique) e a tabela abaixo da figura irá mostrar as grandezas disponíveis.

### Estrelas de referência e céu

Caso sua imagem de referência não possui ainda a correções de placa, pode seguir as instruções da terceira coluna para obter a solução (lembrando que terá que se cadastrar no Astrometry). Independente da correção de placa deixe o seletor da imagem de referência sempre selecionado para o nome dessa imagem, isso vai poupar bastante tempo, mais a frente.

Seleciona na primeira coluna o tipo _src_ e clica nas fontes de luz que serão usadas como referência para redução fotométrica, faça isso sempre na imagem de referência. Após esse passo pode selecionar _sky_ na primeira coluna e escolher pontos fora das fontes de luz para calcular o céu.

Uma vez escolhidas as coordenadas de cada fonte de luz e céu é possível reproduzir essas mesmas coordenadas nas demais figuras clicando na aba desejada e depois no botão _Copiar coordenadas_ acima da imagem, isso agiliza bastante o trabalho.

**Atenção!** Uma vez escolhidas todas as coordenadas em todas a imagens revise a tabela para verificar se os fluxos (coluna _flux_) foram todos calculados, caso exista valores "_na_" basta escolher outro tamanho de abertura fotométrica na primeira coluna e depois voltar ao tamanho desejado, isso ativa o evento que recalcula os fluxos em todas as aberturas selecionadas nas imagens.

### Finalizando a redução

Após todos os passos acima será necessário clicar no botão _Salvar tabela_, isso irá fazer o download de uma arquivo `.xlsx` com os cálculos de fluxos e algumas outras informações.

A redução do objeto termina se fizermos a requisição manualmente, substituindo o palavra `plot` por `reducao` na barra de endereços do navegador, a saida será links para downloads das tabelas com os valores dos índices de cor do objeto e coeficientes de converção ajustados para obter tais índices, bem como a informação da data juliana da imagem (obtida diretamento do cabeçalho do `.fits` claro).

# Referências

Aqui terá as referências para o método de redução utilizado.
