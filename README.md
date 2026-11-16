# Classificador de Imagens feito "na mão"

> Uma rede neural que olha uma foto pequena e diz se nela há **um gato, um cachorro, uma pessoa, um carro, uma moto, um ônibus ou um caminhão** — escrita do zero em TypeScript, **sem nenhuma biblioteca de inteligência artificial**.

---

## 1. A ideia em uma frase

O projeto começou como um exercício de **"gato ou cachorro?"**. Neste micro-capstone ele foi **ampliado para 7 categorias**, usando exatamente as mesmas técnicas, para responder: *até onde dá para ir com uma rede neural construída à mão, e quando vale a pena usar ferramentas prontas?*

| | Antes | Agora |
|---|---|---|
| Pergunta | "É gato ou cachorro?" | "Qual destas 7 coisas é?" |
| Resposta | 1 número (0 = gato, 1 = cachorro) | 7 porcentagens que somam 100% |
| Imagens de treino | 40 | 2.840 |
| Acerto em fotos novas | — | **55,6%** (acaso seria 14,3%) |
| Bibliotecas de IA | nenhuma | nenhuma |

As únicas dependências servem para **abrir arquivos de imagem** (`pngjs`, `jpeg-js`, `heic-decode`) e **rodar TypeScript** (`tsx`). Toda a matemática — convolução, ativação, retropropagação, otimizador — está escrita à mão na pasta `src/`.

---

## 2. Como executar

### Pré-requisitos
- **Node.js 20 ou superior** (testado no 24)
- ~400 MB livres em disco (para as imagens públicas do CIFAR)

### Passo a passo

```bash
npm install
```

**1) Baixar as imagens públicas (CIFAR-10 e CIFAR-100)** e extrair dentro da pasta `data-raw/`:

```bash
mkdir data-raw
```

```bash
curl -L -o data-raw/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
```

```bash
curl -L -o data-raw/cifar-100-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
```

```bash
tar -xzf data-raw/cifar-10-binary.tar.gz -C data-raw
```

```bash
tar -xzf data-raw/cifar-100-binary.tar.gz -C data-raw
```

**2) Separar as imagens nas 7 pastas** (400 treino / 100 validação / 100 teste por classe; gatos e cachorros também mantêm as fotos originais do projeto):

```bash
npm run import-cifar
```

**3) (Opcional) Conferir se a matemática está certa:**

```bash
npm run gradcheck
```

**4) Treinar** (o número é a quantidade máxima de épocas; ~15–20 s por época):

```bash
npm run train -- 30
```

**5) Perguntar ao modelo sobre uma imagem qualquer** (PNG, JPG ou HEIC):

```bash
npm run predict -- dataset/test/motorcycles/cifar_0003.png
```

Saída real do modelo treinado:

```
motorcycles   85.5%  ██████████████████████████
cats           6.5%  ██
cars           5.6%  ██
trucks         1.4%
dogs           0.8%
buses          0.2%
people         0.1%

I have detected: MOTORCYCLES
```

### Todos os comandos

| Comando | O que faz |
|---|---|
| `npm run import-cifar -- 400 100 100` | Converte o CIFAR em PNGs nas pastas `dataset/` (treino, validação, teste por classe) |
| `npm run gradcheck` | Prova que o aprendizado (backpropagation) está calculando certo |
| `npm run summary` | Mostra as camadas da rede e o tamanho de cada etapa |
| `npm run train -- 30` | Treina e salva o modelo em `models/classificador.json` |
| `npm run predict -- <imagem>` | Classifica uma imagem usando o modelo salvo |

> **Quer usar fotos suas?** Basta criar uma pasta com o nome da classe em `dataset/train/`, `dataset/validation/` e `dataset/test/` e colocar as fotos lá. O programa descobre as classes sozinho pelos nomes das pastas.

---

## 3. Como funciona (explicado para leigos)

### 3.1 Para o computador, uma foto é uma planilha de números

Cada imagem é reduzida para **32 × 32 pixels**. Cada pixel tem 3 números (vermelho, verde e azul) entre 0 e 1. Então uma foto vira **3.072 números**. É só isso que a rede "vê".

### 3.2 O caminho de uma imagem dentro da rede

```
 Foto 32×32 (3.072 números)
        │
        ▼
 ┌──────────────────┐  8 "lupas" passeiam pela foto procurando padrões simples:
 │  Convolução 1    │  bordas, cantos, manchas de cor
 └──────────────────┘
        │  ReLU  → descarta o que é negativo ("não achei nada aqui")
        │  Pool  → resume cada quadrado 2×2 pelo valor mais forte (foto encolhe p/ 16×16)
        ▼
 ┌──────────────────┐  16 lupas combinam os padrões anteriores em formas maiores:
 │  Convolução 2    │  rodas, janelas, orelhas, focinhos
 └──────────────────┘
        │  ReLU + Pool  → 16×8×8 = 1.024 números
        ▼
 ┌──────────────────┐  64 "neurônios" olham todos os padrões juntos e
 │  Camada densa    │  formam uma opinião
 └──────────────────┘
        ▼
 ┌──────────────────┐  7 saídas, uma por classe
 │  Camada de saída │
 └──────────────────┘
        ▼
 ┌──────────────────┐  transforma as 7 notas em porcentagens que somam 100%
 │     Softmax      │  → "ônibus 71%, caminhão 18%, carro 6%..."
 └──────────────────┘
```

No total são **~67 mil números ajustáveis** (os "pesos"). Treinar é encontrar bons valores para eles.

### 3.3 Como a rede aprende

Imagine um aluno fazendo um simulado com gabarito:

1. **Chuta** — a rede olha uma foto e dá as 7 porcentagens.
2. **Confere o gabarito** — a *função de perda* (cross-entropy) dá uma nota de quão errado foi o chute. Dar 90% para a classe certa gera um erro pequeno, e dar 5% gera um erro grande.
3. **Descobre de quem é a culpa** — a *retropropagação* (backpropagation) volta camada por camada calculando quanto cada um dos 67 mil pesos contribuiu para o erro.
4. **Corrige um pouquinho** — o otimizador *Adam* ajusta cada peso na direção que diminui o erro.
5. **Repete** com todas as fotos. Uma passada completa é uma **época**.

### 3.4 Por que separar treino, validação e teste?

| Conjunto | Analogia | Para que serve |
|---|---|---|
| **Treino** (400/classe) | Exercícios do livro | A rede aprende com eles |
| **Validação** (100/classe) | Simulado | Mede o progresso a cada época, sem ensinar. O programa guarda a versão da rede que foi melhor no simulado |
| **Teste** (100/classe) | Prova final | Usado uma única vez no fim, com fotos nunca vistas. É a nota "honesta" |

Se a nota no treino sobe mas a da validação cai, a rede está **decorando** em vez de aprender (*overfitting*). O programa avisa quando isso acontece e para sozinho se a validação ficar 6 épocas sem melhorar.

### 3.5 Truque extra: espelhar as fotos

Durante o treino, metade das fotos é espelhada horizontalmente (um carro virado para a esquerda continua sendo um carro). Isso "dobra" a variedade de exemplos de graça e reduz a decoreba.

---

## 4. De onde vêm as imagens

São usados dois bancos públicos de imagens 32×32 da Universidade de Toronto, muito usados em ensino:

| Classe (pasta) | Origem |
|---|---|
| `cats` | CIFAR-10 · cat |
| `dogs` | CIFAR-10 · dog |
| `cars` | CIFAR-10 · automobile |
| `trucks` | CIFAR-10 · truck |
| `buses` | CIFAR-100 · bus |
| `motorcycles` | CIFAR-100 · motorcycle |
| `people` | CIFAR-100 · baby + boy + girl + man + woman (misturados igualmente) |

O script `cli/import-cifar.ts` lê os arquivos binários do CIFAR byte a byte (também sem biblioteca) e salva cada imagem como PNG.

---

## 5. Estrutura do projeto

```
cli/
  import-cifar.ts   converte o CIFAR em pastas de imagens
  train.ts          comando de treino
  predict.ts        comando de previsão
  gradcheck.ts      verificação matemática dos gradientes
src/
  tensor.ts               "planilha" de números com várias dimensões
  convolve2d.ts           camada de convolução (as "lupas")
  activation.ts           ReLU e Sigmoid
  maxPool2xWindow.ts      pooling (resumir 2×2 pelo maior valor)
  dense.ts                camada densa (neurônios totalmente conectados)
  softmax-cross-entropy.ts porcentagens + cálculo do erro para várias classes
  tiny-cnn.ts             monta a rede inteira, camada por camada
  adam.ts / sgd.ts        otimizadores (como corrigir os pesos)
  dataset.ts              lê as pastas, embaralha e espelha imagens
  image-loader.ts         abre PNG/JPG/HEIC e redimensiona para 32×32
  training-loop.ts        o laço de treino, avaliação e matriz de confusão
  train.ts                junta tudo: treinar, testar, salvar, prever
  io.ts                   salva/carrega o modelo em JSON
dataset/{train,validation,test}/<classe>/   as imagens
models/classificador.json                   o modelo treinado
```

Os arquivos `neuron.ts`, `backPropagation.ts`, `finite-diference.ts`, `bce.ts` e `trainable-mlp.ts` são das etapas anteriores do estudo (um neurônio só, classificação binária) e foram mantidos como histórico.

---

## 6. O que foi alterado para sair de 2 para 7 classes

| Mudança | Por quê |
|---|---|
| **Sigmoid + Binary Cross-Entropy → Softmax + Cross-Entropy categórica** | Sigmoid só responde "sim/não". Softmax distribui 100% entre N opções |
| Classes descobertas pelas pastas | Antes `cats`/`dogs` estavam fixos no código |
| Acerto = classe com maior porcentagem (*argmax*) | Antes era "passou de 0,5?" |
| Rede maior (4→8 e 8→16 filtros, 32→64 neurônios) | 7 classes exigem reconhecer mais padrões |
| Inicialização *He* dos pesos | Começo mais equilibrado para redes com ReLU |
| Espelhamento, guardar a melhor época, parada antecipada | Combate a decoreba (*overfitting*) |
| Matriz de confusão | Mostra **quais** classes a rede confunde entre si |
| Convolução e pooling reescritos com acesso direto à memória | Mesma matemática, **~10× mais rápido** |

### Dois bugs encontrados e corrigidos

A verificação de gradientes (`npm run gradcheck`) compara o aprendizado calculado pela rede com uma derivada calculada "na força bruta". Ela revelou que **as camadas de convolução nunca aprendiam nada** no projeto original:

1. **ReLU** — em tensores 3D, `get(i)` com um só índice retornava `undefined`, e o gradiente virava sempre zero.
2. **MaxPool** — o índice do "vencedor" de cada janela nunca avançava, e o gradiente ia todo para o lugar errado.

Ou seja, no "gato vs cachorro" original só as duas camadas finais aprendiam. As "lupas" ficavam com valores aleatórios. Depois da correção, todos os 202 parâmetros do teste batem com diferença menor que 0,000001%.

---

## 7. Resultados

Treino real (CPU comum, ~16 s por época, ~3 minutos no total):

| Época | Acerto no treino | Acerto na validação | |
|---|---|---|---|
| 1 | 46,9% | 44,8% | |
| 3 | 55,6% | 50,4% | |
| 6 | 74,3% | **57,0%** | ← melhor época, pesos guardados |
| 9 | 75,5% | 54,5% | |
| 12 | 87,9% | 56,2% | parada antecipada |

**Nota final na prova (teste, 710 fotos nunca vistas): 55,6% de acerto.**
Chutando ao acaso entre 7 opções, o acerto seria 14,3%. Ou seja, a rede é **~4× melhor que o acaso**.

### Onde ela acerta e onde se confunde (matriz de confusão do teste)

Cada linha é o que a foto **realmente** era, e cada coluna é o que a rede **disse**.

| real \ previsto | ônibus | carro | gato | cachorro | moto | pessoa | caminhão | **acerto** |
|---|---|---|---|---|---|---|---|---|
| **ônibus** | **48** | 16 | 6 | 2 | 1 | 4 | 23 | 48% |
| **carro** | 14 | **64** | 2 | 2 | 3 | 1 | 14 | 64% |
| **gato** | 1 | 6 | **54** | 24 | 3 | 14 | 3 | 51% |
| **cachorro** | 1 | 3 | 34 | **51** | 1 | 13 | 2 | 49% |
| **moto** | 4 | 13 | 3 | 3 | **69** | 0 | 8 | 69% |
| **pessoa** | 5 | 3 | 13 | 15 | 2 | **60** | 2 | 60% |
| **caminhão** | 20 | 18 | 6 | 2 | 2 | 3 | **49** | 49% |

### O que os números dizem

- **Os erros fazem sentido.** A rede quase nunca confunde "bicho" com "veículo". Os erros ficam **dentro de famílias parecidas**:
  - **gato ↔ cachorro** (24 e 34 trocas): pelos, quatro patas, orelhas, tudo parecido em 32×32;
  - **ônibus ↔ caminhão ↔ carro**: caixas grandes com rodas e janelas;
  - **pessoa ↔ animais**: rostos e fundos domésticos.
- **Moto é a classe mais fácil (69%)**: o formato é bem diferente de tudo.
- **A rede decora a partir da 6ª época**: o acerto no treino segue subindo (88%), mas o da validação estaciona em ~55%. Por isso o programa guarda a época 6 e para sozinho.
- Antes das correções de bugs, as "lupas" não aprendiam. Agora elas aprendem e o resultado ficou dentro do esperado para uma rede deste tamanho.

> O log completo do treino fica em `train-log.txt`.

---

## 8. Análise e recomendação

### Limites desta abordagem
- **Classifica, não detecta**: dá um rótulo para a foto inteira. Se houver uma pessoa *e* um carro, ela escolhe um. Para localizar vários objetos com caixas seria preciso um modelo de **detecção** (ex.: família YOLO), uma arquitetura bem mais complexa.
- **32×32 pixels é muito pouco**: até um humano às vezes confunde um caminhão com um ônibus nesse tamanho.
- **Rede pequena e CPU pura**: sem GPU e sem operações em lote, treinar redes maiores à mão fica lento.
- **Poucos dados**: 400 fotos por classe é pouco para aprender do zero.

### Comparação de alternativas

| Alternativa | Esforço | Precisão esperada (7 classes) | Quando usar |
|---|---|---|---|
| **Este projeto (à mão)** | Alto | Moderada | Aprender como uma rede neural funciona por dentro |
| Mesma rede em PyTorch / TensorFlow.js | Baixo | Igual ou um pouco melhor (treina mais rápido, permite redes maiores) | Prototipar arquiteturas maiores rapidamente |
| Modelo pré-treinado (ex.: MobileNet, ResNet) + ajuste fino | Baixo | Alta (> 90%) | Classificação real em produto |
| Detector pré-treinado (ex.: YOLO, treinado no COCO, que já conhece todas as 7 classes) | Baixo | Alta, e ainda **localiza** cada objeto | Câmeras de trânsito, contagem de veículos e pessoas |

### Próximos passos possíveis (ainda à mão)
- **Dropout** (desligar neurônios ao acaso no treino) e **L2** para reduzir a decoreba;
- **Mais fotos por classe**: o CIFAR tem até 5.000 por classe no CIFAR-10;
- **Uma 3ª camada de convolução** e **mini-batches** para um treino mais estável;
- **Taxa de aprendizado decrescente** ao longo das épocas.

### Recomendação
- **Para aprender**: este projeto cumpre o papel. Ele mostra cada engrenagem, e a verificação de gradientes já provou seu valor ao encontrar dois bugs invisíveis.
- **Para um problema real** (ex.: contar pessoas e veículos numa rua): usar um **detector pré-treinado**. Ele já reconhece as 7 categorias, funciona em fotos grandes e encontra vários objetos por imagem. Escrever tudo à mão não compensa para produção.

---

## 9. Glossário rápido

| Termo | Significado simples |
|---|---|
| **Tensor** | Uma tabela de números com várias dimensões (ex.: 3 cores × 32 linhas × 32 colunas) |
| **Convolução** | Uma pequena "lupa" 3×3 que desliza pela imagem procurando um padrão |
| **ReLU** | Mantém os valores positivos e zera os negativos |
| **Pooling** | Encolhe a imagem guardando só o valor mais forte de cada pedaço |
| **Camada densa** | Neurônios que olham todas as informações da etapa anterior |
| **Softmax** | Transforma notas em porcentagens que somam 100% |
| **Loss (perda)** | Nota do erro. Quanto menor, melhor |
| **Backpropagation** | Cálculo de "quem tem culpa" no erro, voltando camada por camada |
| **Adam** | Otimizador: a regra de quanto e como corrigir cada peso |
| **Época** | Uma passada completa por todas as fotos de treino |
| **Overfitting** | Quando a rede decora o treino e vai mal em fotos novas |
| **Matriz de confusão** | Tabela que mostra o que era cada foto *versus* o que a rede disse |
# machine-learning-treinamento-ia-nativo
