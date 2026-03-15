import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // primeira camada da rede neural:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)

    // 80 neuronios = colocamos tudo isso pois há pouca base de treino
    // quanto mais neuronios, mais complexo o modelo, mais preciso, mas mais lento

    // Ativação a Reulo age como um filtro:
    // é como se ela deixasse apenas os dados interessantes seguirem na rede
    // SE a informação que chegou nesse neuronio é positiva, passa para frente
    // SE é negativa ou zero, bloqueia e não passa para frente

    model.add(tf.layers.dense({
        units: 80, // 3 saídas (premium, medium, basic)
        inputShape: [7], // 7 posições de entrada
        activation: 'relu' // função de ativação para classificação
    }));

    // Saída: 3 neuronios = 3 saídas (premium, medium, basic)
    model.add(tf.layers.dense({
        units: 3, // 3 saídas (premium, medium, basic)
        activation: 'softmax' // função de ativação para classificação
    }));

    // Compila o modelo:
    // optimizer: Adam (Adaptative Moment Estimation)
    // é um treinador pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com histórico de erros e acertos

    // loss: Categorical Crossentropy (função de perda para classificação)
    // ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa (o one-hot encoded das categorias)
    // a categoria premium será sempre 1, 0, 0, a medium será 0, 1, 0, e a basic será 0, 0, 1
    
    // metrics: ['accuracy'] (métrica de avaliação)
    // é a medida de quão certo o modelo está
    // quanto mais próximo de 1, melhor o modelo
    // quanto mais próximo de 0, pior o modelo
    // a cada época, o modelo vai tentar melhorar sua precisão
    
    // exemplos clássicos: classificação de imagens, classificação de texto, classificação de voz,
    // recomendação, etc.
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Treinamento do modelo
    // verbose: 0 = não mostra o progresso
    // epochs: 100 = 100 épocas de treinamento
    // shuffle: true = embaralha os dados a cada época, para evitar BIAS (viés) e melhorar a generalização
    // callbacks: {
    //     onEpochEnd: (epoch, logs) => {
    //         console.log(`Epoch ${epoch} - Loss: ${logs.loss} - Accuracy: ${logs.accuracy}`);
    //     }
    // } = callback para mostrar o progresso a cada época
    await model.fit(
        inputXs, 
        outputYs, 
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch} - Loss: ${logs.loss} - Accuracy: ${logs.acc}`);
                }
            }
        }
    );

    return model;
}

async function predict(model, pessoaTensorNormalizado) {
    // transformar um array em um tensor
    const tfInput = tf.tensor2d([pessoaTensorNormalizado]);
    const prediction = await model.predict(tfInput);
    const predictionArray = await prediction.array();
    return predictionArray[0].map((prob, index) => ({prob, index}));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dados melhor
// assim o altoritmo consegue entender melhor os padrões complexos
// dos dados
const model = await trainModel(inputXs, outputYs);

const pessoa = {
    nome: "Zé",
    idade: 28,
    cor: "verde",
    localizacao: "Curitiba"
}

// Normalizar a pessoa
// Exemplo: idade_min = 25, idade_max = 40, idade = (28 - 25) / (40 - 25) = 0.2

const pessoaTensorNormalizado = [
    (pessoa.idade - 25) / (40 - 25),
    1, // azul
    0, // vermelho
    0, // verde
    0, // São Paulo
    1, // Rio
    0 // Curitiba
];

const predictions = await predict(model, pessoaTensorNormalizado);
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(({prob, index}) => ({
        categoria: labelsNomes[index],
        probabilidade: `${(prob*100).toFixed(2)}%`
    }));
console.log(results);



