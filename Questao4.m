close all;
clear all;
clc;
disp('Programa MLP para XOR 2 entradas');
X = [ -1 , 1 , -1 , 1 ;-1 , -1 , 1 , 1 ];
Yd =[ -1 , 1 , 1 , -1 ];
neuronios_camada_escondida = 2; 
disp('Criando a rede MLP...');
net = feedforwardnet(neuronios_camada_escondida); %Camadas ocultas
disp('Configurando a rede...');
net = configure(net,X,Yd); %configura entradas e saídas da rede
net.divideParam.trainRatio = 1; %Amostras para treinamento
net.divideParam.valRatio = 0;%Amostras para classificação
net.divideParam.testRatio = 0;
net.trainFcn = 'traingd'; %Tipo da função de ativação
net.performFcn = 'mse';%Tipo de erro a ser calculado
net.trainParam.epochs = 500000; %Maximo de epocas
net.trainParam.lr = 0.1;%Taxa de Aprendizagem
net.trainParam.mc = 0; %Momento 
tansig.goal = 0.0001;
net.layers{1}.transferFcn = 'tansig';%Tipo da função de ativação em cada camada
net.layers{2}.transferFcn = 'tansig';%Tipo da função de ativação em cada camada
disp('Inicializando a rede neural....');
net = init(net);
disp('Treinando a rede neural...');
[net, tr] = train(net,X,Yd); %Treina a rede neural
plotperform(tr);
disp('Simulando a rede neural treinada...');
Ysaida = sim(net,X);
disp('Resultado: ');
disp(Ysaida);
disp('Calculando o erro da rede neural...');
perf = perform(net,Yd,Ysaida);
disp('Erro: ');
disp(perf);
wb = getwb(net);
[b,IW,LW] = separatewb(net,wb);
b = cell2mat(b);
disp('Bias de todas camadas =');
disp(b);
Wescondida = cell2mat(IW);
disp('Pesos da camada escondida =');
disp(Wescondida);
Wsaida = cell2mat(LW);
disp('Pesos da camada de saida =');
disp(Wsaida);