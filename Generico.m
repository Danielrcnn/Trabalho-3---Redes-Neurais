clear; clc; close all;
load('Dados_Aprox_Trein.mat')
load('Dados_Aprox_Val.mat')
Entradas = x;
Saidas = yd;
[Linhas, Colunas] = size(Entradas);
Neuronios = input('Quantos neur�nios na camada oculta? ');
Pesos = rand(Colunas, Neuronios);
Pesos_Outros_Neuronios = rand(1, Neuronios); 
Momento = 1;
iter=0;
n=0.005;
MAbsoluta=1;
precisao=0.0002;
while(MAbsoluta>=precisao && iter<10000000)
    iter=iter+1;
    
    U = Entradas(:,1)*Pesos(1,:) + Entradas(:,2)*Pesos(2,:) + Entradas(:,3)*Pesos(3,:); %A unica parte do c�digo que n�o esta gen�rica
    
    [Sigmoid_Oculta] = Sigmoid(U);%Fun��o Sigmoid das entradas para os neur�nios ocultos
    [DerivadaOculta] = DerivadaSigmoid(Sigmoid_Oculta); %Derivada da Sigmoid Oculta
    
    Saida = Sigmoid_Oculta(:,:)*Pesos_Outros_Neuronios';%"Ativa��o" dos neur�nios ocultos pra saida
    
    [Sigmoid_OcultaSaida] = Sigmoid(Saida);%Fun��o Sigmoid dos ocultos para a saida
    Erro = Saidas(:,1) - Sigmoid_OcultaSaida(:,1);
    MAbsoluta = mean(Erro.^2)
    ParaPlotarGrafico(iter) = MAbsoluta;
    [DerivadaSaida] = DerivadaSigmoid(Sigmoid_OcultaSaida);%Derivada da Sigmoid Saida
    
    DeltaSaida = Erro(:,1).*DerivadaSaida(:,1);
    
    for i=1:Linhas
        for k=1:Neuronios
            DeltaOculto(i,k) = DerivadaOculta(i,k)*Pesos_Outros_Neuronios(1,k)*DeltaSaida(i,1); %Derivada da Sigmoid
        end
    end
    %Entradas * Delta
    for k=1:Neuronios
       for i=1:Linhas
           SaidaDeltaOculto(k,i) = Sigmoid_Oculta(i,k)*DeltaSaida(i,:); %Da saida pro oculto
       end
           SomatorioDaSaidaDeltaOculto(k,1) = sum(SaidaDeltaOculto(k,:));
    end
    
    Pesos_Outros_Neuronios = (Pesos_Outros_Neuronios*Momento)+(SomatorioDaSaidaDeltaOculto*n)'; %Atualiza��o dos pesos da Saida pra Oculta
    
    [l,c]=size(DeltaOculto);
    for i=1:Colunas
        for k=1:c
            DeltaEntrada(i,k) = dot(DeltaOculto(:,k),Entradas(:,i)'); %Vou deixar isso por enquanto porque Tui que fez e eu n�o entendi muito bem, depois eu vejo isso
        end
    end
     Pesos = (Pesos*Momento)+(DeltaEntrada*n);
end
% Valida��o do treinamento
[Ativacao] = Sigmoid(U);
SaidaFinal = Ativacao*Pesos_Outros_Neuronios';
[Validacao] = Sigmoid(SaidaFinal);

% Valida��o dados classifica��o
UValidacao = xv(:,1)*Pesos(1,:) + xv(:,2)*Pesos(2,:) + xv(:,3)*Pesos(3,:);
[AtivacaoValidacao] = Sigmoid(UValidacao);
[SaidaFinalDadosClassificacao] = AtivacaoValidacao*Pesos_Outros_Neuronios';
[ValidacaodaClassificacao]=Sigmoid(SaidaFinalDadosClassificacao)

figure (1)
X = [1:1:iter];
plot(X,ParaPlotarGrafico), title('EQM x �pocas de treinamento'),xlabel('�pocas'),ylabel('MSE')
figure(2)
VetorQualquer = [1:1:length(ydv)];
plot(VetorQualquer,ydv,'o',VetorQualquer,ValidacaodaClassificacao,'*r')
Variancia = var(ValidacaodaClassificacao-ydv);
ErroMedio = sum((ValidacaodaClassificacao-ydv)/length(ydv))