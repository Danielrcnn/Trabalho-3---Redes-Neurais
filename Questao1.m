clear; clc; close all;
Entradas = load('Entradas.txt');
Saidas = load('Saidas.txt');
[Linhas, Colunas] = size(Entradas);
Pesos = rand(2,3)%[-0.424 -0.740 -0.961; 0.358 -0.577 -0.469]; 
Pesos_Outros_Neuronios = rand(1,3)%[-0.017 -0.893 0.148];
Momento = 1;
iter=0;
n=0.005;
MAbsoluta=1;
Neuronios = 3;
precisao=0.0005;
while(MAbsoluta>=precisao && iter<10000000)
    iter=iter+1;
    
    U = Entradas(:,1)*Pesos(1,:) + Entradas(:,2)*Pesos(2,:);
    
    [Sigmoid_Oculta] = Sigmoid(U);%Função Sigmoid das entradas para os neurônios ocultos
    [DerivadaOculta] = DerivadaSigmoid(Sigmoid_Oculta); %Derivada da Sigmoid Oculta
    
    Saida = Sigmoid_Oculta(:,:)*Pesos_Outros_Neuronios';%"Ativação" dos neurônios ocultos pra saida
    
    [Sigmoid_OcultaSaida] = Sigmoid(Saida);%Função Sigmoid dos ocultos para a saida

    Erro = Saidas(:,1) - Sigmoid_OcultaSaida(:,1);
    MAbsoluta = mean(Erro.^2);
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
    
    Pesos_Outros_Neuronios = (Pesos_Outros_Neuronios*Momento)+(SomatorioDaSaidaDeltaOculto*n)'; %Atualização dos pesos da Saida pra Oculta
    
    [l,c]=size(DeltaOculto);
    for i=1:Colunas
        for k=1:c
            DeltaEntrada(i,k) = dot(DeltaOculto(:,k),Entradas(:,i)'); %Vou deixar isso por enquanto porque Tui que fez e eu não entendi muito bem, depois eu vejo isso
        end
    end

     Pesos = (Pesos*Momento)+(DeltaEntrada*n);
     fprintf('Pesos: W1: %f\t W2: %f\t W3: %f\t W4: %f\t W5: %f\t W6: %f\t W7: %f\t W8: %f\t W9: %f\t\n', Pesos(1,1), Pesos(2,1), Pesos(1,2), Pesos(2,2), Pesos(1,3), Pesos(2,3), Pesos_Outros_Neuronios(1,1), Pesos_Outros_Neuronios(1,2), Pesos_Outros_Neuronios(1,3))
end

% for i=1:Linhas
%     for k=1:3
%         U(i,k) = Entradas(i,1)*Pesos(1,k) + Entradas(i,2)*Pesos(2,k);%Ativação camada oculta
%     end
% end

[Ativacao] = Sigmoid(U);
SaidaFinal = Ativacao*Pesos_Outros_Neuronios';
[Validacao] = Sigmoid(SaidaFinal)

figure (1)
VetorQualquer = [1:1:length(Saidas)];
plot(VetorQualquer,Saidas,'o',VetorQualquer,Validacao,'*r')

figure (2)
X = [1:1:iter];
plot(X,ParaPlotarGrafico), title('MSE x Épocas de treinamento'),xlabel('Épocas'),ylabel('MSE')
%Fazer a topologia da rede e nas linhas dos pesos colocar: W1, W2..., até
%as ultimas linhas de pesos e aqui no código imprimir os pesos de 1 à 9