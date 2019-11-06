clear; clc; close all;
Entradas = load('Entradas.txt');
Saidas = load('Saidas.txt');
[Linhas, Colunas] = size(Entradas);
Pesos = [-0.424 -0.740 -0.961; 0.358 -0.577 -0.469];
Pesos_Outros_Neuronios = [-0.017 -0.893 0.148];
Momento = 1;
iter=0;
n=0.5;
MAbsoluta=1;
MAtual = 0;
MAnterior = 0;
med_ant=0;
precisao=0.05;
while(MAbsoluta>=precisao && iter<1000000)
    iter=iter+1;
    for i=1:Linhas
        for k=1:3
            U(i,k) = Entradas(i,1)*Pesos(1,k) + Entradas(i,2)*Pesos(2,k);%Ativação camada oculta
    	end
    end
    
    [Sigmoid_Oculta] = Sigmoid(U);%Função Sigmoid das entradas para os neurônios ocultos
    [DerivadaOculta] = DerivadaSigmoid(Sigmoid_Oculta); %Derivada da Sigmoid Oculta
    
    Saida = Sigmoid_Oculta(:,:)*Pesos_Outros_Neuronios';%"Ativação" dos neurônios ocultos pra saida
    
    [Sigmoid_OcultaSaida] = Sigmoid(Saida);%Função Sigmoid dos ocultos para a saida

    Erro = Saidas(:,1) - Sigmoid_OcultaSaida(:,1) ;
    MAbsoluta = sum(abs(Erro(:,1)))/Linhas;
    
    [DerivadaSaida] = DerivadaSigmoid(Sigmoid_OcultaSaida);%Derivada da Sigmoid Saida
    
    DeltaSaida = Erro(:,1).*DerivadaSaida(:,1);
    
    for i=1:4
        for k=1:3
            DeltaOculto(i,k) = DerivadaOculta(i,k)*Pesos_Outros_Neuronios(1,k)*DeltaSaida(i,1); %Derivada da Sigmoid
        end
    end
    
    %Entradas * Delta
    for k=1:3
       for i=1:4
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
    
     
%      Sigmoid_Oculta = 0;
%      DerivadaOculta = 0;
%      DerivadaSaida = 0;
       
%      med_ant=MAbsoluta; 
     
     fprintf('Pesos: W1: %f\t W2: %f\t W3: %f\t W4: %f\t W5: %f\t W6: %f\t W7: %f\t W8: %f\t W9: %f\t\n', Pesos(1,1), Pesos(2,1), Pesos(1,2), Pesos(2,2), Pesos(1,3), Pesos(2,3), Pesos_Outros_Neuronios(1,1), Pesos_Outros_Neuronios(1,2), Pesos_Outros_Neuronios(1,3))
%     return
end

for k=1:Linhas
   for i=1:Linhas
        for r=1:3
            U(i,r) = Entradas(i,1)*Pesos(1,r) + Entradas(i,2)*Pesos(2,r);%Ativação camada oculta
        end
   end
   
   for i=1:3
       SomaDosValoresDosNeuronios(1,i) = sum(U(:,i));
   end
   
   T = SomaDosValoresDosNeuronios*Pesos_Outros_Neuronios';
   
   if T>=0
       y(k)=1;
       plot(Entradas(k,1),Entradas(k,2),'go');hold on;
   else
       y(k)=-1;
       plot(Entradas(k,1),Entradas(k,2),'bs');hold on;
   end
end
%Fazer a topologia da rede e nas linhas dos pesos colocar: W1, W2..., até
%as ultimas linhas de pesos e aqui no código imprimir os pesos de 1 à 9