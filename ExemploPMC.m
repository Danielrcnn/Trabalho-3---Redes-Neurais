clear; clc; close all;
Entradas = load('Entradas.txt');
Saidas = load('Saidas.txt');
Pesos = [-0.424 -0.740 -0.961; 0.358 -0.577 -0.469];
[Linhas, Colunas] = size(Entradas);
Pesos_Outros_Neuronios = [-0.017 -0.893 0.148];
Momento = 1;
n=0.3;
for i=1:Linhas
    for k=1:3
        U(i,k) = Entradas(i,1)*Pesos(1,k) + Entradas(i,2)*Pesos(2,k) %Ativação
    end
end
% for i=1:Linhas
%     for k=1:3
%          f(i,k) = 1/(1+exp(-U(i,k))) %Função Sigmoid das entradas para os neurônios ocultos
       [f] = Sigmoid(U)
%     end
% end
for i=1:Linhas
    Saida(i,1) = f(i,:)*Pesos_Outros_Neuronios' %"Ativação" dos neurônios ocultos pra saida
end
for i=1:length(Saida)
    g(i,1) = 1/(1+exp(-Saida(i,1))) %Função Sigmoid dos ocultos para a saida
end

Erro = Saidas(:,1) - g(:,1) 
%Calclar a media absoluta para usar la na frente como criério de parada

for i=1:length(Saidas)
    DerivadaSaida(i,1) = g(i,1)*(1-g(i,1)) %Derivada da Sigmoid
    %slide 30
end

for i=1:length(DerivadaSaida)
   DeltaSaida(i,1) = Erro(i,1)*DerivadaSaida(i,1)
   %slide 30 a frene
end

for i=1:4
    for k=1:3
        DerivadaOculta(i,k) = f(i,k)*(1-f(i,k)) %Derivada da Sigmoid
        %slide 34
    end
end

for i=1:4
    for k=1:3
        DeltaOculto(i,k) = DerivadaOculta(i,k)*Pesos_Outros_Neuronios(1,k)*DeltaSaida(i,1) %Derivada da Sigmoid
        %slide 34
    end
end

%% Entradas * Delta
  for k=1:3
      for i=1:4
        SaidaDeltaOculto(k,i) = f(i,k)*DeltaSaida(i,:) %Da saida pro oculto
      end
      SomatorioDaSaidaDeltaOculto(k,1) = sum(SaidaDeltaOculto(k,:))
  end

  for i=1:length(Pesos_Outros_Neuronios)
      Pesos_Outros_Neuronios(:,i) = (Pesos_Outros_Neuronios(:,i)*Momento)+(SomatorioDaSaidaDeltaOculto(i,:)*n)
  end
  
  [l,c]=size(DeltaOculto)
  for i=1:Linhas
      for k=1:c
          DeltaEntrada(i,k) = Entradas(i,1)*DeltaOculto(i,k) + Entradas(i,2)*DeltaOculto(i,k)
      end
%      SomatorioDaEntradaDeltaOculto(i,k) = sum(DeltaEntrada(i,:))
  end

% for k=1:2
%     for i=1:4
%        EntradaDeltaOculto(i,k) = Entradas(i,k)*DeltaEntrada(i,:) %Da saida pro oculto
%     end
%        SomatorioDaEntradaDeltaOculto(i,k) = sum(EntradaDeltaOculto(k,:))
%  end