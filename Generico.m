clear; clc; close all;
load('Dados_Aprox_Trein.mat')
load('Dados_Aprox_Val.mat')
Entradas = x;
Saidas = yd;
[Linhas, Colunas] = size(Entradas);
Neuronios = input('Quantos neurônios na camada oculta? ');
Pesos = rand(Colunas, Neuronios);
Pesos_Outros_Neuronios = rand(1, Neuronios); 
Momento = 1;
iter=0;
n=0.0025;
MAbsoluta=1;
Contador=0;
precisao=0.0003;
while(MAbsoluta>=precisao && iter<10000000)
    iter=iter+1;
    Contador = Contador + 1;
    
    U = Entradas(:,1)*Pesos(1,:) + Entradas(:,2)*Pesos(2,:) + Entradas(:,3)*Pesos(3,:);
    
    [Sigmoid_Oculta] = Sigmoid(U);%Função Sigmoid das entradas para os neurônios ocultos
    [DerivadaOculta] = DerivadaSigmoid(Sigmoid_Oculta); %Derivada da Sigmoid Oculta
    
    Saida = Sigmoid_Oculta(:,:)*Pesos_Outros_Neuronios';%"Ativação" dos neurônios ocultos pra saida
    
    [Sigmoid_OcultaSaida] = Sigmoid(Saida);%Função Sigmoid dos ocultos para a saida
    Erro = Saidas(:,1) - Sigmoid_OcultaSaida(:,1);
%     MAbsoluta = sum(abs(Erro(:,1)))/Linhas;
%     MAbsoluta = sum((Erro(:,1)).^2)/Linhas
    MAbsoluta = mean(Erro.^2)
    ParaPlotarGrafico(Contador) = MAbsoluta;
    [DerivadaSaida] = DerivadaSigmoid(Sigmoid_OcultaSaida);%Derivada da Sigmoid Saida
    
    DeltaSaida = Erro(:,1).*DerivadaSaida(:,1);
    
    for i=1:Linhas
        for k=1:Neuronios
            DeltaOculto(i,k) = DerivadaOculta(i,k)*Pesos_Outros_Neuronios(1,k)*DeltaSaida(i,1); %Derivada da Sigmoid
        end
    end
%     DeltaOculto = DerivadaOculta*Pesos_Outros_Neuronios'.*DeltaSaida;
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
     
     %fprintf('Pesos: W1: %f\t W2: %f\t W3: %f\t W4: %f\t W5: %f\t W6: %f\t W7: %f\t W8: %f\t W9: %f\t\n', Pesos(1,1), Pesos(2,1), Pesos(1,2), Pesos(2,2), Pesos(1,3), Pesos(2,3), Pesos_Outros_Neuronios(1,1), Pesos_Outros_Neuronios(1,2), Pesos_Outros_Neuronios(1,3))
%     return
% break
end
% Validação do treinamento
[Ativacao] = Sigmoid(U);
SaidaFinal = Ativacao*Pesos_Outros_Neuronios';
[Validacao] = Sigmoid(SaidaFinal)

% Validação dados classificação

UValidacao = xv(:,1)*Pesos(1,:) + xv(:,2)*Pesos(2,:) + xv(:,3)*Pesos(3,:);
[AtivacaoValidacao] = Sigmoid(UValidacao);
[SaidaFinalDadosClassificacao] = AtivacaoValidacao*Pesos_Outros_Neuronios';
[ValidacaodaClassificacao]=Sigmoid(SaidaFinalDadosClassificacao)

figure (1)
X = [1:1:iter];
plot(X,ParaPlotarGrafico)

figure (2)
[Linhas, Colunas] = size(xv);
for k=1:Linhas
   if Validacao(k,1)>=0.5
       plot3(xv(k,1), xv(k,2), xv(k,3), 'go'); grid on; hold on;
   else
       plot3(xv(k,1), xv(k,2), xv(k,3), 'bs'); grid on; hold on;
   end
end
hold on;
P = [0:0.01:1];

% P2=(Pesos(1,1)*Pesos(2,1)*P)/Pesos(3,1);
% P3=(Pesos(1,2)*Pesos(2,2)*P)/Pesos(3,2);
% P4=(Pesos(1,3)*Pesos(2,3)*P)/Pesos(3,3);
% P5=(Pesos(1,4)*Pesos(2,4)*P)/Pesos(3,4);
% P6=(Pesos(1,5)*Pesos(2,5)*P)/Pesos(3,5);
% P7=(Pesos(1,6)*Pesos(2,6)*P)/Pesos(3,6);
% P8=(Pesos(1,7)*Pesos(2,7)*P)/Pesos(3,7);
% P9=(Pesos(1,8)*Pesos(2,8)*P)/Pesos(3,8);
% P10=(Pesos(1,9)*Pesos(2,9)*P)/Pesos(3,9);
% P11=(Pesos(1,10)*Pesos(2,10)*P)/Pesos(3,10);
% plot3(P,F,P2,'r'); hold on;
% plot3(P,F,P3,'b'); hold on;
% plot3(P,F,P4,'m'); hold on;
% plot3(P,F,P5,'--'); hold on;
% plot3(P,F,P6,'bs'); hold on;
% plot3(P,F,P7,'*'); hold on;
% plot3(P,F,P8,'go'); hold on;
% plot3(P,F,P9,'g'); hold on;
% plot3(P,F,P10,'y'); hold on;
% plot3(P,F,P11,'+'); hold on;




return
figure (1)
hold on;
P = [0:0.01:1];
P2=(Pesos(1,1)*P)/Pesos(2,1);
P3=(Pesos(1,2)*P)/Pesos(2,2);
P4=(Pesos(1,3)*P)/Pesos(2,3);
% P5=(Pesos(1,4)*P)/Pesos(2,4);
% P6=(Pesos(1,5)*P)/Pesos(2,5);
% P7=(Pesos(1,6)*P)/Pesos(2,6);
% P8=(Pesos(1,7)*P)/Pesos(2,7);
% P9=(Pesos(1,8)*P)/Pesos(2,8);
% P10=(Pesos(1,9)*P)/Pesos(2,9);
% P11=(Pesos(1,10)*P)/Pesos(2,10);
plot(P,P2,'r'); hold on;
plot(P,P3,'b'); hold on;
plot(P,P4,'m'); hold on;
% plot(P,P5,'--'); hold on;
% plot(P,P6,'bs'); hold on;
% plot(P,P7,'*'); hold on;
% plot(P,P8,'go'); hold on;
% plot(P,P9,'g'); hold on;
% plot(P,P10,'y'); hold on;
% plot(P,P11,'+'); hold on;
for k=1:Linhas
   if Validacao(k,1)>=0.5
       y(k)=1;
       plot(Entradas(k,1),Entradas(k,2),'go');hold on;
   else
       y(k)=-1;
       plot(Entradas(k,1),Entradas(k,2),'bs');hold on;
   end
end

figure (2)
X = [1:1:iter];
plot(X,ParaPlotarGrafico)