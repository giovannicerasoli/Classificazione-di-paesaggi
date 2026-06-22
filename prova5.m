%%

clear all
clc
close all

%% DATASET CON IMMAGINI TRAIN

% path di una cartella con esempi
vehicleDir = 'C:\Users\giova\Giogiò\Universita\MPA\Esame\archive3\archive3\train';
% mette le immagini in un datastore e desume le label dalle cartelle
datatrain = imageDatastore(vehicleDir,'IncludeSubfolders',true,'LabelSource', 'foldernames');
% numero totale di immagini
numtrain = numel(datatrain.Files);
% conta quante ce ne sono per ogni label
countEachLabel(datatrain)
numtrain = numel(datatrain.Files);

%% hog

img = readimage(datatrain, 1);
%operatore hog
% dimensione dell'operatore di HoG (sperimentare)
cellSize = [5 5];%all'aumentare hog meno sensibile
% prova di estrazione del feature vector di HoG
[hog, visualHoG] = extractHOGFeatures(img,'CellSize',cellSize);
% visualizza immagine e features
figure(1)
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
plot(visualHoG);

%%

tic
fprintf('Calcolo features ... \n')
hogFeatureSize = length(hog);
% prepara una matrice per contenere le features del training set
XTrain = zeros(numtrain, hogFeatureSize, 'single');
% ciclo di calcolo
tic
for i = 1:numtrain
    % legge immagine
    img = readimage(datatrain, i);
    % estrazione features
    XTrain(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end
toc
YTrain = datatrain.Labels;
fprintf('ho finito \n')
toc

%%

% visualizzazione esempi con subplot
figure(1)
% dimensione del carattere
size = [174 174];
% ciclo di visualizzazione di esempi random (12 su grigliato 3x4)
for i=1:12
    % genera un random tra 1 e ntrain
    p = random('Discrete Uniform',numtrain);
    % estrae il dato e lo formatta come metrice
    % definisce il subplot
    subplot(3,4,i)
    % plot immagine
    imshow(readimage(datatrain, p))
    % etichetta 
    title(YTrain(p));
end

%% 
%diagramma a barre train

labels = {'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'};
values = [2190, 2269, 2315, 2512, 2274, 2382];

bar(values)
xticklabels(labels)
title('Distribuzione dei dati')
xlabel('Categorie')
ylabel('Valori')

%% DATASET CON IMMAGINI TEST

% path di una cartella con esempi
vehicleDir = 'C:\Users\giova\Giogiò\Universita\MPA\Esame\archive3\archive3\test';
% mette le immagini in un datastore e desume le label dalle cartelle
datatest = imageDatastore(vehicleDir,'IncludeSubfolders',true,'LabelSource', 'foldernames');
% numero totale di immagini
numtest = numel(datatest.Files);
% conta quante ce ne sono per ogni label
countEachLabel(datatest)
numtest = numel(datatest.Files);

%% hog

img = readimage(datatest, 1);
%operatore hog
% dimensione dell'operatore di HoG (sperimentare)
cellSize = [5 5];%all'aumentare hog meno sensibile
% prova di estrazione del feature vector di HoG
[hog, visualHoG] = extractHOGFeatures(img,'CellSize',cellSize);
% visualizza immagine e features
figure(1)
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
plot(visualHoG);

%%

%diagramma a barre test

labels = {'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'};
values = [437, 474, 537, 525, 510, 501];

bar(values)
xticklabels(labels)
title('Distribuzione dei dati')
xlabel('Categorie')
ylabel('Valori')

%%

tic
fprintf('Calcolo features ... \n')
hogFeatureSize = length(hog);
% prepara una matrice per contenere le features del training set
XTest = zeros(numtest, hogFeatureSize, 'single');
% ciclo di calcolo
tic
for i = 1:numtest
    % legge immagine
    img = readimage(datatest, i);
    % estrazione features
    XTest(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end
toc
YTest = datatest.Labels;
fprintf('ho finito \n')
toc

%%

% visualizzazione esempi con subplot
figure(1)
% dimensione del carattere
size = [174 174];
% ciclo di visualizzazione di esempi random (12 su grigliato 3x4)
for i=1:12
    % genera un random tra 1 e ntrain
    p = random('Discrete Uniform',numtest);
    % estrae il dato e lo formatta come metrice
    % definisce il subplot
    subplot(3,4,i)
    % plot immagine
    imshow(readimage(datatest, p))
    % etichetta 
    title(YTest(p));
end

%% 

% valuta la separabilità a priori con t-SNE
fprintf('Valutazione t-SNE ... \n')
T = tsne(XTrain,'NumPCAComponents',25);
% output dei 2 clusters mappati
figure(2)
gscatter(T(:,1),T(:,2),YTrain)

%%

A = cov(XTrain); 
%%
% estrazione autovalori (in ordine crescente)
L = eig(A); 
% cambia ordine, per comodità grafica
L = flip(L);

%% NON VA

% accuratezza della ricostruzione
Acc = cumsum(L);
Acc = Acc/sum(L);
% grafico autovalori e 
figure(2)
subplot(1,2,1);
plot(L)
subplot(1,2,2);
plot(Acc);

%%

tic
% quante features utilizzare ... (sperimentare)
ncomp = 2000;
% PCA
coeff = pca(XTrain,'NumComponents',ncomp);
% proiezione
fprintf('Proiezione su sottospazio ...\n');
XTrain2 = XTrain*coeff;
XTest2 = XTest*coeff;
toc

%%

tic
% training di una foresta
fprintf('Classificazione con RF ...\n');
% numero di alberi (sperimentare)
ntrees = 500;
treeModel = TreeBagger(ntrees,XTrain2,YTrain,'OOBPrediction','on',...
    'MinLeafSize',10);
% grafico OOB error
figure(4)
E = oobError(treeModel);
plot(E);
fprintf('OOBError %f\n',E(end));
% la applica al test
test_pred = predict(treeModel, XTest2);

% oggetto per valutazione prestazioni
test_pred = categorical(test_pred);

YTest = cellstr(YTest);
test_pred = cellstr(test_pred);
cp2 = classperf(YTest, test_pred);

fprintf('Errore test %f\n',cp2.ErrorRate);
% calcola la matrice di confusione
figure(5)
confusionchart(categorical(YTest), categorical(test_pred));
str = sprintf('Random Forest - errore %.4f\n',cp2.ErrorRate);
title(str)
toc

%% NON VA

tic
% training/test classificatore quadratico ottimo
fprintf('Classificazione bayesiana ...\n');
test_class = classify(XTest2,XTrain2,YTrain,'Quadratic');
% oggetto per valutazione prestazioni
cp1 = classperf(YTest, test_class);
fprintf('Errore test %f\n',cp1.ErrorRate);
% statistica errori
figure(3)
confusionchart(YTest, test_class);
str = sprintf('Bayes quadratico - errore %.4f\n',cp1.ErrorRate);
title(str)
toc

%%

tic
% training di una SVM multiclasse
fprintf('Classificazione con SVM ...\n');
svmModel = fitcecoc(XTrain2, YTrain);
fprintf('ResubError %f\n',resubLoss(svmModel));
% la applica al test
test_svm = predict(svmModel, XTest2);
% oggetto per valutazione prestazioni
cp3 = classperf(cellstr(YTest), cellstr(test_svm));
fprintf('Errore test %f\n',cp3.ErrorRate);
% statistica errori
figure(6)
confusionchart(categorical(YTest), categorical(test_svm));
str = sprintf('SVM lineare - errore %.4f\n',cp3.ErrorRate);
title(str)
toc

%% training kNN

tic
fprintf('Classificazione con kNN ...\n');
% numero di NN (sperimentare)
k = 6;
knnModel = fitcknn(XTrain2, YTrain,'NumNeighbors',k);
% applica al test
test_knn = predict(knnModel, XTest2);
% oggetto per valutazione prestazioni
cp4 = classperf(cellstr(YTest), cellstr(test_knn));
fprintf('Errore test %f\n',cp4.ErrorRate);
% statistica errori
figure(7)
confusionchart(categorical(YTest), categorical(test_knn));
str = sprintf('KNN - errore %.4f\n',cp4.ErrorRate);
title(str)
toc

%% training MLP

tic
fprintf('Classificazione con MLP ...\n');
% Converte le etichette di addestramento in indici numerici
[~, YTrainIndices] = ismember(YTrain, unique(YTrain));
% Converte gli indici delle etichette in vettori binari
net_labels = full(ind2vec(YTrainIndices', 6));
% numero di nodi hidden (sperimentare)
nhidden = 30;
% crea ed addestra la rete
net = patternnet(nhidden);
net = train(net, XTrain2', net_labels);
% applica al test
test_out = net(XTest2');
test_net = vec2ind(test_out)'-1;
% oggetto per valutazione prestazioni
% Converti YTest in un cell array di vettori di caratteri
YTest_cell = cellstr(YTest);
% Converti test_net in un cell array di vettori di caratteri
test_net_cell = cellstr(num2str(test_net));
% oggetto per valutazione prestazioni
cp5 = classperf(YTest_cell, test_net_cell);
fprintf('Errore test %f\n', cp5.ErrorRate);
% statistica errori
figure(8)
confusionchart(categorical(YTest_cell), categorical(test_net_cell));
str = sprintf('MultiLayer Net - errore %.4f\n', cp5.ErrorRate);
title(str)
toc

%%

% training MLP
fprintf('Classificazione con MLP ...\n');
% Converte le etichette di addestramento in indici numerici
[~, YTrainIndices] = ismember(YTrain, unique(YTrain));
% prepara i dati
net_labels = full(ind2vec(YTrainIndices'+1,10));
% numero di nodi hidden (sperimentare)
nhidden = 30;
% crea ed addestra la rete
net = patternnet(nhidden);
net = train(net,XTrain2', net_labels);
% applica al test
test_out = net(XTest2');
test_net = vec2ind(test_out)'-1;
% oggetto per valutazione prestazioni
cp5 = classperf(YTest, test_net);
fprintf('Errore test %f\n',cp5.ErrorRate);
% statistica errori
figure(8)
confusionchart(categorical(YTest), categorical(test_net));
str = sprintf('MultiLayer Net - errore %.4f\n',cp5.ErrorRate);
title(str)