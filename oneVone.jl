#uvodjenje potrebnih biblioteka
using GLM
using DataFrames
using CSV
using Lathe.preprocess: TrainTestSplit
using StatsBase	
using MLBase
using StatsModels

using ROC
using Plots

#podaci se ucitavaju u dataframe, dele se na train i test podatke
data = DataFrame(CSV.File("tacke100.csv"))
dataTrain, dataTest = TrainTestSplit(data, .80)
display(describe(data))
display(countmap(data[!, :grupa]))

#grupa zavisi od koordinata x i y, zato se pravi formula:
fm = @formula(grupa ~ x + y)

#postoje 4 grupe tj. klase kojima podaci pripadaju, zato se svaki podatak mora klasifikovati 6 puta (n*(n-1)/2).
#train podaci se kopiraju i filtriraju na 6 razlicitih skupova podataka. Takodje svaki skup sada ima pripadnost klasi 0 ili 1.
dataTrainPrvaDruga = deepcopy(dataTrain)
filter!(row -> row.grupa != 3, dataTrainPrvaDruga)
filter!(row -> row.grupa != 4, dataTrainPrvaDruga)
dataTrainPrvaDruga .= ifelse.(dataTrainPrvaDruga .== 1, 0.0, dataTrainPrvaDruga)
dataTrainPrvaDruga .= ifelse.(dataTrainPrvaDruga .== 2, 1.0, dataTrainPrvaDruga)

dataTrainPrvaTreca = deepcopy(dataTrain)
filter!(row -> row.grupa != 2, dataTrainPrvaTreca)
filter!(row -> row.grupa != 4, dataTrainPrvaTreca)
dataTrainPrvaTreca .= ifelse.(dataTrainPrvaTreca .== 1, 0.0, dataTrainPrvaTreca)
dataTrainPrvaTreca .= ifelse.(dataTrainPrvaTreca .== 3, 1.0, dataTrainPrvaTreca)

dataTrainPrvaCetvrta = deepcopy(dataTrain)
filter!(row -> row.grupa != 2, dataTrainPrvaCetvrta)
filter!(row -> row.grupa != 3, dataTrainPrvaCetvrta)
dataTrainPrvaCetvrta .= ifelse.(dataTrainPrvaCetvrta .== 1, 0.0, dataTrainPrvaCetvrta)
dataTrainPrvaCetvrta .= ifelse.(dataTrainPrvaCetvrta .== 4, 1.0, dataTrainPrvaCetvrta)

dataTrainDrugaTreca = deepcopy(dataTrain)
filter!(row -> row.grupa != 1, dataTrainDrugaTreca)
filter!(row -> row.grupa != 4, dataTrainDrugaTreca)
dataTrainDrugaTreca .= ifelse.(dataTrainDrugaTreca .== 2, 0.0, dataTrainDrugaTreca)
dataTrainDrugaTreca .= ifelse.(dataTrainDrugaTreca .== 3, 1.0, dataTrainDrugaTreca)

dataTrainDrugaCetvrta = deepcopy(dataTrain)
filter!(row -> row.grupa != 1, dataTrainDrugaCetvrta)
filter!(row -> row.grupa != 3, dataTrainDrugaCetvrta)
dataTrainDrugaCetvrta .= ifelse.(dataTrainDrugaCetvrta .== 2, 0.0, dataTrainDrugaCetvrta)
dataTrainDrugaCetvrta .= ifelse.(dataTrainDrugaCetvrta .== 4, 1.0, dataTrainDrugaCetvrta)

dataTrainTrecaCetvrta = deepcopy(dataTrain)
filter!(row -> row.grupa != 1, dataTrainTrecaCetvrta)
filter!(row -> row.grupa != 2, dataTrainTrecaCetvrta)
dataTrainTrecaCetvrta .= ifelse.(dataTrainTrecaCetvrta .== 3, 0.0, dataTrainTrecaCetvrta)
dataTrainTrecaCetvrta .= ifelse.(dataTrainTrecaCetvrta .== 4, 1.0, dataTrainTrecaCetvrta)

#prave se 6 regresora za svaku grupu podataka
logisticRegressorPrvaDruga = glm(fm, dataTrainPrvaDruga, Binomial(), ProbitLink())
logisticRegressorPrvaTreca = glm(fm, dataTrainPrvaTreca, Binomial(), ProbitLink())
logisticRegressorPrvaCetvrta = glm(fm, dataTrainPrvaCetvrta, Binomial(), ProbitLink())
logisticRegressorDrugaTreca = glm(fm, dataTrainDrugaTreca, Binomial(), ProbitLink())
logisticRegressorDrugaCetvrta = glm(fm, dataTrainDrugaCetvrta, Binomial(), ProbitLink())
logisticRegressorTrecaCetvrta = glm(fm, dataTrainTrecaCetvrta, Binomial(), ProbitLink())

#koristimo kreirane regresore i predvidjamo podatke nad istim test skupom za svaki regresor
dataPredictTestPrvaDruga = predict(logisticRegressorPrvaDruga, dataTest)
dataPredictTestPrvaTreca = predict(logisticRegressorPrvaTreca, dataTest)
dataPredictTestPrvaCetvrta = predict(logisticRegressorPrvaCetvrta, dataTest)
dataPredictTestDrugaTreca = predict(logisticRegressorDrugaTreca, dataTest)
dataPredictTestDrugaCetvrta = predict(logisticRegressorDrugaCetvrta, dataTest)
dataPredictTestTrecaCetvrta = predict(logisticRegressorTrecaCetvrta, dataTest)

#ovaj skup nam sluzi za odredjivanje klase 
dataPredictTestClass = repeat(0:0, length(dataTest.grupa))

#koristicemo niz score kako bi pratili koliko puta se podatak plasirao u datu klasu i biracemo klasu sa najvise plasiranja kao klasu tog podatka
for i in 1:length(dataPredictTestClass)

	score = [0, 0, 0, 0]

	#prva i druga klasa
	if dataPredictTestPrvaDruga[i] < 0.5
		score[1] += 1
	else
		score[2] += 1
	end

	#prva i treca klasa
	if dataPredictTestPrvaTreca[i] < 0.5
		score[1] += 1
	else
		score[3] += 1
	end

	#prva i cetvrta klasa
	if dataPredictTestPrvaCetvrta[i] < 0.5
		score[1] += 1
	else
		score[4] += 1
	end

	#druga i treca klasa
	if dataPredictTestDrugaTreca[i] < 0.5
		score[2] += 1
	else
		score[3] += 1
	end

	#druga i cetvrta klasa
	if dataPredictTestDrugaCetvrta[i] < 0.5
		score[2] += 1
	else
		score[4] += 1
	end
	
	#treca i cetvrta klasa
	if dataPredictTestTrecaCetvrta[i] < 0.5
		score[3] += 1
	else
		score[4] += 1
	end

	global dataPredictTestClass[i] = argmax(score)

end

#racunanje kvaliteta klasifikacije
FPTest1 = 0
FNTest1 = 0
TPTest1 = 0
TNTest1 = 0

FPTest2 = 0
FNTest2 = 0
TPTest2 = 0
TNTest2 = 0

FPTest3 = 0
FNTest3 = 0
TPTest3 = 0
TNTest3 = 0

FPTest4 = 0
FNTest4 = 0
TPTest4 = 0
TNTest4 = 0

for i in 1:length(dataPredictTestClass)
	
    #za prvu klasu
	if dataPredictTestClass[i] == 1 && dataTest.grupa[i] == 1
		global TPTest1 += 1
	elseif dataPredictTestClass[i] == 1 && dataTest.grupa[i] != 1
		global FPTest1 += 1
	elseif dataPredictTestClass[i] != 1 && dataTest.grupa[i] == 1
		global FNTest1 += 1
    else
        global TNTest1 += 1
    end

	#za drugu klasu
	if dataPredictTestClass[i] == 2 && dataTest.grupa[i] == 2
		global TPTest2 += 1
	elseif dataPredictTestClass[i] == 2 && dataTest.grupa[i] != 2
		global FPTest2 += 1
	elseif dataPredictTestClass[i] != 2 && dataTest.grupa[i] == 2
		global FNTest2 += 1
    else
        global TNTest2 += 1
    end

	#za trecu klasu
	if dataPredictTestClass[i] == 3 && dataTest.grupa[i] == 3
		global TPTest3 += 1
	elseif dataPredictTestClass[i] == 3 && dataTest.grupa[i] != 3
		global FPTest3 += 1
	elseif dataPredictTestClass[i] != 3 && dataTest.grupa[i] == 3
		global FNTest3 += 1
    else
        global TNTest3 += 1
    end

	#za cetvrtu klasu
	if dataPredictTestClass[i] == 4 && dataTest.grupa[i] == 4
		global TPTest4 += 1
	elseif dataPredictTestClass[i] == 4 && dataTest.grupa[i] != 4
		global FPTest4 += 1
	elseif dataPredictTestClass[i] != 4 && dataTest.grupa[i] == 4
		global FNTest4 += 1
	else
		global TNTest4 += 1
	end
end

TNSum = TNTest1 + TNTest2 + TNTest3 + TNTest4
TPSum = TPTest1 + TPTest2 + TPTest3 + TPTest4
FNSum = FNTest1 + FNTest2 + FNTest3 + FNTest4
FPSum = FPTest1 + FPTest2 + FPTest3 + FPTest4

# Accuracy (preciznost) = (TP+TN)/(TP+TN+FP+FN)
accuracyTest = (TPSum + TNSum) / (TPSum + TNSum + FPSum + FNSum)

spec1 = TNTest1/(TNTest1+FPTest1) 
spec2 = TNTest2/(TNTest2+FPTest2)
spec3 = TNTest3/(TNTest3+FPTest3)
spec4 = TNTest4/(TNTest4+FPTest4)

specificityTest = TNSum/(TNSum+FPSum)

#ispis preciznosti celog sistema i specificnosti za svaku od klasa
println("Preciznost za test skup je $(round(accuracyTest; digits = 3))")
println("\nSpecificnost za sve klase: ")
println("1: $spec1")
println("2: $spec2")
println("3: $spec3")
println("4: $spec4")
println("\nSpecificnost za ceo test skup:")
println(round(specificityTest; digits = 3))


