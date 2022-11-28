import pandas
import pandas as pds
import numpy as npy


data = pds.read_csv('train_old.csv')

# Nombre
print(data)
mort = 0
survivant = 0

# Recherche le nombre de survivant
nb_survive = data.query(expr="Survived == 1")

# Recherche le nombre de mort
nb_mort = data.query(expr="Survived == 0")

print(len(nb_survive), "Survivant")
print(len(nb_mort), "Mort")

# recherche les éléments null
print(len(data.isna().query(expr="Survived == True")), "Valeur NUlL pour la colonne Survived")

# Calcul de l'age moyen des passagers
print(len(data.isna().query(expr="Age == True")), "Valeur NUlL pour la colonne age")
# Check valeur null
mean_age = round(data['Age'].mean())
print("Age moyen :", mean_age)

# On remplace la valeur par la valeur moyenne
# inplace permet de modifier de permanante les données
print("***** Replace valeur  ********")
print(data['Age'].head(15))
data.Age.fillna(mean_age, inplace=True)
print(data['Age'].head(15))

print("***** Nombre de mort par classe ********")
# Afficher le nb de personne dans chacune des classes et les morts par classe
nb_classe_1 = data.query(expr="Pclass == 1")
nb_classe_1_mort = data.query(expr="Pclass == 1 and Survived == 0")
nb_classe_2 = data.query(expr="Pclass == 2")
nb_classe_2_mort = data.query(expr="Pclass == 2 and Survived == 0")
nb_classe_3 = data.query(expr="Pclass == 3")
nb_classe_3_mort = data.query(expr="Pclass == 3 and Survived == 0")
print(len(nb_classe_1), "passagers en classe 1 et ", len(nb_classe_1_mort), " mort")
print(len(nb_classe_2), "passagers en classe 2 et ", len(nb_classe_2_mort), " mort")
print(len(nb_classe_3), "passagers en classe 3 et ", len(nb_classe_3_mort), " mort")

# Afficher le %de personnnes qui ont survécu pour chacune des classes
print("***** Pourcentage de mort par classe ********")
#Classe 3
pourcentage_mort_classe_3 = round((len(nb_classe_3_mort)/len(nb_classe_3))*100)
print(pourcentage_mort_classe_3, '% de mort dans la classe 3')
#Classe 2
pourcentage_mort_classe_2 = round((len(nb_classe_2_mort)/len(nb_classe_2))*100)
print(pourcentage_mort_classe_2, '% de mort dans la classe 2')

#Classe 1
pourcentage_mort_classe_1 = round((len(nb_classe_1_mort)/len(nb_classe_1))*100)
print(pourcentage_mort_classe_1, '% de mort dans la classe 1')

# Remplacer "Male" et "female" par 1 et 0
# car les biblio pandas et numpy ne travaille que sur des valeurs numériques

print("***** Changemenet Male par 1  ********")
data.replace('male', 1, inplace=True)
print("***** Changemenet female par 0  ********")
data.replace('female', 0, inplace=True)

print("********** DROP COLOMNE ************")
# Supprimer les colonnes qui seront pas utilisés contenant des valeurs textuelles ou ne semblait
data.drop(['Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
print(data.head(15))

#Lancement des algorithme
#Modèle Knn (K-Neighbor classifier = voisin les plus proches)
#build knn model for best score