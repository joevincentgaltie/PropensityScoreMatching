# PropensityScoreMatching

L'appariement par score de propension est une méthode d'ébaluation des politiques publiques. Elle permet de comparer les réalisations d’une variable, observées dans deux échantillons distincts, en ajustant des différences de composition en termes de caractéristiques observables entre ces échantillons. Cet ajustment vise à réduire le biais engendré par les effets de sélection liés aux caractéristiques observables des bénéficiaires 1 pour estimer l’effet causal d’un traitement.

## Fonctionnalités

- PropensityScoreMatcher: Cette classe permet de réaliser une correspondance entre les individus traités et les individus témoins sur la base des scores de propension. Elle offre des méthodes pour ajuster un modèle de régression logistique, vérifier graphiquement la cohérence des scores de propension, effectuer la correspondance et générer un rapport sur les résultats de la correspondance.

- PropensityScoreMatcherTS: Une extension de la classe PropensityScoreMatcher conçue pour les données de séries temporelles. Elle prend en compte les variations temporelles pour effectuer la correspondance entre les individus traités et les individus témoins. Elle offre des fonctionnalités similaires à PropensityScoreMatcher, mais adaptées aux données de séries temporelles.

 - DoubleDiff: Cette classe permet de générer des graphiques et d'estimer les intervalles de confiance par bootstrap pour les données de séries temporelles avec deux ou plusieurs groupes. Elle facilite la comparaison des moyennes et des intervalles de confiance des différentes groupes au fil du temps.

### Installation


### Exemple d'utilisation 

Le notebook example.ipynb offre un exemple d'utilisation dans le cas de séries temporelles et non temporelles. 

### Contributions

Les contributions sont les bienvenues ! 


