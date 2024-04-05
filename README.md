# PropensityScoreMatching

## Fonctionnalités

- PropensityScoreMatcher: Cette classe permet de réaliser une correspondance entre les individus traités et les individus témoins sur la base des scores de propension. Elle offre des méthodes pour ajuster un modèle de régression logistique, vérifier graphiquement la cohérence des scores de propension, effectuer la correspondance et générer un rapport sur les résultats de la correspondance.

- PropensityScoreMatcherTS: Une extension de la classe PropensityScoreMatcher conçue pour les données de séries temporelles. Elle prend en compte les variations temporelles pour effectuer la correspondance entre les individus traités et les individus témoins. Elle offre des fonctionnalités similaires à PropensityScoreMatcher, mais adaptées aux données de séries temporelles.

 - DoubleDiff: Cette classe permet de générer des graphiques et d'estimer les intervalles de confiance par bootstrap pour les données de séries temporelles avec deux ou plusieurs groupes. Elle facilite la comparaison des moyennes et des intervalles de confiance des différentes groupes au fil du temps.
