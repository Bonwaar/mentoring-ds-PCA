import React, { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const ACPPresentation = () => {
  const [currentSlide, setCurrentSlide] = useState(0);

  const slides = [
    {
      title: "Analyse en Composantes Principales (ACP)",
      subtitle: "Une méthode simple pour comprendre vos données",
      content: null
    },
    {
      title: "Qu'est-ce que l'ACP ?",
      content: (
        <div className="space-y-6">
          <p className="text-xl">L'ACP est une technique qui permet de <span className="font-semibold text-blue-600">simplifier des données complexes</span></p>
          
          <div className="bg-blue-50 p-6 rounded-lg">
            <p className="text-lg mb-4">🎯 <span className="font-semibold">Objectif principal :</span></p>
            <p className="text-lg">Transformer beaucoup de variables corrélées en quelques nouvelles variables indépendantes appelées <span className="font-semibold">"composantes principales"</span></p>
          </div>
          
          <div className="grid grid-cols-2 gap-4 mt-6">
            <div className="bg-red-50 p-4 rounded-lg">
              <p className="font-semibold text-red-700 mb-2">Avant l'ACP</p>
              <p>10 variables difficiles à visualiser et comprendre</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <p className="font-semibold text-green-700 mb-2">Après l'ACP</p>
              <p>2-3 composantes qui résument l'essentiel</p>
            </div>
          </div>
        </div>
      )
    },
    {
      title: "Comment ça marche ? (en simple)",
      content: (
        <div className="space-y-5">
          <div className="bg-purple-50 p-5 rounded-lg">
            <p className="font-semibold text-purple-700 mb-3">Étape 1 : Standardisation</p>
            <p>On met toutes les variables à la même échelle (moyenne = 0, écart-type = 1)</p>
          </div>
          
          <div className="bg-blue-50 p-5 rounded-lg">
            <p className="font-semibold text-blue-700 mb-3">Étape 2 : Recherche des directions</p>
            <p>On cherche les directions où les données varient le plus</p>
            <p className="text-sm mt-2 italic">C'est comme trouver le meilleur angle de vue pour une photo</p>
          </div>
          
          <div className="bg-green-50 p-5 rounded-lg">
            <p className="font-semibold text-green-700 mb-3">Étape 3 : Création des composantes</p>
            <p>Chaque composante est une combinaison des variables originales</p>
            <p className="text-sm mt-2">CP1 = 0.3×Var1 + 0.5×Var2 + 0.2×Var3 + ...</p>
          </div>
          
          <div className="bg-yellow-50 p-5 rounded-lg">
            <p className="font-semibold text-yellow-700 mb-3">Étape 4 : Sélection</p>
            <p>On garde seulement les composantes qui expliquent le plus de variance</p>
          </div>
        </div>
      )
    },
    {
      title: "Le Graphe des Individus",
      content: (
        <div className="space-y-5">
          <p className="text-lg font-semibold text-gray-700">Projection des observations sur les 2 premières composantes principales</p>
          
          <div className="bg-blue-50 p-5 rounded-lg">
            <p className="font-semibold text-blue-700 mb-3">📊 Ce qu'on y voit :</p>
            <ul className="space-y-2 ml-4">
              <li>• Chaque point = une observation (un individu)</li>
              <li>• Les points proches = individus similaires</li>
              <li>• Les points éloignés = individus différents</li>
            </ul>
          </div>
          
          <div className="bg-green-50 p-5 rounded-lg">
            <p className="font-semibold text-green-700 mb-3">🔍 Interprétations possibles :</p>
            <ul className="space-y-2 ml-4">
              <li>• <span className="font-semibold">Clusters</span> : Des groupes naturels dans les données</li>
              <li>• <span className="font-semibold">Outliers</span> : Points isolés = observations atypiques</li>
              <li>• <span className="font-semibold">Structure</span> : Forme générale de la distribution</li>
            </ul>
          </div>
        </div>
      )
    },
    {
      title: "Le Graphe des Variables",
      content: (
        <div className="space-y-5">
          <p className="text-lg font-semibold text-gray-700">Cercle de corrélation des variables</p>
          
          <div className="bg-purple-50 p-5 rounded-lg">
            <p className="font-semibold text-purple-700 mb-3">📊 Ce qu'on y voit :</p>
            <ul className="space-y-2 ml-4">
              <li>• Chaque flèche = une variable originale</li>
              <li>• Plus la flèche est longue = mieux représentée</li>
              <li>• Position sur le cercle = contribution aux composantes</li>
            </ul>
          </div>
          
          <div className="bg-orange-50 p-5 rounded-lg">
            <p className="font-semibold text-orange-700 mb-3">🔍 Interprétations :</p>
            <ul className="space-y-2 ml-4">
              <li>• <span className="font-semibold">Angle faible entre flèches</span> : Variables corrélées positivement</li>
              <li>• <span className="font-semibold">Angle à 180°</span> : Variables corrélées négativement</li>
              <li>• <span className="font-semibold">Angle à 90°</span> : Variables indépendantes</li>
            </ul>
          </div>
        </div>
      )
    },
    {
      title: "Applications Pratiques de l'ACP",
      content: (
        <div className="space-y-4">
          <div className="bg-red-50 p-5 rounded-lg">
            <p className="font-semibold text-red-700 mb-3">1️⃣ Résoudre la Multicolinéarité</p>
            <p className="mb-2"><span className="font-semibold">Problème :</span> En régression linéaire, des variables corrélées créent de l'instabilité</p>
            <p><span className="font-semibold">Solution ACP :</span> Les composantes principales sont orthogonales (indépendantes) → plus de corrélation !</p>
          </div>
          
          <div className="bg-green-50 p-5 rounded-lg">
            <p className="font-semibold text-green-700 mb-3">2️⃣ Réduire la Dimensionnalité</p>
            <p className="mb-2"><span className="font-semibold">Problème :</span> Trop de features → temps de calcul élevé, overfitting</p>
            <p><span className="font-semibold">Solution ACP :</span> Passer de 10 variables à 3 composantes en gardant 80% de l'information</p>
          </div>
          
          <div className="bg-blue-50 p-5 rounded-lg">
            <p className="font-semibold text-blue-700 mb-3">3️⃣ Améliorer les Modèles ML</p>
            <p>• Moins de features = entraînement plus rapide</p>
            <p>• Moins de bruit = meilleure généralisation</p>
            <p>• Visualisation facilitée en 2D/3D</p>
          </div>
        </div>
      )
    },
    {
      title: "Lab 1 : Données Fortement Corrélées",
      content: (
        <div className="space-y-5">
          <div className="bg-blue-100 p-5 rounded-lg">
            <p className="font-semibold text-blue-800 mb-3">🎯 Configuration</p>
            <ul className="space-y-2 ml-4">
              <li>• 10 variables quantitatives</li>
              <li>• Forte corrélation entre les variables</li>
              <li>• 200 observations</li>
            </ul>
          </div>
          
          <div className="bg-green-100 p-5 rounded-lg">
            <p className="font-semibold text-green-800 mb-3">📈 Résultats attendus</p>
            <ul className="space-y-2 ml-4">
              <li>• 3-4 composantes principales suffisent</li>
              <li>• ~80% de variance expliquée avec peu de composantes</li>
              <li>• Variables regroupées sur le cercle de corrélation</li>
              <li>• Clusters potentiels sur le graphe des individus</li>
            </ul>
          </div>
          
          <p className="text-center text-lg font-semibold text-gray-600 mt-4">
            ➡️ Démonstration pratique dans le notebook
          </p>
        </div>
      )
    },
    {
      title: "Lab 2 : Données Faiblement Corrélées",
      content: (
        <div className="space-y-5">
          <div className="bg-orange-100 p-5 rounded-lg">
            <p className="font-semibold text-orange-800 mb-3">🎯 Configuration</p>
            <ul className="space-y-2 ml-4">
              <li>• 10 variables quantitatives</li>
              <li>• Très faible corrélation entre les variables</li>
              <li>• 200 observations</li>
            </ul>
          </div>
          
          <div className="bg-red-100 p-5 rounded-lg">
            <p className="font-semibold text-red-800 mb-3">📉 Résultats attendus</p>
            <ul className="space-y-2 ml-4">
              <li>• Beaucoup de composantes nécessaires</li>
              <li>• Variance répartie uniformément</li>
              <li>• Variables dispersées sur le cercle (angles à 90°)</li>
              <li>• Pas de structure claire dans les données</li>
            </ul>
          </div>
          
          <div className="bg-purple-100 p-5 rounded-lg">
            <p className="font-semibold text-purple-800 mb-3">💡 Conclusion</p>
            <p>L'ACP est moins efficace quand les variables sont indépendantes !</p>
          </div>
        </div>
      )
    },
    {
      title: "Points Clés à Retenir",
      content: (
        <div className="space-y-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="font-semibold text-blue-700">✓ L'ACP simplifie des données complexes</p>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg">
            <p className="font-semibold text-green-700">✓ Elle fonctionne mieux avec des variables corrélées</p>
          </div>
          
          <div className="bg-purple-50 p-4 rounded-lg">
            <p className="font-semibold text-purple-700">✓ Les graphes permettent d'explorer visuellement les données</p>
          </div>
          
          <div className="bg-orange-50 p-4 rounded-lg">
            <p className="font-semibold text-orange-700">✓ Utile pour la régression et le machine learning</p>
          </div>
          
          <div className="bg-red-50 p-4 rounded-lg">
            <p className="font-semibold text-red-700">✓ Toujours vérifier le % de variance expliquée</p>
          </div>
          
          <p className="text-center text-2xl font-bold text-gray-700 mt-8">
            Questions ?
          </p>
        </div>
      )
    }
  ];

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-100 p-8">
      <div className="max-w-5xl mx-auto bg-white rounded-2xl shadow-2xl overflow-hidden">
        {/* Slide Content */}
        <div className="p-12 min-h-[600px]">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            {slides[currentSlide].title}
          </h1>
          {slides[currentSlide].subtitle && (
            <p className="text-2xl text-gray-600 mb-8">{slides[currentSlide].subtitle}</p>
          )}
          <div className="text-gray-700">
            {slides[currentSlide].content}
          </div>
        </div>

        {/* Navigation */}
        <div className="bg-gray-100 p-6 flex items-center justify-between">
          <button
            onClick={prevSlide}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            disabled={currentSlide === 0}
          >
            <ChevronLeft size={20} />
            Précédent
          </button>

          <div className="flex gap-2">
            {slides.map((_, index) => (
              <div
                key={index}
                className={`h-2 w-2 rounded-full transition-all ${
                  index === currentSlide ? 'bg-blue-600 w-8' : 'bg-gray-400'
                }`}
              />
            ))}
          </div>

          <button
            onClick={nextSlide}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            disabled={currentSlide === slides.length - 1}
          >
            Suivant
            <ChevronRight size={20} />
          </button>
        </div>

        {/* Slide Counter */}
        <div className="text-center py-2 bg-gray-50 text-gray-600 text-sm">
          Slide {currentSlide + 1} / {slides.length}
        </div>
      </div>
    </div>
  );
};

export default ACPPresentation;