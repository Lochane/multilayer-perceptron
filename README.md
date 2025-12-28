# multilayer-perceptron

# Concepts fondamentaux d’un réseau de neurones

Cette section explique les notions essentielles pour comprendre et coder un réseau de neurones, que ce soit **from scratch** ou avec des frameworks comme PyTorch ou TensorFlow.

---

## 1. Poids (Weights)

* **Définition :** Coefficient qui mesure l’importance d’une entrée (feature) pour un neurone.
* **Rôle :** Plus le poids est grand, plus la feature influence la sortie du neurone.

* **Formule mathématique :**
z = w1*x1 + w2*x2 + ... + wn*xn + b


---

## 2. Biais (Bias)

* **Définition :** Constante ajoutée au neurone pour décaler la fonction d’activation.
* **Rôle :** Permet au neurone de produire une sortie non nulle même si toutes les entrées sont nulles.

* **Formule :**
z = w1*x1 + w2*x2 + ... + wn*xn + b

---

## 3. Fonction d’activation (Activation Function)

* **Définition :** Transforme la sortie linéaire (z = w \cdot x + b) en une valeur non-linéaire.
* **Importance :** Sans activation, plusieurs couches équivaudraient à une seule couche (incapable de modéliser des relations complexes).

* **Exemples courants :**

| Fonction | Formule                                    | Intervalle               |
| -------- | ------------------------------------------ | ------------------------ |
| Sigmoïde | `sigma(z) = 1 / (1 + exp(-z))`             | (0,1)                    |
| ReLU     | `ReLU(z) = max(0, z)`                      | [0,∞)                    |
| Softmax  | `softmax(z_i) = exp(z_i) / sum_j exp(z_j)` | somme = 1 (probabilités) |


---

## 4. Gradient

* **Définition :** Dérivée de la loss par rapport aux poids ou biais.
* **Rôle :** Indique **comment ajuster les poids et biais pour réduire l’erreur**.

* **Formule pour un neurone sigmoïde avec loss L :**

dL/dw_i = (a - y) * sigma'(z) * x_i
dL/db   = (a - y) * sigma'(z)

or 

sigma'(z) = sigma(z) * (1 - sigma(z))


---

## 5. Concepts mathématiques essentiels

1. **Multiplication matricielle / vecteurs**
   a = f(Wx + b)

* (W) = matrice des poids
* (b) = vecteur des biais
* (f) = fonction d’activation

2. **Fonction de loss**

* Mesure l’erreur du modèle
* Exemples :

  * Classification binaire : cross-entropy
  * Classification multi-classes : categorical cross-entropy
  * Régression : mean squared error

3. **Gradient descent**

* Met à jour les poids pour minimiser la loss :
* w = w - lr * dL/dw

* lr = learning rate

4. **Backpropagation**

* Calcul des gradients couche par couche pour ajuster tous les poids et biais

5. **Non-linéarité**

* Essentielle pour permettre aux réseaux profonds de modéliser des relations complexes

---

### 💡 Résumé mnémotechnique

* **Poids** = combien chaque entrée compte
* **Biais** = où placer la limite de décision
* **Activation** = transformation non linéaire de la sortie
* **Gradient** = direction pour corriger les erreurs

---

