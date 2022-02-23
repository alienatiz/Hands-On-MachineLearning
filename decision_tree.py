import graphviz
from graphviz import Source
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz  # conda install python-graphviz

from settings import *

# 6.1 Train decision tree and visualize the model
iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

export_graphviz(tree_clf)
print(export_graphviz(tree_clf))

export_graphviz(
        tree_clf,
        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
)

Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))

with open("images/decision_trees/iris_tree.dot", encoding='UTF-8') as f:
        dot_graph = f.read()
        dot = graphviz.Source(dot_graph)
        dot.format = 'png'
        dot.render(filename='iris_tree', directory='images/decision_trees', cleanup=True)

# 6.2 Predict
# 6.3 Estimate the probability of the class
tree_clf.predict_proba([[5, 1.5]])
print("tree_clf.predict_proba([[5, 1.5]]): \n", tree_clf.predict_proba([[5, 1.5]]))
tree_clf.predict([[5, 1.5]])
print("tree_clf.predict([[5, 1.5]]): \n", tree_clf.predict([[5, 1.5]]))
