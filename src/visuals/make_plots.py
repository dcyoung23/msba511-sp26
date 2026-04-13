import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from pandas.plotting import parallel_coordinates
from sklearn.metrics import silhouette_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.decomposition import PCA


def create_elbow_chart(range_n_clusters, params, data):
    inertia = []   
    for n_clusters in range_n_clusters: 
        kmeans = KMeans(n_clusters=n_clusters, 
                        **params).fit(data) 
        # Average SS per cluster
        inertia.append(kmeans.inertia_ / n_clusters) 
    inertias = pd.DataFrame({'n_clusters': range_n_clusters, 'inertia': inertia}) 
    ax = inertias.plot(x='n_clusters', y='inertia')
    plt.xlabel('Number of clusters (k)') 
    plt.ylabel('Average Within-Cluster Squared Distances') 
    plt.ylim((0, 1.1 * inertias.inertia.max())) 
    ax.legend().set_visible(False)
    return inertias


def plot_silhouette_scores(range_n_clusters, params, data):
    silhouette_scores = []  
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, 
                        **params).fit(data) 
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)       
    plt.plot(range_n_clusters, silhouette_scores)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel("Avg Silhouette Score")


def create_profile_chart(params, data, ax):
    kmeans = KMeans(**params).fit(data) 
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns)
    centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
    parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', 
                         linewidth=5, ax=ax)
    plt.xlim(-0.5,len(data.columns)-0.5)
    plt.legend(loc='best')


def visualize_clusters_pca(params, data):
    kmeans = KMeans(**params).fit(data) 
    cluster_assignments = kmeans.fit_predict(data)
    colors = cm.Set1(np.linspace(0, 1, params['n_clusters']))
    cmap = matplotlib.colors.ListedColormap(colors)
    pca = PCA(n_components=2)
    pc1, pc2 = zip(*pca.fit_transform(data))
    plt.scatter(pc1, pc2, c=cluster_assignments.tolist(), cmap=cmap)
    plt.xlabel('PC1')
    plt.ylabel('PC2')


def plot_confusion_matrix(clf, X, y, ax, title):
    ConfusionMatrixDisplay.from_estimator(clf, X, y, cmap=plt.cm.Blues, colorbar=False, ax=ax)
    plt.title(title)
    plt.tight_layout()


def plot_feature_importances(features, importances, ax, title, top_n=None):
    importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
    }).sort_values('Importance', ascending=True)
    if top_n:
        importances_df = importances_df.head(top_n)
    ax.barh(importances_df['Feature'], importances_df['Importance'])
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')


def plot_metric_by_threshold_cv(
    pipeline,
    X,
    y,
    metric,
    thresholds=None,
    cv=5,
    metric_name='Customer Metric',
    plot_title=None
):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    if plot_title is None:
        plot_title = f"{metric_name} by Probability Threshold (CV={cv})"

    def evaluate_metric(metric_obj, y_true, y_pred):
        # Plain metric function
        if hasattr(metric_obj, "__name__"):
            return metric_obj(y_true, y_pred)

        # sklearn scorer object from make_scorer
        elif hasattr(metric_obj, "_score_func"):
            kwargs = getattr(metric_obj, "_kwargs", {})
            return metric_obj._score_func(y_true, y_pred, **kwargs)

        else:
            raise TypeError(
                "metric must be either a plain metric function or a scorer created with make_scorer()."
            )

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)

    oof_proba = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv_strategy,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    scores = []

    for threshold in thresholds:
        y_pred = (oof_proba >= threshold).astype(int)
        score = evaluate_metric(metric, y, y_pred)
        scores.append(score)

    results_df = pd.DataFrame({
        "threshold": thresholds,
        "score": scores
    })

    best_idx = results_df["score"].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]
    best_score = results_df.loc[best_idx, "score"]

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best {metric_name}: {best_score:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df["threshold"], results_df["score"], marker="o", markersize=3)
    plt.axvline(best_threshold, linestyle="--", alpha=0.7,
                label=f"Best threshold = {best_threshold:.2f}")
    plt.xlabel("Probability Threshold")
    plt.ylabel(metric_name)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    return results_df


