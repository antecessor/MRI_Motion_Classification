# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Modelling Algorithms
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels


class DataAnalysisUtils:

    def plotCorrelationHeatMap(self, dataFrame):
        f = mpl.figure(figsize=(16, 16))
        ax = f.add_subplot(1, 1, 1)

        sns.heatmap(dataFrame.corr(), annot=True,
                    linewidths=.03, fmt='.1f', ax=ax)
        mpl.show()
        return f

    def plotUnivariateDistribution(self, df, column):
        sns.set(rc={'figure.figsize': (9, 7)})
        sns.distplot(df[column])

    def plotBivariateDistribution(self, df, var, target, **kwargs):
        row = kwargs.get('row', None)
        col = kwargs.get('col', None)
        facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
        facet.map(sns.kdeplot, var, shade=True)
        facet.set(xlim=(0, df[var].max()))
        facet.add_legend()

    def plotPairAllFeatureByHue(self, dataFrame, hue):
        sns.pairplot(dataFrame, hue=hue)
        mpl.savefig('relation.png')

    def plotJoint(self, dataFrame, columnX, columnY, size=6):
        sns.jointplot(x=columnX, y=columnY, data=dataFrame,
                      size=size, kind='kde', color='#800000', space=0)

    def plotScatterAllFeatures(self, dataFrame):
        scatter_matrix = pd.plotting.scatter_matrix(dataFrame, figsize=(14, 14))
        for ax in scatter_matrix.ravel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=15, rotation=0)
            ax.set_ylabel(ax.get_ylabel(), fontsize=15, rotation=90)

        mpl.savefig('relation.png')
        mpl.figure()

    def plotColumnVersusColumn(self, dataFrame, columnX, columnY, kind='scatter', color='red'):
        'kind : line|scatter'
        dataFrame.plot(kind=kind, x=columnX, y=columnY, color=color)
        mpl.xlabel(columnX)
        mpl.ylabel(columnY)

    def plotHistograms(self, df, variables, n_rows, n_cols):
        fig = mpl.figure(figsize=(16, 12))
        for i, var_name in enumerate(variables):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            df[var_name].hist(bins=13, ax=ax)
            # + ' ' + var_name ) #var_name+" Distribution")
            ax.set_title(var_name)
            # ax.set_xticklabels([], visible=False)
            # ax.set_yticklabels([], visible=False)
        fig.tight_layout()  # Improves appearance a bit.

        mpl.show()
        return fig

    def plotCategories(self, df, cat, target, **kwargs):
        row = kwargs.get('row', None)
        col = kwargs.get('col', None)
        facet = sns.FacetGrid(df, row=row, col=col)
        facet.map(sns.barplot, cat, target)
        facet.add_legend()

    def describeMore(self, df):
        var = []
        l = []
        t = []
        for x in df:
            var.append(x)
            l.append(len(pd.value_counts(df[x])))
            t.append(df[x].dtypes)
        levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
        levels.sort_values(by='Levels', inplace=True)
        return levels

    def plotModelVarImp(self, model, X, y):
        imp = pd.DataFrame(
            model.feature_importances_,
            columns=['Importance'],
            index=X.columns
        )
        imp = imp.sort_values(['Importance'], ascending=True)
        imp[: 10].plot(kind='barh')
        print(model.score(X, y))

    def boxPlotOnTwoColumn(self, dataFrame, column, columnBy):
        f, ax = mpl.subplots(figsize=(12, 8))
        fig = sns.boxplot(x=column, y=columnBy, data=dataFrame)

    def convertColumnsToRow(self, dataFrame, id, columns):
        return pd.melt(frame=dataFrame, id_vars=id, value_vars=columns)

    def convertRowsToColumn(self, melted, id, columns, values):
        return melted.pivot(index=id, columns=columns, values=values)

    def concatDataFramesFromRow(self, dataFrame1, dataFrame2):
        return pd.concat([dataFrame1, dataFrame2], axis=0, ignore_index=True)

    def concatDataFramesFromColumn(self, dataFrame1, dataFrame2):
        return pd.concat([dataFrame1, dataFrame2], axis=1)

    def checkMissingData(self, df):
        flag = df.isna().sum().any()
        if flag == True:
            total = df.isnull().sum()
            percent = (df.isnull().sum()) / (df.isnull().count() * 100)
            output = pd.concat([total, percent], axis=1,
                               keys=['Total', 'Percent'])
            data_type = []
            # written by MJ Bahmani
            for col in df.columns:
                dtype = str(df[col].dtype)
                data_type.append(dtype)
            output['Types'] = data_type
            return (np.transpose(output))
        else:
            return (False)

    def randomForestClassifierGridSearch(self, X_train, y_train, estimator=[4, 6, 9], depth=[2, 3, 5, 10], sampleSplit=[2, 3, 5], sampleLeaf=[1, 5, 8]):
        rfc = RandomForestClassifier()

        # Choose some parameter combinations to try
        parameters = {'n_estimators': estimator,
                      'max_features': ['log2', 'sqrt', 'auto'],
                      'criterion': ['entropy', 'gini'],
                      'max_depth': depth,
                      'min_samples_split': sampleSplit,
                      'min_samples_leaf': sampleLeaf
                      }

        # Type of scoring used to compare parameter combinations
        acc_scorer = make_scorer(accuracy_score)

        # Run the grid search
        grid_obj = GridSearchCV(rfc, parameters, scoring=acc_scorer)
        grid_obj = grid_obj.fit(X_train, y_train)

        # Set the clf to the best combination of parameters
        rfc = grid_obj.best_estimator_

        # Fit the best algorithm to the data.
        rfc.fit(X_train, y_train)

        return rfc

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=mpl.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = mpl.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        mpl.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def plotGeoData(self, lati, longi, df, column, dtick=10):
        scl = [0, "rgb(150,0,90)"], [0.125, "rgb(0, 0, 200)"], [0.25, "rgb(0, 25, 255)"], [0.375, "rgb(0, 152, 255)"], [0.5, "rgb(44, 255, 150)"], [0.625, "rgb(151, 255, 0)"],
        [0.75, "rgb(255, 234, 0)"], [0.875, "rgb(255, 111, 0)"], [1, "rgb(255, 0, 0)"]
        fig = go.Figure(data=go.Scattergeo(
            lon=longi,
            lat=lati,
            text=df[column],
            marker=dict(
                color=df[column],
                colorscale=scl,
                reversescale=True,
                opacity=0.7,
                colorbar=dict(
                    titleside="right",
                    outlinecolor="rgba(68, 68, 68, 0)",
                    title=column,
                    dtick=dtick)
            )
        ))

        fig.update_layout(
            geo=dict(

                showland=True,
                landcolor="rgb(212, 212, 212)",
                subunitcolor="rgb(140, 255, 0)",
                countrycolor="rgb(150, 255, 100)",
                showlakes=True,
                showcoastlines=True,
                lakecolor="rgb(0, 150, 255)",

                resolution=50,

                lonaxis=dict(
                    showgrid=True,
                    gridwidth=0.5,
                    range=[-180.0, -55.0],
                    dtick=5
                ),
                lataxis=dict(
                    showgrid=True,
                    gridwidth=0.5,
                    range=[45, 85],
                    dtick=5
                )
            ),
        )
        fig.show()
        nameOfFile = column + '_MAP.png'
        fig.write_image(nameOfFile)
