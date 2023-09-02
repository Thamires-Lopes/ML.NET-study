using Microsoft.ML;
using Microsoft.ML.Data;
using ML.NET_study.Entities;

string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

var mlContext = new MLContext(seed: 0);

var dataView = LoadData(mlContext, _dataPath);

var model = TrainModel(mlContext, dataView);

SaveModel(model, dataView, _modelPath);

var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

IDataView LoadData(MLContext mlContext, string dataPath)
{
    return mlContext.Data.LoadFromTextFile<IrisData>(dataPath, hasHeader: false, separatorChar: ',');
}

ITransformer TrainModel(MLContext mlContext, IDataView dataView)
{
    string featuresColumnName = "Features";
    var pipeline = mlContext.Transforms
        .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
        .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

   return pipeline.Fit(dataView);
}

void SaveModel(ITransformer model, IDataView dataView, string modelPath)
{
    using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
    {
        mlContext.Model.Save(model, dataView.Schema, fileStream);
    }
}

