using Microsoft.ML;
using ML.NET_study.Entities;

class Program
{
    static void Main(string[] args)
    {
        string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");

        var mlContext = new MLContext(seed: 0);
        var dataView = LoadData(mlContext, _dataPath);

        var trainTestSplit = mlContext.Data.TrainTestSplit(dataView);
        var trainData = trainTestSplit.TrainSet;
        var testData = trainTestSplit.TestSet;

        var model = TrainModel(mlContext, trainData);

        var predictions = model.Transform(testData);

        var metrics = mlContext.Clustering.Evaluate(predictions);

        Console.WriteLine("Metrics:");
        Console.WriteLine(metrics.DaviesBouldinIndex);
        Console.WriteLine(metrics.AverageDistance);
        Console.WriteLine(metrics.NormalizedMutualInformation);

        IDataView LoadData(MLContext mlContext, string dataPath)
        {
            return mlContext.Data.LoadFromTextFile<IrisData>(dataPath, hasHeader: false, separatorChar: ',');
        }

        ITransformer TrainModel(MLContext mlContext, IDataView trainData)
        {
            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

            return pipeline.Fit(trainData);
        }
    }
}
