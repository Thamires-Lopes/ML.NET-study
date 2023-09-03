using Microsoft.ML;
using ML.NET_study.Entities;

class Program
{
    static void Main()
    {
        string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");

        var mlContext = new MLContext(seed: 0);
        var dataView = LoadData(mlContext, _dataPath);
        var (trainData, testData) = SplitData(mlContext, dataView);

        string[] featuresColumnsNames = { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" };

        var model = TrainModel(mlContext, trainData, featuresColumnsNames, 3);

        var predictions = model.Transform(testData);

        Evalute(mlContext, predictions);
    }

    private static IDataView LoadData(MLContext mlContext, string dataPath)
    {
        return mlContext.Data.LoadFromTextFile<IrisData>(dataPath, hasHeader: false, separatorChar: ',');
    }

    private static (IDataView trainData, IDataView testData) SplitData(MLContext mlContext, IDataView data)
    {
        var trainTestSplit = mlContext.Data.TrainTestSplit(data);
        var trainData = trainTestSplit.TrainSet;
        var testData = trainTestSplit.TestSet;

        return (trainData, testData);
    }

    private static ITransformer TrainModel(MLContext mlContext, IDataView trainData, string[] features, int numberCluster)
    {
        string featuresColumnName = "Features";
        var pipeline = mlContext.Transforms
            .Concatenate(featuresColumnName, features)
            .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: numberCluster));

        return pipeline.Fit(trainData);
    }

    private static void Evalute(MLContext mlContext, IDataView predictions)
    {
        var metrics = mlContext.Clustering.Evaluate(predictions);

        Console.WriteLine("Metrics:");
        Console.WriteLine($"Davies-Bouldin Index: {metrics.DaviesBouldinIndex}");
        Console.WriteLine($"Average Distance: {metrics.AverageDistance}");
        Console.WriteLine($"Normalized Mutual Information: {metrics.NormalizedMutualInformation}");
    }
}
