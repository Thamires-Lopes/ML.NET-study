using BenchmarkDotNet.Attributes;
using Microsoft.ML;
using ML.NET_study.Entities;

namespace ML.NET_study.Algorithms
{
    [MemoryDiagnoser]
    public class KMeans
    {
        string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Dry_Bean_Dataset.xlsx");

        [Benchmark]
        public void Run()
        {
            var mlContext = new MLContext(seed: 0);
            var dataView = LoadData(mlContext, _dataPath);

            string[] featuresColumnsNames = { "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation", "Eccentricity",
                                          "ConvexArea", "Eccentricity", "EquivDiameter", "Extent", "Solidity", "Roundness", "Compactness",
                                          "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"};

            var model = TrainModel(mlContext, dataView, featuresColumnsNames, 5);

            var predictions = model.Transform(dataView);

            Evalute(mlContext, predictions);
        }

        private static IDataView LoadData(MLContext mlContext, string dataPath)
        {
            return mlContext.Data.LoadFromTextFile<BeanData>(dataPath, hasHeader: true, separatorChar: ',');
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
            //Console.WriteLine($"Normalized Mutual Information: {metrics.NormalizedMutualInformation}"); //Only calculates if label column is provided
        }

    }
}
