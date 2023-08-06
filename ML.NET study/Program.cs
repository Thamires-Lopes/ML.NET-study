using Microsoft.ML;
using ML.NET_study.Entities;
using static Microsoft.ML.DataOperationsCatalog;

var _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");

var mlContext = new MLContext();

var splitDataView = LoadData(mlContext);

var model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

Evaluate(mlContext, model, splitDataView.TestSet);

UseModelWithSingleItem(mlContext, model);

UseModelWithBatchItems(mlContext, model);

TrainTestData LoadData(MLContext mlContext)
{
    var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
    var splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

    return splitDataView;
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text
                        .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();

    return model;
}

void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    var predictions = model.Transform(splitTestSet);

    var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");
}

void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    var predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

    var sampleStatement = new SentimentData
    {
        SentimentText = "This was a very bad steak"
    };

    var resultPrediction = predictionFunction.Predict(sampleStatement);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}

void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    var sentiments = new[]
    {
        new SentimentData
        {
            SentimentText = "This was a horrible meal"
        },
        new SentimentData
        {
            SentimentText = "I love this spaghetti."
        }
    };

    var batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

    var predictions = model.Transform(batchComments);

    var predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine();

    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

    foreach (var prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }

    Console.WriteLine("=============== End of predictions ===============");
}