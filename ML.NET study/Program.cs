using Microsoft.ML;
using ML.NET_study.Entities;

//path with train data
string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
//path with test data
string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
//path with trained model
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

var mlContext = new MLContext(seed: 0);

var model = Train(mlContext, _trainDataPath);

Evaluate(mlContext, model);

TestSinglePrediction(mlContext, model);

ITransformer Train(MLContext mlContext, string trainDataPath)
{
    var dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(trainDataPath, hasHeader: true, separatorChar: ','); //Load train data
    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount") //Transform the column that will be predict in label
                                       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId")) //Transform categorical data values into numbers to use this data like features
                                       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode")) //Transform categorical data values into numbers to use this data like features
                                       .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType")) //Transform categorical data values into numbers to use this data like features
                                       .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded")) //Combine all the feature columns to the Features column (it's the column that will be used in prediction)
                                       .Append(mlContext.Regression.Trainers.FastTree()); //Use regression machine learning task (FastTree)

    var model = pipeline.Fit(dataView); //Trains model by transforming the dataset and applying the training

    return model;
}

void Evaluate(MLContext mlContext, ITransformer model)
{
    var dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ','); //Load test data
    var predictions = model.Transform(dataView); //The transform method makes predictions
    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score"); //Evaluate the method and returns metrics

    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");

    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}"); //Takes values between 0 and 1. Closer to 1 is better

    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}"); //The lower it is, the better the model is
    Console.WriteLine();
}

void TestSinglePrediction(MLContext mlContext, ITransformer model)
{
    //This creates a function to predict single examples. It is not thread-safe. For producton environments we should use PredictionEnginePool
    var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
    
    var taxiTripSample = new TaxiTrip()
    {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripTime = 1140,
        TripDistance = 3.75f,
        PaymentType = "CRD",
        FareAmount = 0 // To predict. Actual/Observed = 15.5
    };

    var prediction = predictionFunction.Predict(taxiTripSample);

    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"                  Test Single Prediction                              ");
    Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
    Console.WriteLine($"**********************************************************************");
}